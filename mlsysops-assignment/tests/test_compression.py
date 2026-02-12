import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest
import torch
from src.compression import get_compressor, TopKCompressorCPU, TopKCompressorGPU

def roundtrip(comp, tensor):
    v, idx, shape = comp.compress(tensor)
    return comp.decompress(v, idx, shape)

class TestTopKCPU:
    def test_topk_selects_largest_magnitudes(self):
        comp = TopKCompressorCPU(ratio=0.1)
        t = torch.arange(100, dtype=torch.float32) - 50
        v, idx, _ = comp.compress(t)
        assert len(v) == 10
        min_selected = v.abs().min().item()
        assert (t.abs() >= min_selected - 1e-6).sum() >= 10

    def test_compress_decompress_shape(self):
        comp = TopKCompressorCPU(ratio=0.05)
        t = torch.randn(4, 8, 8)
        assert roundtrip(comp, t).shape == t.shape

    def test_compression_ratio(self):
        comp = TopKCompressorCPU(ratio=0.02)
        v, idx, _ = comp.compress(torch.randn(1000))
        assert len(v) == max(1, int(1000 * 0.02))

    def test_zero_tensor(self):
        comp = TopKCompressorCPU(ratio=0.1)
        t = torch.zeros(100)
        out = roundtrip(comp, t)
        assert out.shape == t.shape
        assert torch.all(out == 0)

    def test_single_spike(self):
        comp = TopKCompressorCPU(ratio=0.01)
        t = torch.zeros(1000); t[500] = 999.0
        v, idx, shape = comp.compress(t)
        assert 500 in idx.tolist()
        out = comp.decompress(v, idx, shape)
        assert abs(out[500].item() - 999.0) < 1e-4

    def test_negative_signs_preserved(self):
        comp = TopKCompressorCPU(ratio=0.1)
        t = -torch.arange(1, 101, dtype=torch.float32)
        v, _, _ = comp.compress(t)
        assert (v < 0).all()

    def test_multidimensional(self):
        comp = TopKCompressorCPU(ratio=0.05)
        t = torch.randn(3, 32, 32)
        assert roundtrip(comp, t).shape == (3, 32, 32)

    def test_stats_tracking(self):
        comp = TopKCompressorCPU(ratio=0.1)
        for _ in range(5):
            roundtrip(comp, torch.randn(500))
        assert comp.stats.total_compress_calls == 5
        assert comp.stats.compression_ratio > 1.0

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError): TopKCompressorCPU(ratio=0.0)
        with pytest.raises(ValueError): TopKCompressorCPU(ratio=1.5)

    def test_factory_cpu(self):
        from src.compression import get_compressor
        assert isinstance(get_compressor(ratio=0.01, device="cpu"), TopKCompressorCPU)

    def test_full_ratio(self):
        comp = TopKCompressorCPU(ratio=1.0)
        t = torch.randn(50)
        v, _, _ = comp.compress(t)
        assert len(v) == 50

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_compressor(self):
        comp = TopKCompressorGPU(ratio=0.1)
        t = torch.randn(500).cuda()
        out = comp.decompress(*comp.compress(t))
        assert out.device.type == "cuda" and out.shape == t.shape
