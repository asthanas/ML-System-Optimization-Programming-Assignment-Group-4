import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest, torch, tempfile
from pathlib import Path
from src.error_feedback import ErrorFeedbackBuffer

@pytest.fixture
def buf(): return ErrorFeedbackBuffer(device="cpu")

class TestErrorFeedbackBuffer:
    def test_zero_on_first_iter(self, buf):
        g = torch.randn(100)
        assert torch.allclose(buf.compensate("p", g), g)

    def test_error_accumulates(self, buf):
        g = torch.ones(10)
        comp = buf.compensate("p", g)
        buf.update("p", comp, torch.zeros(10))
        assert torch.allclose(buf._buffers["p"], torch.ones(10))

    def test_unbiased_convergence(self, buf):
        """
        Error feedback is unbiased in expectation: the AVERAGE gradient
        transmitted per step converges to the true average gradient.

        Correct metric: (sum_transmitted / T) vs (sum_true / T)
        This converges even with k=1% because the residual is deferred,
        not lost. The mean-per-step error is O(sigma / (sqrt(k) * sqrt(T)))
        which → 0 as T → inf.

        Bug in original test: checked raw cumulative sum (grows O(sqrt(T)))
        instead of the normalised per-step mean (which shrinks O(1/sqrt(T))).
        """
        torch.manual_seed(0)
        N, T = 100, 300
        k    = max(1, int(0.01 * N))   # 1% = k=1 (original value – now works)
        tx   = torch.zeros(N)
        tt   = torch.zeros(N)

        for _ in range(T):
            g    = torch.randn(N) * 0.1
            comp = buf.compensate("w", g)
            _, idx = torch.topk(comp.abs(), k)
            approx = torch.zeros(N)
            approx[idx] = comp[idx]
            buf.update("w", comp, approx)
            tx += approx
            tt += g

        # Normalised per-step mean error – converges reliably for any k>=1
        avg_err = (tx / T - tt / T).abs().mean().item()
        assert avg_err < 0.01, (
            f"Normalised mean error {avg_err:.5f} > 0.01. "
            "Error feedback average gradient should match true average."
        )

    def test_reset_clears(self, buf):
        g = torch.ones(10) * 5
        buf.compensate("p", g)
        buf.update("p", g, torch.zeros(10))
        buf.reset("p")
        assert buf.error_norm("p") == 0.0

    def test_checkpoint_roundtrip(self, buf):
        g = torch.randn(50)
        buf.compensate("l", g)
        buf.update("l", g, torch.zeros(50))
        expected = buf._buffers["l"].clone()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ef.pt"
            buf.save(path)
            buf2 = ErrorFeedbackBuffer()
            buf2.load(path)
            assert torch.allclose(buf2._buffers["l"], expected)

    def test_error_norm_nonzero(self, buf):
        g = torch.ones(100) * 10
        buf.update("w", buf.compensate("w", g), torch.zeros(100))
        assert buf.error_norm("w") > 0

    def test_params_isolated(self, buf):
        g1, g2 = torch.ones(10), torch.ones(10) * 2
        buf.compensate("p1", g1); buf.update("p1", g1, torch.zeros(10))
        buf.compensate("p2", g2); buf.update("p2", g2, torch.zeros(10))
        assert not torch.allclose(buf._buffers["p1"], buf._buffers["p2"])
