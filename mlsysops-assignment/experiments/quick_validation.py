import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
Quick validation: 5 smoke tests, no downloads required.
Run from ANY directory:  python experiments/quick_validation.py
"""
import sys, time, torch, torch.nn as nn

print("=" * 60)
print("Compressed-DDP Quick Validation")
print("=" * 60)

PASS="  [PASS]"; FAIL="  [FAIL]"; results=[]

def check(name, fn):
    t0=time.perf_counter()
    try:
        fn()
        print(f"{PASS} {name}  ({(time.perf_counter()-t0)*1000:.0f} ms)")
        results.append(True)
    except Exception as e:
        print(f"{FAIL} {name}"); print(f"         {e}"); results.append(False)

def test_imports():
    from src.compression import get_compressor
    from src.error_feedback import ErrorFeedbackBuffer
    from src.communication import DistributedBackend
    from src.models import get_model
    from src.utils import detect_device, get_platform_info
    info = get_platform_info()
    print(f"         torch={info['torch']}  cuda={info['cuda_available']}")
check("Module imports", test_imports)

def test_compression():
    from src.compression import get_compressor
    comp = get_compressor(ratio=0.01, device="cpu")
    t = torch.randn(10_000)
    v, idx, shape = comp.compress(t)
    out = comp.decompress(v, idx, shape)
    assert out.shape == t.shape, f"shape mismatch"
    assert len(v) == 100, f"expected k=100 got {len(v)}"
    print(f"         {comp.stats}")
check("CPU Top-K compression", test_compression)

def test_error_feedback():
    from src.error_feedback import ErrorFeedbackBuffer
    buf = ErrorFeedbackBuffer()
    g   = torch.randn(500)
    assert torch.allclose(buf.compensate("w", g), g)
    buf.update("w", g, torch.zeros(500))
    assert buf.error_norm("w") > 0
check("Error feedback buffer", test_error_feedback)

def test_model():
    from src.models import get_model
    m = get_model("simple_cnn")
    out = m(torch.randn(4, 1, 28, 28))
    assert out.shape == (4, 10)
    print(f"         params={sum(p.numel() for p in m.parameters()):,}")
check("SimpleCNN forward pass", test_model)

def test_training_step():
    from src.models import get_model
    from src.compression import get_compressor
    from src.error_feedback import ErrorFeedbackBuffer
    from src.communication import DistributedBackend
    import torch.optim as optim
    m    = get_model("simple_cnn")
    comp = get_compressor(ratio=0.01, device="cpu")
    ef   = ErrorFeedbackBuffer()
    back = DistributedBackend(compressor=comp, error_buffer=ef)
    opt  = optim.SGD(m.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    x,y  = torch.randn(8,1,28,28), torch.randint(0,10,(8,))
    opt.zero_grad()
    loss = crit(m(x), y)
    loss.backward()
    back.allreduce_gradients(m.named_parameters())
    opt.step()
    print(f"         loss={loss.item():.4f}")
check("Compressed training step", test_training_step)

if torch.cuda.is_available():
    def test_gpu():
        from src.compression import get_compressor
        comp = get_compressor(ratio=0.01, device="cuda")
        t = torch.randn(10_000).cuda()
        out = comp.decompress(*comp.compress(t))
        assert out.device.type=="cuda" and out.shape==t.shape
    check("GPU Top-K compression", test_gpu)
else:
    print("  [SKIP] GPU test – no CUDA device")

print()
passed=sum(results); total=len(results)
print(f"Results: {passed}/{total} checks passed")
if passed < total:
    print("Some checks failed.")
    sys.exit(1)
else:
    print("All checks passed! Project is ready.")
