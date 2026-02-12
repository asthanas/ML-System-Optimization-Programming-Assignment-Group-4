import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Compression throughput benchmark - FIXED for Python 3.13"""
import csv, time, torch
from pathlib import Path
from src.compression import get_compressor
from src.utils import detect_device, get_platform_info

SIZES  = [1_000_000, 10_000_000, 25_000_000]
RATIOS = [0.001, 0.01, 0.1]
TRIALS = 5

def main():
    device = detect_device()
    print("="*70)
    print("Compression Benchmark (Fixed for Python 3.13)")
    print("="*70)
    print(f"Device: {device}  |  {get_platform_info()}")

    results = []
    for numel in SIZES:
        for ratio in RATIOS:
            comp = get_compressor(ratio=ratio, device=str(device))
            t    = torch.randn(numel, device=device)
            for _ in range(2): comp.compress(t)   # warm-up
            comp.reset_stats()
            times=[]
            for _ in range(TRIALS):
                t0=time.perf_counter()
                v,idx,shape=comp.compress(t)
                comp.decompress(v,idx,shape)
                times.append((time.perf_counter()-t0)*1000)
            k   = comp._k(numel)
            bw  = round((1 - k*(4+8)/(numel*4))*100, 1)
            row = {"numel":numel,"ratio":ratio,"k":k,"device":str(device),
                   "mean_ms":round(sum(times)/TRIALS,2),"bw_saving_pct":bw}
            results.append(row)
            print(f"  {numel/1e6:.0f}M  ratio={ratio}  {row['mean_ms']}ms  saved={bw}%")

    out = Path("experiments/results"); out.mkdir(parents=True, exist_ok=True)
    with open(out/"compression_benchmark.csv","w",newline="") as f:
        w2=csv.DictWriter(f,fieldnames=results[0].keys()); w2.writeheader(); w2.writerows(results)
    print(f"\n✅ Saved: experiments/results/compression_benchmark.csv")

if __name__ == "__main__":
    main()
