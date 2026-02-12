import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Training speed & accuracy benchmark - FIXED for Python 3.13 + macOS"""
import csv, time, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from src.models import get_model
from src.data import get_dataloaders
from src.compression import get_compressor
from src.error_feedback import ErrorFeedbackBuffer
from src.communication import DistributedBackend
from src.utils import detect_device

device = detect_device()
EPOCHS=3; BATCH=64

CONFIGS=[
    {"name":"baseline",    "compress":False},
    {"name":"topk_0.1",    "compress":True,"ratio":0.1},
    {"name":"topk_0.01",   "compress":True,"ratio":0.01},
    {"name":"topk_0.001",  "compress":True,"ratio":0.001},
]

def run(cfg):
    print(f"\nRunning {cfg['name']}...")
    torch.manual_seed(42)
    model = get_model("simple_cnn").to(device)

    # FIX: Use num_workers=0 to avoid multiprocessing issues on macOS
    tl,vl = get_dataloaders("mnist", batch_size=BATCH, num_workers=0)

    opt   = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    comp  = get_compressor(ratio=cfg.get("ratio",0.01),device=str(device)) if cfg["compress"] else None
    ef    = ErrorFeedbackBuffer() if cfg["compress"] else None
    back  = DistributedBackend(compressor=comp,error_buffer=ef)
    times,accs=[],[]

    for epoch in range(EPOCHS):
        t0=time.perf_counter(); model.train()
        for x,y in tl:
            x,y=x.to(device),y.to(device); opt.zero_grad()
            crit(model(x),y).backward()
            back.allreduce_gradients(model.named_parameters()); opt.step()
        times.append(time.perf_counter()-t0)

        model.eval(); c=n=0
        with torch.no_grad():
            for x,y in vl:
                x,y=x.to(device),y.to(device)
                c+=(model(x).argmax(1)==y).sum().item(); n+=y.size(0)
        accs.append(100.0*c/n)
        print(f"  [{cfg['name']}] epoch {epoch+1}/{EPOCHS}  t={times[-1]:.1f}s  acc={accs[-1]:.1f}%")

    return {"name":cfg["name"],"mean_epoch_s":round(sum(times)/len(times),2),"final_val_acc":round(accs[-1],2)}

def main():
    print("="*70)
    print("Training Benchmark (Fixed for Python 3.13)")
    print("="*70)

    results=[]
    for cfg in CONFIGS:
        results.append(run(cfg))

    base_t=results[0]["mean_epoch_s"]
    for r in results: r["speedup"]=round(base_t/r["mean_epoch_s"],2)

    out=Path("experiments/results"); out.mkdir(parents=True,exist_ok=True)
    with open(out/"training_benchmark.csv","w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=results[0].keys()); wr.writeheader(); wr.writerows(results)
    print(f"\n✅ Saved: experiments/results/training_benchmark.csv")

    print("\nSummary:")
    for r in results:
        print(f"  {r['name']:15s}  epoch={r['mean_epoch_s']}s  acc={r['final_val_acc']}%  speedup={r['speedup']}x")

if __name__ == "__main__":
    main()
