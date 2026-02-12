import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""Theoretical scalability model. Outputs: experiments/results/scalability_analysis.csv"""
import csv
from pathlib import Path

PARAMS_M  = 11.0; DTYPE_B=4; COMPUTE_T=0.5
NETWORKS  = {"1_Gbps":1e9/8, "10_Gbps":10e9/8, "NVLink":600e9/8}
WORKERS   = [1,2,4,8,16,32]
RATIOS    = {"baseline":1.0, "1pct_compression":0.01}
results   = []

for bw_name,bw in NETWORKS.items():
    for ratio_name,ratio in RATIOS.items():
        for P in WORKERS:
            grad_b = PARAMS_M*1e6*DTYPE_B*ratio
            comm_t = 2*grad_b/bw if P>1 else 0.0
            total_t= COMPUTE_T+comm_t
            eff    = (COMPUTE_T/P)/total_t if P>1 else 1.0
            results.append({"network":bw_name,"ratio":ratio_name,"workers":P,
                            "comm_s":round(comm_t,4),"total_s":round(total_t,4),
                            "efficiency":round(eff,4),"speedup":round(COMPUTE_T/total_t*P,2)})

out=Path("experiments/results"); out.mkdir(parents=True,exist_ok=True)
with open(out/"scalability_analysis.csv","w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=results[0].keys()); wr.writeheader(); wr.writerows(results)
print("Saved: experiments/results/scalability_analysis.csv")
print("\nE(8) summary:")
for r in results:
    if r["workers"]==8: print(f"  {r['network']:12s} {r['ratio']:25s} E={r['efficiency']:.1%}")

try:
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,(bw_name,_) in zip(axes,NETWORKS.items()):
        for rn,c in zip(RATIOS,["steelblue","coral"]):
            sub=[r for r in results if r["network"]==bw_name and r["ratio"]==rn]
            ax.plot([r["workers"] for r in sub],[r["efficiency"] for r in sub],
                    marker="o",label=rn,color=c)
        ax.set_title(bw_name); ax.set_xlabel("Workers"); ax.set_ylabel("Efficiency")
        ax.set_ylim(0,1.05); ax.axhline(0.7,color="red",linestyle="--",alpha=0.5)
        ax.legend(fontsize=8); ax.set_xticks(WORKERS)
    plt.tight_layout()
    plt.savefig(out/"scalability_analysis.png",dpi=150)
    print("Plot saved.")
except ImportError:
    pass
