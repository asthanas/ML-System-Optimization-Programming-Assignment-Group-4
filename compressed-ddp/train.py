import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

"""
train.py – Main training script.
Usage: python train.py --model simple_cnn --dataset mnist --epochs 5
       python train.py --model simple_cnn --dataset mnist --epochs 5 --compress --ratio 0.01
"""
import argparse, logging, time
from pathlib import Path
import torch, torch.nn as nn, torch.optim as optim
from src.models import get_model
from src.data import get_dataloaders
from src.compression import get_compressor
from src.error_feedback import ErrorFeedbackBuffer
from src.communication import DistributedBackend, setup_distributed, cleanup_distributed
from src.metrics import MetricsTracker
from src.utils import detect_device, get_platform_info, save_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")

def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",    default="simple_cnn", choices=["simple_cnn","resnet18","resnet50"])
    p.add_argument("--dataset",  default="mnist",      choices=["mnist","cifar10"])
    p.add_argument("--data-dir", default="data/")
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--epochs",    type=int,   default=10)
    p.add_argument("--batch-size",type=int,   default=64)
    p.add_argument("--lr",        type=float, default=0.01)
    p.add_argument("--momentum",  type=float, default=0.9)
    p.add_argument("--wd",        type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--compress",  action="store_true")
    p.add_argument("--ratio",     type=float, default=0.01)
    p.add_argument("--no-error-feedback", action="store_true")
    p.add_argument("--backend",   default="gloo", choices=["gloo","nccl"])
    p.add_argument("--world-size",type=int, default=1)
    p.add_argument("--rank",      type=int, default=0)
    p.add_argument("--device",    default="auto")
    p.add_argument("--checkpoint-dir", default="checkpoints/")
    p.add_argument("--log-dir",        default="runs/")
    p.add_argument("--resume",         default=None)
    p.add_argument("--seed", type=int, default=42)
    return p

def train_epoch(model, loader, optimizer, criterion, backend, device, epoch, tracker):
    model.train()
    total_loss = correct = total = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        backend.allreduce_gradients(model.named_parameters())
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
        if i % 100 == 0:
            tracker.update("train/loss", loss.item(), epoch*len(loader)+i)
    acc = 100.0 * correct / total
    log.info("Epoch %d  train loss=%.4f  acc=%.2f%%", epoch, total_loss/total, acc)
    return acc

@torch.no_grad()
def validate(model, loader, criterion, device, epoch, tracker):
    model.eval()
    total_loss = correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out  = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
    acc = 100.0 * correct / total
    tracker.update("val/acc", acc, epoch)
    log.info("Epoch %d  val   loss=%.4f  acc=%.2f%%", epoch, total_loss/total, acc)
    return acc

def main():
    args   = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = detect_device(args.device)
    log.info("Platform: %s", get_platform_info())
    log.info("Device:   %s", device)
    setup_distributed(backend=args.backend, rank=args.rank, world_size=args.world_size)
    model = get_model(args.model, args.num_classes).to(device)
    log.info("Model %s: %.2fM params", args.model,
             sum(p.numel() for p in model.parameters())/1e6)
    compressor   = get_compressor(ratio=args.ratio, device=str(device)) if args.compress else None
    error_buffer = ErrorFeedbackBuffer(device=str(device)) if (args.compress and not args.no_error_feedback) else None
    backend = DistributedBackend(compressor=compressor, error_buffer=error_buffer,
                                 world_size=args.world_size, rank=args.rank)
    train_loader, val_loader = get_dataloaders(
        args.dataset, args.batch_size, args.data_dir, args.rank, args.world_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    run_name  = f"{args.model}_{args.dataset}{'_compressed' if args.compress else ''}"
    tracker   = MetricsTracker(log_dir=str(Path(args.log_dir)/run_name))
    best_acc  = 0.0
    backend.broadcast_parameters(model)
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, optimizer, criterion, backend, device, epoch, tracker)
        val_acc = validate(model, val_loader, criterion, device, epoch, tracker)
        scheduler.step()
        if args.rank == 0:
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            save_checkpoint(
                {"epoch":epoch,"model_state_dict":model.state_dict(),
                 "optimizer_state_dict":optimizer.state_dict(),"best_acc":best_acc},
                Path(args.checkpoint_dir)/run_name/f"epoch_{epoch:03d}.pt",
                is_best=is_best)
    log.info("Done. Best val acc: %.2f%%", best_acc)
    tracker.save_csv(Path(args.log_dir)/f"{run_name}_metrics.csv")
    tracker.close()
    cleanup_distributed()

if __name__ == "__main__":
    main()
