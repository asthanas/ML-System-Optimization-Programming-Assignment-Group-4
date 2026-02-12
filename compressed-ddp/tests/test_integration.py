import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch, torch.nn as nn, torch.optim as optim
from src.models import get_model
from src.data import get_dataloaders
from src.compression import get_compressor
from src.error_feedback import ErrorFeedbackBuffer
from src.communication import DistributedBackend

def _train(model, loader, device, comp=None, ef=None, steps=50):
    back = DistributedBackend(compressor=comp, error_buffer=ef)
    opt  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    model.train(); losses=[]
    for i,(x,y) in enumerate(loader):
        if i>=steps: break
        x,y=x.to(device),y.to(device)
        opt.zero_grad()
        loss=crit(model(x),y); loss.backward()
        back.allreduce_gradients(model.named_parameters())
        opt.step(); losses.append(loss.item())
    return losses

class TestIntegration:
    def test_baseline_converges(self):
        torch.manual_seed(42)
        m = get_model("simple_cnn")
        dl,_ = get_dataloaders("mnist", batch_size=32)
        ls = _train(m, dl, torch.device("cpu"), steps=50)
        assert sum(ls[:10])/10 > sum(ls[-10:])/10, "loss should decrease"

    def test_compressed_converges(self):
        torch.manual_seed(42)
        m = get_model("simple_cnn")
        dl,_ = get_dataloaders("mnist", batch_size=32)
        ls = _train(m,dl,torch.device("cpu"),
                    get_compressor(ratio=0.01,device="cpu"),
                    ErrorFeedbackBuffer(), steps=50)
        assert sum(ls[:10])/10 > sum(ls[-10:])/10

    def test_accuracy_comparable(self):
        device = torch.device("cpu")
        def run(compress):
            torch.manual_seed(42)
            m = get_model("simple_cnn")
            dl,vl = get_dataloaders("mnist", batch_size=64)
            c  = get_compressor(ratio=0.01,device="cpu") if compress else None
            ef = ErrorFeedbackBuffer() if compress else None
            _train(m,dl,device,c,ef,steps=100)
            m.eval(); correct=total=0
            with torch.no_grad():
                for x,y in vl:
                    correct+=(m(x).argmax(1)==y).sum().item(); total+=y.size(0)
                    if total>=1000: break
            return 100.0*correct/total
        assert abs(run(False)-run(True)) < 15.0
