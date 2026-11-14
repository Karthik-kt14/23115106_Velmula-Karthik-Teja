import argparse,csv,torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn import functional as F
from .dataset import PetDataset
from .models import build_model,count_parameters
def epoch(m,dl,opt,dev,train):
    m.train(train); ls=[]; inter=den=0
    for x,y,v in dl:
        x,y,v=x.to(dev),y.to(dev),v.to(dev)
        with torch.set_grad_enabled(train):
            z=m(x); raw=F.binary_cross_entropy_with_logits(z,y,reduction="none"); loss=(raw*v).sum()/v.sum().clamp_min(1)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        ls.append(loss.item()); q=(torch.sigmoid(z)>.5)&v; t=(y>.5)&v; inter+=(q&t).sum().item(); den+=q.sum().item()+t.sum().item()
    return sum(ls)/len(ls),2*inter/(den+1e-8)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--model",choices=["unet","dilated_unet"],required=True); p.add_argument("--output",required=True); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=8); a=p.parse_args()
    torch.manual_seed(42); dev="cuda" if torch.cuda.is_available() else "cpu"; m=build_model(a.model).to(dev); opt=torch.optim.Adam(m.parameters(),lr=.001)
    tr=DataLoader(PetDataset(a.data,"train",True),batch_size=a.batch_size,shuffle=True,num_workers=2); va=DataLoader(PetDataset(a.data,"val"),batch_size=a.batch_size)
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); best=-1; stale=0; rows=[]
    for e in range(1,a.epochs+1):
        tl,td=epoch(m,tr,opt,dev,True); vl,vd=epoch(m,va,opt,dev,False); row=[e,tl,td,vl,vd]; rows.append(row); print(row)
        if vd>best: best=vd; stale=0; torch.save(m.state_dict(),out/"best.pt")
        else: stale+=1
        if stale>=5: break
    with open(out/"training_log.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","train_loss","train_dice","val_loss","val_dice"]); w.writerows(rows)
    print("device",dev,"parameters",count_parameters(m),"best_val_dice",best)
if __name__=="__main__": main()
