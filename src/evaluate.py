import argparse,json,time,torch
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import PetDataset
from .models import build_model,count_parameters
def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--model",choices=["unet","dilated_unet"],required=True); p.add_argument("--output",required=True); a=p.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); m=build_model(a.model).to(dev); m.load_state_dict(torch.load(a.checkpoint,map_location=dev)); m.eval()
    dl=DataLoader(PetDataset(a.data,"test"),batch_size=1); sums={k:0. for k in ["iou","dice","precision","recall"]}; n=0
    with torch.no_grad():
        for x,y,v in dl:
            z=torch.sigmoid(m(x.to(dev)))>.5; y=y.to(dev)>.5; v=v.to(dev)
            tp=(z&y&v).sum().item(); fp=(z&~y&v).sum().item(); fn=(~z&y&v).sum().item()
            sums["iou"]+=tp/(tp+fp+fn+1e-8); sums["dice"]+=2*tp/(2*tp+fp+fn+1e-8); sums["precision"]+=tp/(tp+fp+1e-8); sums["recall"]+=tp/(tp+fn+1e-8); n+=1
    dummy=torch.randn(1,3,256,256,device=dev)
    with torch.no_grad():
        for _ in range(10):m(dummy)
        if dev.type=="cuda":torch.cuda.synchronize()
        ts=[]
        for _ in range(50):
            t=time.perf_counter();m(dummy)
            if dev.type=="cuda":torch.cuda.synchronize()
            ts.append((time.perf_counter()-t)*1000)
    result={"metrics":{k:v/n for k,v in sums.items()},"pet_iou_miou":sums["iou"]/n,"parameters":count_parameters(m),"latency_ms_mean":sum(ts)/len(ts),"device":str(dev),"latency_setup":"batch=1, 256x256, 10 warmup, 50 repeats"}
    Path(a.output).mkdir(parents=True,exist_ok=True); (Path(a.output)/"metrics.json").write_text(json.dumps(result,indent=2)); print(result)
if __name__=="__main__":main()
