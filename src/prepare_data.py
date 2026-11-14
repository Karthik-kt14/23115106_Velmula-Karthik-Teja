import argparse,tarfile,urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

URLS={"images.tar.gz":"https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
      "annotations.tar.gz":"https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"}
def extract(src,dst):
    with tarfile.open(src,"r:gz") as t: t.extractall(dst,filter="data")
def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="data/oxford_pet"); p.add_argument("--raw",default="data/raw"); a=p.parse_args()
    raw=Path(a.raw); raw.mkdir(parents=True,exist_ok=True)
    for n,u in URLS.items():
        if not (raw/n).exists(): print("Downloading",n); urllib.request.urlretrieve(u,raw/n)
    ex=raw/"extracted"; ex.mkdir(exist_ok=True)
    if not (ex/"images").exists(): extract(raw/"images.tar.gz",ex)
    if not (ex/"annotations").exists(): extract(raw/"annotations.tar.gz",ex)
    ann=ex/"annotations"; ims=ex/"images"; tri=ann/"trimaps"
    read=lambda f:[x.split()[0] for x in f.read_text().splitlines() if x.strip()]
    tv=read(ann/"trainval.txt"); te=read(ann/"test.txt"); tr,va=train_test_split(tv,test_size=.15,random_state=42)
    for split,ids in {"train":tr,"val":va,"test":te}.items():
        for sub in ["images","masks"]:(Path(a.output)/split/sub).mkdir(parents=True,exist_ok=True)
        for i in ids:
            ip=ims/f"{i}.jpg"; mp=tri/f"{i}.png"
            if not ip.exists() or not mp.exists(): raise FileNotFoundError(i)
            im=Image.open(ip).convert("RGB").resize((256,256),Image.Resampling.BILINEAR)
            m=np.array(Image.open(mp).resize((256,256),Image.Resampling.NEAREST))
            # foreground=1, background=0, boundary=255 (ignored)
            y=np.full(m.shape,255,dtype=np.uint8); y[m==1]=1; y[m==2]=0
            im.save(Path(a.output)/split/"images"/f"{i}.jpg",quality=95); Image.fromarray(y).save(Path(a.output)/split/"masks"/f"{i}.png")
        print(split,len(ids))
if __name__=="__main__": main()
