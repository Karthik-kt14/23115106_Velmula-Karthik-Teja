from pathlib import Path
import numpy as np, torch, random
from torch.utils.data import Dataset
from PIL import Image
class PetDataset(Dataset):
    def __init__(self,root,split,augment=False):
        self.d=Path(root)/split; self.files=sorted((self.d/"images").glob("*.jpg")); self.augment=augment
        if not self.files: raise RuntimeError(f"No images: {self.d}")
    def __len__(self): return len(self.files)
    def __getitem__(self,k):
        p=self.files[k]; x=np.asarray(Image.open(p).convert("RGB"),dtype=np.float32)/255
        y=np.asarray(Image.open(self.d/"masks"/(p.stem+".png")),dtype=np.uint8)
        if self.augment and random.random()<.5: x=x[:,::-1].copy(); y=y[:,::-1].copy()
        return torch.tensor(x.transpose(2,0,1).copy()),torch.tensor((y==1)[None].astype(np.float32)),torch.tensor((y!=255)[None])
