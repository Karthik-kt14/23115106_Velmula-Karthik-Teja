import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, a,b,d=1):
        super().__init__(); p=d
        self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=p,dilation=d,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),
                               nn.Conv2d(b,b,3,padding=p,dilation=d,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self,bottleneck_dilation=1):
        super().__init__(); self.pool=nn.MaxPool2d(2)
        self.e1=ConvBlock(3,32); self.e2=ConvBlock(32,64); self.e3=ConvBlock(64,128); self.e4=ConvBlock(128,256)
        self.b=ConvBlock(256,512,bottleneck_dilation)
        self.u4=nn.ConvTranspose2d(512,256,2,2); self.d4=ConvBlock(512,256)
        self.u3=nn.ConvTranspose2d(256,128,2,2); self.d3=ConvBlock(256,128)
        self.u2=nn.ConvTranspose2d(128,64,2,2); self.d2=ConvBlock(128,64)
        self.u1=nn.ConvTranspose2d(64,32,2,2); self.d1=ConvBlock(64,32)
        self.head=nn.Conv2d(32,1,1)
    def forward(self,x):
        a=self.e1(x); b=self.e2(self.pool(a)); c=self.e3(self.pool(b)); d=self.e4(self.pool(c)); z=self.b(self.pool(d))
        z=self.d4(torch.cat([self.u4(z),d],1)); z=self.d3(torch.cat([self.u3(z),c],1))
        z=self.d2(torch.cat([self.u2(z),b],1)); z=self.d1(torch.cat([self.u1(z),a],1))
        return self.head(z)

def build_model(name):
    if name=="unet": return UNet(1)
    if name=="dilated_unet": return UNet(2)
    raise ValueError(name)

def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
