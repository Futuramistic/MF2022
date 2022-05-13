from torch import nn
import torch
import torchvision.transforms

class BasicSRModel(nn.Module):
    def __init__(self, num_blocks=10):
        super(BasicSRModel, self).__init__()
        self.first = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),padding=1)
        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1))
            self.blocks.append(nn.LeakyReLU())
        # Output
        self.outpout = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=(3,3),padding=1)
    
    def forward(self,x): 
        x = torchvision.transforms.Resize(size=(int(x.size(2)*2),int(x.size(3)*2)))(x)
        x = self.first(x)
        for layer in self.blocks:
            x = layer(x)
        x = self.outpout(x)
        return x

class BasicSRModelSkip(nn.Module):
    def __init__(self, num_blocks=10):
        super(BasicSRModelSkip, self).__init__()
        self.first = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(3,3),padding=1)
        # Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1))
            self.blocks.append(nn.LeakyReLU())
        # Output
        self.outpout = nn.Conv2d(in_channels=64,out_channels=3,kernel_size=(3,3),padding=1)
    
    def forward(self,x):
        x = torchvision.transforms.Resize(size=(int(x.size(2)*2),int(x.size(3)*2)))(x)
        pre = self.first(x)
        x = pre
        for layer in self.blocks:
            x = layer(x)
        x = x+pre
        x = self.outpout(x)
        return x