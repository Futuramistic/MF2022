from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import torch

class VGG19_54(nn.Module):
    def __init__(self):
        super(VGG19_54,self).__init__()
        self.model = nn.Sequential(*list(vgg19(pretrained=True).features.children())[:35])

    def forward(self,x):
        return self.model(x)

class DenseRB(nn.Module):

    def __init__(self, filters: int, res:float = 0.2):
        super(DenseRB, self).__init__()
        self.res = res

        def RB(in_channels, activation = True):
            layers = [nn.Conv2d(in_channels=in_channels,
                                out_channels=filters,
                                kernel_size=(3,3),
                                stride=(1,1),
                                padding=(1,1),
                                bias=True)]
            if activation:
                layers+=[nn.LeakyReLU()]
            return nn.Sequential(*layers)
        
        self.block1 = RB(in_channels=filters)
        self.block2 = RB(in_channels=2*filters)
        self.block3 = RB(in_channels=3*filters)
        self.block4 = RB(in_channels=4*filters)
        self.block5 = RB(in_channels=5*filters, activation=False)
        self.model = [self.block1, self.block2, self.block3, self.block4, self.block5]

    def forward(self,x):
        inputs = x
        for block in self.model:
            output = block(inputs)
            inputs = torch.cat([inputs,output], 1)
        return output.mul(self.res) + x

class RRDB(nn.Module):
    def __init__(self, filters: int, res: float= 0.2):
        super(RRDB, self).__init__()
        self.res = res
        self.dense = nn.Sequential(
            DenseRB(filters),
            DenseRB(filters),
            DenseRB(filters)
        )
    
    def forward(self,x):
        return self.dense(x).mul(self.res) + x

class UpBlock(nn.Module):
    def __init__(self, filters: int):
        super(UpBlock,self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(filters, filters, (3,3), (1,1), (1,1), bias=True),
                nn.LeakyReLU()
                )
        
    def forward(self,x):
        return F.interpolate(self.model(x), scale_factor=2, mode="nearest")

class Generator(nn.Module):
    def __init__(self, channels=3, filters=64, num_res=16, num_up=1):
        super(Generator, self).__init__()

        self.first = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.Sequential(*[RRDB(filters) for _ in range(num_res)])

        self.second = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        self.upsampling = nn.Sequential(*[UpBlock(filters) for _ in range(num_up)])

        self.output = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self,x):
        out1 = self.first(x)
        out = self.blocks(out1)
        out2 = self.second(out)
        out = torch.add(out1,out2)
        out = self.upsampling(out)
        out = self.output(out)
        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, features=64, double = True):
        super(DiscriminatorBlock, self).__init__()
        self. out_features= features*2 if double else features
        self.block = nn.Sequential(
            nn.Conv2d(features, features, (3,3), (1,1), (1,1)),
            nn.BatchNorm2d(features),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(features, self.out_features, (3,3), (2,2), (1,1)),
            nn.BatchNorm2d(self.out_features),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self,x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, features = 64):
        super(Discriminator,self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3,features, (3,3), (1,1), (1,1), bias=True),
            nn.LeakyReLU(0.2, True)
        )
        self.blocks = nn.Sequential(
            DiscriminatorBlock(features),
            DiscriminatorBlock(2*features),
            DiscriminatorBlock(4*features),
            DiscriminatorBlock(8*features, double=False),
        )
        self.last_convo = nn.Conv2d(8*features, 1, (3,3),(1,1),(1,1))

    def forward(self,x):
        out = self.first(x)
        out = self.blocks(out)
        out = self.last_convo(out)
        return out
