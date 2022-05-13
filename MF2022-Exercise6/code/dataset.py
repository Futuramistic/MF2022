import imp
from torch import FloatTensor, tensor
import torch.utils.data 
import torchvision.io 
import torchvision.transforms
import torch

from os.path import isfile, join

from enum import Enum

class Dataset(Enum):
    TRAIN = 0 # Training
    VALID = 1 # Validation

class SRDDataset(torch.utils.data.Dataset):
    def __init__(self,dir,img_paths, device, 
                    type: Dataset, interpolation: torchvision.transforms.InterpolationMode = torchvision.transforms.InterpolationMode.BILINEAR):
        self.img_dir = dir
        self.device = device
        self.images_paths = img_paths
        self.type = type
        self.interpolation = interpolation
        
    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        img_path = self.images_paths[idx]
        label = torchvision.io.read_image(join(self.img_dir,img_path), mode = torchvision.io.ImageReadMode.RGB).to(self.device)
        if self.type is Dataset.TRAIN:
            label = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(size=(64,64)),
                torchvision.transforms.ColorJitter(brightness=0.2,contrast=0.2,saturation=0.2,hue=0.2)
                ])(label)
        image = torchvision.transforms.Resize(size=(int(label.size(1)/2),int(label.size(2)/2)),interpolation=self.interpolation)(label)
        if isinstance(label,torch.cuda.ByteTensor) or isinstance(label,torch.ByteTensor):
            label = torchvision.transforms.ConvertImageDtype(dtype=torch.float32)(label)
        if isinstance(image,torch.cuda.ByteTensor) or isinstance(image,torch.ByteTensor):
            image = torchvision.transforms.ConvertImageDtype(dtype=torch.float32)(image)
        return image, label