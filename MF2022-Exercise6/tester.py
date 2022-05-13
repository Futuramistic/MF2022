import torch
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import SSIM
import torchvision

class Tester():
    def __init__(self, model):
        self.model = model    
    def test(self,test_dataloader, model_name:str="model"):
        test_l1 = 0.
        test_PSNR = 0.
        test_SSIM = 0.
        images = None
        for i,batch in enumerate(tqdm(test_dataloader, leave=True, position=0)):
            low_res, high_res = batch
            torch.no_grad()
            high_res_pred = self.model(low_res)
            if images is None:
                images = high_res_pred
            else:
                images = torch.concat([images,high_res_pred])
            test_l1     += (nn.L1Loss()(high_res_pred,high_res)).item()
            test_PSNR   += (-10*torch.log10((nn.MSELoss()(high_res_pred,high_res))).item())
            test_SSIM   += SSIM(data_range=1.0)(high_res_pred,high_res).item()

            baseline = torch.clamp(low_res[0, ...].mul(255),0,255)
            baseline = torchvision.transforms.Resize(size=(int(low_res.size(dim=2)*2),int(low_res.size(dim=3)*2)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(baseline)
            torchvision.io.write_png(torch.clamp(high_res[0, ...].mul(255),0,255).byte(), "results/"+model_name+"/test"+str(i)+".png")
            torchvision.io.write_png(torch.clamp(high_res_pred[0, ...].mul(255),0,255).byte(), "results/"+model_name+"/result"+str(i)+".png")
            torchvision.io.write_png(baseline.byte(), "results/"+model_name+"/base_bilin"+str(i)+".png")
        test_l1 /= len(test_dataloader)
        test_SSIM /= len(test_dataloader)
        test_PSNR /= len(test_dataloader)
        print(f'Test (L1): {test_l1:.3f} Test (PSNR): {test_PSNR:.3f}  Test (SSIM): {test_SSIM:.3f}')