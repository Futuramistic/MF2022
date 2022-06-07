import torch
from torch import bilinear
import torch.nn as nn
from tqdm import tqdm
from pytorch_msssim import SSIM
import torchvision

class Tester():
    def __init__(self, model):
        self.model = model

        # Model needs to be on CPU to output images
        self.model.to('cpu')    

    def test(self,test_dataloader, model_name:str="model"):

        test_l1 = 0.
        test_PSNR = 0.
        test_SSIM = 0.

        bilinear_l1 = 0.
        bilinear_PSNR = 0.
        bilinear_SSIM = 0.

        bicubic_l1 = 0.
        bicubic_PSNR = 0.
        bicubic_SSIM = 0.

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

            baseline1 = torchvision.transforms.Resize(size=(int(low_res.size(dim=2)*2),int(low_res.size(dim=3)*2)), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)(low_res)
            baseline2 = torchvision.transforms.Resize(size=(int(low_res.size(dim=2)*2),int(low_res.size(dim=3)*2)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)(low_res)

            bilinear_l1     += (nn.L1Loss()(baseline1,high_res)).item()
            bilinear_PSNR   += (-10*torch.log10((nn.MSELoss()(baseline1,high_res))).item())
            bilinear_SSIM   += SSIM(data_range=1.0)(baseline1,high_res).item()

            bicubic_l1 += (nn.L1Loss()(baseline2,high_res)).item()
            bicubic_PSNR   += (-10*torch.log10((nn.MSELoss()(baseline2,high_res))).item())
            bicubic_SSIM   += SSIM(data_range=1.0)(baseline2,high_res).item()

            torchvision.io.write_png(torch.clamp(high_res_pred[0, ...].mul(255),0,255).byte(), "../results/"+model_name+"/result"+str(i)+".png")
            torchvision.io.write_png(torch.clamp(baseline1[0,...].mul(255),0,255).byte(), "../results/base/bilinear/result"+str(i)+".png")
            torchvision.io.write_png(torch.clamp(baseline2[0,...].mul(255),0,255).byte(), "../results/base/bicubic/result"+str(i)+".png")
        test_l1 /= len(test_dataloader)
        test_SSIM /= len(test_dataloader)
        test_PSNR /= len(test_dataloader)

        bicubic_l1 /= len(test_dataloader)
        bicubic_SSIM /= len(test_dataloader)
        bicubic_PSNR /= len(test_dataloader)

        bilinear_l1 /= len(test_dataloader)
        bilinear_SSIM /= len(test_dataloader)
        bilinear_PSNR /= len(test_dataloader)

        print(f'Test:       (L1):   {test_l1:.3f}   (PSNR): {test_PSNR:.3f} (SSIM): {test_SSIM:.3f}')
        print(f'Bilinear:   (L1):   {bilinear_l1:.3f}   (PSNR): {bilinear_PSNR:.3f} (SSIM): {bilinear_SSIM:.3f}')
        print(f'Bicubic:    (L1):   {bicubic_l1:.3f}    (PSNR): {bicubic_PSNR:.3f}  (SSIM): {bicubic_SSIM:.3f}')