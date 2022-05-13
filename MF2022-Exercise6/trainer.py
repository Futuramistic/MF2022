import torch
import torch.optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_msssim import SSIM
from model import BasicSRModel

class SRTrainer():

    def __init__(self, model: nn.Module, loss_fn: nn.modules.loss._Loss,lr: float):

        self.loss_fn = loss_fn
        self.lr = lr
        self.model = model
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr)
        self.writer = SummaryWriter()

    def train(self,train_loader,val_loader=[],num_epochs:int=32,model_name:str="model"):
        for epoch in range(num_epochs):

            iteration_loss = 0.
            for batch in tqdm(train_loader, leave=True, position=0):
                low_res, high_res = batch
                torch.enable_grad()
                self.optimizer.zero_grad()
                high_res_pred = self.model(low_res)
                loss = self.loss_fn(high_res_pred, high_res)
                loss.backward()
                self.optimizer.step()
                iteration_loss += loss.item()
            iteration_loss /= len(train_loader)

            val_l1 = 0.
            val_PSNR = 0.
            val_SSIM = 0.
            for batch in tqdm(val_loader, leave=True, position=0):
                low_res, high_res = batch
                torch.no_grad()
                high_res_pred = self.model(low_res)
                val_l1 += (nn.L1Loss()(high_res_pred,high_res)).item()
                val_PSNR+=(-10*torch.log10((nn.MSELoss()(high_res_pred,high_res))).item())
                val_SSIM += SSIM(data_range=1.0)(high_res_pred,high_res).item()

            val_l1      /= len(val_loader)
            val_PSNR    /= len(val_loader)
            val_SSIM    /= len(val_loader)
            
            self.writer.add_scalar("Loss/train", iteration_loss, epoch)
            self.writer.add_scalar("Loss/validation/L1",     val_l1, epoch)
            self.writer.add_scalar("Loss/validation/PSNR", val_PSNR, epoch)
            self.writer.add_scalar("Loss/validation/SSIM", val_SSIM, epoch)

            print(f'Epoch {epoch+1}/{num_epochs} Training Loss: {iteration_loss:.3f} Valid (L1): {val_l1:.3f} Valid (PSNR): {val_PSNR:.3f}  Valid (SSIM): {val_SSIM:.3f}')
            if (epoch%5==0) or (epoch==num_epochs-1):
                torch.save(self.model.state_dict(),"./model/"+model_name+str(epoch)+"_"+str(self.lr))
        self.writer.close()