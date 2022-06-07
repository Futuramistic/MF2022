import torch
import torch.optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from pytorch_msssim import SSIM
import numpy as np
from esrgan import VGG19_54
from model import BasicSRModel

class SRTrainer():

    def __init__(self, model: nn.Module, loss_fn: nn.modules.loss._Loss,lr: float):

        self.loss_fn = loss_fn
        self.lr = lr
        self.model = model
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),lr=self.lr)
        self.writer = SummaryWriter()

    def train(self,train_loader,val_loader=[],num_epochs:int=32,model_name:str="model", save:bool = True):
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
            if save and ((epoch%10==0) or (epoch==num_epochs-1)):
                torch.save(self.model.state_dict(),"../model/"+model_name+str(epoch)+"_"+str(self.lr))
        self.writer.close()

class GANTrainer():

    def __init__(self, generator: nn.Module, discriminator: nn.Module, lr=1e-4, eta = 1e-2, lambd = 5e-3, alpha = 1):

        self.gen = generator
        self.dis = discriminator
        self.lr = lr
        self.eta = eta
        self.lambd = lambd
        self.alpha = alpha
        self.gen_loss = nn.L1Loss()
        self.gan_loss = nn.BCEWithLogitsLoss()
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()),lr=self.lr)
        self.optimizer_D = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dis.parameters()),lr=self.lr)
        self.scheduler_G = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G,0.99,-1)
        self.scheduler_D = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D, 0.99, -1)
    def train(self,train_loader,val_loader=[],num_epochs:int=32,warm_up_batches:int=100, device = "cpu", save = True):
        
        vgg = VGG19_54().to(device)
        
        for epoch in range(num_epochs):

            iteration_loss_G_color = 0.
            iteration_loss_G_content = 0.
            iteration_loss_G = 0.
            iteration_loss_D = 0.
            iteration_loss_G_fool = 0.

            for i,batch in enumerate(tqdm(train_loader, leave=True, position=0)):
                num_batches = epoch * len(train_loader) + i
                low_res, high_res = batch
                torch.enable_grad()

                # Generator loss
                self.optimizer_G.zero_grad()

                # Generate and compare
                high_gen = self.gen(low_res)
                loss_color = self.gen_loss(high_gen, high_res)

                if num_batches < warm_up_batches:
                    loss_color.backward()
                    self.optimizer_G.step()
                    iteration_loss_G_color += loss_color.item()
                    continue
                
                # Content
                real_content = vgg(high_res)
                fake_content = vgg(high_gen).detach()
                loss_content = self.gen_loss(fake_content,real_content)

                # Fool discriminator
                pred_fake    = self.dis(high_gen)
                pred_real   = self.dis(high_res).detach().clone()

                ones    = torch.autograd.Variable(torch.Tensor(np.ones((low_res.size(0),1,4,4))).to(device), requires_grad = False)
                zeros   = torch.autograd.Variable(torch.Tensor(np.zeros((low_res.size(0),1,4,4))).to(device), requires_grad = False)
                loss_fool = self.gan_loss(pred_fake - pred_real.mean(0, keepdim=True),ones)
                loss_G =  self.alpha*loss_content + self.lambd*loss_fool + self.eta*loss_color
                
                loss_G.backward()
                self.optimizer_G.step()

                iteration_loss_G += loss_G.item()
                iteration_loss_G_color += loss_color.item()
                iteration_loss_G_fool += loss_fool.item()
                iteration_loss_G_content += loss_content.item()

                # Discriminator
                self.optimizer_D.zero_grad()
                pred_fake    = self.dis(high_gen.detach().clone())
                pred_real    = self.dis(high_res)

                loss_D = (self.gan_loss(pred_real - pred_fake.mean(0, keepdim=True), ones) + self.gan_loss(pred_fake - pred_real.mean(0, keepdim=True), zeros))/2
                loss_D.backward()
                self.optimizer_D.step()
                iteration_loss_D += loss_D.item()

            self.scheduler_D.step()
            self.scheduler_G.step()
            iteration_loss_G_color  /= len(train_loader)
            iteration_loss_G        /= len(train_loader)
            iteration_loss_D        /= len(train_loader)
            iteration_loss_G_fool   /= len(train_loader)
            iteration_loss_G_content/= len(train_loader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'\t Generator loss: {iteration_loss_G:.4e}')
            print(f'\t \t Content loss: \t\t {iteration_loss_G_content:.4e}\t factor: {self.alpha:1e}') 
            print(f'\t \t Color loss: \t\t {iteration_loss_G_color:.4e} \t factor: {self.eta:1e}') 
            print(f'\t \t Adversarial loss: \t {iteration_loss_G_fool:.4e} \t factor: {self.lambd:1e}')
            print(f'\t Discriminator loss: {iteration_loss_D:.4e}')

            # Save both models
            if save and (epoch%5==0) or (epoch==num_epochs-1):
                torch.save(self.gen.state_dict(),"../model/generator"+str(epoch)+"_"+str(self.lr))
                torch.save(self.gen.state_dict(),"../model/discriminator"+str(epoch)+"_"+str(self.lr))