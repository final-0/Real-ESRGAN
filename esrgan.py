import numpy as np
import random
import torch
import torch.nn as nn
import os
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from esrgan_1d import *
from esrgan_2d import * 
from esrgan_loss import * 
from esrgan_Dnet import *
from esrgan_Gnet import *
from esrgan_data import *
from esrgan_noise import *

#===variable===#
block = 23
learning_rate = 0.0001
epochs = 20
data_name = "../../data/train"

#===setting===#
os.makedirs("train_images", exist_ok=True)
os.makedirs("saved_models5", exist_ok=True)

cuda = torch.cuda.is_available()
Device = 'cuda' if torch.cuda.is_available() else 'cpu'

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
dataloader = DataLoader(RealESRGANDataset(data_name), batch_size=1, shuffle=True, num_workers=8)

if cuda:
    criterion_MSE = torch.nn.MSELoss().cuda()
    criterion_L1 = L1Loss().cuda()
    criterion_pe = PerceptualLoss().cuda()
    criterion_gan = GANLoss().cuda()
    net_G = Generator(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=block, num_grow_ch=32).cuda()
    net_D = Discriminator().cuda()
    jpeger = DiffJPEG(differentiable=False).cuda()  
    usm_sharpener = USMSharp().cuda() 

optimizer_G = torch.optim.Adam(net_G.parameters(), lr=learning_rate, betas=(0.9, 0.99))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=learning_rate, betas=(0.9, 0.99))
sample = 50

#===train===#
for epoch in range(0,epochs):
    for i, img in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i

        Img_gt = Variable(img['gt'].type(Tensor)) 
        Img_usm = Variable(usm_sharpener(Img_gt).type(Tensor)) 
        kernel1 = Variable(img['kernel1'].type(Tensor))
        kernel2 = Variable(img['kernel2'].type(Tensor))
        sinc_kernel = Variable(img['sinc_kernel'].type(Tensor))

        chunk_dim = 2
        a_x_split = torch.chunk(Img_gt, chunk_dim, dim=2)
        chunks_gt = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_gt.append(c_)
        a_x_split = torch.chunk(Img_usm, chunk_dim, dim=2)
        chunks_usm = []
        for cnk in a_x_split:
            cnks = torch.chunk(cnk, chunk_dim, dim=3)
            for c_ in cnks:
                chunks_usm.append(c_)
        
        chunks_lq = []
        chunks_sr = []

        """
            chunks_gt[]  正解画像
            chunks_usm[] sharpフィルター
            chunks_lq[]  低解像度
            chunks_sr[]  生成画像
        """
        for j in range(len(chunks_gt)):

            ori_h, ori_w = chunks_gt[j].size()[2:4]
            out = chunks_gt[j]
            
            #===first noise===#
            out = filter2D(out, kernel1)
            updown_type = random.choices(['up', 'down', 'keep'], [0.2, 0.7, 0.1])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, 1.5)
            elif updown_type == 'down':
                scale = np.random.uniform(0.5, 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)

            noise_prob = 0.4
            if np.random.uniform() < 0.5:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range = [1,10],
                    clip=True,
                    rounds=False,
                    gray_prob=noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range = [0.05,3],
                    gray_prob=noise_prob,
                    clip=True,
                    rounds=False)
            
            #===second noise===#
            if np.random.uniform() < 0.8:
                out = filter2D(out, kernel2)
            updown_type = random.choices(['up', 'down', 'keep'], [0.3, 0.4, 0.3])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, 1.2)
            elif updown_type == 'down':
                scale = np.random.uniform(0.7, 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(int(ori_h/4*scale), int(ori_w/4*scale)), mode=mode)
        
            if np.random.uniform() < 0.5:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range = [1,10],
                    clip=True,
                    rounds=False,
                    gray_prob=noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range = [0.05,2.5],
                    gray_prob=noise_prob,
                    clip=True,
                    rounds=False)
            
            #===else===#
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h//4, ori_w//4), mode=mode)
            out = filter2D(out, sinc_kernel)

            lq = out
            chunks_lq.append(lq)
            lq = lq.contiguous()

            if batches_done < 50:
                optimizer_G.zero_grad()
                loss_pixel = criterion_MSE(net_G(lq), chunks_gt[j])
                loss_pixel.backward()
                optimizer_G.step()
                continue

            #==LOSS==#
            L1_gt = chunks_usm[j] 
            Pe_gt = chunks_usm[j] 
            Gan_gt = chunks_gt[j] 
            loss_dict = OrderedDict()
    
        #---generator---#
            optimizer_G.zero_grad()
            Img_out = net_G(lq)
            chunks_sr.append(Img_out)
            Loss_G = 0
            # pixel loss 
            loss_pix = criterion_MSE(Img_out, L1_gt)
            Loss_G += loss_pix
            # perceptual loss
            loss_pe = criterion_pe(Img_out, Pe_gt)
            Loss_G += loss_pe
            # gan loss
            fake_g = net_D(Img_out)
            loss_gan = criterion_gan(fake_g, True, is_disc=False)
            Loss_G += loss_gan

            Loss_G.backward()
            optimizer_G.step()

        #---discriminator---#
            optimizer_D.zero_grad()
            Loss_D = 0
            # real
            real_d = net_D(Gan_gt)
            loss_real = criterion_gan(real_d, True, is_disc=True)
            # fake
            fake_d = net_D(Img_out.detach().clone()) 
            loss_fake = criterion_gan(fake_d, False, is_disc=True)
            
            Loss_D += loss_real
            Loss_D += loss_fake
            Loss_D.backward()
            optimizer_D.step()

        #---progress---#
            print(
                "[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (
                    epoch,
                    i,
                    len(dataloader),
                    loss_fake.item(),
                    Loss_G.item()
                )
            )

        if batches_done % sample == 0 and batches_done != 0:
            up_u = torch.cat((chunks_usm[0],chunks_usm[1]),3)
            down_u = torch.cat((chunks_usm[2],chunks_usm[3]),3)
            usm_hr = torch.cat((up_u,down_u),2) 
            up_g = torch.cat((chunks_sr[0],chunks_sr[1]),3)
            down_g = torch.cat((chunks_sr[2],chunks_sr[3]),3)
            gen_hr = torch.cat((up_g,down_g),2) 
            print(gen_hr.size())
            up_h = torch.cat((chunks_gt[0],chunks_gt[1]),3)
            down_h = torch.cat((chunks_gt[2],chunks_gt[3]),3)
            Img_hr = torch.cat((up_h,down_h),2) 
            print(Img_hr.size())
            up_l = torch.cat((chunks_lq[0],chunks_lq[1]),3)
            down_l = torch.cat((chunks_lq[2],chunks_lq[3]),3)
            Img_lr = torch.cat((up_l,down_l),2) 
            print(Img_lr.size())

            Img_lr = nn.functional.interpolate(Img_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            Img_lr = make_grid(Img_lr, nrow=1, normalize=True)
            Img_hr = make_grid(Img_hr, nrow=1, normalize=True)
            
            img_grid = torch.cat((Img_lr, Img_hr, gen_hr),-1)
            save_image(img_grid, "train_images/%d.png" % batches_done)

#===save model===#
torch.save(net_G.state_dict(), "saved_models5/generator.pth")
torch.save(net_D.state_dict(), "saved_models5/discriminator.pth")
