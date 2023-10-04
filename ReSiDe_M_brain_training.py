# -*- coding: utf-8 -*-
"""
Created on Wed Oct 04 00:37:34 2021

@author: sizhu 
"""

import sys
# sys.path.append('./')
import logging
import pathlib
import random
import shutil
import time
import os
import h5py
import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import torch.nn as nn
import torch
import torch.nn.init as init
#from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import os, glob, re
import torch.optim as optim
from pMRI_2D import pMRI_2D
from skimage.restoration import (denoise_wavelet, estimate_sigma)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def NMSE(true,b):
    y = 20*np.log10(np.linalg.norm(true-b)/np.linalg.norm(true))
    return y

def complex_random_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = np.random.randint(0,data.shape[-3]-shape[0]+1)
    h_from = np.random.randint(0,data.shape[-2]-shape[1]+1)
    # w_from = (data.shape[-3] - shape[0]) // 2
    # h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :], w_from,w_to, h_from,h_to

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.bias = nn.Parameter(torch.zeros(num_features))

        # self.running_mean = torch.zeros(num_features)
        self.register_buffer('running_mean', torch.zeros(num_features))

    def forward(self, inp):
        size = list(inp.size())
        beta = self.bias.view(1, self.num_features, 1, 1)

        if self.training:
            avg = torch.mean(inp, dim=3)
            avg = torch.mean(avg, dim=2)
            avg = torch.mean(avg, dim=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * avg
        else:
            avg = self.running_mean.repeat(size[0], 1)

        output = inp - avg.view(1, self.num_features, 1, 1)
        output = output + beta

        return output

    def extra_repr(self):
        return '{num_features}, momentum={momentum} '.format(**self.__dict__)
    
    
class BasicNet(nn.Module):
    def __init__(self):
        layers = []
        imchannel = 2
        filternum = 128
        filtersize = 3
        depth = 3
        super(BasicNet, self).__init__()        
        layers.append(nn.utils.spectral_norm(nn.Conv2d(imchannel, filternum, filtersize, padding=1, bias=True), n_power_iterations=20))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(filternum, filternum, filtersize, padding=1, bias=False), n_power_iterations=20))
            layers.append(MeanOnlyBatchNorm(filternum,momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.utils.spectral_norm(nn.Conv2d(filternum, imchannel, filtersize, padding=1, bias=False), n_power_iterations=20))
        self.cnn = nn.Sequential(*layers)
        self.init_weights()
    def forward(self,x):
        y = x
        out = self.cnn(x)
        return y-out
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)




class create_datasets():
    def __init__(self,midvar,snr):
        self.keep_slices_o = []
        self.keep_slices_i = []
        for i in range(16):
            a = midvar[i]/abs(np.real(midvar[i])).max()
            sigma = np.linalg.norm(a)/np.sqrt(midvar[i].size)/(10**(snr/20))/np.sqrt(2)
            data_out = torch.from_numpy(np.stack((a.real, a.imag), axis=-1)).float()
            for i in range(144):
                data_o, w_from,w_to, h_from,h_to = complex_random_crop(data_out,(64,64))
                data_i = data_o + sigma*torch.randn(data_o.size())
                self.keep_slices_o.append(data_o.permute(2,0,1))
                self.keep_slices_i.append(data_i.permute(2,0,1))
    def __len__(self):
        return len(self.keep_slices_o)
    def __getitem__(self, index):
        outp = self.keep_slices_o[index]
        inp = self.keep_slices_i[index]
        return inp, outp
    
def create_data_loaders():

    train_loader = DataLoader(
        create_datasets(midvar,snr),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader


def powerite(pMRI, n):
    q = np.random.randn(*n)
    q = q/np.linalg.norm(q.flatten())
    th = 1e-3
    err = np.inf
    uest = np.inf
    while err > th:
        q = pMRI.multTr(pMRI.mult(q))
        unew = np.linalg.norm(q.flatten())
        err = abs(np.log(uest/unew))
        uest = unew
        q = q/np.linalg.norm(q.flatten())
    return uest

def apply_denoiser(x,model):
    x = np.fft.ifftshift(np.fft.ifftshift(x,axes=1),axes = 0)
    x_norm = np.expand_dims(x,axis = 0)
    x_im = np.expand_dims(x_norm ,axis = 1)
    x_im = np.concatenate((np.real(x_im),np.imag(x_im)),1)
    x_im = np.array(x_im,dtype = 'float32')
    x_im = torch.from_numpy(x_im).cuda()
    w = model(x_im).cpu()
    w = w.detach().numpy().astype(np.float32)
    w = np.squeeze(w[:,0,:,:]+1j*w[:,1,:,:])
    w = np.fft.fftshift(np.fft.fftshift(w,axes=0),axes = 1)
    return w



if __name__ == '__main__':
    kdata = []
    lsq = []
    S = []
    pMRI = []
    x = []
    z = []
    w = []
    p = []
    gamma_p = []
    midvar = []
    xold = []
    noise_power = 0
    rho = 1  
    savenmse = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    snr = 10
    ep = 10
    # samp = np.fft.fftshift(np.fft.fftshift(loadmat(os.getcwd()+'/Brain/T1/data_for_training/R4_gro.mat')['samp'],axes = 0), axes = 1)
    # samp_3 = np.tile(np.expand_dims(samp,axis=2),[1,1,8])    
    for i in range(1,17):
        samp = np.fft.fftshift(np.fft.fftshift(loadmat(os.getcwd()+'/Brain/T1/data_for_training/R4_randoml_k'+str(i)+'.mat')['samp'],axes = 0), axes = 1)
        samp_3 = np.tile(np.expand_dims(samp,axis=2),[1,1,8])    
        kdata.append(loadmat(os.getcwd()+'/Brain/T1/data_for_training/k_'+str(i)+'.mat')['k_full'])
        noise_power = noise_power + np.var(np.concatenate((kdata[i-1][:4,:,:].reshape(-1),kdata[i-1][-4:,:,:].reshape(-1),kdata[i-1][4:-4,:4,:].reshape(-1),kdata[i-1][4:-4,-4:,:].reshape(-1))))
        S.append(np.squeeze(loadmat(os.getcwd()+'/Brain/T1/data_for_training/t1_r4_randoml_map_k'+str(i)+'.mat')['map'])) 
        lsq.append(loadmat(os.getcwd()+'/Brain/T1/data_for_training/im_'+str(i)+'.mat')['imtrue'])
        kdata[i-1] = np.fft.fftshift(np.fft.fftshift(kdata[i-1],axes = 0), axes = 1)
        kdata[i-1] = kdata[i-1]*samp_3
        S[i-1] = np.fft.fftshift(np.fft.fftshift(S[i-1],axes = 0), axes = 1)  
        kdata_u = kdata[i-1].flatten('F')    
        kdata[i-1] = kdata_u[np.where(samp_3.flatten('F')>0)]
        pMRI.append(pMRI_2D(S[i-1], samp))
        x.append(pMRI[i-1].multTr(kdata[i-1]))
        w.append(np.fft.fftshift(np.fft.fftshift(x[i-1],axes=0),axes = 1))
        z.append(pMRI[i-1].mult(x[i-1])-kdata[i-1])
        p.append(powerite(pMRI[i-1],x[i-1].shape))
        gamma_p.append(rho/p[i-1])
        xold.append(x[i-1])
        midvar.append(x[i-1])
    for ite in range(0,80):   
        # if ite < 15:
        #     snr = 10
        #     ep = 10
        # if 15 <= ite < 30:
        #     snr = 15
        #     ep = 10
        # if 30 <= ite < 45:
        #     snr = 20
        #     ep = 10
        # if 45 <= ite < 60:
        #     snr = 25
        #     ep = 10
        # if 60 <= ite:
        #     snr = 30
        #     ep = 10
        for i in range(16):
            xold[i] = x[i]
            midvar[i] = xold[i]-1/rho*pMRI[i].multTr(z[i])
            midvar[i] = np.fft.ifftshift(np.fft.ifftshift(midvar[i],axes=1),axes = 0)
        model = BasicNet()
        model.train()
        device = torch.device('cuda:0')
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.001)
        scheduler = MultiStepLR(opt, milestones=[200, 300], gamma=0.5)
        train_loader = create_data_loaders()
        for epoch in range(0,ep):
            for train_iter, Data in enumerate(train_loader):
                x_batch,y_batch = Data
                out = model(x_batch.to(device, dtype=torch.float))
                loss = F.mse_loss(out,y_batch.to(device, dtype=torch.float), reduction='sum')
                opt.zero_grad()
                loss.backward()
                opt.step()
        torch.save(model,os.getcwd()+'/Brain/T1/data_for_testing/T1_r4_randoml/reside_m_net_auto/pymodel_%03d.pth' % (ite+1))  
        noisepower_avg = 0
        for i in range(16): 
            midvar_norm = midvar[i]/np.abs(np.real(midvar[i])).max()
            midvar_im = np.expand_dims(np.expand_dims(midvar_norm,axis = 0) ,axis = 1)
            midvar_im = np.concatenate((np.real(midvar_im),np.imag(midvar_im)),1)
            midvar_im = np.array(midvar_im,dtype = 'float32')
            midvar_im = torch.from_numpy(midvar_im).cuda()
            midout = model(midvar_im).cpu().detach().numpy().astype(np.float32)
            midout = np.squeeze(midout[:,0,:,:]+1j*midout[:,1,:,:])
            w[i] = midout* np.abs(np.real(midvar[i])).max()
            x[i] = np.fft.fftshift(np.fft.fftshift(w[i],axes=0),axes = 1)       
            s = 2*x[i]-xold[i]
            z[i] = 1/(1+gamma_p[i])*z[i]+gamma_p[i]/(1+gamma_p[i])*(pMRI[i].mult(s)-kdata[i])  
            noisepower_avg = noisepower_avg + np.linalg.norm(pMRI[i].mult(x[i])-kdata[i])**2/kdata[i].size
            nmse_i = NMSE(lsq[i],w[i])
            savenmse[i].append(nmse_i)
            print(nmse_i)   
        file_name = os.getcwd()+'/Brain/T1/data_for_testing/T1_r4_randoml/reside_m_net_auto/im_'+str(ite)+'.mat'
        savemat(file_name,{'x':w})              
        savemat(os.getcwd()+'/Brain/T1/data_for_testing/T1_r4_randoml/reside_m_net_auto/nmse.mat',{'nmse':savenmse}) 
        para = noisepower_avg/noise_power/0.7
        if ite > 2:
            snr = snr*para**0.1
        print("snr: " + repr(snr))



