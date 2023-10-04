# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 00:37:34 2021

@author: Shafique
"""

import sys
sys.path.append('/')
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
    savenmse = []
  
    i=21
    k_full = loadmat(os.getcwd()+'/datafortesting/T1/k_'+str(i)+'.mat')['k_full']
    S = np.squeeze(loadmat(os.getcwd()+'/datafortesting/T1/T1_r4_randoml/t1_r4_randoml_map_k'+str(i)+'.mat')['map'])   
    imtrue = loadmat(os.getcwd()+'/datafortesting/T1/im_'+str(i)+'.mat')['imtrue']
    samp = loadmat(os.getcwd()+'/datafortesting/T1/T1_r4_randoml/R4_randoml_k'+str(i)+'.mat')['samp'] 
    # samp = loadmat(os.getcwd()+'/datafortesting/T1/T1_r4_gro/R4_gro.mat')['samp'] 

    k_samp = k_full*np.expand_dims(samp,axis=2)
    
    k_shifted = np.fft.fftshift(np.fft.fftshift(k_samp,axes = 0), axes = 1)
    samp = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)
    samp_shifted = np.tile(np.expand_dims(samp,axis=2),[1,1,np.size(k_full,2)]) 
    kdata = k_shifted.flatten('F')    
    kdata = kdata[np.where(samp_shifted.flatten('F')>0)]
    
    
    S = np.fft.fftshift(np.fft.fftshift(S,axes = 0), axes = 1)
    pMRI = pMRI_2D(S, samp)         
    x = pMRI.multTr(kdata)

    rho = 1  
    w = np.fft.fftshift(np.fft.fftshift(x,axes=0),axes = 1)
    z = pMRI.mult(x)-kdata
    p = powerite(pMRI,x.shape)
    gamma_p = rho/p
    device = torch.device('cuda:0')    
    for ite in range(80):     
        model = BasicNet()
        model = torch.load(os.getcwd()+'/datafortesting/T1/T1_r4_randoml/reside_m_net_auto/pymodel_%03d.pth' % (ite+1))
        model = model.to(device)        
        model.eval()
        xold = x
        midvar = xold-1/rho*pMRI.multTr(z)
        midvar = np.fft.ifftshift(np.fft.ifftshift(midvar,axes=1),axes = 0)
        midvar_norm = midvar/np.abs(np.real(midvar)).max()
        midvar_norm = np.expand_dims(midvar_norm,axis = 0)
        midvar_im = np.expand_dims(midvar_norm ,axis = 1)
        midvar_im = np.concatenate((np.real(midvar_im),np.imag(midvar_im)),1)
        midvar_im = np.array(midvar_im,dtype = 'float32')
        midvar_im = torch.from_numpy(midvar_im).cuda()
        w = model(midvar_im).cpu()
        w = w.detach().numpy().astype(np.float32)
        w = np.squeeze(w[:,0,:,:]+1j*w[:,1,:,:])
        w = w* np.abs(np.real(midvar)).max()
        x = np.fft.fftshift(np.fft.fftshift(w,axes=0),axes = 1)       
        s = 2*x-xold
        z = 1/(1+gamma_p)*z+gamma_p/(1+gamma_p)*(pMRI.mult(s)-kdata)  
        nmse_i = NMSE(imtrue,w)
        print(nmse_i) 
        savenmse.append(nmse_i)
#    savemat('sigma.mat',{'sigma':savesigma}) 


        
        file_name = os.getcwd()+'/datafortesting/T1/T1_r4_randoml/reside_m_k'+str(i)+'_auto/im_'+str(ite)+'.mat'
        savemat(file_name,{'x':w})  

    savemat(os.getcwd()+'/datafortesting/T1/T1_r4_randoml/reside_m_k'+str(i)+'_auto/nmse.mat',{'nmse':savenmse}) 









