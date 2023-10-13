# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 01:18:41 2023

@author: sizhu
"""

import sys
# sys.path.append('./')
import logging
import random
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
from pMRI_Op_2D_t import pMRI_Op_2D_t
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import os, glob, re
import torch.optim as optim
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
        beta = self.bias.view(1, self.num_features, 1, 1, 1)

        if self.training:
            avg = torch.mean(inp, dim=4)
            avg = torch.mean(avg, dim=3)
            avg = torch.mean(avg, dim=2)
            avg = torch.mean(avg, dim=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * avg
        else:
            avg = self.running_mean.repeat(size[0], 1)

        output = inp - avg.view(1, self.num_features, 1, 1, 1)
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
        layers.append(nn.utils.spectral_norm(nn.Conv3d(imchannel, filternum, filtersize, padding=1, bias=True), n_power_iterations=20))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth):
            layers.append(nn.utils.spectral_norm(nn.Conv3d(filternum, filternum, filtersize, padding=1, bias=False), n_power_iterations=20))
            layers.append(MeanOnlyBatchNorm(filternum,momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.utils.spectral_norm(nn.Conv3d(filternum, imchannel, filtersize, padding=1, bias=False), n_power_iterations=20))
        self.cnn = nn.Sequential(*layers)
        self.init_weights()
    def forward(self,x):
        y = x
        out = self.cnn(x)
        return y-out
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
def downsample_data(data, patterns):
    if np.ndim(patterns)==3:
        mask_pattern = {}
        y = []
        for i in range(0,np.size(patterns,2)):
            mask_pattern[i] = np.where(patterns[:,:,i].T.flatten()>0)
            for j in range(0,np.size(data,2)):
                tmp = data[:,:,j,i].T.flatten()
                tmp = tmp[mask_pattern[i]]
                y = np.append(y,tmp)
    elif np.ndim(patterns)==4:
        data = data*np.transpose(np.expand_dims(patterns,axis=4),(0,1,2,4,3))
        y = data[abs(data)>0]
    return y
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
# if __name__ == '__main__':
device = torch.device('cuda:0') 
samp = np.float32(loadmat(os.getcwd()+'/Perfusion/MRXCAT/data/perf_phantom_samp_R4.mat')['samp'])
samp_shifted = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)          
rho = 1
nite = 80
for i in range(17,22):
    savenmse = []
    kdata = loadmat(os.getcwd()+'/Perfusion/MRXCAT/data/perf_phantom_'+str(i)+'_k.mat')['k_full']
    lsq = loadmat(os.getcwd()+'/Perfusion/MRXCAT/data/perf_phantom_'+str(i)+'_imtrue.mat')['imtrue']
    S = np.squeeze(loadmat(os.getcwd()+'/Perfusion/MRXCAT/data/perf_phantom_'+str(i)+'_map_R4.mat')['map'])
    x0 = loadmat(os.getcwd()+'/Perfusion/MRXCAT/data/perf_phantom_'+str(i)+'_x0_R4.mat')['x0'] 
    x0 = np.fft.fftshift(np.fft.fftshift(x0,axes = 0), axes = 1)
    x0 = np.tile(np.expand_dims(x0,axis=2),[1,1,np.size(samp_shifted,2)])  
    kdata = kdata*np.expand_dims(samp,axis=2)
    S = np.tile(np.expand_dims(S,axis = 3),[1,1,1,np.size(samp,2)])              
    kdata = np.fft.fftshift(np.fft.fftshift(kdata,axes = 0), axes = 1)
    S = np.fft.fftshift(np.fft.fftshift(S,axes = 0), axes = 1)   
    k_avg = kdata.sum(axis = 3)/samp_shifted.sum(axis = 2,keepdims=1)
    kdata = downsample_data(kdata,samp_shifted)
    pMRI = pMRI_Op_2D_t(S, samp_shifted)
    x = x0
    z = pMRI.mult(x)-kdata
    w = np.fft.fftshift(np.fft.fftshift(x,axes=0),axes = 1)
    p = powerite(pMRI,x.shape)
    gamma = rho*p 
    for ite in range(nite):     
        model = BasicNet()
        model = torch.load(os.getcwd()+'/Perfusion/MRXCAT/data/pymodel_%03d.pth' % (ite+1)) 
        model = model.to(device)        
        model.eval()
        xold = x
        midvar = xold-rho*pMRI.multTr(z)
        midvar = np.fft.ifftshift(np.fft.ifftshift(midvar,axes=1),axes = 0)
        midvar_norm = np.expand_dims(np.expand_dims(midvar/np.abs(np.real(midvar)).max(),axis = 0),axis = 1)
        midvar_im = torch.from_numpy(np.array(np.concatenate((np.real(midvar_norm),np.imag(midvar_norm)),1),dtype = 'float32')).cuda()
        midout = model(midvar_im).cpu().detach().numpy().astype(np.float32)
        midout = np.squeeze(midout[:,0,:,:,:]+1j*midout[:,1,:,:,:])
        w = midout* np.abs(np.real(midvar)).max()
        x = np.fft.fftshift(np.fft.fftshift(w,axes=0),axes = 1) 
        s = 2*x-xold
        z = gamma_p/(1+gamma_p)*z+1/(1+gamma_p)*(pMRI.mult(s)-kdata)
        nmse_i = NMSE(lsq,w)
        savenmse.append(nmse_i)
        print('normalized mean square error of x'+str(i)+': ' + repr(nmse_i)) 
        file_name = os.getcwd()+'/Perfusion/MRXCAT/data/reside_m_k'+str(i)+'_auto/im_'+str(ite)+'.mat'
        savemat(file_name,{'x':w})  
        savemat(os.getcwd()+'/Perfusion/MRXCAT/data/reside_m_k'+str(i)+'_auto/nmse.mat',{'nmse':savenmse})  

































