# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 01:18:41 2021

@author: sizhu
"""
import sys
sys.path.append('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp')
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
def complex_random_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-4]
    assert 0 < shape[1] <= data.shape[-3]
    assert 0 < shape[2] <= data.shape[-2]
    w_from = np.random.randint(0,data.shape[-4]-shape[0]+1)
    h_from = np.random.randint(0,data.shape[-3]-shape[1]+1)
    l_from = np.random.randint(0,data.shape[-2]-shape[2]+1)
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    l_to = l_from + shape[2]
    return data[..., w_from:w_to, h_from:h_to, l_from:l_to,:], w_from,w_to, h_from,h_to, l_from,l_to
class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.bias = nn.Parameter(torch.zeros(num_features))
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
class create_datasets():
    def __init__(self,midvar,snr):
        self.keep_slices_o = []
        self.keep_slices_i = []
        for i in range(16):
            a = midvar[i]/abs(np.real(midvar[i])).max()
            sigma = np.linalg.norm(a)/np.sqrt(midvar[i].size)/(10**(snr/20))/np.sqrt(2)
            data_out = torch.from_numpy(np.stack((a.real, a.imag), axis=-1)).float()
            for i in range(288):
                data_o, w_from,w_to, h_from,h_to, l_from,l_to = complex_random_crop(data_out,(64,64,20))
                data_i = data_o + sigma*torch.randn(data_o.size())
                self.keep_slices_o.append(data_o.permute(3,0,1,2))
                self.keep_slices_i.append(data_i.permute(3,0,1,2))
    def __len__(self):
        return len(self.keep_slices_o)
    def __getitem__(self, index):
        outp = self.keep_slices_o[index]
        inp = self.keep_slices_i[index]
        return inp, outp
def create_data_loaders():
    train_loader = DataLoader(
        create_datasets(midvar,snr),
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader
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
if __name__ == '__main__':
    kdata = []
    lsq = []
    S = []
    pMRI = []
    x = []
    z = []
    w = []
    p = []
    gamma = []
    midvar = []
    savenmse = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    xold = []
    samp = np.float32(loadmat('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp/perfusion/MRXCAT/perf_phantom_samp_R4.mat')['samp'])
    samp_shifted = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)  
    x_ini = loadmat('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp/perfusion/MRXCAT/recon/testing/R4/reside_m_net_auto/im_10.mat')['x'] 
    rho = 1
    ep = 10
    snr = 5
    for i in range(1,17):
        kdata.append(loadmat('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp/perfusion/MRXCAT/recon/training/perf_phantom_'+str(i)+'_k.mat')['k_full'])
        lsq.append(loadmat('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp/perfusion/MRXCAT/recon/training/perf_phantom_'+str(i)+'_imtrue.mat')['imtrue'])   
        S.append(np.squeeze(loadmat('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp/perfusion/MRXCAT/recon/training/perf_phantom_'+str(i)+'_map_R4.mat')['map']))
        x0 = x_ini[i-1,:,:,:]
        # x0 = loadmat('C:/Users/sizhu/OneDrive - The Ohio State University/Documents_OSU_Research/unsupervised pnp/perfusion/MRXCAT/recon/training/perf_phantom_'+str(i)+'_x0_R4.mat')['x0'] 
        x0 = np.fft.fftshift(np.fft.fftshift(x0,axes = 0), axes = 1)
        # x0 = np.tile(np.expand_dims(x0,axis=2),[1,1,np.size(samp_shifted,2)])    
        kdata[i-1] = kdata[i-1]*np.expand_dims(samp,axis=2)
        S[i-1] = np.tile(np.expand_dims(S[i-1],axis = 3),[1,1,1,np.size(samp,2)])              
        kdata[i-1] = np.fft.fftshift(np.fft.fftshift(kdata[i-1],axes = 0), axes = 1)
        S[i-1] = np.fft.fftshift(np.fft.fftshift(S[i-1],axes = 0), axes = 1)           
        kdata[i-1] = downsample_data(kdata[i-1],samp_shifted)
        pMRI.append(pMRI_Op_2D_t(S[i-1], samp_shifted))
        x.append(x0)
        z.append(pMRI[i-1].mult(x[i-1])-kdata[i-1])
        w.append(np.fft.fftshift(np.fft.fftshift(x[i-1],axes=0),axes = 1))
        p.append(powerite(pMRI[i-1],x[i-1].shape))
        gamma.append(rho/p[i-1]) 
        xold.append(x[i-1])
        midvar.append(x[i-1])
    for ite in range(10,60):   
        # if ite < 8:
        #     snr = 5
        #     ep = 10
        # if 8 <= ite < 60:
        #     snr = 10
        #     ep = 10
        noisepower_avg = 0
        for ii in range(16):
            xold[ii] = x[ii]
            midvar[ii] = xold[ii]-1/rho*pMRI[ii].multTr(z[ii])
            midvar[ii] = np.fft.ifftshift(np.fft.ifftshift(midvar[ii],axes=1),axes = 0)            
        model = BasicNet()
        model.train()
        device = torch.device('cuda:0')
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=0.0001)
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
        torch.save(model,os.getcwd()+'/perfusion/MRXCAT/recon/testing/R4/reside_m_net_auto_test/pymodel_%03d.pth' % (ite+1))          
        for i in range(16):
            midvar_norm = np.expand_dims(np.expand_dims(midvar[i]/np.abs(np.real(midvar[i])).max(),axis = 0),axis = 1)
            midvar_im = torch.from_numpy(np.array(np.concatenate((np.real(midvar_norm),np.imag(midvar_norm)),1),dtype = 'float32')).cuda()
            midout = model(midvar_im).cpu().detach().numpy().astype(np.float32)
            midout = np.squeeze(midout[:,0,:,:,:]+1j*midout[:,1,:,:,:])
            w[i] = midout* np.abs(np.real(midvar[i])).max()
            x[i] = np.fft.fftshift(np.fft.fftshift(w[i],axes=0),axes = 1) 
            s = 2*x[i]-xold[i]
            z[i] = 1/(1+gamma[i])*z[i]+gamma[i]/(1+gamma[i])*(pMRI[i].mult(s)-kdata[i])
            nmse_i = NMSE(lsq[i],w[i])
            savenmse[i].append(nmse_i)            
            print('normalized mean square error of x'+str(i)+': ' + repr(nmse_i))
            file_name = os.getcwd()+'/perfusion/MRXCAT/recon/testing/R4/reside_m_net_auto_test/im_'+str(ite)+'.mat'
            savemat(file_name,{'x':w})  
            savemat(os.getcwd()+'/perfusion/MRXCAT/recon/testing/R4/reside_m_net_auto_test/nmse.mat',{'nmse':savenmse}) 
            noisepower_avg = noisepower_avg + np.var(pMRI[i].mult(x[i])-kdata[i])
        del model
        para = noisepower_avg/16/6.1252/0.9
        if ite > 2:
            snr = snr*para**0.1
            print("snr: " + repr(snr))
































