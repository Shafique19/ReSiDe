#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 15:11:32 2019

@author: tesla
"""
import numpy as np
from scipy import signal
import sys
# sys.path.append('/home/tesla/sizhuo')

class pMRI_2D:
    
    def __init__(self,maps,pattern):
        self.C = maps
        self.frame_size = [np.size(self.C,0),np.size(self.C,1)]
        patterns = np.expand_dims(pattern,axis=2)
        patterns = np.tile(patterns,(1,1,np.size(self.C,2)))
        self.mask_patterns = (patterns.T==1)
    def mult(self,x):
       
        x = np.expand_dims(x, axis=2)
        fft_x = np.fft.fft2(x*self.C,axes=(0,1),norm='ortho')
        y = fft_x.T[self.mask_patterns] 
        return y
    
    def multTr(self,x):
        
        y = np.zeros((self.frame_size[0],self.frame_size[1],np.size(self.C,2)),dtype='complex64')
        y.T[self.mask_patterns] = x
        y_ifft = np.fft.ifft2(y,axes=(0,1),norm='ortho')
        y = np.sum(y_ifft*np.conj(self.C), axis = 2)
        return y 
    
    def multC(self,x):     

        x_ = x.reshape(self.frame_size[0],self.frame_size[1],1,order = 'f')
        fft_x = np.fft.fft2(x_*self.C,axes=(0,1),norm='ortho')
        y1 = fft_x.T[self.mask_patterns]
        Y = np.zeros((self.frame_size[0],self.frame_size[1],np.size(self.C,2)),dtype='complex64')
        Y.T[self.mask_patterns] = y1
        Y_ifft = np.fft.ifft2(Y,axes=(0,1),norm='ortho')
        Y1 = np.sum(Y_ifft*np.conj(self.C), axis = 2)
        Y1 = Y1.T.flatten()
        Y = Y1+x
        return Y
    
    
