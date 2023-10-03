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

class pMRI_Op_2D_t:
    
    def __init__(self,maps,pattern):
        self.C = maps
        self.frame_size = [np.size(self.C,0),np.size(self.C,1)]
        self.Q = np.size(self.C,3)
#         self.uniform_var = 0
        patterns = np.expand_dims(pattern,axis=2)
        patterns = np.tile(patterns,(1,1,np.size(self.C,2),1))
        self.mask_patterns = (patterns.T==1)# why are we taking transpose here?
#         self.N = self.frame_size[0]*self.frame_size[1]*self.Q
#         if self.uniform_var==0:
#             self.CSq = abs(self.C**2)/(self.N/self.Q)
#             self.CSqTr = self.CSq[:,:,:,0]
#             self.CSq = self.CSq[:,:,:,0].reshape((np.size(self.CSq,0)*np.size(self.CSq,1),np.size(self.CSq,2)), order="F")
#             self.CSq = self.CSq.T
#         else:
#             self.CSq = []
#         
#         self.Fro2 = 1/(self.N*np.size(self.C,2))
    # performs Ax
    # args:
    #       x: numpy arrray of size (y, x, 1, frame)
    def mult(self,x):
       
        x = np.expand_dims(x, axis=2)
        fft_x = np.fft.fft2(x*self.C,axes=(0,1),norm='ortho')
        y = fft_x.T[self.mask_patterns] # here?
        return y
    
    def multTr(self,x):
        
        y = np.zeros((self.frame_size[0],self.frame_size[1],np.size(self.C,2),self.Q),dtype='complex64')
        y.T[self.mask_patterns] = x # and here?
        y_ifft = np.fft.ifft2(y,axes=(0,1),norm='ortho')
        y = np.sum(y_ifft*np.conj(self.C), axis = 2)
        return y # .T
    
    def multC(self,x):     
#        print(x)
#        print(x.shape)
        x_ = x.reshape(self.frame_size[0],self.frame_size[1],1,self.Q,order = 'f')
#        print(x_.shape)
##        x = np.expand_dims(x, axis=2)
#        
        fft_x = np.fft.fft2(x_*self.C,axes=(0,1),norm='ortho')
        y1 = fft_x.T[self.mask_patterns] # here?
#        y2 = x.T.flatten()
##        y = np.append(y1,y2)
#        
        Y = np.zeros((self.frame_size[0],self.frame_size[1],np.size(self.C,2),self.Q),dtype='complex64')
        Y.T[self.mask_patterns] = y1 # and here?
        Y_ifft = np.fft.ifft2(Y,axes=(0,1),norm='ortho')
        Y1 = np.sum(Y_ifft*np.conj(self.C), axis = 2)
#        print(Y1.shape)
        Y1 = Y1.T.flatten()
#        print(Y1.shape)
#        Y2 = y2
#        Y = np.append(Y1,Y2)
        Y = Y1+x
#        print(Y.shape)
        return Y