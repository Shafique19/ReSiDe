#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:00:37 2019

@author: tesla
"""
import numpy as np
from scipy.io import loadmat, savemat
from scipy import signal
import sys
from pMRI_Op_2D_t import pMRI_Op_2D_t
from denoiser_torch_new import CNN_denoiser, BasicNet, MeanOnlyBatchNorm
#from denoiser_torch_new_complex import CNN_denoiser, BasicNet, ComplexConv3d
from scipy.sparse.linalg import LinearOperator, cg
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from skimage import measure
import os
import time


def NMSE(true,b):
    y = 20*np.log10(np.linalg.norm(true-b)/np.linalg.norm(true))
    return y


def ifft2_shift(x):
    y = np.sqrt(np.size(x,0)*np.size(x,1))*np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x),axes=(0,1)))
    return y
    
def sensCorrect(x, s, opt):
    Nc = np.size(s,2)
    if opt == 1:
        xc = x
        sc = s
    elif opt == 2:
        phs = np.arctan2(np.imag(x),np.real(x))
        map = np.exp(-1j*phs)
        xc = x*map
        sc = s*np.tile(np.expand_dims(np.conj(map),axis=2),(1,1,Nc))
    elif opt == 3:
        avgPhase = np.arctan2(np.imag(np.sum(x)),np.real(np.sum(x)))
        tmp = abs(x), np.exp(1j*avgPhase)
        phs = np.arccos((np.real(x)*np.real(tmp) + np.imag(x)*np.imag(tmp))/(abs(x)*abs(tmp)))
        map = 2*(phs>(np.pi/2))-1
        xc = x*map
        sc = s*np.tile(np.expand_dims(np.conj(map),axis=2),(1,1,Nc))      
                
    return xc, sc

def WalshCoilCombine(I, **kwargs):
    if len(kwargs) >=2:
        print("Incorrect number of input arguments")
    elif len(kwargs) == 0:
        fil = 3.0
        opt = 2.0
    else:
        fil = kwargs['fil']
        opt = 2.0
    if fil < 1:
        fil = 9.0
        print("filter size is set to 9")
    fil = 2*np.floor(fil/2)+1 # ensure filter size is odd
    
    FE = np.size(I,0)
    PE = np.size(I,1)
    Nc = np.size(I,2)
    cI = np.zeros([FE,PE],dtype = 'complex64')
    S = np.zeros([FE,PE,Nc],dtype = 'complex64')
    covI = np.zeros([FE,PE,Nc,Nc],dtype = 'complex64')
    for k in range(0,Nc):
        covI[:,:,k,k] = I[:,:,k]*np.conj(I[:,:,k])
        for j in range(0,k):
            covI[:,:,k,j] = I[:,:,k]*np.conj(I[:,:,j])
            covI[:,:,j,k] = np.conj(covI[:,:,k,j])
    fil2 = np.ones([int(fil),int(fil)],dtype = 'complex64')
    covIs = np.zeros([FE,PE,Nc,Nc],dtype = 'complex64')
    for i in range(0,Nc):
        for j in range(0,Nc):
            covIs[:,:,i,j] = signal.convolve2d(covI[:,:,i,j], fil2, mode='same')
    for i in range(0,PE):
        for j in range(0,FE):
            w,v = np.linalg.eigh((covIs[j,i,:,:]+np.matrix.getH(covIs[j,i,:,:]))/2)
            cI[j,i] = np.matmul(I[j,i,:].reshape((1,Nc)), np.conj(v[:,[-1]]))
            S[j,i,:] = v[:,[-1]].reshape((1,1,Nc))        
    cI, S = sensCorrect(cI, S, opt)
    return cI, S

        
def normImage(kdata,samp):
    weights = samp.sum(axis = 2,keepdims = 1)
    weights[weights==0] = np.inf
    time_av = ifft2_shift(np.sum(kdata,axis = 3)/weights)
    time_av, S = WalshCoilCombine(time_av,fil=3.0)
       
    x_50 = np.percentile(abs(np.array(time_av).flatten()),60)
    tmp = time_av[abs(time_av)>=x_50]
    scale = abs(tmp).mean()
    
    return scale, time_av
    
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
    
    
def obj(x,pMRI,z,u,b,rho):
#    return np.linalg.norm(pMRI_Op_2D_t(x,maps,samp).mult()-b)
    return 1/2*np.linalg.norm(pMRI.mult(x)-b)**2 + rho/2*np.linalg.norm(x-z+u)**2
def dx_obj(x,pMRI,z,u,b,rho):
#    return pMRI_Op_2D_t(x,maps,samp).mult()
    return pMRI.multTr(pMRI.mult(x)-b) + rho*(x-z+u)

def red_agd(x0, b, param, CNN, pMRI, lsq):
    lamred = param['lambda']
    max_ite = param['iter']
    mu = param['mu']
    a_nmse = []
    a_dist = []
    x = x0
    y1 = x
    t1 = 1
    print("red_agd")
    for k in range(max_ite):
        xold = x
        f_x = CNN.denoise(x)
        grad1 = pMRI.multTr(pMRI.mult(x) - b)
        grad2 = lamred*(x - f_x)
        grad = grad1 + grad2
        y0 = y1
        y1 = x - mu*grad
        t0 = t1
        t1 = (1+ np.sqrt(1 + 4*t0**2))/2
        x = y1 + (t0 - 1)/t1*(y1-y0)
        a_dist.append(np.linalg.norm(x - xold))
        x_i = np.fft.ifftshift(np.fft.ifftshift(x,axes=1),axes = 0)
        nmse_i = NMSE(lsq,x_i)
        a_nmse.append(nmse_i)
#        print(k)
#        if k == max_ite-1:
        print("normalized mean square error of x: " + repr(nmse_i))
        
    return x, a_nmse, a_dist
    
def pnp_pg(x0, b, param, CNN, pMRI, lsq):
    max_ite = param['iter']
    mu = param['mu']
    a_nmse = []
    a_dist = []
    x = x0
    y1 = x
    t1 = 1
    print("pnp_pg")
    for k in range(max_ite):
        xold = x
#        f_x = CNN.denoise(x)
        grad = pMRI.multTr(pMRI.mult(x) - b)
        y0 = y1
        y1 = CNN.denoise(x - mu*grad)
        t0 = t1
        t1 = (1+ np.sqrt(1 + 4*t0**2))/2
        x = y1 + (t0 - 1)/t1*(y1-y0)
        a_dist.append(np.linalg.norm(x - xold))
        x_i = np.fft.ifftshift(np.fft.ifftshift(x,axes=1),axes = 0)
        nmse_i = NMSE(lsq,x_i)
        a_nmse.append(nmse_i)
        savemat('90r10.mat',{'x':x_i})
#        print(k)
#        if k == max_ite-1:
        print("normalized mean square error of x: " + repr(nmse_i))
        
    return x, a_nmse, a_dist


def admm(x0,b,param,CNN, pMRI, lsq):
    lamred = param['lambda']
    max_ite = param['iter']
    rho = param['rho']
    loss_solver = param['loss_solver']
    RED = param['RED']
    m1 = param['m1']
    m2 = param['m2']
    scale = param['scale']
    a_nmse = []
    a_dist = []
    x = x0
    z = x0
    u = np.zeros(x.shape)
    
    
    if loss_solver == 'adam':
        mu = 10
        tau = 2
        ABSTOL = np.exp(0.0001)
        RELTOL = np.exp(0.01)
    else:
        At_b = pMRI.multTr(b)
    
    for k in range(0,max_ite):
        print(k)
        xold = x

        # x update
        if loss_solver == 'adam':
            step = 1
            step_ind = 0
            obj_pass = 0
            
            obj_start = obj(x,pMRI,z,u,b,rho)
            # print('Starting Objective value '+str(obj_start))
            gradient = dx_obj(x,pMRI,z,u,b,rho)
            while obj_pass ==0 and step_ind < 10:
                x_new = x - step*gradient
                obj_end = obj(x_new,pMRI,z,u,b,rho)
                if obj_start >= obj_end:
                    obj_pass = 1
                    x = x_new
                else:
                    step = 0.5*step
                step_ind = step_ind+1
#            print("ite = 0" + "  step size =" + repr(step) + "  objective value=" + repr(obj_end))
            if k == 1:
                num_ind = 50
            elif k < 10:
                num_ind = 50
            else:
                num_ind = 50
            s = gradient
            stepPrev = 0.1
            for ind in range(num_ind):
                step = 10*stepPrev
                step_ind = 1
                obj_pass = 0
                obj_start = obj(x,pMRI,z,u,b,rho)
                
                gradientPrev = gradient
                gradient = dx_obj(x,pMRI,z,u,b,rho)
                b_conj = np.matmul(np.expand_dims(gradient.T.flatten(),axis = 0),np.expand_dims(gradient.T.flatten()-gradientPrev.T.flatten(),axis=1))/(np.matmul(np.expand_dims(gradientPrev.T.flatten(),axis = 0),np.expand_dims(gradientPrev.T.flatten(),axis=1))+2**(-52))
                s = gradient+b_conj*s
                while obj_pass==0 and step_ind<10:
                    x_new = x-step*s
                    obj_end = obj(x_new,pMRI ,z,u,b,rho)
                    if obj_start >= obj_end:
                        obj_pass = 1
                        x = x_new
                    else:
                        step = 0.5*step
                    step_ind = step_ind+1
                stepPrev = step
#                print("ite = " + repr(ind) + "  step size =" + repr(step) + "  objective value=" + repr(obj_end))
                
                if (obj_start-obj_end)/obj_start<0.001:
                    break
        elif loss_solver == 'CG':
            z_ = (z-u).T.flatten()
            b_ = pMRI.multTr(b)
            y = z_+b_.T.flatten()
            A = LinearOperator((z.size,z.size), matvec=pMRI.multC)
            t = time.time()
            x, info = cg(A,y,x0 = x.T.flatten())
            print(info)
            elapsed_prox = time.time()-t
            x = x.reshape(z.shape,order = 'f')
            
            
            
        elif loss_solver == 'GD':
            step_size = 0.0125;
            t = time.time()
            for ind in range(32):
                gradient = dx_obj(x,pMRI,z,u,b,rho)
                x = x - step_size*gradient
            elapsed_prox = time.time()-t
#            print("norm_of_x_after = " + repr(np.linalg.norm(x)))
#            print("norm_of_ax-b = " + repr(1/2*np.linalg.norm(pMRI.mult(x)-b)**2))
        else:
            for k1 in range(m1):
                c = At_b + rho*(z - u)
                A_x = pMRI.multTr(pMRI.mult(x)) + rho*x
                res = c - A_x
                a_res = pMRI.multTr(pMRI.mult(res)) + rho*res
                mu_opt  = np.mean(res*res)/np.mean(res*a_res)
                x = x + mu_opt*res
        a_dist.append(np.linalg.norm(x - xold))
        x_i = np.fft.ifftshift(np.fft.ifftshift(x,axes=1),axes = 0)
        nmse_i = NMSE(lsq,x_i)
        x_i = x_i*scale
        a_nmse.append(nmse_i)
        if k == 149:
            savemat('150.mat',{'x':x_i})
        print("normalized mean square error of x: " + repr(nmse_i))
        
                
        # z update
        zold = z

        if RED:
            zStar = rho*(x+u)
            for ind in range(m2):    
                zBar = CNN.denoise(z)
                z = 1/(rho+lamred)*(lamred*zBar+zStar)
        else:
            z = CNN.denoise(x+u)
            
        z_i = np.fft.ifftshift(np.fft.ifftshift(z,axes=1),axes = 0)    
        print("normalized mean square error of z: " + repr(NMSE(lsq,z_i)))
        # u update
        u = u+(x-z)
        
        
        # These adaptive features of ADMM were implemented by ADAM
        if loss_solver == 'adam':
            r_norm = np.linalg.norm(x-z)
            s_norm = np.linalg.norm(-rho*(z-zold))
            if r_norm > mu*s_norm:
                rho = tau*rho
                u = u/tau
            elif s_norm > mu*r_norm:
                rho = rho/tau
                u = u*tau
            
    return x, a_nmse, a_dist
    
def PDS(x0, b, param, CNN, pMRI, lsq):
    max_ite = param['iter']
    rho = param['rho']
    scale = param['scale']
    a_nmse = []
    a_dist = []
    a_rho = []
    x = x0
    z = pMRI.mult(x)-b
    p = powerite(pMRI,x.shape)
    gamma = rho/p
    
    for k in range(0,max_ite):
##        print(k)
#        if k > 19 and k%5==0:
#            discrepancy = 1/0.6*1/np.std(pMRI.mult(x)-b)**2
#            print(discrepancy)            
##            ax_y = pMRI.mult(x)-b
##            discrepancy = 1/np.std(ax_y[abs(ax_y)<2*np.std(ax_y)])**2
##            print(discrepancy)            
##            discrepancy = 1*b.size/np.linalg.norm(pMRI.mult(x)-b)**2
##            print(discrepancy)
#            rho = rho*discrepancy
#            gamma = rho/p
##        discrepancy = 1/np.std(pMRI.mult(x)-b)**2
##        print(discrepancy)
##        print(1/rho)

        xold = x
        x = CNN.denoise(xold-1/rho*pMRI.multTr(z))
        s = 2*x-xold
        z = 1/(1+gamma)*z+gamma/(1+gamma)*(pMRI.mult(s)-b)
        a_rho.append(rho)
        

        x_i = np.fft.ifftshift(np.fft.ifftshift(x,axes=1),axes = 0)
        nmse_i = NMSE(lsq,x_i)
        print("normalized mean square error of x: " + repr(nmse_i))
        a_nmse.append(nmse_i)
#        x_i = x_i*scale
#        mask = lsq > abs(lsq).max()*0.05
#        lsq_ = lsq*mask
#        x_i_ = x_i*mask
#        size_x = x_i.shape
#        x1 = round(size_x[0]*0.25)
#        x2 = round(size_x[1]*0.25)
#        ssim = measure.compare_ssim(abs(lsq_[x1:-x1,x2:-x2,:]),abs(x_i_[x1:-x1,x2:-x2,:]))
        

        print("normalized mean square error of x: " + repr(nmse_i))
#       print(ssim)
    return x_i, a_nmse, a_dist, a_rho

def main():

    
    kdata = loadmat('test data/HY/FID14520kx_ky_c_f1')['y']
    samp = np.float32(loadmat('test data/HY/FID14520sampR10.mat')['samp'])
    lsq = loadmat('test data/HY/FID14520_slc1_SCR_R1_inversA.mat')['xHat']        



    kdata = kdata*np.expand_dims(samp,axis=2)
    weights_b = samp.sum(axis = 2,keepdims = 1)
    weights_b[weights_b==0] = np.inf
    time_av_b = ifft2_shift(np.sum(kdata,axis = 3)/weights_b)
    x0, maps = WalshCoilCombine(time_av_b,fil=1)

    
 
    maps = np.tile(np.expand_dims(maps,axis = 3),[1,1,1,np.size(samp,2)])
    x0 = np.tile(np.expand_dims(x0,axis=2),[1,1,np.size(samp,2)])
    x0  = x0 + 0.010*abs(x0).max()*np.random.randn(x0.shape[0],x0.shape[1],x0.shape[2])
    kdata = np.fft.fftshift(np.fft.fftshift(kdata,axes = 0), axes = 1)
    samp = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)
    x0 = np.fft.fftshift(np.fft.fftshift(x0,axes = 0), axes = 1)
    maps = np.fft.fftshift(np.fft.fftshift(maps,axes = 0), axes = 1)
    kdata = downsample_data(kdata,samp)
    
    
    
    CNN = CNN_denoiser()
    pMRI = pMRI_Op_2D_t(maps, samp)
    
    
    
    # Run PDS
    param = {}
    param['scale'] = 1
    param['iter'] = 200
    
    param['rho'] = 64
    x, a_pds_nmse, a_pds_dist, a_rho = PDS(x0,kdata,param, CNN,pMRI,lsq) 
    file_name = 'FID11652_slc1' +'_28dB_nu_'+str(1/a_rho[-1])+'_200.mat'
    savemat(file_name,{'x':x})
    
#    savemat('rho_norm.mat',{'rho':a_rho})
#    savemat('nmse_norm.mat',{'nmse':a_pds_nmse})
#    for r in range(-2,8):
#        param['rho'] = 2**(-r)    
#        print(2**(r))
#   
#        x, a_pds_nmse, a_pds_dist, a_rho, ssim = PDS(x0,kdata,param, CNN,pMRI,lsq) 
#        file_name = 'FID14461_R8_28dB'+'_rho_'+str(a_rho[-1])+'_nmse_'+str(a_pds_nmse[-1])+'_ssim_'+str(ssim)+'.mat'
#        
#        savemat(file_name,{'x':x})

     
    # Run ADMM (Adam's version)
#    param = {}
#    param['lambda'] = lambdaRED
#    param['iter'] = iterations
#    param['rho'] = 1
#    param['loss_solver'] = 'adam'
#    param['RED'] = True
#    param['m1'] = None
#    param['m2'] = 1
#    
#    # param['mu'] = 0.1
##    x, a_redadmm_adam_nmse, a_redadmm_adam_dist = admm(x0,kdata,param, CNN,pMRI,lsq)
##    savemat('24540_1000.mat',{'x':x})
#    # Run ADMM (Romano version)
#    param = {}
#    param['lambda'] = lambdaRED
#    param['iter'] = iterations
#    param['rho'] = 1
#    param['loss_solver'] = 'GD'
#    param['RED'] = True
#    param['m1'] = 1
#    param['m2'] = 2
#    
##    x, a_redadmm_romano_nmse, a_redadmm_romano_dist = admm(x0,kdata,param, CNN,pMRI,lsq)
#    
#    # Run PnP-ADMM
#    param = {}
#    # TODO control trade-off
#    param['lambda'] = None
#    param['iter'] = iterations
#    param['rho'] = 1
#    param['loss_solver'] = 'adam'
#    param['RED'] = False
#    param['m1'] = 10
#    param['m2'] = 1
#    param['scale'] = scale
#
##    x, a_pnpadmm_nmse, a_pnpadmm_dist = admm(x0,kdata,param, CNN,pMRI,lsq)
#    savemat('nu_025.mat',{'nmse':a_pnpadmm_nmse})
#    # Run PnP-PG
#    param = {}
#    param['iter'] = iterations
#    # Step size controls loss, denoiser trade off
#    param['mu'] = 0.26
#
##    x, a_pnppg_nmse, a_pnppg_dist = pnp_pg(x0,kdata,param, CNN,pMRI,lsq)
##    savemat('pnppg_nmse.mat',{'pnppg_nmse':a_pnppg_nmse})
#    # Run RED-AGD
#    param = {}
#    param['lambda'] = lambdaRED
#    param['iter'] = iterations
#    # Step size controls loss, denoiser trade off
#    param['mu'] = 0.17
#
##    x, a_redagd_nmse, a_redagd_dist = red_agd(x0,kdata,param, CNN,pMRI,lsq)
#    
#    # plot nmse
#    
#    a_iter = range(iterations)
#    plt.semilogx(a_iter, a_pnppg_nmse, '.-')
#    plt.legend(['PnP-PG'])
#    plt.xlabel('iter')
#    plt.ylabel('nmse')
#    plt.show()
#    
#    # plot update distance
#    
#    a_iter = range(iterations)
#    plt.semilogx(a_iter, a_redadmm_adam_dist, '.-',
#                 a_iter, a_pnpadmm_dist, '.-')
#    plt.legend(['R-ADMM-A', 'PnP-ADMM'])
#    plt.xlabel('iter')
#    plt.ylabel('update dist')
#    plt.show()
#    
#    
#    plt.semilogx(a_iter, a_redadmm_adam_nmse, '.-',
#                a_iter, a_redadmm_romano_nmse, '.-',
#                a_iter, a_pnpadmm_nmse, '.-',
#                a_iter, a_pnppg_nmse, '.-',
#                a_iter, a_redagd_nmse, '.-')
#    plt.legend(['R-ADMM-A', 'R-ADMM-R', 'PnP-ADMM',
#                    'PnP-PG', 'RED-AGD'])
#    plt.xlabel('iter')
#    plt.ylabel('nmse')
#    plt.show()
#    
#    # plot update distance
#    
#    a_iter = range(iterations)
#    plt.semilogx(a_iter, a_redadmm_adam_dist, '.-',
#                a_iter, a_redadmm_romano_dist, '.-',
#                a_iter, a_pnpadmm_dist, '.-',
#                a_iter, a_pnppg_dist, '.-',
#                a_iter, a_redagd_dist, '.-')
#    plt.legend(['R-ADMM-A', 'R-ADMM-R', 'PnP-ADMM',
#                    'PnP-PG', 'RED-AGD'])
#    plt.xlabel('iter')
#    plt.ylabel('update dist')
#    plt.show()


if __name__ == "__main__":
    main()
    
    
    
def test_pMRI(kdata, x0, samp, maps):
    pMRI = pMRI_Op_2D_t(maps, samp)
    Ax = pMRI.mult(x0)
    print(Ax)
    AAx = pMRI.multTr(Ax)
    print(np.sum(AAx))


    
