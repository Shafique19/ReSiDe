import os
import numpy as np
import torch
from torch.nn import functional as F
from scipy.io import savemat
import torch.optim as optim
import parser_ops, preprocess
import models
import functions as fun

study = 'Brain'                        # Brain or Perfusion
samp_pattern = 'GRO'                   # GRO or randomlines
mode = 'T1'                            # T1, T2, MRXCAT, Perf


parser = parser_ops.get_parser('reside-s',study,mode)
args = parser.parse_args()
snr = args.snr
rho = args.rho 


for i in range(17,22):
    print('Running '+study+' '+mode+' with '+samp_pattern+'Samp on Dataset '+str(i))
    x, kdata, lsq, pMRI, dimension, noise_power = preprocess.data_load(i,study,samp_pattern,mode)
    z = pMRI.mult(x)-kdata
    p = fun.powerite(pMRI,x.shape)
    gamma_p = rho/p
    for ite in range(args.iterations): 
        xold = x
        midvar = xold-1/rho*pMRI.multTr(z)
        midvar = np.fft.fftshift(np.fft.fftshift(midvar,axes=0),axes = 1)       
            
        if dimension == '2D':
            model = models.BasicNet2D()
        if dimension == '3D':
            model = models.BasicNet3D()
        model.train()
        device = torch.device('cuda:0')
        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=args.learning_rate)
        train_loader = fun.create_data_loaders(midvar,snr,dimension,args)
        for epoch in range(0,args.ep):
            for train_iter, Data in enumerate(train_loader):
                x_batch,y_batch = Data
                out = model(x_batch.to(device, dtype=torch.float))
                loss = F.mse_loss(out,y_batch.to(device, dtype=torch.float), reduction='sum')
                opt.zero_grad()
                loss.backward()
                opt.step()
        # torch.save(model,os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/pymodel_%03d.pth' % (ite+1))
            
        midvar_norm = midvar/np.abs(np.real(midvar)).max()            
        im = fun.apply_denoiser(midvar_norm, model)
        im = im* np.abs(np.real(midvar)).max() 
        x = np.fft.fftshift(np.fft.fftshift(im,axes=0),axes = 1)
        s = 2*x-xold
        z = 1/(1+gamma_p)*z+gamma_p/(1+gamma_p)*(pMRI.mult(s)-kdata)
        para = np.linalg.norm(pMRI.mult(x)-kdata)**2/np.count_nonzero(kdata)/noise_power/args.tau      
        if ite > 2:
            snr = snr*para**args.alpha
        print("training snr: " + repr(snr))
        nmse_i = fun.NMSE(lsq,im)
        print("normalized mean square error of x: " + repr(nmse_i))
    file_name = os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/im_'+str(i)+'_snr_'+str(-nmse_i)+'.mat'
    savemat(file_name,{'x':im})




