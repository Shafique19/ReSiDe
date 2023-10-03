import os
import numpy as np
from pMRI_operator import pMRI_operator
from scipy.io import loadmat







def data_load(i,study,samp_pattern,mode):
    
    if study == 'Brain':
        dimension = '2D'
        k_full = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/kspaces_and_true_images/k_'+str(i)+'.mat')['k_full']        
        lsq = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/kspaces_and_true_images/im_'+str(i)+'.mat')['imtrue']           
        if samp_pattern == 'GRO':
            S = np.squeeze(loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/Gro_samp_and_sens_maps/r4_gro_map_k'+str(i)+'.mat')['map'])    
            samp = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/Gro_samp_and_sens_maps/R4_gro.mat')['samp']                                                                                                                                                                                                                                                                                                                   
        if samp_pattern == 'randomlines':
            S = np.squeeze(loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/Random_samp_and_sens_maps/r4_randoml_map_k'+str(i)+'.mat')['map'])
            samp = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/Random_samp_and_sens_maps/R4_randoml_k'+str(i)+'.mat')['samp']
            
        noise_power = np.var(np.concatenate((k_full[:4,:,:].reshape(-1),k_full[-4:,:,:].reshape(-1),k_full[4:-4,:4,:].reshape(-1),k_full[4:-4,-4:,:].reshape(-1))))
        k_samp = k_full*np.expand_dims(samp,axis=2)
        k_shifted = np.fft.fftshift(np.fft.fftshift(k_samp,axes = 0), axes = 1)
        samp_ = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)
        samp_shifted = np.tile(np.expand_dims(samp_,axis=2),[1,1,np.size(k_full,2)]) 
        kdata = k_shifted
        # kdata = k_shifted.flatten('F')    
        # kdata = kdata[np.where(samp_shifted.flatten('F')>0)]    
        S = np.fft.fftshift(np.fft.fftshift(S,axes = 0), axes = 1)
        pMRI = pMRI_operator(S, samp_shifted)         
        x = pMRI.multTr(kdata)
        
    if study == 'Perfusion':
        dimension = '3D'
        if mode == 'MRXCAT':
            kdata = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/perf_phantom_'+str(i)+'_k.mat')['k_full']
            lsq = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/perf_phantom_'+str(i)+'_imtrue.mat')['imtrue']
            S = np.squeeze(loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/perf_phantom_'+str(i)+'_map_R4.mat')['map'])
            x0 = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/perf_phantom_'+str(i)+'_x0_R4.mat')['x0']
            samp = np.float32(loadmat(os.getcwd()+'/'+study+'/'+mode+'/perf_phantom_samp_R4.mat')['samp'])
            samp_ = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)
            samp_shifted = np.tile(np.expand_dims(samp_,axis=2),(1,1,np.size(S,2),1))
            if i == 17:
                noise_power = 6.2087
            if i == 18:
                noise_power = 6.0969
            if i == 19:
                noise_power = 6.0223
            if i == 20:
                noise_power = 6.0819
            if i == 21:
                noise_power = 6.1727
            
        if mode == 'Perf':
            kdata = loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/k_'+str(i)+'.mat')['k']
            samp = np.float32(loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/samp_'+str(i)+'.mat')['samp'])  
            S = np.squeeze(loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/map_'+str(i)+'.mat')['map'])
            x0 = np.squeeze(loadmat(os.getcwd()+'/'+study+'/'+mode+'/data_for_testing/x0_'+str(i)+'.mat')['x0'])
            samp_ = np.fft.fftshift(np.fft.fftshift(samp,axes = 0), axes = 1)
            samp_shifted = np.tile(np.expand_dims(samp_,axis=2),(1,1,np.size(S,2),1))
            noise_power = 1

        kdata = kdata*np.expand_dims(samp,axis=2)
        S = np.tile(np.expand_dims(S,axis = 3),[1,1,1,np.size(samp,2)])              
        kdata = np.fft.fftshift(np.fft.fftshift(kdata,axes = 0), axes = 1)
        S = np.fft.fftshift(np.fft.fftshift(S,axes = 0), axes = 1)
        x0 = np.fft.fftshift(np.fft.fftshift(x0,axes = 0), axes = 1)
        x = np.tile(np.expand_dims(x0,axis=2),[1,1,np.size(samp,2)])
        pMRI = pMRI_operator(S, samp_shifted)
        


        
    return x, kdata, lsq, pMRI, dimension, noise_power

