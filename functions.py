import numpy as np
import torch
from torch.utils.data import DataLoader



def NMSE(true,b):
    y = 20*np.log10(np.linalg.norm(true-b)/np.linalg.norm(true))
    return y

def complex_random_crop_2d(data, shape):
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = np.random.randint(0,data.shape[-3]-shape[0]+1)
    h_from = np.random.randint(0,data.shape[-2]-shape[1]+1)
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :], w_from,w_to, h_from,h_to
def complex_random_crop_3d(data, shape):
    assert 0 < shape[0] <= data.shape[-4]
    assert 0 < shape[1] <= data.shape[-3]
    assert 0 < shape[2] <= data.shape[-2]
    w_from = np.random.randint(0,data.shape[-4]-shape[0]+1)
    h_from = np.random.randint(0,data.shape[-3]-shape[1]+1)
    l_from = np.random.randint(0,data.shape[-2]-shape[2]+1)
    # w_from = (data.shape[-3] - shape[0]) // 2
    # h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    l_to = l_from + shape[2]
    return data[..., w_from:w_to, h_from:h_to, l_from:l_to,:], w_from,w_to, h_from,h_to, l_from,l_to


class create_datasets():
    def __init__(self,midvar,snr,dimension, args):
        self.keep_slices_o = []
        self.keep_slices_i = []
        mid = midvar/abs(np.real(midvar)).max()
        sigma = np.linalg.norm(mid)/np.sqrt(mid.size)/(10**(snr/20))/np.sqrt(2)
        data_out = torch.from_numpy(np.stack((mid.real, mid.imag), axis=-1)).float()
        for i in range(args.patches):
            if dimension == '2D':
                data_o, w_from,w_to, h_from,h_to = complex_random_crop_2d(data_out,args.patchSize)
                data_i = data_o + sigma*torch.randn(data_o.size())
                self.keep_slices_o.append(data_o.permute(2,0,1))
                self.keep_slices_i.append(data_i.permute(2,0,1))
            if dimension == '3D':       
                data_o, w_from,w_to, h_from,h_to, l_from,l_to = complex_random_crop_3d(data_out,(64,64,20))
                data_i = data_o + sigma*torch.randn(data_o.size())
                self.keep_slices_o.append(data_o.permute(3,0,1,2))
                self.keep_slices_i.append(data_i.permute(3,0,1,2))
    def __len__(self):
        return len(self.keep_slices_o)
    def __getitem__(self, index):
        outp = self.keep_slices_o[index]
        inp = self.keep_slices_i[index]
        return inp, outp
    
def create_data_loaders(midvar,snr,dimension,args):

    train_loader = DataLoader(
        create_datasets(midvar,snr,dimension,args),
        batch_size=args.batchSize,
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
    x = np.expand_dims(x,axis = (0,1))
    x_im = np.array(np.concatenate((np.real(x),np.imag(x)),1),dtype = 'float32')
    x_im = torch.from_numpy(x_im).cuda()
    w = model(x_im).cpu().detach().numpy().astype(np.float32)
    w = np.squeeze(w[:,0,:,:]+1j*w[:,1,:,:])
    
    return w

