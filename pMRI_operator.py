import numpy as np



class pMRI_operator:
    
    def __init__(self,maps,pattern):
        self.C = maps
        self.frame_size = [np.size(self.C,0),np.size(self.C,1)]
        self.patterns = pattern


    def mult(self,x):
       
        x = np.expand_dims(x, axis=2)
        y = np.fft.fft2(x*self.C,axes=(0,1),norm='ortho')
        y = y*self.patterns
        return y    

    def multTr(self,y):
        y = y*self.patterns
        y_ifft = np.fft.ifft2(y,axes=(0,1),norm='ortho')
        y = np.sum(y_ifft*np.conj(self.C), axis = 2)
        return y     
    
    