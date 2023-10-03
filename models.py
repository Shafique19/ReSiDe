import torch.nn as nn
import torch.nn.init as init
import torch

class MeanOnlyBatchNorm_2d(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm_2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.bias = nn.Parameter(torch.zeros(num_features))
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
    
class MeanOnlyBatchNorm_3d(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm_3d, self).__init__()
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
class BasicNet2D(nn.Module):
    def __init__(self):
        layers = []
        imchannel = 2
        filternum = 128
        filtersize = 3
        depth = 3
        super(BasicNet2D, self).__init__()        
        layers.append(nn.utils.spectral_norm(nn.Conv2d(imchannel, filternum, filtersize, padding=1, bias=True), n_power_iterations=20))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth):
            layers.append(nn.utils.spectral_norm(nn.Conv2d(filternum, filternum, filtersize, padding=1, bias=False), n_power_iterations=20))
            layers.append(MeanOnlyBatchNorm_2d(filternum,momentum=0.95))
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
                    
class BasicNet3D(nn.Module):
    def __init__(self):
        layers = []
        imchannel = 2
        filternum = 128
        filtersize = 3
        depth = 3
        super(BasicNet3D, self).__init__()        
        layers.append(nn.utils.spectral_norm(nn.Conv3d(imchannel, filternum, filtersize, padding=1, bias=True), n_power_iterations=20))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth):
            layers.append(nn.utils.spectral_norm(nn.Conv3d(filternum, filternum, filtersize, padding=1, bias=False), n_power_iterations=20))
            layers.append(MeanOnlyBatchNorm_3d(filternum,momentum=0.95))
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