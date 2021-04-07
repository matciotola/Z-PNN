# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:15:52 2020

@author: sergv
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:25:47 2020

@author: sergv
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:25:47 2020

@author: sergv
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
#import numpy as np

def mse_loss(output,ref):
    loss = 0.5*torch.sum((output-ref)**2)
    return loss

def mae_loss(output,ref):
    loss = torch.sum(abs(output-ref))
    return loss

class Net(nn.Module):
    def __init__(self,layers):
        super(Net, self).__init__()
        
#        Conv2d(ch_in,ch_out,k,k)
        self.conv1 = nn.Conv2d(layers[0][0],layers[0][1],layers[0][2])
        # torch.nn.init.normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(layers[1][0],layers[1][1],layers[1][2])
        # torch.nn.init.normal_(self.conv2.weight)
        self.conv3 = nn.Conv2d(layers[2][0],layers[2][1],layers[2][2])        
        # torch.nn.init.normal_(self.conv3.weight)
        self.layer = [self.conv1,self.conv2,self.conv3]
        self.layers = layers
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
        


#import torch.nn as nn
#import torch.nn.functional as F
#import torch
##import numpy as np
#def mae_loss(output,target):
#    loss = torch.sum(torch.abs(output-target))
#    return loss
#
#
#class Net(nn.Module):
#    def __init__(self,layers):
#        super(Net, self).__init__()
#        
#
#        sh = [layer.shape for layer in layers]
#        self.conv1 = nn.Conv2d(sh[0][-2],sh[0][-1],sh[0][-3])
#        self.conv2 = nn.Conv2d(sh[2][-2],sh[2][-1],sh[2][-3])
#        self.conv3 = nn.Conv2d(sh[4][-2],sh[4][-1],sh[4][-3])
#        
#        
#    def forward(self, x):
#        x = F.relu(self.conv1(x))
#        x = F.relu(self.conv2(x))
#        x = self.conv3(x)
#        return x
    
        

