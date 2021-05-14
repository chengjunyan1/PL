# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

from __future__ import print_function
import torch,os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from scipy import misc
import OOD.calMetric as m
import OOD.calData as d
#CUDA_DEVICE = 0

start = time.time()
#loading data sets

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])




# loading neural network

# Name of neural networks
# Densenet trained on CIFAR-10:         densenet10
# Densenet trained on CIFAR-100:        densenet100
# Densenet trained on WideResNet-10:    wideresnet10
# Densenet trained on WideResNet-100:   wideresnet100
#nnName = "densenet10"

#imName = "Imagenet"



criterion = nn.CrossEntropyLoss()



def testood(name, net1, dataName, num_workers, indis="CIFAR-10", CUDA_DEVICE=0, epsilon=0.0014, temperature=1000):
    
    assert dataName in ["Imagenet","Imagenet_resize","LSUN","LSUN_resize",
                    "iSUN","Gaussian","Uniform"]
    net1.cuda(CUDA_DEVICE)
    
    if dataName != "Uniform" and dataName != "Gaussian":
        testsetout = torchvision.datasets.ImageFolder("./data/{}".format(dataName), transform=transform)
        testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                         shuffle=False, num_workers=num_workers)

    # if indis=='MNIST':
    #     testloaderIn = torch.utils.data.DataLoader(
    #         torchvision.datasets.MNIST(root='./data', train=False, download=True,
    #             transform=transforms.Compose([transforms.ToTensor(),])),
    #         batch_size=1, shuffle=False,num_workers=num_workers, pin_memory=True)

    if indis=="CIFAR-10": 
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=num_workers)
    if indis=="CIFAR-100":
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=num_workers)
    
    
    path='./OOD/scores/'+name+'/'+dataName
    if not os.path.exists(path): os.makedirs(path)

    if dataName == "Gaussian": d.testGaussian(path, net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, dataName, epsilon, temperature)
    elif dataName == "Uniform": d.testUni(path, net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, dataName, epsilon, temperature)
    else: d.testData(path, net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, dataName, epsilon, temperature) 
    m.metric(path, indis, dataName)







