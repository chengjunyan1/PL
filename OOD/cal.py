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
    # transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),
])



criterion = nn.CrossEntropyLoss()



def testood(name, net1, dataName, num_workers, indis, CUDA_DEVICE=0, epsilon=0.0014, temperature=1000):
    
    assert dataName in ["Imagenet","Imagenet_resize","LSUN","LSUN_resize",
                    "iSUN","Gaussian","Uniform","cifar","svhn"]
    net1.cuda(CUDA_DEVICE)
    
    if dataName != "Uniform" and dataName != "Gaussian":
        if dataName=="cifar":
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            testloaderOut = torch.utils.data.DataLoader(testset, batch_size=1,
                shuffle=False, num_workers=num_workers, pin_memory=True)
        elif dataName=='svhn':
            testloaderOut = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root='./data', split='test', 
                transform=transforms.Compose([transforms.ToTensor(),]), download=True),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            testsetout = torchvision.datasets.ImageFolder("./data/{}".format(dataName), transform=transform)
            testloaderOut = torch.utils.data.DataLoader(testsetout, batch_size=1,
                                            shuffle=False, num_workers=num_workers)

    assert indis in ['cifar','svhn']

    if indis=="cifar": 
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        testloaderIn = torch.utils.data.DataLoader(testset, batch_size=1,
            shuffle=False, num_workers=num_workers, pin_memory=True)
    if indis=='svhn':
        testloaderIn = torch.utils.data.DataLoader(torchvision.datasets.SVHN(root='./data', split='test', 
            transform=transforms.Compose([transforms.ToTensor(),]), download=True),
            batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    path='./OOD/scores/'+name+'/'+dataName
    if not os.path.exists(path): os.makedirs(path)

    if dataName == "Gaussian": d.testGaussian(path, net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, dataName, epsilon, temperature)
    elif dataName == "Uniform": d.testUni(path, net1, criterion, CUDA_DEVICE, testloaderIn, testloaderIn, dataName, epsilon, temperature)
    else: d.testData(path, net1, criterion, CUDA_DEVICE, testloaderIn, testloaderOut, dataName, epsilon, temperature) 
    return m.metric(path, indis, dataName)







