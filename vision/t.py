import torch


from vgg import vgg11
from resnet import resnet18
from mobilenet import mobilenet_v3_small


resnet=resnet18()
vgg=vgg11()
mobilenet=mobilenet_v3_small()


q=torch.randn(2,3,32,32)

w1=resnet(q)
w2=vgg(q)
w3=mobilenet(q)

