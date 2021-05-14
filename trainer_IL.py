import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from IL.data_loader import iCIFAR10, iCIFAR100
from IL.model import *
from ML.triplet_margin_loss import TripletMarginLoss as TML
from ML.n_pairs_loss import NPairsLoss as NPL
from pytorch_metric_learning import distances
from OOD.cal import testood
import torchattacks


def show_images(images):
    N = images.shape[0]
    fig = plt.figure(figsize=(1, N))
    gs = gridspec.GridSpec(1, N)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()

transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def model_loader(args,model):
    save_dir=os.path.join(args.save_dir, args.group+'_IL')
    save_dir=os.path.join(save_dir, args.loss)
    resume_path=os.path.join(save_dir, args.name+'_checkpoint.th')
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        args.start_epoch = checkpoint['start_epoch']
        args.start_class = checkpoint['start_class']
        model.load_state_dict(checkpoint['state_dict'])
        model.exemplar_sets=checkpoint['exemplar_sets']
        model.n_classes=checkpoint['n_classes']
        model.n_known=checkpoint['n_known']
    else: print("=> no checkpoint found at '{}'".format(resume_path))
    return model,args

def save_checkpoint(state, filename): torch.save(state, filename)

def dist_helper(dist):
    assert dist in ['dotproduct']
    if dist=='dotproduct': return distances.DotProductSimilarity()

def model_helper(args):
    if args.loss=='icarl': return iCaRLNet(2048, 1, args.learning_rate)
    elif args.loss=='dce': return DCENet(2048, 1, args.learning_rate)
    elif args.loss=='tla': return MLNet(2048, 1, args.learning_rate, TML())
    elif args.loss=='nla': return MLNet(2048, 1, args.learning_rate, NPL())
    elif args.loss=='pl': return PLNet(2048, 1, args.learning_rate, dist_helper(args.dist))

def main(model,args,attack=None):
    K=args.K*args.ratio
    total_classes=10
    if args.resume or args.evaluate: model,args=model_loader(args,model)

    for s in range(args.start_class, total_classes, args.num_classes):
        if not args.evaluate:
            print('______________________________________________________________')
            print("Loading training examples for classes", range(s, s+args.num_classes))
            train_set = iCIFAR10(root='./data',
                                train=True,
                                classes=range(s,s+args.num_classes),
                                download=True,
                                transform=transform_test,
                                ratio=args.ratio)
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=args.num_workers)

        test_set = iCIFAR10(root='./data',
                            train=False,
                            classes=range(args.num_classes),
                            download=True,
                            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers)

        save_dir=os.path.join(args.save_dir, args.group+'_IL')
        save_dir=os.path.join(save_dir, args.loss)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        if not args.evaluate:
            # Update representation via BackProp
            do_class=args.start_epoch!=args.num_epochs
            if do_class:
                print('********** Update **********')
                model.update_representation(train_set,args,attack)
            else: print('********** Class trained **********')
            args.start_epoch=0

            m = K // model.n_classes

            # Reduce exemplar sets for known classes
            model.reduce_exemplar_sets(m)

            # Construct exemplar sets for new classes
            print('///////////////////////////////////')
            for y in range(model.n_known, model.n_classes):
                print("Constructing exemplar set for class-"+str(y)+'...')
                images = train_set.get_image_class(y)
                model.construct_exemplar_set(images, m, transform_test)
                print("Done")
            for y, P_y in enumerate(model.exemplar_sets):
                print("Exemplar set for class-"+str(y)+':', P_y.shape)
                #show_images(P_y[:10])
            model.n_known = model.n_classes
            print("model classes:",model.n_known)
            print('///////////////////////////////////')

            if do_class:
                total = 0.0
                correct = 0.0
                for indices, images, labels in train_loader:
                    images = Variable(images).cuda()
                    preds = model.classify(images, transform_test)
                    total += labels.size(0)
                    correct += (preds.data.cuda() == labels.cuda()).sum()
                print('Train Accuracy:', (100 * correct / total).item())

        if args.evaluate: print('evaluating, num classes:',model.n_classes)
        total = 0.0
        correct = 0.0
        for indices, images, labels in test_loader:
            images = Variable(images).cuda()
            preds = model.classify(images, transform_test)
            total += labels.size(0)
            correct += (preds.data.cuda() == labels.cuda()).sum()
        print('Test Accuracy:', (100 * correct / total).item())
        
        if attack is not None:
            total = 0.0
            correct = 0.0
            for indices, images, labels in test_loader:
                images = Variable(images).cuda()
                images = attack(images, labels.cuda())
                preds = model.classify(images, transform_test)
                total += labels.size(0)
                correct += (preds.data.cuda() == labels.cuda()).sum()
            print('Test Robustness:', (100 * correct / total).item())

        if args.evaluate: return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype learning')
    parser.add_argument('-f')
    parser.add_argument('--name', type=str,  default='model', help='model name')
    parser.add_argument('--group', type=str,  default='standard', help='experiment group')
    parser.add_argument('--loss', type=str,  default='', help='loss')
    parser.add_argument('--dist', type=str,  default='dotproduct', help='distance metric')
    parser.add_argument('--AT', type=bool,  default=False, help='use adversarial training')
    parser.add_argument('--num_classes', default=1, type=int, help='new classes num')
    parser.add_argument('--K', default=2000, type=int, help='total number of exemplars')
    parser.add_argument('--num_workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--num_epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--ratio', default=1.0, type=float, help='training data ratio')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start-class', default=0, type=int, metavar='N',
                        help='manual class number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, 
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--resume', type=bool,  default=False,
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', type=bool, default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--save-dir', dest='save_dir', default='save_temp',
                        help='The directory used to save the trained models', type=str)
    args = parser.parse_args()


    # Hyper Parameters
    args.name='test'
    args.group='test'
    args.num_classes = 1
    args.num_workers=0
    args.batch_size=32
    args.num_epochs=1
    args.learning_rate = 0.002
    args.ratio=1.0 # for FSL
    args.loss='pl'
    args.dist='dotproduct' # only for PL
    args.evaluate=False
    args.resume=False
    model = model_helper(args).cuda()


    """ 1. IL """
    # main(model,args)
    

    """ 2. ATIL """
    # args.AT=True # whether use AT 
    # # atk = torchattacks.PGD(model, eps=0.3, alpha=2/255, steps=20)
    # atk = torchattacks.FGSM(model, eps=0.3)
    # main(model,args,atk)


    """ 3. OOD Test on a Trained model (w/wo ODIN) """
    # dataname='Imagenet'
    # assert dataname in ["Imagenet","Imagenet_resize","LSUN","LSUN_resize",
    #                 "iSUN","Gaussian","Uniform"]
    # model = model_helper(args).cuda()
    # model,_=model_loader(args,model)
    # expname=args.group+'_IL_'+args.loss+'_'+args.name
    # model.eval()
    # testood(expname,model,dataname,args.num_workers)
    

    """ 4. FSIL """
    # args.ratio=0.1 
    
    
    
    
    
    