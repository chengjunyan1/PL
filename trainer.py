import argparse
import os
import shutil
import time,random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import vision_models as models
from ML.triplet_margin_loss import TripletMarginLoss as TML
from ML.n_pairs_loss import NPairsLoss as NPL
from pytorch_metric_learning import distances
import torchattacks
from OOD.cal import testood


def name_helper(args):
    if args.loss=='PL':
        savename=args.model+'-D'+str(args.D)+'_'+args.name
        savename+='_'+args.lossdist+'-'+args.normdist+'_C'+str(args.C)
        savename+='_a'+str(args.ploption[0])+'-b'+str(args.ploption[1])
        savename+='_advnorm-'+str(args.adv_norm)
    else:
        savename=args.model+'-D'+str(args.D)+'_'+args.loss+'_'+args.name
    return savename

def model_loader(args,model,eval=False):
    best_prec1=0
    args.start_epoch=0
    save_dir=os.path.join(args.save_dir, args.group)
    save_dir=os.path.join(save_dir, args.dataset)
    savename=name_helper(args)
    if eval: 
        resume_path=os.path.join(save_dir, savename+'_best.th')
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['state_dict'])
        else: print("=> no checkpoint found at '{}'".format(resume_path))
        return model
    else: 
        resume_path=os.path.join(save_dir, savename+'_checkpoint.th')
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
        else: print("=> no checkpoint found at '{}'".format(resume_path))
        return model,args,best_prec1


def main(args, model, attack=None):
    save_dir=os.path.join(args.save_dir, args.group)
    save_dir=os.path.join(save_dir, args.dataset)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    # model = torch.nn.DataParallel(model)
    # model.cuda()

    best_prec1=0
    # optionally resume from a checkpoint
    if args.evaluate: model=model_loader(args,model,eval=True)
    elif args.resume: model,args,best_prec1=model_loader(args,model,eval=False)
    
    cudnn.benchmark = True

    if args.dataset=='cifar':
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        d=datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                # normalize,
            ]), download=True)
        indexes=torch.tensor(random.sample(range(d.data.shape[0]),int(args.ratio*d.data.shape[0])))
        d.data=d.data[indexes]
        d.targets=torch.Tensor(d.targets).long().index_select(0,indexes)
        train_loader = torch.utils.data.DataLoader(d,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset=='svhn':
        d=datasets.SVHN(root='./data', split='train', transform=transforms.Compose([
                # transforms.RandomCrop([54, 54]),
                transforms.ToTensor(),
            ]), download=True)
        indexes=torch.tensor(random.sample(range(d.data.shape[0]),int(args.ratio*d.data.shape[0])))
        d.data=d.data[indexes]
        d.labels=torch.Tensor(d.labels).long().index_select(0,indexes)
        train_loader = torch.utils.data.DataLoader(d,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root='./data', split='test', transform=transforms.Compose([
                transforms.ToTensor(),
            ]), download=True),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset=='mnist':
        d=datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,))
                            ]))
        indexes=torch.tensor(random.sample(range(d.data.shape[0]),int(args.ratio*d.data.shape[0])))
        d.data=d.data.index_select(0,indexes)
        d.targets=d.targets.index_select(0,indexes)
        train_loader = torch.utils.data.DataLoader(d,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=False, download=True,
                           transform=transforms.Compose([
                                transforms.ToTensor(),
                                # transforms.Normalize((0.1307,), (0.3081,))
                            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': args.lr}], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.evaluate:
        print('Evaluating...')
        validate(args, val_loader, model)
        if attack: validate(args, val_loader, model, attack)
        return

    ts=time.time()
    for epoch in range(args.start_epoch, args.epochs):
        te=time.time()
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader, model, optimizer, epoch, attack)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(args, val_loader, model)
        robust=0
        if attack: robust=validate(args, val_loader, model, attack)

        # remember best prec@1 and save checkpoint
        is_best = prec1+robust > best_prec1
        best_prec1 = max(prec1+robust, best_prec1)
        print('Epoch',epoch+1,'/',args.epochs,'time:',time.time()-te)

        savename=name_helper(args)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(save_dir, savename+'_checkpoint.th'))

        if is_best:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(save_dir, savename+'_best.th'))
    print('Accomplished. Total time:',time.time()-ts)

def train(args, train_loader, model, optimizer, epoch, attack=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    robust = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        bs=input.size(0)

        # compute output
        adversarial_inputs=None
        if args.AT and attack:
            model.eval()
            if args.loss=='PL' and args.adv_norm:
                adversarial_inputs = attack(input_var, target_var)
            else:
                adversarial_inputs = attack(input_var[bs//2:], target_var[bs//2:])
                input_var = torch.cat((input_var[:bs//2], adversarial_inputs), dim=0)
                # target_var=torch.cat((target_var, target_var), dim=0)
            model.train()

        if args.loss=='DCE':
            features, centers, output= model(input_var,True)
            loss=model.loss(output,target_var,features,centers) 
        elif args.loss in ['TLA','NLA']:
            output, embeds= model(input_var,True)
            loss=model.loss(output,embeds,target_var)
        elif args.loss=='PL':
            output, distance, x= model(input_var,True)
            if args.AT and attack and args.adv_norm:
                output_adv, distance_adv, x_adv= model(adversarial_inputs,True)
                loss=model.loss(output,x,distance,target_var,x_adv,option=args.ploption)
            else: 
                loss=model.loss(output,x,distance,target_var,option=args.ploption)
        else:
            output,_ = model(input_var,True)
            loss = model.loss(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        robust1=0
        if args.AT and attack:
            if args.loss=='PL' and args.adv_norm:
                prec1 = accuracy(output.data, target_var)[0]
                robust1 = accuracy(output_adv.data, target_var)[0]
            else:    
                prec1 = accuracy(output.data[:bs//2], target_var[:bs//2])[0]
                robust1 = accuracy(output.data[bs//2:], target_var[bs//2:])[0]
            robust.update(robust1.item(), bs)
        else:
            prec1 = accuracy(output.data, target_var)[0]

        losses.update(loss.item(), bs)
        top1.update(prec1.item(), bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Robust {robust.val:.4f} ({robust.avg:.4f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1,robust=robust))


def validate(args, val_loader, model, attack=None):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    top1 = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()

        if attack: input_var = attack(input_var, target_var)

        with torch.no_grad():
            output = model(input_var)

        output = output.float()
        prec1 = accuracy(output.data, target)[0]
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if not attack: print(' Accuracy {top1.avg:.3f}'.format(top1=top1))
    else: print(' Robustness {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'): torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def atk_helper(args,model,eps):
    if args.atk=='pgd': return torchattacks.PGD(model, eps=eps, alpha=2/255, steps=20)
    if args.atk=='pgdl2': 
        eps=3 if args.dataset=='mnist' else 2
        return torchattacks.PGDL2(model, eps=eps, alpha=2/255, steps=20)
    if args.atk=='pgdrs': return torchattacks.PGD(model, eps=eps, alpha=2/255, steps=7,random_start=True)
    elif args.atk=='fgsm': return torchattacks.FGSM(model, eps=eps)
    elif args.atk=='bim': return torchattacks.BIM(model, eps=eps, alpha=1/255, steps=20)
    else: print('No attack.'); return None

def model_helper(args):
    assert args.loss in ['DCE','vanilla','PL','TLA','NLA']
    if args.loss=='DCE': print('Using DCE Loss')
    elif args.loss=='PL': print('Using PL Loss')
    elif args.loss=='TLA': print('Using Triplet Loss')
    elif args.loss=='NLA': print('Using N-pair Loss')
    elif args.loss=='vanilla': print('Using Vanilla model')
    modelname='conv' if args.dataset=='mnist' else args.model 
    print('Using',modelname,'model')
    if args.loss=='DCE': return models.DCEmodel(modelname,args.D).cuda()
    elif args.loss=='PL': return models.PLmodel(modelname,args.C,args.D,args.lossdist,args.normdist,args.preddist).cuda()
    elif args.loss=='TLA': return models.MLmodel(TML(),modelname,args.D).cuda()
    elif args.loss=='NLA': return models.MLmodel(NPL(),modelname,args.D).cuda()
    elif args.loss=='vanilla': return models.Vanillamodel(modelname,args.D).cuda()


def AR_test(atks,losses,dataset,backbones):
    print('\n','*'*50,'\nAdversarial robustness test start.\n')
    args.evaluate=True
    for d in dataset:
        print('===================== Dataset: '+d+' =====================')
        args.dataset=d
        if d=='mnist': args.epochs=10;eps=0.3;backbone=['conv']
        if d=='cifar': args.epochs=200;eps=8/255;backbone=backbones
        if d=='svhn': args.epochs=200;eps=8/255;backbone=backbones
        for m in backbone:
            if d!='mnist' and m=='conv': continue
            args.model=m
            print('\n+----------------------- Backbone: '+m,' --------------------+')
            for args.atk in atks:
                print('\n~~~~~~~~~~~~~~~~~~~~~~ Attack: '+args.atk.upper()+' ~~~~~~~~~~~~~~~~~~~~~~')
                for i in losses:
                    print('______________________ Loss: '+i+' ____________________')
                    args.loss=i
                    model=model_helper(args)
                    atk=atk_helper(args,model,eps)
                    main(args,model,atk)
    
def OOD_test(ood_dataset,losses,dataset,backbones):
    print('\n','*'*50,'\nOOD robustness test start.\n')
    for m in backbones:
        if m=='conv': continue
        args.model=m
        print('+----------------------- Backbone: '+m,' --------------------+\n')
        for indis in dataset:
            if indis=='mnist': continue
            print('===================== Indis: '+indis+' =====================\n')
            args.dataset=indis
            for dataname in ood_dataset:
                for i in losses:
                    print('______________________ Loss: '+i+' ____________________')
                    args.loss=i
                    model=model_helper(args)
                    model=model_loader(args,model,eval=True) # it will load model based on args
                    expname=args.group+'_'+indis+'_'+args.model+'-D'+str(args.D)+'_'+args.loss+'_'+args.name # not useful actually
                    model.eval()
                    testood(expname,model,dataname,args.workers,indis)
                    
def trainer(args,losses,dataset,backbones):
    print('\n','*'*50,'\nTraining start.\n')
    if args.AT: print('Training with attack: '+args.atk.upper())
    for d in dataset:
        print('===================== Dataset: '+d+' =====================')
        args.dataset=d
        if d=='mnist': args.epochs=10;eps=0.3;backbone=['conv']
        if d=='cifar': args.epochs=200;eps=8/255;backbone=backbones
        if d=='svhn': args.epochs=200;eps=8/255;backbone=backbones
        for m in backbone:
            if d!='mnist' and m=='conv': continue
            args.model=m
            print('+----------------------- Backbone: '+m,' --------------------+')
            for i in losses:
                print('______________________ Loss: '+i+' ____________________')
                args.loss=i
                atk=None
                if args.AT:
                    print('Attack: '+args.atk.upper())
                    args.batch_size=64 if i=='TLA' else 256
                    atk=atk_helper(args,eps)
                model=model_helper(args)
                main(args,model,atk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prototype learning')
    parser.add_argument('-f')
    parser.add_argument('--name', type=str,  default='model', help='model name')
    parser.add_argument('--group', type=str,  default='standard', help='experiment group')
    parser.add_argument('--loss', type=str,  default='', help='loss')
    parser.add_argument('--model', type=str,  default='resnet', help='distance metric')
    parser.add_argument('--atk', type=str,  default='', help='atk')
    parser.add_argument('--C', type=int,  default=1, help='number of prototypes')
    parser.add_argument('--D', type=int,  default=64, help='d_model')
    parser.add_argument('--lossdist', type=str,  default='L2', help='loss distance metric')
    parser.add_argument('--normdist', type=str,  default='L2', help='norm distance metric')
    parser.add_argument('--preddist', type=str,  default='L2', help='pred distance metric')
    parser.add_argument('--adv_norm', type=bool,  default=True, help='seperate adv norm term')
    parser.add_argument('--ploption', type=list,  default=[0.1,0.1], help='[a,b]')
    parser.add_argument('--AT', type=bool,  default=False, help='use adversarial training')
    parser.add_argument('--dataset', type=str,  default='mnist', help='dataset')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--ratio', default=1.0, type=float, help='training data ratio')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, 
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-4,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', type=int, default=50,
                        metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', type=bool,  default=False,
                        metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', type=bool, default=False,
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', type=bool, default=False,
                        help='use pre-trained model')
    parser.add_argument('--save-dir', dest='save_dir', default='save_temp',
                        help='The directory used to save the trained models', type=str)
    parser.add_argument('--save-every', dest='save_every', default=10,
                        help='Saves checkpoints at every specified number of epochs', type=int)
    args = parser.parse_args()


    """
    Naming convention:
        args.name: experiment name, arbitrary
        args.group: 
            woat: no AT
            fgsm: trained with fgsm
            ...
    """

    args.name='test' 
    args.group='woat' #work only for non-PL models
    args.batch_size=256
    args.lr=1e-2
    args.ratio=1.0 # For FSL
    
    args.adv_norm=True # use a seperate adv norm term
    args.C=2
    args.D=64
    args.lossdist='L2'
    args.normdist='L2'
    args.preddist='L2'
    args.ploption=[0.1,0.2] # a,b

    # backbones=['resnet','vgg','mobilenet','conv']
    backbones=['mobilenet']
    # losses=['PL','vanilla','DCE','TLA','NLA']
    losses=['PL','vanilla']
    # dataset=['mnist','cifar','svhn']
    dataset=['mnist']


    args.resume=True
    args.evaluate=False
    
    """ Train """
    args.AT=False # whether do AT
    args.atk=None
    trainer(args,losses,dataset,backbones)


    """ Adversarial Robustness Test """
    atks=['fgsm','pgd','bim','pgdrs','pgdl2']
    AR_test(atks,losses,dataset,backbones)
    

    """ OOD Test on a Trained model (w/wo ODIN) """
    ood_dataset=["Imagenet","Imagenet_resize","LSUN","LSUN_resize",
                    "iSUN","Gaussian","Uniform"]
    OOD_test(ood_dataset,losses,dataset,backbones)



    