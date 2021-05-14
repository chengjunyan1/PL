import torch,os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from PIL import Image

from IL.resnet import resnet18
from pytorch_metric_learning import distances


def save_checkpoint(state, filename): torch.save(state, filename)

class ILBase(nn.Module):
    def __init__(self, feature_size, n_classes):
        super(ILBase, self).__init__()
        self.feature_extractor = resnet18()
        self.feature_extractor.fc =\
            nn.Linear(self.feature_extractor.fc.in_features, feature_size)
        self.exemplar_sets = []

    def construct_exemplar_set(self, images, m, transform):
        features = []
        for img in images:
            with torch.no_grad():
                x = transform(Image.fromarray(img)).cuda()
            feature = self.feature_extractor(x.unsqueeze(0)).data.cpu().numpy()
            feature = feature / np.linalg.norm(feature) # Normalize
            features.append(feature[0])

        features = np.array(features)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

        exemplar_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in range(int(m)):
            S = np.sum(exemplar_features, axis=0)
            phi = features
            mu = class_mean
            mu_p = 1.0/(k+1) * (phi + S)
            mu_p = mu_p / np.linalg.norm(mu_p)
            i = np.argmin(np.sqrt(np.sum((mu - mu_p) ** 2, axis=1)))
            exemplar_set.append(images[i])
            exemplar_features.append(features[i])
        self.exemplar_sets.append(np.array(exemplar_set))

    def reduce_exemplar_sets(self, m):
        for y, P_y in enumerate(self.exemplar_sets):
            self.exemplar_sets[y] = P_y[:int(m)]

    def combine_dataset_with_exemplars(self, dataset):
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = P_y
            exemplar_labels = [y] * len(P_y)
            dataset.append(exemplar_images, exemplar_labels)


class iCaRLNet(ILBase):
    def __init__(self, feature_size, n_classes, learning_rate):
        super(iCaRLNet, self).__init__(feature_size, n_classes)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)
        self.n_classes = n_classes
        self.n_known = 0
        # Learning method
        self.cls_loss = nn.CrossEntropyLoss()
        self.dist_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        # Means of exemplars
        self.compute_means = True
        self.exemplar_means = []

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.bn(x)
        x = self.ReLU(x)
        x = self.fc(x)
        return x

    def increment_classes(self, n):
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def classify(self, x, transform):
        batch_size = x.size(0)
        if self.compute_means:
            print("Computing mean of exemplars...")
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                for ex in P_y:
                    ex = Variable(transform(Image.fromarray(ex)), volatile=True).cuda()
                    feature = self.feature_extractor(ex.unsqueeze(0))
                    feature = feature.squeeze()
                    feature.data = feature.data / feature.data.norm() # Normalize
                    features.append(feature)
                features = torch.stack(features)
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm() # Normalize
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print("Done")

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means) # (n_classes, feature_size)
        means = torch.stack([means] * batch_size) # (batch_size, n_classes, feature_size)
        means = means.transpose(1, 2) # (batch_size, feature_size, n_classes)

        feature = self.feature_extractor(x) # (batch_size, feature_size)
        for i in range(feature.size(0)): # Normalize
            feature.data[i] = feature.data[i] / feature.data[i].norm()
        feature = feature.unsqueeze(2) # (batch_size, feature_size, 1)
        feature = feature.expand_as(means) # (batch_size, feature_size, n_classes)

        dists = (feature - means).pow(2).sum(1).squeeze() #(batch_size, n_classes)
        _, preds = dists.min(1)
        return preds

    def update_representation(self, dataset, args, attack=None):
        num_workers, batch_size, num_epochs=args.num_workers,args.batch_size,args.num_epochs
        self.compute_means = True
        # Increment number of weights in final fc layer
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self.cuda()
        print(len(new_classes),"new classes")
        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
        # Store network outputs with pre-update parameters
        q = torch.zeros(len(dataset), self.n_classes).cuda()
        for indices, images, labels in loader:
            images = Variable(images).cuda()
            indices = indices.cuda()
            g = torch.sigmoid(self.forward(images))
            q[indices] = g.data
        q = Variable(q).cuda()

        # Run network training
        save_dir=os.path.join(args.save_dir, args.group+'_IL')
        save_dir=os.path.join(save_dir, args.loss)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        optimizer = self.optimizer
        for epoch in range(args.start_epoch,num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                pred = self.forward(images)
                loss=self.icarl_loss(pred,labels,q,indices)
                if args.AT and attack is not None:
                    images_adv = attack(images, labels)
                    pred_adv = self.forward(images_adv)
                    loss+=self.cls_loss(pred_adv, labels)

                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, 
                            len(dataset)//batch_size, loss.float().item()))
            print('Epoch [{:d}/{:d}], samples: {:d}, Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, len(dataset), loss.float().item()))
            
            save_checkpoint({
                'start_epoch': epoch + 1,
                'start_class': self.n_classes - 1,
                'state_dict': self.state_dict(),
                'exemplar_sets': self.exemplar_sets,
                'n_classes': self.n_classes,
                'n_known': self.n_known,
            }, filename=os.path.join(save_dir, args.name+'_checkpoint.th'))
            print('* Epoch Checkpoint saved. *')


    def icarl_loss(self,g,labels,q,indices):
        loss = self.cls_loss(g, labels) # CE loss
        if self.n_known > 0: # Distilation loss for old classes
            g = torch.sigmoid(g)
            q_i = q[indices]
            dist_loss = sum(self.dist_loss(g[:,y], q_i[:,y])\
                    for y in range(self.n_known))
            loss += dist_loss
        return loss


class DCENet(ILBase):
    def __init__(self, feature_size, n_classes, learning_rate):
        super(DCENet, self).__init__(feature_size, n_classes)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(feature_size, 2)
        self.dce=dce_loss(10,2)
        self.apply(_weights_init)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        self.n_classes = n_classes
        self.n_known = 0

    def forward(self, x, embed=False, scale=2):
        x = self.feature_extractor(x)
        x = self.bn(x)
        features = self.preluip1(self.ip1(x))
        centers,distance=self.dce(features)
        if embed: return features,centers[:,:self.n_classes],distance[:,:self.n_classes] #features, centers,distance
        return distance[:,:self.n_classes]

    def classify(self, x, transform=None): return self(x).max(1)[1]

    def update_representation(self, dataset, args, attack=None):
        num_workers, batch_size, num_epochs=args.num_workers,args.batch_size,args.num_epochs
        self.compute_means = True
        # Increment number of weights in final fc layer
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.n_classes += len(new_classes)
        self.cuda()
        print(len(new_classes),"new classes")
        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

        # Run network training
        save_dir=os.path.join(args.save_dir, args.group+'_IL')
        save_dir=os.path.join(save_dir, args.loss)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        optimizer = self.optimizer
        for epoch in range(args.start_epoch,num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                if args.AT and attack is not None:
                    images_adv = attack(images, labels)
                    images = torch.cat((images, images_adv), dim=0)
                    labels=torch.cat((labels, labels), dim=0)
                features, centers, output= self.forward(images,True)
                loss=self.loss(output,labels,features,centers) 

                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, 
                            len(dataset)//batch_size, loss.float().item()))
            print('Epoch [{:d}/{:d}], samples: {:d}, Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, len(dataset), loss.float().item()))
                    
            save_checkpoint({
                'start_epoch': epoch + 1,
                'start_class': self.n_classes - 1,
                'state_dict': self.state_dict(),
                'exemplar_sets': self.exemplar_sets,
                'n_classes': self.n_classes,
                'n_known': self.n_known,
            }, filename=os.path.join(save_dir, args.name+'_checkpoint.th'))
            print('* Epoch Checkpoint saved. *')

    def loss(self,distance,label,features,centers,reg=0.001):  
        loss1 = self.criterion(distance, label)
        loss2=regularization(features, centers, label)
        return loss1+reg*loss2


class MLNet(ILBase):
    def __init__(self, feature_size, n_classes, learning_rate, mlloss):
        super(MLNet, self).__init__(feature_size, n_classes)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.mlloss=mlloss
        self.ReLU = nn.ReLU()
        self.fc = nn.Linear(feature_size, n_classes, bias=False)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        self.n_classes = n_classes
        self.n_known = 0

    def forward(self, x, embed=False):
        x = self.feature_extractor(x)
        x = self.bn(x)
        pred=self.fc(x)
        if embed: return pred,x
        return pred

    def classify(self, x, transform=None): return self(x).max(1)[1]

    def increment_classes(self, n):
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features+n, bias=False)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

    def update_representation(self, dataset, args, attack=None):
        num_workers, batch_size, num_epochs=args.num_workers,args.batch_size,args.num_epochs
        self.compute_means = True
        # Increment number of weights in final fc layer
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.increment_classes(len(new_classes))
        self.cuda()
        print(len(new_classes),"new classes")
        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

        # Run network training
        save_dir=os.path.join(args.save_dir, args.group+'_IL')
        save_dir=os.path.join(save_dir, args.loss)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        optimizer = self.optimizer
        for epoch in range(args.start_epoch,num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                if args.AT and attack is not None:
                    images_adv = attack(images, labels)
                    images = torch.cat((images, images_adv), dim=0)
                    labels=torch.cat((labels, labels), dim=0)
                output, embeds= self.forward(images,True)
                loss=self.loss(output,embeds,labels)
                
                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, 
                            len(dataset)//batch_size, loss.float().item()))
            print('Epoch [{:d}/{:d}], samples: {:d}, Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, len(dataset), loss.float().item()))
                    
            save_checkpoint({
                'start_epoch': epoch + 1,
                'start_class': self.n_classes - 1,
                'state_dict': self.state_dict(),
                'exemplar_sets': self.exemplar_sets,
                'n_classes': self.n_classes,
                'n_known': self.n_known,
            }, filename=os.path.join(save_dir, args.name+'_checkpoint.th'))
            print('* Epoch Checkpoint saved. *')

    def loss(self,pred,embeds,y,a=0.3,b=1e-4): 
        if self.n_classes==1: return self.criterion(pred, y)
        return self.criterion(pred, y)+self.mlloss(embeds,y,a,b)


class PLNet(ILBase):
    def __init__(self, feature_size, n_classes, learning_rate, distance):
        super(PLNet, self).__init__(feature_size, n_classes)
        self.bn = nn.BatchNorm1d(feature_size, momentum=0.01)
        self.embeds=nn.Parameter(torch.randn(10,feature_size),requires_grad=True)
        self.distance=distance
        self.L2dist=distances.LpDistance(power=2)
        self.apply(_weights_init)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate,
                                    weight_decay=0.00001)
        self.n_classes = n_classes
        self.n_known = 0

    def forward(self, x, embed=False, scale=2):
        x = self.feature_extractor(x)
        x = self.bn(x)
        distance=self.distance(x, self.embeds)
        pred=-self.L2dist(x, self.embeds)
        if embed: return pred[:,:self.n_classes],distance[:,:self.n_classes],x
        return pred[:,:self.n_classes]

    def classify(self, x, transform=None): return self(x).max(1)[1]

    def update_representation(self, dataset, args, attack=None):
        num_workers, batch_size, num_epochs=args.num_workers,args.batch_size,args.num_epochs
        self.compute_means = True
        # Increment number of weights in final fc layer
        classes = list(set(dataset.train_labels))
        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.n_classes += len(new_classes)
        self.cuda()
        print(len(new_classes),"new classes")
        # Form combined training set
        self.combine_dataset_with_exemplars(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

        # Run network training
        save_dir=os.path.join(args.save_dir, args.group+'_IL')
        save_dir=os.path.join(save_dir, args.loss)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        optimizer = self.optimizer
        for epoch in range(args.start_epoch,num_epochs):
            for i, (indices, images, labels) in enumerate(loader):
                images = Variable(images).cuda()
                labels = Variable(labels).cuda()
                indices = indices.cuda()

                optimizer.zero_grad()
                if args.AT and attack is not None:
                    images_adv = attack(images, labels)
                    images = torch.cat((images, images_adv), dim=0)
                    labels=torch.cat((labels, labels), dim=0)
                _, distance, x= self.forward(images,True)
                loss=self.loss(x,distance,labels)

                loss.backward()
                optimizer.step()
                if (i+1) % 100 == 0:
                    print('Epoch [{:d}/{:d}], Iter [{:d}/{:d}] Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, 
                            len(dataset)//batch_size, loss.float().item()))
            print('Epoch [{:d}/{:d}], samples: {:d}, Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, len(dataset), loss.float().item()))
                    
            save_checkpoint({
                'start_epoch': epoch + 1,
                'start_class': self.n_classes - 1,
                'state_dict': self.state_dict(),
                'exemplar_sets': self.exemplar_sets,
                'n_classes': self.n_classes,
                'n_known': self.n_known,
            }, filename=os.path.join(save_dir, args.name+'_checkpoint.th'))
            print('* Epoch Checkpoint saved. *')

    def loss(self,x,distance,y): 
        l2norm=torch.sum(self.L2dist(x,self.embeds)[:,:self.n_classes],1)
        return pl_loss(l2norm,y,distance,self.n_classes)



""" Utils """

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class dce_loss(torch.nn.Module):
    def __init__(self, n_classes,feat_dim,init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes=n_classes
        self.feat_dim=feat_dim
        self.centers=nn.Parameter(torch.randn(self.feat_dim,self.n_classes).cuda(),requires_grad=True)
        if init_weight: nn.init.kaiming_normal_(self.centers)
    def forward(self, x):
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers
        return self.centers, -dist

def regularization(features, centers, labels):
    distance=(features-torch.t(centers)[labels])
    distance=torch.sum(torch.pow(distance,2),1, keepdim=True)
    distance=(torch.sum(distance, 0, keepdim=True))/features.shape[0]
    return distance

def pl_loss(l2norm,y,distance,N_class=10):
    targets=torch.nn.functional.one_hot(y,num_classes=N_class)
    loss=npair_loss(targets,distance,N_class)+0.1*l2norm
    return torch.mean(loss)

def gather_nd(x,y,w):
    pos=torch.cat(torch.where(y==w)).reshape(2,-1)
    return x[pos[0,:], pos[1,:]]
    
def npair_loss(y,dist,K=10):
    pos = gather_nd(dist,y,1).reshape(-1,1)
    neg = gather_nd(dist,y,0).reshape(-1,K-1) if K>1 else 0
    return torch.log(1+torch.sum(torch.exp(neg-pos),-1))