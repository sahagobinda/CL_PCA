from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from collections import OrderedDict
import numpy as np
import os
import os.path
import pdb
from logger import Logger
import math 
from torch.autograd import Variable
from torch.nn.functional import normalize

###----------------------------learning rate adjustment ---------------------------------###########
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 after specified epochs"""
    print('Learning rate decayed by 10 at epoch {}'.format(epoch))
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']=args.lr
        else:
            param_group['lr'] /= 10   
        #print ('Printing adjusted learning rate {:.4f}'.format(param_group['lr']))  
  
###----------------------------PCA functions -------------------------------------------###########
def run_PCA(activations,key_idx, components, threshold=0.99):
        """PCA Compression step 
        Output: PCA transformation matrix that will be applied to filters 
                Optimum number of filters to keep"""

        ##--- Make the activations matrix size/shape compitable for PCA analysis :: see paper for discussions on shapes of the activavtions -- ##
        a=activations.cpu().numpy().swapaxes(1,2).swapaxes(2,3)
        a_shape=a.shape
        #print('reshaped ativations are of shape',a.shape)
        pca = PCA(n_components=components, whiten=False) #number of components should be equal to the number of filters ## made whiten==False ##
        pca.fit(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])) # activation should be of shape (N*H*W,M)
        
        #a_trans=pca.transform(a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3]))        
        #print('explained variance ratio is:',pca.explained_variance_ratio_)
        #plt.plot(np.cumsum(pca.explained_variance_ratio_))
        #plt.show()  #saves the PCA figure.

        ## ------Determine the optimal number of filters --------- ##
        optimal_num_filters=np.sum(np.cumsum(pca.explained_variance_ratio_)<threshold) 
        #print('number of filters required to explain {0} variance is: {1}'.format(threshold, optimal_num_filters))

        return optimal_num_filters,pca

def PCA_transformation(activations, model_paramA, num_filterA, layer, threshold=0.999):
        '''PCA Transformation step, in this part compression on residual space happens 
        output is the transformation matrix (x1) that will be applied to filters '''

        pca_filterA = model_paramA['pca_comp_final']  #list of 5 arrays          
        a=activations.cpu().numpy().swapaxes(1,2).swapaxes(2,3)
        a_shape=a.shape
        a_reshaped = a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])

        ## Zero-out the activations corresponding to the filters from the previous tasks 
        a_reshaped [:,0:int(num_filterA[layer])]=0 
        ## DO PCA on rest of the dimention (residual space)
        components = int(pca_filterA[layer].shape[1]-num_filterA[layer]) 
        pca = PCA(n_components=components, whiten=False) #number of components should be equal to the number of filters ## made whiten==False ##
        pca.fit(a_reshaped) #this should be N*H*W,M
        partial_num_filters=np.sum(np.cumsum(pca.explained_variance_ratio_)<threshold)
 
        ## construct the transformation matrix, x1        
        x1=np.zeros_like(pca_filterA[layer])
        for ik in range (int(num_filterA[layer])):
            x1[ik,ik]=1     
        x1[int(num_filterA[layer]):int(num_filterA[layer])+partial_num_filters,]=pca.components_[0:partial_num_filters,]

        return x1

def filter_selection(activations,num_filterA, layer, threshold=0.995):
        '''Filter selection step after PCA Transformamtion, in this part optimal number of filter is decided -- output is the optimal number of filters for pruning '''
        
        a=activations.cpu().numpy().swapaxes(1,2).swapaxes(2,3)
        a_shape=a.shape
        a_reshaped = a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])

        ## Taking diagonal elements of the activavtion matrix :: Eq. 2 in the paper 
        v_val=np.diag(np.matmul(np.transpose(a_reshaped),a_reshaped))
        
        var_ratio=v_val/(v_val).sum()

        cum_var=0
        for ii in range (var_ratio.shape[0]):
            if cum_var < threshold:
                cum_var += var_ratio[ii]
                opt_filterB = ii+1
        
        if opt_filterB < int (num_filterA[layer]):
            opt_filterB = int(num_filterA[layer])
        
        return opt_filterB

###########-------------------------------training---------------------------------------#########

def train_next(args, model, device, train_loader, optimizer, epoch,logger,loss_hist,filter_num):
    """Trains the model with data of the current task"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # pdb.set_trace()
        target = target - int(args.split*args.num_out)
        data, target = data.to(device), target.to(device)
        output,_ = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target) 
        loss.backward()
        if (args.split >0):
            # clears the gradient update of the frozen filters (core space)
            for k,param in enumerate(model.parameters()):  
                if (k%2==0 and k<10):
                    param.grad[0:int(filter_num[k]),:,:,:]=0
                elif (k%2==1 and k<10):
                    param.grad[0:int(filter_num[k])]=0

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_hist.append(loss.item())                   
    return loss_hist
    
def train_next_pca(args, model, device, train_loader, optimizer, epoch,layer,loss_hist,optimal_num_filters, filter_num):
    """Trains the model with data of the current task after PCA"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target - int(args.split*args.num_out)
        data, target = data.to(device), target.to(device)
        output,_ = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target) 
        loss.backward()

        if (args.split > 0):
            # clears the gradient update of the frozen filters (core space)
            for k,param in enumerate(model.parameters()):  
                if (k%2==0 and k<10):   
                    param.grad[0:int(filter_num[k]),:,:,:]=0
                elif (k%2==1 and k<10):
                    param.grad[0:int(filter_num[k])]=0
        
        k=0; ik =0; jk =0
        for k,param in enumerate(model.parameters()):
            # clears the gradient update of the pruned filters from the privious layers     
            if (k<=((2*layer)+1)):
                if (k%2==0):   
                    param.grad[int(optimal_num_filters[ik]):,:,:,:]=0
                    ik +=1
                elif (k%2==1):
                    param.grad[int(optimal_num_filters[jk]):]=0
                    jk +=1

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_hist.append(loss.item())
    return loss_hist     

def train_classifier(args, model, device, train_loader, optimizer, epoch):
    '''Training the output classifer only'''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target - int(args.split*args.num_out)
        data, target = data.to(device), target.to(device)
        output,_ = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(output, target) 
        loss.backward()
 
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

######-------------------------------Testing-----------------------------------------##############                
def call_test_model (model,test_masked_model,args,device, test_loader,sudo, lx, keep_classifier_list, t):

    filter_num = [lx[0],lx[0],lx[1],lx[1],lx[2],lx[2],lx[3],lx[3],lx[4],lx[4]]
    print('-' * 16)
    print ('Useful Filters for task - {} are {}'.format (t+1, lx))
    for k, p in enumerate(model.parameters()):
        if (k%2==0 and k<10 ):
            temp =p.detach().cpu().numpy()
            if k==0:
                temp [int(filter_num[k]):,:,:,:] = 0
            else:
                temp [int(filter_num[k]):,:,:,:] = 0
                temp [0:int(filter_num[k]),int(filter_num[k-2]):,:,:] = 0
            test_masked_model.state_dict()[list(test_masked_model.state_dict().keys())[k]].copy_(torch.Tensor(temp))
            
        elif (k%2==1 and k<10 ): 
            temp =p.detach().cpu().numpy()
            temp [int(filter_num[k]):] = 0
            test_masked_model.state_dict()[list(test_masked_model.state_dict().keys())[k]].copy_(torch.tensor(temp)) 
        
        elif (k >= 10):
            temp =p.detach().cpu().numpy()
            if (k==10):
                temp  = keep_classifier_list [t*2]
            elif (k==11):
                temp  = keep_classifier_list [(t*2)+1]
            test_masked_model.state_dict()[list(test_masked_model.state_dict().keys())[k]].copy_(torch.Tensor(temp)) 
    
    test_masked_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target - (t*args.num_out) ## check if correct 
            data, target = data.to(device), target.to(device)
            output,_= test_masked_model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('-' * 16)

def test(args, model, device, test_loader,loss_test): ## main test function 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target - (args.split*args.num_out) ## check if correct 
            data, target = data.to(device), target.to(device)
            output,_= model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    loss_test.append(test_loss)
    return loss_test
  
def test_acc_save(args, model, device, test_loader,loss_test): 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = target - (args.split*args.num_out)
            data, target = data.to(device), target.to(device)
            output,_ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    loss_test.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

