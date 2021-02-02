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
import math 
from torch.autograd import Variable
from torch.nn.functional import normalize
import sys
import time
import math
import torch.nn.init as init
import scipy as sp

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
def network_compression_PCA(activations,key_idx, components, threshold=0.99):
    """ - PCA step adated from: sklearn.decomposition.PCA
	  - Used to compress the space after training 1st task 
        - returns the optimum number of filters needed in each layer deciced by given threshold """

    a=activations.cpu().numpy().swapaxes(1,2).swapaxes(2,3)
    a_shape=a.shape
    a_reshaped = a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])
    # mean-normalize the reshaped activation 
    a_col_mean = np.mean(a_reshaped,axis=0)
    a_reshaped = (a_reshaped - a_col_mean) 

    ## Determine Total variance in the space 
    ua,sa,vha=np.linalg.svd(a_reshaped,full_matrices=False)
    var_total=np.sum(sa**2)
    print('-' * 30)
    print ('Total Variance in full activation space:',var_total) 

    var_ratio = (sa**2)/var_total
    ## ------Determine the optimal number of filters --------- ##
    optimal_num_filters=np.sum(np.cumsum(var_ratio)<threshold) 
    #print('number of filters required to explain {0} variance is: {1}'.format(threshold, optimal_num_filters))
    return optimal_num_filters,vha

def projection_subtraction_PCA(activations, pca_filterA, num_filterA, layer, threshold=0.999):
    '''projection_subtraction_PCA algorithm 
	- PCA step adated from: sklearn.decomposition.PCA
       Inputs -
     - layer : layer index 
     - num_filterA : list contatining number of core filters in each layer 
     - pca_filterA[layer].shape[1] : total available filters in the given layer 
     - num_filterA[layer] : frozen core filters in the given layer'''
 
    if int(pca_filterA[layer].shape[1]) > int(num_filterA[layer]): 
        a=activations.cpu().numpy().swapaxes(1,2).swapaxes(2,3)
        a_shape=a.shape
        a_reshaped = a.reshape(a_shape[0]*a_shape[1]*a_shape[2],a_shape[3])
        a_col_mean = np.mean(a_reshaped,axis=0)
        # mean-normalize the reshaped activation 
        a_reshaped = (a_reshaped - a_col_mean)
        a_s=a_reshaped.copy()  
        a_d=a_reshaped.copy()

        ## find the basis of the Core space 
        a_reshaped [:,int(num_filterA[layer]):]=0
        u,s,vh=np.linalg.svd(a_reshaped,full_matrices=False) 
        r = int(num_filterA[layer])
        # Projection-subtraction 
        xb=np.dot(u[:,0:r].transpose(),a_s)
        z =np.dot(u[:,0:r],xb)        
        a_reshaped_new = a_s-z 
        
        ## Find the variance in core space only 
        var_core_only=np.sum(s**2)
        print('-' * 30)
        print ('Variance in core space only:',var_core_only) 
        ## Find the variance in the residual only 
        a_d [:,:int(num_filterA[layer])]=0
        ud,sd,vhd=np.linalg.svd(a_d,full_matrices=False)
        var_res_only=np.sum(sd**2)
        print ('Variance in residual space only:',var_res_only) 
        print ('Total Variance partial sums:',(var_core_only+var_res_only)) 

        ## Remaining Residual variance 
        ur,sr,vhr=np.linalg.svd(a_reshaped_new,full_matrices=False)
        var_res_rem=np.sum(sr**2)
        print ('Remaining Variance in Residual space:',var_res_rem)
        
        ## variance retention criterion
        cum_var = (var_res_only-var_res_rem)/var_res_only 
        var_ratio =  (sr**2)/ var_res_only 
        opt_filter = int(num_filterA[layer])

        for ii in range (var_ratio.shape[0]):
            if cum_var < threshold:
                cum_var += var_ratio[ii]
                opt_filter += 1
            else:
                break
        partial_num_filters = opt_filter - int(num_filterA[layer])
        print('Number of filters needs to be added to the Core from Residual : ',partial_num_filters)
        print('-' * 30)        

    else:
        print ('Resource exhausted in layer- {}'.format(layer+1))
        opt_filter = int(num_filterA[layer])

    return opt_filter

###########-------------------------------training---------------------------------------#########

def train_next(args, model, device, train_loader, optimizer, epoch ,loss_hist,filter_num):
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
    
def train_next_pca(args, model, device, train_loader, optimizer, epoch,loss_hist,optimal_num_filters, filter_num):
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
        
        k=0; 
        ik =0; jk=0;
        for k,param in enumerate(model.parameters()):
            # clears the gradient update of the pruned filters from the privious layers     
            if (k%2==0 and k<10):   
                param.grad[int(optimal_num_filters[ik]):,:,:,:]=0
                ik +=1
            elif (k%2==1 and k<10):
                param.grad[int(optimal_num_filters[jk]):]=0
                jk +=1

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            loss_hist.append(loss.item())
    return loss_hist     


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
    accuracy = 100. * correct / len(test_loader.dataset)    
    return accuracy
  

