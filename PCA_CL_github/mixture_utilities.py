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
import random
from torch.nn.functional import relu, avg_pool2d

###----------------------------learning rate adjustment ---------------------------------###########

def adjust_learning_rate(optimizer, epoch, args):
    # print('--- Learning rate decayed at {} epoch {} ---'.format(epoch))
    for param_group in optimizer.param_groups:
        if (epoch ==1):
            param_group['lr']  = args.lr
        else:
            param_group['lr'] /= args.lr_factor  
            
###---------------------------- Activation Collection for ResNet ------------------------###########
def act_collect_resnet (net, device, x, y): # x is from xtrain @check
    # Collect activations by forward pass
    net.eval()
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    b=r[0:1000] # Take 1000 examples 
    example_data = x[b].to(device)
    example_out  = net(example_data)

    act_list =[]
    act_list_final =[]
    act_list.extend([net.act['conv_in'], net.layer1[0].act['conv_0'], net.layer1[0].act['conv_1'], net.layer1[1].act['conv_0'],    net.layer1[1].act['conv_1'],
                                         net.layer2[0].act['conv_0'], net.layer2[0].act['conv_1'], net.layer2[0].act['short_cut'], net.layer2[1].act['conv_0'], net.layer2[1].act['conv_1'],
                                         net.layer3[0].act['conv_0'], net.layer3[0].act['conv_1'], net.layer3[0].act['short_cut'], net.layer3[1].act['conv_0'], net.layer3[1].act['conv_1'],
                                         net.layer4[0].act['conv_0'], net.layer4[0].act['conv_1'], net.layer4[0].act['short_cut'], net.layer4[1].act['conv_0'], net.layer4[1].act['conv_1']])
    
    scale =[25,25,25,25,25, 100,100,100,100,100, 250,250,250,250,250, 1000,1000,1000,1000,1000]
    for i in range (len(act_list)):
        activations = act_list[i][0:scale[i]]
        act_list_final.append(activations)
    act_list = []
    
    return act_list_final 
###----------------------------PCA functions -------------------------------------------###########
def network_compression_PCA (activations, threshold=0.99):
    """ - PCA step adated from: sklearn.decomposition.PCA
	   - Used to compress the space after training 1st task 
         - returns the optimum number of filters needed in each layer deciced by given threshold """
    a=activations.detach().cpu().numpy().swapaxes(1,2).swapaxes(2,3)
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
    return optimal_num_filters

def projection_subtraction_pca(activations, original_filter, num_filterA, layer, threshold=0.99):
    '''projection_subtraction_PCA algorithm 
     - PCA step adated from: sklearn.decomposition.PCA
       Inputs -
     - layer : layer index 
     - num_filterA : list contatining number of core filters in each layer 
     - original_filter[layer] : total available filters in the given layer 
     - num_filterA[layer] : frozen core filters in the given layer'''
        
    if int(original_filter[layer]) > int(num_filterA[layer]): 
        a=activations.detach().cpu().numpy().swapaxes(1,2).swapaxes(2,3)
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
        print('-' * 30)

        ## variance retention criterion
        cum_var = (var_res_only-var_res_rem)/var_res_only 
        var_ratio =  (sr**2)/ var_res_only 
        opt_filter = int(num_filterA[layer])

        print('-' * 30)
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
        print ('Resource exhausted at layer- {}'.format(layer+1))
        opt_filter = int(num_filterA[layer])

    return opt_filter

###########-------------------------------training---------------------------------------#########

def train (args, model, device, x,y, optimizer,criterion, epoch, task_id, filter_list, retrain=False):
    model.train()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    # Loop batches
    for i in range(0,len(r),args.batch_size_train):
        if i+args.batch_size_train<=len(r): b=r[i:i+args.batch_size_train]
        else: b=r[i:]
        data = x[b]
        data, target = data.to(device), y[b].to(device)
        optimizer.zero_grad()        
        output = model(data)
        loss = criterion(output[task_id], target)        
        loss.backward()
        
        if task_id > 0:
            for k, param in enumerate(model.parameters()):
                if len(param.size())==4:
                    # Conv Params 
                    if retrain:
                        param.grad.data[0:int(filter_list[task_id-1][k]),:,:,:]=0
                        param.grad.data[int(filter_list[task_id][k]):,:,:,:]   =0
                    else:
                        param.grad.data[0:int(filter_list[task_id-1][k]),:,:,:]=0
                elif len(param.size())==1:
                    # BN params 
                    if retrain:
                        param.grad.data[0:int(filter_list[task_id-1][k])]=0
                        param.grad.data[int(filter_list[task_id][k]):]   =0
                    else:
                        param.grad.data[0:int(filter_list[task_id-1][k])]=0

        if task_id == 0 and retrain:
            for k, param in enumerate(model.parameters()):
                if len(param.size())==4:
                    param.grad.data[int(filter_list[task_id][k]):,:,:,:]=0
                elif len(param.size())==1:
                    param.grad.data[int(filter_list[task_id][k]):]=0

        optimizer.step()
        

######-------------------------------Testing-----------------------------------------##############                
def test_masked_model (args, model, masked_model, device, xtest, ytest, criterion, filter_list, task_id):

    for k, (p,p_masked) in enumerate(zip(model.parameters(),masked_model.parameters())):
        if len(p.size())==4 or len(p.size())==1 :
            temp_masked = p_masked.detach().cpu().numpy()
            temp        = p.detach().cpu().numpy()
            temp [int(filter_list[task_id][k]):] = 0
            p_masked.data.copy_(torch.Tensor(temp)) 

    loss, acc = test(args, masked_model, device, xtest, ytest, criterion, task_id)
    return loss, acc


def test(args, model, device, x, y, criterion, task_id):
    model.eval()
    total_loss = 0
    total_num = 0 
    correct = 0
    r=np.arange(x.size(0))
    np.random.shuffle(r)
    r=torch.LongTensor(r).to(device)
    with torch.no_grad():
        # Loop batches
        for i in range(0,len(r),args.batch_size_test):
            if i+args.batch_size_test<=len(r): b=r[i:i+args.batch_size_test]
            else: b=r[i:]
            data = x[b]
            data, target = data.to(device), y[b].to(device)
            output = model(data)
            loss = criterion(output[task_id], target)#, reduction='sum').item() # sum up batch loss
            pred = output[task_id].argmax(dim=1, keepdim=True) # get the index of the max log-probability
            
            correct    += pred.eq(target.view_as(pred)).sum().item()
            total_loss += loss.data.cpu().numpy().item()*len(b)
            total_num  += len(b)

    acc = 100. * correct / total_num
    final_loss = total_loss / total_num
    return final_loss, acc
  



