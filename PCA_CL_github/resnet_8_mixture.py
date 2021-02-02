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
from torch.nn.functional import relu, avg_pool2d
import mixture_8_datasets
from mixture_utilities import*
import random
from copy import deepcopy
import time 
########-------------------------------------------- Model Class --------------------------------##################
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        xs  = x.clone()
        out = self.conv1(x)
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = relu(self.bn1(out))
        
        out = self.conv2(out)
        self.count = self.count % 2 
        self.act['conv_{}'.format(self.count)] = out
        self.count +=1
        out = self.bn2(out)
        
        for k, layer in enumerate(self.shortcut):
            if k==0:
                xs = layer(xs)
                self.act['short_cut'] = xs
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
#         self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        
        self.taskcla = taskcla
        self.linear=torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.linear.append(nn.Linear(nf * 8 * block.expansion, n, bias=False))
        self.act = OrderedDict()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = self.conv1(x.view(bsz, 3, 32, 32))
        self.act['conv_in'] = out
        out = relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
#         out = self.linear(out)
        y=[]
        for t,i in self.taskcla:
            y.append(self.linear[t](out))
        return y

def ResNet18(taskcla, nf=32):
    return ResNet(BasicBlock, [2, 2, 2, 2], taskcla, nf)

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

########--------------------------------------------Main Function --------------------------------##################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size-train', type=int, default=64, metavar='TR',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--batch-size-test', type=int, default=128, metavar='TE',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--train-epoch', type=int, default=150, metavar='T',
                        help='number of epochs for train step (default: 10)')
    parser.add_argument('--retrain-epoch', type=int, default=150, metavar='RT',
                        help='number of epochs for train step (default: 10)')
    parser.add_argument('--lr_min', type=float, default=1e-4, metavar='LRM',
                        help='minimum lr rate (default: 10)')
    parser.add_argument('--lr_patience', type=int, default=5, metavar='LRP',
                        help='hold before decaying lr (default: 5)')
    parser.add_argument('--lr_factor', type=int, default=3, metavar='LRF',
                        help='lr decay factor (default: 3)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--train', action='store_true', default=False, 
                        help='enables network training')
    parser.add_argument('--retrain', action='store_true', default=False, 
                        help='enables network retraining')
    parser.add_argument('--call-test', action='store_true', default=False, 
                        help='set TRUE if need network testing')
    parser.add_argument('--save-model', action='store_true', default=False, 
                        help='set TRUE if need network saving')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--param_dir', default='Mixture_task', type=str,   
                        help='Pre-trained weights directory')
    parser.add_argument('--file_suffix', type=int, default=1, metavar='N',
                        help='adds suffix to file name')

    parser.add_argument('--var_kept', nargs='+', type=float)
    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)

    # model directories 
    model_root     = os.path.expanduser('PARAM/'+args.param_dir)
    running_param_fname   = os.path.join(model_root, 'mixture_rp_{}.pth'.format(args.file_suffix)) 
    param_fnameB   = os.path.join(model_root, 'mixture_{}.pth'.format(args.file_suffix))
    param_fnameB_rt= os.path.join(model_root, 'mixture_rt_{}.pth'.format(args.file_suffix))
    checkpoint_fname = os.path.join(model_root, 'mixture_ckp_{}.pth'.format(args.file_suffix))
    # seeds and device setup 
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}
    
    ##############################################################################
    #########                      DataSet and Model                  ############ 
    ##############################################################################
    # Load data 
    data,taskcla,inputsize = mixture_8_datasets.get(seed=args.seed, pc_valid=0.1)
    # Initialize Model 
    tstart = time.time()
    tpca = []
    model   = ResNet18(taskcla,40).to(device)
    oldmodel= ResNet18(taskcla,40).to(device) # For network initialization purpose 
    print(model)

    original_filter_list =[]
    for param in model.parameters():
        if len(param.size()) == 4:
            original_filter_list.append(param.shape[0])

    criterion = torch.nn.CrossEntropyLoss()
    # start training loop for tasks 
    for k,ncla in taskcla:
        task_id = k
        if task_id < 8:
            # load current task data 
            print('*'*100)
            print('Task {:2d} ({:s})'.format(k,data[k]['name']))
            print('*'*100)
            xtrain=data[k]['train']['x']
            ytrain=data[k]['train']['y']
            xvalid=data[k]['valid']['x']
            yvalid=data[k]['valid']['y']
            xtest =data[k]['test']['x']
            ytest =data[k]['test']['y']
            print (xtrain.shape, xvalid.shape, xtest.shape)

            ##############################################################################
            #########                         TRAIN                           ############ 
            ##############################################################################
            if (args.train):            
                if task_id == 0:
                    filter_list  = []
                    conv_filter_list = []
                    accuracy_train =[]
                    accuracy_retrain =[]
                else:
                    # Load model from last task
                    model_param = torch.load(param_fnameB_rt) 
                    model.load_state_dict(model_param['state_dict'])
                    # load filter and accuracy statistics 
                    running_param = torch.load(running_param_fname)
                    filter_list = running_param ['filter_list'] 
                    conv_filter_list = running_param ['conv_filter_list']
                    accuracy_train = running_param ['accuracy_train']
                    accuracy_retrain = running_param ['accuracy_retrain']                

                best_loss=np.inf
                lr = args.lr    
                best_model=get_model(model)
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)#, weight_decay=5e-4)
                best_optim=get_model(optimizer)

                for epoch in range(1, args.train_epoch+1):
                    train (args, model, device, xtrain, ytrain, optimizer,criterion, epoch, task_id, filter_list)
                    tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion, task_id)
                    print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch,tr_loss,tr_acc),end='')
                    # Valid
                    valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, task_id)
                    print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                    # Adapt lr
                    if valid_loss<best_loss:
                        best_loss=valid_loss
                        best_model=get_model(model)
                        best_optim=get_model(optimizer)
                        patience=args.lr_patience
                        print(' *',end='')
                    else:
                        patience-=1
                        if patience<=0:
                            lr/=args.lr_factor
                            print(' lr={:.1e}'.format(lr),end='')
                            if lr<args.lr_min:
                                print()
                                break
                            patience=args.lr_patience
                            # Check if copying back model makes sense ?
                            # set_model_(model, best_model)
                            # set_model_(optimizer, best_optim)
                            adjust_learning_rate(optimizer, epoch, args)
                    print()

                set_model_(model,best_model)
                set_model_(optimizer, best_optim)
                test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, task_id)
                print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))

                ## Saving the test accuracies in the list 
                accuracy_train.append(test_acc)
                if (args.save_model):
                    print('Saving base Model after TRAINING...')
                    model_param_train = {'state_dict': model.state_dict() } 
                    torch.save(model_param_train,param_fnameB) 
            ##############################################################################
            #########                       RETRAIN                           ############ 
            ##############################################################################
            if (args.retrain): 
                if (args.train==False) :
                    # load model of current task after training 
                    model_param = torch.load(param_fnameB) 
                    model.load_state_dict(model_param['state_dict'])
                    
                    if (task_id > 0):
                        # load filter and accuracy statistics 
                        running_param = torch.load(running_param_fname)
                        filter_list = running_param ['filter_list'] 
                        conv_filter_list = running_param ['conv_filter_list']
                        accuracy_train = running_param ['accuracy_train']
                        accuracy_retrain = running_param ['accuracy_retrain']   
                    else:
                        filter_list  = []
                        conv_filter_list = []
                        accuracy_train =[]
                        accuracy_retrain =[]

                optimal_num_filters = np.zeros(3*len(original_filter_list))
                conv_filters = np.zeros(len(original_filter_list))
                # Collect activations
                t_pca_start = time.time()
                act_list  = act_collect_resnet (model, device, xtrain, ytrain)
                for idk in range (len(act_list)):
                    print(act_list[idk].shape)
                i =0 
                for k, param in enumerate (model.parameters()):
                    if len(param.size())==4:

                        if task_id ==0:
                            conv_filters[i] = network_compression_PCA (act_list[i], args.var_kept[i])
                        else:
                            conv_filters[i] = projection_subtraction_pca (act_list[i], original_filter_list,\
                                                conv_filter_list[task_id-1], i, args.var_kept[i])

                        optimal_num_filters [3*i:3*(i+1)] = conv_filters[i] 
                        temp = param.data.detach().cpu().numpy()
                        temp [int(optimal_num_filters[k]):] = 0
                        param.data.copy_(torch.Tensor(temp))
                        i += 1
                    elif len(param.size())==1:
                        temp = param.data.detach().cpu().numpy()
                        temp [int(optimal_num_filters[k]):] = 0
                        param.data.copy_(torch.Tensor(temp))
                
                t_pca_stop = time.time()
                tpca.append((t_pca_stop-t_pca_start))

                filter_list.append(optimal_num_filters)
                conv_filter_list.append(conv_filters)
                print ('Task {}  filter set: {}'.format(task_id, conv_filter_list[task_id]))

                best_loss=np.inf
                lr = args.lr      

                best_model=get_model(model)
                feature_list = []
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)#, weight_decay=5e-4)
                best_optim=get_model(optimizer)
                # test(args, model, device, xtest, ytest, criterion, task_id )  
                for epoch in range(1, args.retrain_epoch+1):
                    train (args, model, device, xtrain, ytrain, optimizer,criterion, epoch, task_id, filter_list, True)
                    tr_loss,tr_acc = test(args, model, device, xtrain, ytrain, criterion, task_id)
                    print('Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% |'.format(epoch,tr_loss,tr_acc),end='')
                    # Valid
                    valid_loss,valid_acc = test(args, model, device, xvalid, yvalid,  criterion, task_id)
                    print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, valid_acc),end='')
                    # Adapt lr
                    if valid_loss<best_loss:
                        best_loss=valid_loss
                        best_model=get_model(model)
                        best_optim=get_model(optimizer)
                        patience=args.lr_patience
                        print(' *',end='')
                    else:
                        patience-=1
                        if patience<=0:
                            lr/=args.lr_factor
                            print(' lr={:.1e}'.format(lr),end='')
                            if lr<args.lr_min:
                                print()
                                break
                            patience=args.lr_patience
                            # Check if copying back model makes sense ?
                            # set_model_(model, best_model)
                            # set_model_(optimizer, best_optim)
                            adjust_learning_rate(optimizer, epoch, args)
                    print()

                set_model_(model,best_model)
                set_model_(optimizer, best_optim)
                test_loss, test_acc = test(args, model, device, xtest, ytest,  criterion, task_id)
                print('Test: loss={:.3f} , acc={:5.1f}%'.format(test_loss,test_acc))
                accuracy_retrain.append(test_acc)

                ##############################################################################
                #########                       Post Process                      ############ 
                ##############################################################################
                # Model post processing 
                running_param = {'filter_list' : filter_list, 'conv_filter_list' : conv_filter_list,\
                                  'accuracy_train' : accuracy_train, 'accuracy_retrain' : accuracy_retrain }

                # Randomly reinitialize the pruned model  
                for k, (p,p_old) in enumerate(zip(model.parameters(),oldmodel.parameters())):
                    if len(p.size())==4 or len(p.size())==1 :
                        temp_old=p_old.detach().cpu().numpy()
                        temp    =p.detach().cpu().numpy()
                        temp [int(filter_list[task_id][k]):] = temp_old[int(filter_list[task_id][k]):]
                        p.data.copy_(torch.Tensor(temp))
            
                # save and print the model statistics 
                model_param = {'state_dict': model.state_dict()}
                if (args.save_model):
                    print('Saving Model after Retraining...')
                    torch.save(model_param,param_fnameB_rt)
                    torch.save(running_param,running_param_fname)
                print ('Training Accuracy:',accuracy_train)
                print ('Retraining Accuracy:',accuracy_retrain)
            ##############################################################################
            #########                          TEST                          ############ 
            ##############################################################################
            if (args.call_test):
                ''' This part of the code test the trained model with task hint '''
                print ('-'*30)

                running_param = torch.load(running_param_fname)
                filter_list = running_param ['filter_list'] 
                # load Trained model 
                model_param = torch.load(param_fnameB_rt) 
                model.load_state_dict(model_param['state_dict'])
                
                masked_model= ResNet18(taskcla,40).to(device) 
                masked_model.load_state_dict(model_param['state_dict'])
                t_loss, t_acc = test_masked_model (args, model, masked_model, device,\
                                                  xtest, ytest, criterion, filter_list, task_id)  
                print ('Masked Test - Task : {} | Loss : {:.3f}, Accuracy : {:5.1f}%'.format(task_id, t_loss, t_acc))

    tend = time.time()
    print('[Elapsed time = {:.1f} s]'.format((tend-tstart)))
    print ('>> Total time spent in PCA : {} sec <<'.format(sum(tpca)))
    print (tpca)       


if __name__ == '__main__':
    main()
