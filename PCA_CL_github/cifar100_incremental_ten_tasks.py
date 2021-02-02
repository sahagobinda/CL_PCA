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
from cifar_utilities import*
import time 

########-------------------------------------------- Model Class --------------------------------##################
class Net(nn.Module):
    def __init__(self,num_classes=10):
        super(Net, self).__init__()
        self.act=OrderedDict()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.drop_outA = nn.Dropout(0.15)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128,128,3)
        self.drop_outB = nn.Dropout(0.15)
        self.conv5 = nn.Conv2d(128,256,2)
        self.fc1 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        self.act['conv1_pre_relu']=x
        x = F.relu(x)
        x = self.conv2(x)
        self.act['conv2_pre_relu']=x
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop_outA(x)
        
        x = self.conv3(x)
        self.act['conv3_pre_relu']=x
        x = F.relu(x)
        x = self.conv4(x)
        self.act['conv4_pre_relu']=x
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.drop_outB(x)
        
        x = self.conv5(x)
        self.act['conv5_pre_relu']=x
        x = F.relu(x)
        x = F.avg_pool2d(x, 2, 2)
        #x = x.view(-1, 2048)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        self.act['fc1_output']=x
        return F.log_softmax(x, dim=1),x


########--------------------------------------------Main Function --------------------------------##################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--train_epoch', type=int, default=10, metavar='N',
                        help='number of epochs for train step (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--full_task', type=int, default=0, metavar='N',
                        help='If full dataset is to be simulated (default: 0)')
    parser.add_argument('--call_test', type=int, default=0, metavar='N',
                            help='if 1 calls masked test (default: 0)')
    parser.add_argument('--split', type=int, default=0, metavar='N',
                        help='Split task ID; 0: for first task; 1: for 2nd task, 4: for 5th task (default: 0)')
    parser.add_argument('--train', type=int, default=1, metavar='N',
                        help='If 1 trains the model else loads (default: 1)')
    parser.add_argument('--retrain', type=int, default=1, metavar='N',
                        help='If 1 retrains the model with PCA compression (default: 1)')
    parser.add_argument('--rand_init', type=int, default=0, metavar='N',
                        help='If 1 randomly initialize the weights after PCA (default: 1)')
    parser.add_argument('--save-model', type=int, default=0, metavar='N',
                        help='For Saving the current Model: 1==save')
    parser.add_argument('--param_dir', default='CIFAR100_task', type=str,   
                        help='Pre-trained weights directory')
    parser.add_argument('--file_suffix', type=int, default=1, metavar='N',
                        help='adds suffix to file name')
    parser.add_argument('--num_out', type=int, default=10, metavar='N',
                        help='Number of output units for this task (default: 2)')
    parser.add_argument('--epoch_list', nargs='+', type=int)
    parser.add_argument('--var_kept', nargs='+', type=float)
    args = parser.parse_args()
    print('='*100)
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    print('='*100)
    
    # model directories 
    model_root     = os.path.expanduser('PARAM/'+args.param_dir)
    running_param_fname   = os.path.join(model_root, 'cifar100_rp_{}.pth'.format(args.file_suffix)) 
    param_fnameB   = os.path.join(model_root, 'cifar100_multi_{}.pth'.format(args.file_suffix))
    param_fnameB_rt= os.path.join(model_root, 'cifar100_multi_rt_{}.pth'.format(args.file_suffix))
    checkpoint_fname = os.path.join(model_root, 'cifar100_ckp_{}.pth'.format(args.file_suffix))

    # seeds and device setup 
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}
    
########-------------------------------------------- Train dataset --------------------------------##################
    taskA=[];taskB=[];taskC=[];taskD=[];taskE=[];taskF=[];taskG=[];taskH=[];taskI=[];taskJ=[]
    test_taskA=[];test_taskB=[];test_taskC=[];test_taskD=[];test_taskE=[];test_taskF=[];test_taskG=[];test_taskH=[];test_taskI=[];test_taskJ=[]
    
    train_dataset= datasets.CIFAR100(root='./data', train=True, download=True,transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    
    if (args.train==1 and args.full_task==1):
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True, **kwargs)
    else:    
        for batch_id, (inputs,labels) in enumerate (train_dataset):
            if labels < 10:
                taskA.append((inputs,labels))
            elif labels>=10 and labels<20:
                taskB.append((inputs,labels))
            elif labels>=20 and labels<30:
                taskC.append((inputs,labels))
            elif labels>=30 and labels<40:
                taskD.append((inputs,labels))
            elif labels>=40 and labels<50:
                taskE.append((inputs,labels))
            elif labels>=50 and labels<60:
                taskF.append((inputs,labels))
            elif labels>=60 and labels<70:
                taskG.append((inputs,labels))
            elif labels>=70 and labels<80:
                taskH.append((inputs,labels))
            elif labels>=80 and labels<90:
                taskI.append((inputs,labels))
            elif labels>=90 and labels<100:
                taskJ.append((inputs,labels))
        
        train_task =[taskA,taskB,taskC,taskD,taskE,taskF,taskG,taskH,taskI,taskJ]
        
    ########-------------------------------------------- test dataset --------------------------------##################   
    test_dataset= datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
    if (args.full_task==1):
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        for batch_id, (inputs,labels) in enumerate (test_dataset):
            if labels < 10:
                test_taskA.append((inputs,labels))
            elif labels>=10 and labels<20:
                test_taskB.append((inputs,labels))
            elif labels>=20 and labels<30:
                test_taskC.append((inputs,labels))
            elif labels>=30 and labels<40:
                test_taskD.append((inputs,labels))
            elif labels>=40 and labels<50:
                test_taskE.append((inputs,labels))
            elif labels>=50 and labels<60:
                test_taskF.append((inputs,labels))
            elif labels>=60 and labels<70:
                test_taskG.append((inputs,labels))
            elif labels>=70 and labels<80:
                test_taskH.append((inputs,labels))
            elif labels>=80 and labels<90:
                test_taskI.append((inputs,labels))
            elif labels>=90 and labels<100:
                test_taskJ.append((inputs,labels))
        
        test_task =[test_taskA,test_taskB,test_taskC,test_taskD,test_taskE,test_taskF,test_taskG,test_taskH,test_taskI,test_taskJ]

########-------------------------------------------- Model and Optimaizer --------------------------------################## 
    model   = Net().to(device)
    # creating a model for network initialization purpose 
    oldmodel= Net().to(device) 
    print(model)
    # creating lists to save network accuracies 
    accuracy_train =[]
    accuracy_retrain =[]
    activation_list =[]
    scale =[100,100,500,500,1000]
    for idx in range (0,10):
        args.split = idx 
        print('Classifing {}-{} | Task id-{}'.format(args.num_out*args.split,args.num_out*(args.split+1)-1,idx+1))
        print('Loading split CIFAR100 train task {}...'.format(args.split+1))
        train_loader = torch.utils.data.DataLoader(train_task[args.split],batch_size=args.batch_size, shuffle=True, **kwargs) ## train_task name  
        train_loader_sudo = torch.utils.data.DataLoader(train_task[args.split],batch_size=1000, shuffle=True, **kwargs)
        print('Loading split CIFAR100 test task {}...'.format(args.split+1))
        test_loader = torch.utils.data.DataLoader(test_task[args.split],batch_size=1000, shuffle=True, **kwargs) ## test_task name
        
    #######--------------------------------------------------train------------------------------------###################      
        if (args.train ==1):            
            if args.split > 0 :
                ## Loading model from already learned tasks 
                if (args.split>1):
                    args.lr = 0.001
                    args.train_epoch = 100
                    args.epoch_list = [150] 
                else:
                    args.lr = 0.01
                model_param = torch.load(param_fnameB_rt) #...............Load model of T-1 task
                model.load_state_dict(model_param['state_dict'])
                
                running_param = torch.load(running_param_fname) # load running filter and class Weight and bias
                keep_classifier_list = running_param['classifier_list']
                opt_filter_list = running_param['opt_filter_list']
                lx= opt_filter_list[args.split-1] ## this is an array
                filter_num = [lx[0],lx[0],lx[1],lx[1],lx[2],lx[2],lx[3],lx[3],lx[4],lx[4]]
                pca_comp_final = running_param['pca_comp']
            
            else:
                args.lr = 0.01
                print('Traning from the scratch model with learning rate:',args.lr)
                keep_classifier_list = []
                opt_filter_list = []
                filter_num =[]
                pca_comp_final=[]
            
            print('Starting learning rate for Training task {} is {}'.format(idx+1,args.lr))
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # using SGD with momentum
            loss_hist=[]
            loss_test=[]
            best_accuracy = 0.0
            best_epoch = 0 
            for epoch in range(1, args.train_epoch + 1): 
                if (args.split<2 and (epoch == 25 or epoch == 65) ):                
                    adjust_learning_rate(optimizer, epoch, args) 
                elif (args.split > 1 and epoch == int(args.train_epoch*0.9)):
                    adjust_learning_rate(optimizer, epoch, args) 
                loss_hist=train_next(args, model, device, train_loader, optimizer, epoch,loss_hist,filter_num)
                temp_accuracy=test(args, model, device, test_loader,loss_test) 
                if (temp_accuracy > best_accuracy):
                    best_accuracy = temp_accuracy
                    best_epoch = epoch
                    model_temp_param = {'state_dict': model.state_dict()}
                    torch.save(model_temp_param,checkpoint_fname)
            print('best Training accuracy: {} at {} epoch'.format(best_accuracy,best_epoch))
            model_load_param = torch.load(checkpoint_fname)
            model.load_state_dict(model_load_param['state_dict'])

            ## Saving the test accuracies in the list 
            acc=test(args, model, device, test_loader,loss_test)
            accuracy_train.append(acc)
            if (args.save_model):
                if (args.split ==0):
                    model_param_train = {'state_dict': model.state_dict() } 
                else:
                    model_param_train  = {'state_dict': model.state_dict()}
                print('Saving base Model after TRAINING...')
                torch.save(model_param_train,param_fnameB)   
        
        if (args.call_test ==1):
            ''' This part of the code test the trained model with task hint 
            e.g. if idx=1 then this is 2nd task and calling this part will  
            return the test accuracy for the incrementally learned 2nd task'''
            sudo=[]
            running_param = torch.load(running_param_fname)
            keep_classifier_list = running_param['classifier_list']
            opt_filter_list = running_param['opt_filter_list']
            
            model_param = torch.load(param_fnameB_rt) #...............Load model
            model.load_state_dict(model_param['state_dict'])
            
            test_masked_model= Net().to(device) 
            test_masked_model.load_state_dict(model_param['state_dict'])
            call_test_model (model,test_masked_model, args , device, test_loader, sudo, opt_filter_list[args.split],keep_classifier_list,idx)
            
        ############----------------------------------------Re-train------------------------------------###################  
        if (args.retrain == 1): 
            sudo=[]
            loss_hist=[]
            loss_test=[]
            optimal_num_filters = np.zeros(5)

            args.lr = 0.01
            
            if (args.train == 0) :
                model_param = torch.load(param_fnameB) #...............Load old model after training 
                model.load_state_dict(model_param['state_dict'])
                
                if (args.split > 0):
                    running_param = torch.load(running_param_fname)# load running filter and class Weight and bias
                    keep_classifier_list = running_param['classifier_list']
                    opt_filter_list = running_param['opt_filter_list']
                    lx= opt_filter_list[args.split-1] ## this is an array
                    filter_num = [lx[0],lx[0],lx[1],lx[1],lx[2],lx[2],lx[3],lx[3],lx[4],lx[4]]
                    pca_comp_final = running_param['pca_comp']
                    
                elif args.split ==0:
                    keep_classifier_list = []
                    opt_filter_list = []
                    filter_num =[]
                    pca_comp_final=[]
                
            weight_tx_list=[]
            bias_tx_list=[]
            #collecting activation for PCA
            test(args, model, device, train_loader_sudo,sudo) 
            for i in range(5): # Here '5' corresponds to 5 convolutional layers 
                out_size    =model.state_dict()[list(model.state_dict().keys())[i*2]].size(0)
                inp_size    =model.state_dict()[list(model.state_dict().keys())[i*2]].size(1)
                k_size      =model.state_dict()[list(model.state_dict().keys())[i*2]].size(2)

                #########------------------------------------- Run PCA ------------------------------------############
                if (args.split == 0):
                    print ('PCA Compression at layer {}'.format(i+1))                  
  
                    ## ----- First PCA Compression step (For 1st Task)---- ##
                    optimal_num_filters[i],pca_xform=network_compression_PCA(model.act[list(model.act.keys())[i]][0:scale[i]],0,out_size,threshold=args.var_kept[i]) 
                    pca_comp_final.append(pca_xform)
                    print('Opt_num_filter for task 1: ', optimal_num_filters) 
                    filter1 =model.state_dict()[list(model.state_dict().keys())[i*2]].cpu().numpy().reshape(out_size,inp_size*k_size*k_size).swapaxes(0,1)
                    out_filt=filter1
                    bias1   =model.state_dict()[list(model.state_dict().keys())[(i*2)+1]].cpu().numpy()
                    out_bias=bias1   

                    ### Zeroing out certain portion of weights keeping the required amount of filters from the task                 
                    out_filt[:,int(optimal_num_filters[i]):]=0
                    out_bias[int(optimal_num_filters[i]):]=0
                    out_filt=out_filt.reshape(inp_size,k_size,k_size,out_size).swapaxes(0,3).swapaxes(2,3).swapaxes(2,1)
                    
                    ## saving for later use 
                    if args.rand_init == 1:
                        print('Random Initialization-----')
                        weight_init=oldmodel.state_dict()[list(model.state_dict().keys())[i*2]].cpu().numpy()
                        bias_init  =oldmodel.state_dict()[list(model.state_dict().keys())[(i*2)+1]].cpu().numpy()
                        weight_init[int(optimal_num_filters[i]):,:,:,:]=0
                        bias_init[int(optimal_num_filters[i]):]=0
                        weight_tx_list.append(weight_init)
                        bias_tx_list.append(bias_init)

                    else:
                        weight_tx_list.append(out_filt)
                        bias_tx_list.append(out_bias)
                
                else:           
                    print ('Projection-Subtraction-PCA Step at layer {}'.format(i+1))          
                    optimal_num_filters[i]=projection_subtraction_PCA(model.act[list(model.act.keys())[i]][0:scale[i]],pca_comp_final,lx, i, threshold=args.var_kept[i])
                    print ('Optimal filter list: {} after layer {} PCAs'.format (optimal_num_filters,i+1))
                    
                    ## Back Up the filters 
                    filter1_backup = model.state_dict()[list(model.state_dict().keys())[i*2]].cpu().numpy()
                    bias1_backup = model.state_dict()[list(model.state_dict().keys())[(i*2)+1]].cpu().numpy() 

                    # copies of filters
                    filter1 =model.state_dict()[list(model.state_dict().keys())[i*2]].cpu().numpy().reshape(out_size,inp_size*k_size*k_size).swapaxes(0,1)
                    bias1   =model.state_dict()[list(model.state_dict().keys())[(i*2)+1]].cpu().numpy()
                    out_filt=filter1
                    out_bias=bias1  

                    ## Zeroing out certain portion of weights 
                    out_filt[:,int(optimal_num_filters[i]):]=0
                    out_bias[int(optimal_num_filters[i]):]=0
                    out_filt=out_filt.reshape(inp_size,k_size,k_size,out_size).swapaxes(0,3).swapaxes(2,3).swapaxes(2,1)
                    ## saving for later use 
                    if args.rand_init == 1:
                        print('Random Initialization-----')
                        weight_init=oldmodel.state_dict()[list(model.state_dict().keys())[i*2]].cpu().numpy()
                        bias_init  =oldmodel.state_dict()[list(model.state_dict().keys())[(i*2)+1]].cpu().numpy()
                        
                        weight_init[0:int(filter_num[i*2]),:,:,:]=filter1_backup[0:int(filter_num[i*2]),:,:,:]
                        bias_init[0:int(filter_num[i*2])]=bias1_backup[0:int(filter_num[i*2])]

                        weight_init[int(optimal_num_filters[i]):,:,:,:]=0
                        bias_init[int(optimal_num_filters[i]):]=0
                        weight_tx_list.append(weight_init)
                        bias_tx_list.append(bias_init)

                    else:
                        weight_tx_list.append(out_filt)
                        bias_tx_list.append(out_bias)
                
            #########------------------------------------- Retraining after PCA ------------------------------------############
            for ik in range(5):
                ## Applying all the transformations --- Check what is the differnce with random initialization !!
                model.state_dict()[list(model.state_dict().keys())[ik*2]].copy_(torch.Tensor(weight_tx_list[ik]))
                model.state_dict()[list(model.state_dict().keys())[(ik*2)+1]].copy_(torch.Tensor(bias_tx_list[ik]))

            print ('Starting learning rate for retraining Task{} is {}'.format(idx+1,args.lr))
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum) # using SGD with momentum 
            
            best_accuracy=0.0
            best_epoch = 0 
            for epoch in range(1, args.epoch_list[0] + 1):   
                if ( args.split<2 and  (epoch == int(args.epoch_list[0]*0.35) or epoch == int(args.epoch_list[0]*0.9)) ):
                	adjust_learning_rate(optimizer,epoch,args)
                elif ( args.split>1 and  (epoch == int(args.epoch_list[0]*0.2) or epoch == int(args.epoch_list[0]*0.9))) :
                    adjust_learning_rate(optimizer,epoch,args)
     
                loss_hist=train_next_pca(args, model, device, train_loader, optimizer, epoch,loss_hist,optimal_num_filters, filter_num)
                temp_accuracy=test(args, model, device, test_loader,loss_test) 
                if (temp_accuracy > best_accuracy):
                    best_accuracy = temp_accuracy
                    best_epoch = epoch
                    model_temp_param = {'state_dict': model.state_dict()}
                    torch.save(model_temp_param,checkpoint_fname)
            print('best Training accuracy: {} at {} epoch'.format(best_accuracy,best_epoch))
            model_load_param = torch.load(checkpoint_fname)
            model.load_state_dict(model_load_param['state_dict'])
            
            ##--------------------------------------------------------Saving --------------------------------------------------------## 
            # saving test accuracy after retraining 
            acc=test(args, model, device, test_loader,loss_test)
            accuracy_retrain.append(acc)
      
            lx= optimal_num_filters
            print ('Current Task  filter list:',lx)
            filter_num = [lx[0],lx[0],lx[1],lx[1],lx[2],lx[2],lx[3],lx[3],lx[4],lx[4]]           
            
            # saving task specific classifer weights and corresponding filter statistics 
            fc_weight = model.fc1.weight.detach().cpu().numpy()
            fc_bias   = model.fc1.bias.detach().cpu().numpy()
            
            opt_filter_list.append(optimal_num_filters)
            keep_classifier_list.append(fc_weight)
            keep_classifier_list.append(fc_bias)  
            running_param = {'opt_filter_list' : opt_filter_list, 'classifier_list' : keep_classifier_list, 'pca_comp' : pca_comp_final }

            ## Initialize the pruned out filters of the original network with random initializaton values before training on new task  
            for k, (p,p_old) in enumerate(zip(model.parameters(),oldmodel.parameters())):
                if (k%2==0 and k<10):
                    temp_old=p_old.detach().cpu().numpy()
                    temp    =p.detach().cpu().numpy()
                    temp [int(filter_num[k]):,:,:,:] = temp_old[int(filter_num[k]):,:,:,:]
                    model.state_dict()[list(model.state_dict().keys())[k]].copy_(torch.Tensor(temp))
                elif (k%2==1 and k<10): 
                    temp_old=p_old.detach().cpu().numpy()
                    temp    =p.detach().cpu().numpy()
                    temp [int(filter_num[k]):] = temp_old[int(filter_num[k]):]
                    model.state_dict()[list(model.state_dict().keys())[k]].copy_(torch.tensor(temp)) 
            
            # save and print the model statistics 
            model_param = {'state_dict': model.state_dict()}
            if (args.save_model):
                print('Saving Model...')
                torch.save(model_param,param_fnameB_rt)
                torch.save(running_param,running_param_fname)
            print ('Training Accuracy:',accuracy_train)
            print ('Retraining Accuracy:',accuracy_retrain)
        print ('Opt_filter_list:',opt_filter_list)


if __name__ == '__main__':
    main()
