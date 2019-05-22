#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 06:16:09 2019

@author: derar
"""
import torch
import os
import sys
import pickle
import random
import argparse
import logging

import torchvision.models as models
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda as cuda
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from model_vgglstm import VGG,AlexNet
from data.dataloader import get_dataloader

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 20)')
parser.add_argument('--batchSize', action='store', default=1, type=int, help='batch size (default: 1)')
parser.add_argument('--windowSize', action='store', default=25, type=int, help='number of frames (default: 25)')
parser.add_argument('--h_dim', action='store', default=256, type=int, help='LSTM hidden layer dimension (default: 256)')
parser.add_argument('--lr','--learning-rate',action='store',default=0.01, type=float,help='learning rate (default: '
                                                                                          '0.01)')
parser.add_argument('--train_fold',action='store',default=0, type=int,help='Training Fold (default: 0)')
parser.add_argument('--test_fold',action='store',default=0, type=int,help='Testing Fold (default: 0)')

parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='AlexNet', const='AlexNet',nargs='?', choices=['VGG', 'AlexNet'], help="net model(default:VGG)")

arg = parser.parse_args()

def main():
    if torch.cuda.is_available() and arg.useGPU_F:
        torch.cuda.set_device(arg.gpu_num)
        torch.cuda.current_device()
    
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
    model_path = 'model/model_LSTM'+str(arg.lr)+'_'+arg.net+'.pt'
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    ch = logging.FileHandler('log/logfile_LSTM'+str(arg.lr)+'_'+arg.net+'.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    logger.info("Batch Size: {}".format(arg.batchSize))
    logger.info("Window Size: {}".format(arg.windowSize))
    logger.info("Hidden Layer Dimension: {}".format(arg.h_dim))
    logger.info("GPU num: {}".format(arg.gpu_num))
    
    #root_dir = 'UCF11_split'
    #train_path = root_dir+'/train'
    #test_path = root_dir+'/test'
    num_of_classes=10
    
    trainLoader = get_dataloader(fold=[arg.train_fold], batch_size=1, shuffle=True, db_prepped=True)
    testLoader =  get_dataloader(fold=[arg.test_fold], batch_size=1, shuffle=True, db_prepped=True)
    trainSize = len(trainLoader)
    testSize =  len(testLoader)
    
    if arg.net == 'VGG':
        model = VGG(arg.h_dim, num_of_classes)
    elif arg.net =='AlexNet':
        model = AlexNet(arg.h_dim, num_of_classes)
    
    if arg.useGPU_f:
        model.cuda()
    
    optimizer = optim.Adam(model.parameters(),lr=arg.lr)
    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    if arg.useGPU_f:
        hidden = ( Variable(torch.randn(1,arg.batchSize,arg.h_dim).cuda(),requires_grad=False),
                   Variable(torch.randn(1,arg.batchSize,arg.h_dim).cuda(),requires_grad=False))
    else:
        hidden = ( Variable(torch.randn(1,arg.batchSize,arg.h_dim),requires_grad=False),
                   Variable(torch.randn(1,arg.batchSize,arg.h_dim),requires_grad=False))
     
    min_acc=0.0
    ##########################
    ##### Start Training #####
    ##########################
    epochs = arg.epochs if arg.train_f else 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for batchIdx,(windowBatch,labelBatch) in enumerate(trainLoader):
            #loss=0.0
            if arg.useGPU_f:
                y=torch.zeros(arg.batchSize, num_of_classes).cuda()
                windowBatch = Variable(windowBatch.cuda(),requires_grad=True).float()
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False).long()
            else:
                y=torch.zeros(arg.batchSize, num_of_classes)
                windowBatch = Variable(windowBatch,requires_grad=True).float()
                labelBatch = Variable(labelBatch,requires_grad=False).long()

            windowSize = windowBatch.shape[1]
            for i in range(windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                temp,hidden = model(imgBatch,hidden)
                (h,c) = hidden
                hidden = (h.detach(), c.detach())
                #loss_ = criterion(temp,labelBatch)
                #loss+=loss_.data
                y += temp
            
            Y=y/windowSize
            #loss = Variable(loss.cuda(),requires_grad=True)
            loss = criterion(Y,labelBatch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            _,pred = torch.max(Y,1) ### prediction should after averging the array
            train_acc = (pred == labelBatch.data).sum()
            train_acc = 100.0*train_acc.data.cpu().numpy()/arg.batchSize
            #print('train acc', train_acc, 'train loss', loss.data.cpu())

            if batchIdx%50==0:
                logger.info("epochs:{}, train loss:{}, train acc:{}".format(epoch, loss.data.cpu(), train_acc))
        
        ########################
        ### Start Validation ###
        ########################
        model.eval()
        val_acc=0.0
        for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader):
            if arg.useGPU_f:
                y=torch.zeros(arg.batchSize, num_of_classes).cuda()
                windowBatch = Variable(windowBatch.cuda(),requires_grad=False).float()
                labelBatch = Variable(labelBatch.cuda(),requires_grad=False).long()
            else:
                y=torch.zeros(arg.batchSize, num_of_classes)
                windowBatch = Variable(windowBatch,requires_grad=False).float()
                labelBatch = Variable(labelBatch,requires_grad=False).long()

            windowSize = windowBatch.shape[1]
            for i in range(windowSize):
                imgBatch = windowBatch[:,i,:,:,:]
                temp,hidden = model(imgBatch,hidden)
                (h,c) = hidden
                hidden = (h.detach(), c.detach())
                #loss_ = criterion(temp,labelBatch)
                #loss+=loss_.data
                y += temp
            
            Y=y/windowSize
            loss = criterion(Y,labelBatch)

            _,pred = torch.max(Y,1)
            val_acc = (pred == labelBatch.data).sum()
            val_acc = 100.0*val_acc.data.cpu().numpy()/arg.batchSize


        logger.info("==> val loss:{}, val acc:{}".format(loss.data.cpu().numpy(),val_acc))
        
        if val_acc>min_acc:
            min_acc=val_acc
            torch.save(model.state_dict(), model_path)
            
    ##########################
    ##### Start Testing #####
    ##########################   
    model.eval()
    torch.no_grad()
    test_acc=0.0
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        
    for batchIdx,(windowBatch,labelBatch) in enumerate(testLoader):
        if arg.useGPU_f:
            y=torch.zeros(arg.batchSize, num_of_classes).cuda()
            windowBatch = Variable(windowBatch.cuda(),requires_grad=False).float()
            labelBatch = Variable(labelBatch.cuda(),requires_grad=False).long()
        else:
            y=torch.zeros(arg.batchSize, num_of_classes)
            windowBatch = Variable(windowBatch,requires_grad=False).float()
            labelBatch = Variable(labelBatch,requires_grad=False).long()

        windowSize = windowBatch.shape[1]
        for i in range(windowSize):
            imgBatch = windowBatch[:,i,:,:,:]
            temp,hidden = model(imgBatch,hidden)
            (h,c) = hidden
            hidden = (h.detach(), c.detach())
            #loss_ = criterion(temp,labelBatch)
            #loss+=loss_.data
            y += temp

        Y=y/windowSize
        loss = criterion(Y,labelBatch)
        _,pred = torch.max(y,1)
        test_acc += (pred == labelBatch.data).sum()
    test_acc = 100.0*test_acc.data.cpu().numpy()/testSize
    
    logger.info("==> test loss:{}, test acc:{}".format(loss.data.cpu().numpy(),test_acc))


if __name__ == "__main__":
    main()
