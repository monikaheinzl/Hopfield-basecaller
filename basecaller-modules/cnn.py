#!/usr/bin/env python
"""
BSD 2-Clause License

Copyright (c) 2021 (monika.heinzl@edumail.at)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np 
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import pickle
import random
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import math
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.tensorboard import SummaryWriter
import itertools 
import seaborn as sns
import pandas as pd
import argparse
from distutils.util import strtobool
import json
import math


class BasicBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size_branch1, kernel_size_branch2, stride, 
        padding, batch_norm=False, input_bias_cnn=True):
        super(BasicBlock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size_branch1 = kernel_size_branch1
        self.kernel_size_branch2 = kernel_size_branch2
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.input_bias_cnn = input_bias_cnn
        print(output_channel, kernel_size_branch1, kernel_size_branch2, kernel_size_branch1//2)
        # TODO: 3x3 convolution -> relu
        #the input and output channel number is channel_num
        if self.batch_norm:
            self.branch1 = nn.Sequential(
                nn.Conv1d(self.input_channel, self.output_channel,
                kernel_size=kernel_size_branch1, stride=1, padding=kernel_size_branch1//2, bias=self.input_bias_cnn), # padding=SAME
                nn.BatchNorm1d(self.output_channel)
            )
        else:
            self.branch1 = nn.Sequential(
                nn.Conv1d(self.input_channel, self.output_channel,
                kernel_size=kernel_size_branch1, stride=1, padding=kernel_size_branch1//2, bias=self.input_bias_cnn)
            )
        
        if self.batch_norm:
            self.branch2 = nn.Sequential(
                # layer1
                nn.Conv1d(self.input_channel, self.output_channel,
                kernel_size=1, stride=1, padding=1//2, bias=self.input_bias_cnn), # padding=SAME
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU(),
                # layer2
                nn.Conv1d(self.output_channel, self.output_channel,
                    kernel_size=kernel_size_branch2, stride=1, padding=kernel_size_branch2//2, bias=self.input_bias_cnn), # padding=SAME
                nn.BatchNorm1d(self.output_channel),
                nn.ReLU(),
                # layer3
                nn.Conv1d(self.output_channel, self.output_channel,
                kernel_size=1, stride=1, padding=1//2, bias=self.input_bias_cnn), # padding=SAME
                nn.BatchNorm1d(self.output_channel),
            )
        else:
            self.branch2 = nn.Sequential(
            # layer1
                nn.Conv1d(self.input_channel, self.output_channel,
                    kernel_size=1, stride=1, padding=1//2, bias=self.input_bias_cnn), # padding=SAME
                nn.ReLU(),
                # layer2
                nn.Conv1d(self.output_channel, self.output_channel,
                    kernel_size=kernel_size_branch2, stride=1, padding=kernel_size_branch2//2, bias=self.input_bias_cnn), # padding=SAME
                nn.ReLU(),
                # layer3
                nn.Conv1d(self.output_channel, self.output_channel,
                    kernel_size=1, stride=1, padding=1//2, bias=self.input_bias_cnn), # padding=SAME
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        indata_cp = self.branch1(x)
        conv_layer3 = self.branch2(x)
        out = indata_cp + conv_layer3
        out = self.relu(out)
        return out

class SimpleCNN_res(torch.nn.Module):
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, res_layer, layers, dropout=False, dropout_p=0.5, dropout_input=False):
        super(SimpleCNN_res, self).__init__()
        self.res_layer = res_layer
        self.layers = layers
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_input = dropout_input

        if self.dropout_input:
            self.drop_in = nn.Dropout(self.dropout_p)

        if self.layers >= 1:
            self.layer1 = self.res_layer[0]
        if self.layers >= 2:
            self.layer2 = self.res_layer[1]
        if self.layers >= 3:
            self.layer3 = self.res_layer[2]
        if self.layers >= 4:
            self.layer4 = self.res_layer[3]
        if self.layers >= 5:
            self.layer5 = self.res_layer[4]
        
        if self.dropout:
            self.drop_out = nn.Dropout(self.dropout_p)  

    def forward(self, x, x_len):
        x = x.float()
        #print("in", x.size())
        if self.dropout_input:
            x = self.drop_in(x)

        if self.layers >= 1:
            # Conv 1
            x = self.layer1(x)

        if self.layers >= 2:
            # Conv 2
            x = self.layer2(x)

        if self.layers >= 3:
            # Conv 3
            x = self.layer3(x)

        if self.layers >= 4:
            # Conv 4
            x = self.layer4(x)

        if self.layers >= 5:
            # Conv 5
            x = self.layer5(x)

        if self.dropout:
            x = self.drop_out(x)
        x_len_out = x_len
        return(x, x_len_out)
class SimpleCNN(torch.nn.Module):
    #Our batch shape for input x is (batch size, 1, seq length)
    def __init__(self, input_channel, output_channel, kernel_size, stride, 
        padding, pooling="average", layers=1, 
        batch_norm=False, dropout=False, dropout_p=0.5, dropout_input=False, input_bias_cnn=True):
        super(SimpleCNN, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_bias_cnn = input_bias_cnn

        self.kernel_pool = 2
        self.pooling = pooling
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_p = dropout_p
        self.dropout_input = dropout_input
        self.counter=0

        if self.dropout_input:
            self.drop_in = nn.Dropout(self.dropout_p)

        if self.layers >= 1:
            # Convolution 1
            #Input channels = 1, output channels = 16
            self.conv1 = torch.nn.Conv1d(self.input_channel, self.output_channel[0], #16*2 
                                         kernel_size=self.kernel_size[0], stride=self.stride[0], 
                                         padding=self.padding[0], bias=self.input_bias_cnn)
            # Batch Normalization 1
            if self.batch_norm:
                # Batch Normalization 1
                self.conv1_bn = nn.BatchNorm1d(self.output_channel[0])
            
            # Max pool 1
            if self.pooling == "average":
                self.pool = torch.nn.AvgPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
            elif self.pooling == "max":
                self.pool = torch.nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
            
        if self.layers >= 2:
            # Convolution 2
            self.conv2 = torch.nn.Conv1d(self.output_channel[0], self.output_channel[1], 
                                         kernel_size=self.kernel_size[1], stride=self.stride[1], 
                                         padding=self.padding[1], bias=self.input_bias_cnn)            
            # Batch Normalization 2
            if self.batch_norm:
                # Batch Normalization 2
                self.conv2_bn = nn.BatchNorm1d(self.output_channel[1])
            
            # Max pool 2
            if self.pooling == "average":
                self.pool2 = torch.nn.AvgPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
            elif self.pooling == "max":
                self.pool2 = torch.nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
        
        if self.layers >= 3:
            # Convolution 3
            self.conv3 = torch.nn.Conv1d(self.output_channel[1], self.output_channel[2], 
                                         kernel_size=self.kernel_size[2], stride=self.stride[2], 
                                         padding=self.padding[2], bias=self.input_bias_cnn)
            if self.batch_norm:
                # Batch Normalization 2
                self.conv3_bn = nn.BatchNorm1d(self.output_channel[2])
            
            # Max pool 3
            if self.pooling == "average":
                self.pool3 = torch.nn.AvgPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
            elif self.pooling == "max":
                self.pool3 = torch.nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K

        if self.layers >= 4:
            # Convolution 3
            self.conv4 = torch.nn.Conv1d(self.output_channel[2], self.output_channel[3], 
                                         kernel_size=self.kernel_size[3], stride=self.stride[3], 
                                         padding=self.padding[3], bias=self.input_bias_cnn)
            if self.batch_norm:
                # Batch Normalization 2
                self.conv4_bn = nn.BatchNorm1d(self.output_channel[3])
            
            # Max pool 3
            if self.pooling == "average":
                self.pool4 = torch.nn.AvgPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
            elif self.pooling == "max":
                self.pool4 = torch.nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K

        if self.layers >= 5:
            # Convolution 3
            self.conv5 = torch.nn.Conv1d(self.output_channel[3], self.output_channel[4], 
                                         kernel_size=self.kernel_size[4], stride=self.stride[4], 
                                         padding=self.padding[4], bias=self.input_bias_cnn)
            if self.batch_norm:
                # Batch Normalization 2
                self.conv5_bn = nn.BatchNorm1d(self.output_channel[4])
            
            # Max pool 3
            if self.pooling == "average":
                self.pool5 = torch.nn.AvgPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
            elif self.pooling == "max":
                self.pool5 = torch.nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K


        if self.layers >= 6:
            # Convolution 3
            self.conv6 = torch.nn.Conv1d(self.output_channel[4], self.output_channel[5], 
                                         kernel_size=self.kernel_size[5], stride=self.stride[5], 
                                         padding=self.padding[5], bias=self.input_bias_cnn)
            if self.batch_norm:
                # Batch Normalization 2
                self.conv6_bn = nn.BatchNorm1d(self.output_channel[5])
            
            # Max pool 3
            if self.pooling == "average":
                self.pool6 = torch.nn.AvgPool1d(kernel_size=self.kernel_pool) # sequence len/K
            # O=W/K = 50/50 = 1
            elif self.pooling == "max":
                self.pool6 = torch.nn.MaxPool1d(kernel_size=self.kernel_pool) # sequence len/K

        if self.dropout:
            self.drop_out = nn.Dropout(self.dropout_p)
        

    def forward(self, x, x_len):
        x = x.float()
        #print("in", x.size())
        if self.dropout_input:
            x = self.drop_in(x)

        if self.layers >= 1:
            # Conv 1
            if self.batch_norm:
                x = F.relu(self.conv1_bn(self.conv1(x)))
            else:
                x = F.relu(self.conv1(x))
            x_len_out = outputLen_Conv(x_len, self.conv1.kernel_size[0], self.conv1.padding[0], self.conv1.stride[0], self.conv1.dilation[0])
            
            # Max pool1
            if self.pooling == "average":
                x = self.pool(x)
                x_len_out = outputLen_AvgPool(x_len_out, self.pool.kernel_size[0], self.pool.padding[0], self.pool.stride[0])
            elif self.pooling == "max":
                x = self.pool(x)
                x_len_out = outputLen_MaxPool(x_len_out, self.pool.kernel_size, self.pool.padding, self.pool.stride, self.pool.dilation)
            #print("after layer 1", x.size())

        if self.layers >= 2:
            # Conv 2
            if self.batch_norm:
                x = F.relu(self.conv2_bn(self.conv2(x)))
            else:
                x = F.relu(self.conv2(x))
            x_len_out = outputLen_Conv(x_len_out, self.conv2.kernel_size[0], self.conv2.padding[0], self.conv2.stride[0], self.conv2.dilation[0])
            
            # Max pool2
            if self.pooling == "average":
                x = self.pool2(x)
                x_len_out = outputLen_AvgPool(x_len_out, self.pool2.kernel_size[0], self.pool2.padding[0], self.pool2.stride[0])
            elif self.pooling == "max":
                x = self.pool2(x)
                x_len_out = outputLen_MaxPool(x_len_out, self.pool2.kernel_size, self.pool2.padding, self.pool2.stride, self.pool2.dilation)
            #print("after layer 2", x.size())

        if self.layers >= 3:
            # Conv 3
            if self.batch_norm:
                x = F.relu(self.conv3_bn(self.conv3(x)))
            else:
                x = F.relu(self.conv3(x))
            x_len_out = outputLen_Conv(x_len_out, self.conv3.kernel_size[0], self.conv3.padding[0], self.conv3.stride[0], self.conv3.dilation[0])
            # Max pool3
            if self.pooling == "average":
                x = self.pool3(x)
                x_len_out = outputLen_AvgPool(x_len_out, self.pool3.kernel_size[0], self.pool3.padding[0], self.pool3.stride[0])
            elif self.pooling == "max":
                x = self.pool3(x)
                x_len_out = outputLen_MaxPool(x_len_out, self.pool3.kernel_size, self.pool3.padding, self.pool3.stride, self.pool3.dilation)
            #print("after layer 3", x.size())

        if self.layers >= 4:
            # Conv 3
            if self.batch_norm:
                x = F.relu(self.conv4_bn(self.conv4(x)))
            else:
                x = F.relu(self.conv4(x))
            x_len_out = outputLen_Conv(x_len_out, self.conv4.kernel_size[0], self.conv4.padding[0], self.conv4.stride[0], self.conv4.dilation[0])
            # Max pool3
            if self.pooling == "average":
                x = self.pool4(x)
                x_len_out = outputLen_AvgPool(x_len_out, self.pool4.kernel_size[0], self.pool4.padding[0], self.pool4.stride[0])
            elif self.pooling == "max":
                x = self.pool4(x)
                x_len_out = outputLen_MaxPool(x_len_out, self.pool4.kernel_size, self.pool4.padding, self.pool4.stride, self.pool4.dilation)
            #print("after layer 3", x.size())

        if self.layers >= 5:
            # Conv 3
            if self.batch_norm:
                x = F.relu(self.conv5_bn(self.conv5(x)))
            else:
                x = F.relu(self.conv5(x))
            x_len_out = outputLen_Conv(x_len_out, self.conv5.kernel_size[0], self.conv5.padding[0], self.conv5.stride[0], self.conv5.dilation[0])
            # Max pool3
            if self.pooling == "average":
                x = self.pool5(x)
                x_len_out = outputLen_AvgPool(x_len_out, self.pool5.kernel_size[0], self.pool5.padding[0], self.pool5.stride[0])
            elif self.pooling == "max":
                x = self.pool5(x)
                x_len_out = outputLen_MaxPool(x_len_out, self.pool5.kernel_size, self.pool5.padding, self.pool5.stride, self.pool5.dilation)
            #print("after layer 3", x.size())

        if self.layers >= 6:
            # Conv 3
            if self.batch_norm:
                x = F.relu(self.conv6_bn(self.conv6(x)))
            else:
                x = F.relu(self.conv6(x))
            x_len_out = outputLen_Conv(x_len_out, self.conv6.kernel_size[0], self.conv6.padding[0], self.conv6.stride[0], self.conv6.dilation[0])
            # Max pool3
            if self.pooling == "average":
                x = self.pool6(x)
                x_len_out = outputLen_AvgPool(x_len_out, self.pool6.kernel_size[0], self.pool6.padding[0], self.pool6.stride[0])
            elif self.pooling == "max":
                x = self.pool6(x)
                x_len_out = outputLen_MaxPool(x_len_out, self.pool6.kernel_size, self.pool6.padding, self.pool6.stride, self.pool6.dilation)
            #print("after layer 3", x.size())

        if self.dropout:
            x = self.drop_out(x)
        
        if self.counter == 0:
            print("set size after CNN", x.size())
        self.counter += 1
            
        return(x, x_len_out)

def outputLen_Conv(input_len, kernel_size, padding, stride, dilation):
    out = torch.floor(((input_len + (2*padding) - (dilation*(kernel_size - 1)) -1)/float(stride)) + 1)
    return(out)
def outputLen_AvgPool(input_len, kernel_size, padding, stride):
    out = torch.floor(((input_len + (2*padding) - kernel_size)/float(stride)) + 1)
    return(out)
def outputLen_MaxPool(input_len, kernel_size, padding, stride, dilation):
    out = torch.floor(((input_len + (2*padding) - (dilation*(kernel_size - 1)) -1)/float(stride)) + 1)
    return(out)