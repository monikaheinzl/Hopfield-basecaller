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


import h5py    
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
from polyleven import levenshtein


sys.path.insert(0, '/basecaller-modules')
from early_stopping import EarlyStopping
from cnn import SimpleCNN, BasicBlock, SimpleCNN_res
from cnn import outputLen_Conv, outputLen_AvgPool, outputLen_MaxPool
plt.switch_backend('agg')

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __eq__(self, other):
        return self.start <= other <= self.end

def make_argparser():
    parser = argparse.ArgumentParser(description='Nanopore Basecaller')
    parser.add_argument('-i', '--input', required = True,
                        help="File path to the pickle input file.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output folder name")

    parser.add_argument('-g', '--gpu_port', type=int, default=1,
                        help="Port on GPU mode")
    parser.add_argument('-s', '--set_seed', type=int, default=1234,
                        help="Set seed")

    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help="Number of epochs")
    parser.add_argument('-v', '--make_validation', type=int, default=1000,
                        help="Make every n updates evaluation on the validation set")

    # CNN arguments
    parser.add_argument("--input_bias_cnn", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)

    parser.add_argument('-c', '--channel_number', type=int, default=256,
                        help="Number of output channels in Encoder-CNN")
    parser.add_argument('-l', '--cnn_layers', type=int, default=1,
                        help="Number of layers in Encoder-CNN")
    parser.add_argument('--pooling_type', default="None",
                        help="Pooling type in Encoder-CNN")
    parser.add_argument('--strides', nargs='+', type=int, default=[1, 1, 1],
                        help="Strides in Encoder-CNN")
    parser.add_argument('--kernel', nargs='+', type=int, default=[11, 11, 11],
                        help="Kernel sizes in Encoder-CNN")
    parser.add_argument("--dropout_cnn", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--dropout_input", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--drop_prob', type=float, default=Range(0.0, 1.0),
                        help="Dropout probability Encoder-CNN")
    parser.add_argument("--batch_norm", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    # LSTM arguments
    parser.add_argument("--attention", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)
    parser.add_argument('--dropout', type=float, default=Range(0.0, 1.0),
                        help="Dropout probability Encoder-LSTM")
    parser.add_argument('-u', '--hidden_units', type=int, default=256,
                        help="Number of hidden units in Encoder-Decoder-LSTM")
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help="Number of layers in Encoder-LSTM")
    parser.add_argument('--forget_bias_encoder', type=str, default="0",
                        help="Set forget gate bias in Encoder-LSTM")
    parser.add_argument('--forget_bias_decoder', type=str, default="0",
                        help="Set forget gate bias in Decoder-LSTM")
    parser.add_argument("--bi_lstm", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    # teacher forcing
    parser.add_argument("--reduced_tf", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    parser.add_argument('--tf_ratio', type=float, default=Range(0.0, 1.0),
                        help="Teacher forcing ratio. Default=1, TF on")

    parser.add_argument('--weight_decay', type=float, default=0,
                        help="Weight decay")
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Weight decay")
    parser.add_argument("--reduce_lr", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    parser.add_argument('--gradient_clip', default="None",
                        help="Gradient clipping")

    # early stopping
    parser.add_argument("--early_stop", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--patience', type=int, default=25,
                        help="Patience in early stopping")

    parser.add_argument("--call", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    parser.add_argument("--editD", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)


    return parser

# Network
# -----------
# * CNN-Encoder
# * LSTM-Encoder
# * LSTM-Decoder

# The Encoder
# -----------
class LSTMCellEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCellEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
#        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(h.size(0), -1)
        c = c.view(c.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        gates = self.i2h(x) + self.h2h(h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
    
        cy = (forgetgate * c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        
        return hy, (hy, cy), (torch.mean(ingate).cpu(),torch.mean(forgetgate).cpu(),
                              torch.mean(cellgate).cpu(), torch.mean(outgate).cpu())

class LSTMencoder(nn.Module):
    #Our batch shape for input x is [batch, seq_len, input_dim]
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=5,
                    num_layers=2, own_cell_encoder = False, bidirectional=False, port=1, dropout=0):
        super(LSTMencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.own_cell_encoder = own_cell_encoder
        self.bidirectional = bidirectional
        self.port = port
        self.dropout=dropout

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
#        self.lstm_p = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        

    def init_hidden(self):
       # This is what we'll initialise our hidden state as
       return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).requires_grad_().cuda(self.port),
               torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).requires_grad_().cuda(self.port))

    def forward(self, x, x_lengths):
        #hidden = self.init_hidden()
        # Forward pass through LSTM layer
        # shape of lstm_in: [batch, seq_len, input_dim]
        # shape of lstm_out: [batch, seq_len, output_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        
        # Sort instances by sequence length in descending order
        #print("in length", x_lengths)
        sorted_len, sorted_idx = x_lengths.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(x)
        sorted_inputs = x.gather(0, index_sorted_idx.long()) # [batch, seq_len, input_dim]
        
        
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_seq = nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_len.cpu().numpy(), batch_first=True)
        # Define the LSTM layer
        lstm_out, hidden = self.lstm(packed_seq) # [seq_len, batch, input_dim]
        # undo the packing operation
        unpacked, unpacked_len = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # targets oder batch sortieren
        # unsort the output
        _, original_idx = sorted_idx.sort(0, descending=False)
        
        # unsort hiddens
        original_idx = original_idx.cpu()
        unsorted_idx_hiddens = original_idx.view(1, -1, 1).expand_as(hidden[0])
        unsorted_hiddens1 = hidden[0][:, original_idx, :]#.gather(0, unsorted_idx_hiddens.long())
        unsorted_hiddens2 = hidden[1][:, original_idx, :]#.gather(0, unsorted_idx_hiddens.long())
        hidden_original = (unsorted_hiddens1, unsorted_hiddens2)#.cpu()
        unpacked_original = unpacked[original_idx, :, :]

        return hidden_original, unpacked_original


# The Decoder
# -----------

class LSTMCellDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, port=1):
        super(LSTMCellDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.i2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.port = port

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        h, c = hidden
        h = h.view(x.size(0), -1)
        c = c.view(x.size(0), -1)
        x = x.view(x.size(0), -1)
        # Linear mappings
        gates = self.i2h(x) + self.h2h(h)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
    
        cy = (forgetgate * c) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        #print("hy", hy.size())
        hy = torch.unsqueeze(hy, 0).cuda(self.port) # (batch, 1, feature)
        cy = torch.unsqueeze(cy, 0).cuda(self.port) # (batch, 1, feature)
        return hy, (hy, cy), (torch.mean(ingate).cpu(), torch.mean(forgetgate).cpu(),
                              torch.mean(cellgate).cpu(), torch.mean(outgate).cpu())

class LSTMdecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=5,
                    num_layers=2, own_cell_decoder = False, bidirectional = False, port=1, attention=True, dropout=0):
        super(LSTMdecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.own_cell_decoder = own_cell_decoder
        self.bidirectional = bidirectional
        self.port=port
        self.attention = attention
        self.dropout = dropout
        
        # Define the LSTM layer
        if self.bidirectional:
            if self.attention:
                self.lstm_unroll = nn.LSTM(self.input_dim+self.hidden_dim*2, self.hidden_dim*2, self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout)
            else:
                self.lstm_unroll = nn.LSTM(self.input_dim, self.hidden_dim*2, self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout)   
        else:
            if self.attention:
                self.lstm_unroll = nn.LSTM(self.input_dim+self.hidden_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout)
            else:
                self.lstm_unroll = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True, bidirectional=False, dropout=self.dropout)

        # Define the output layer
        if self.bidirectional:
            if self.attention:
                # attention
                self.attn = nn.Linear(self.hidden_dim * 4, 1)
            #self.attn = nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2)
            self.linear = nn.Linear(self.hidden_dim*2, self.output_dim)
        else:
            if self.attention:
                # attention
                self.attn = nn.Linear(self.hidden_dim * 2, 1)
            #self.attn = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, encoder_outputs, true_x_len, max_label_len, y, teacher_forcing_ratio, hidden,
                true_y_len, y_class, random_value, mode_type, beam_width=1):
        # Forward pass through LSTM layer
        # shape of lstm_in: [batch, input_dim]
        # shape of lstm_out: [seq_len, batch, output_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).        
        if self.bidirectional:
            hidden1 = hidden[0]
            hidden2 = hidden[1]
            h1 = torch.cat([hidden1[0:hidden1.size(0):2], hidden1[1:hidden1.size(0):2]], 2)
            h2 = torch.cat([hidden2[0:hidden2.size(0):2], hidden2[1:hidden2.size(0):2]], 2)
            hidden = (h1, h2)

        batch_size = true_y_len.size(0)
        max_for = range(max_label_len)
        outputs = torch.zeros(max_label_len, batch_size, self.output_dim).cuda(self.port) # [max. target length, batch size, hidden dim]
        max_in_batch = int(max(true_y_len.cpu()))

        start_seq = True
        input_decoder = torch.zeros(batch_size, self.input_dim).cuda(self.port) # SOS token = 1 [batch size, SOS token]
        for i in max_for:
            # Stop looping if we got to the last element in the batch
            
            if i == max_in_batch:
                break
        
            # when seq length (i) >= true seq length
            if i >= true_y_len[-1].item() and len(true_y_len) > 1: 
                not_padded_batches = i < true_y_len # get indices of not padded sequences 
                true_y_len =  true_y_len[not_padded_batches] # remove smallest element = last one
                #print(true_y_len, true_y_len.size())
                input_decoder = input_decoder[not_padded_batches, :] # get only batches that are NOT padded
                h = hidden[0][:, not_padded_batches, :]
                c = hidden[1][:, not_padded_batches, :]
                hidden = (h, c)
                label = y[not_padded_batches, :, i] # batch-i, features, seq len
                y = y[not_padded_batches, :, :] # batch-i, features, seq len
                encoder_outputs = encoder_outputs[not_padded_batches, :, :]
            else:
                label = y[:, :, i] # batch, features, seq len
            
            if self.attention:  
                # ATTENTION MECHANISM
                attn_weights = F.softmax(
                        self.attn(torch.cat((encoder_outputs[:, i, :], hidden[0][0]), dim=1)), dim=1) # encoder batch first: b,len,hidden, hidden: 2,b,hidden
                attn_applied = torch.bmm(attn_weights.type(torch.FloatTensor).unsqueeze(1).cuda(self.port), 
                    encoder_outputs[:, i, :].type(torch.FloatTensor).unsqueeze(1).cuda(self.port))
                input_decoder = torch.cat((input_decoder, attn_applied.squeeze(1)), dim=1) # input_decoder: b,hidden; attn_applied: b, 1, hidden
            
            input = torch.unsqueeze(input_decoder.type(torch.FloatTensor), 1).cuda(self.port) # (batch, seq, feature), 
            lstm_in_unroll, hidden = self.lstm_unroll(input, hidden) # [batch size, seq, hidden dim]

            lstm_in_unroll = self.linear(lstm_in_unroll.view(-1, lstm_in_unroll.size(2))) # (batch_size*seq, out_dim)
            input_decoder = F.log_softmax(lstm_in_unroll, dim=1) # (batch_size*seq, out_dim) --> (batch_size, out_dim)

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # put on position: seq, 0:true_batch_size, features
            # rest: seq, true_batch_size:max_seq_len, features filled with 0
            outputs[i, 0:input_decoder.size(0), :] = input_decoder # [max. target length, batch size, output dim]
            top1 = input_decoder.argmax(1)  # get argmax of prediction
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            onehot = torch.zeros(len(top1), self.output_dim).cuda(self.port)        
            onehot[:, top1] = 1 # one hot encode input  
            input_decoder = label if teacher_force else onehot #  [batch size, out dim]

        return outputs, hidden


# Sequence to sequence model
# -----------


class Seq2Seq(nn.Module):
    def __init__(self, cnn_encoder, lstm_encoder, lstm_decoder):
        super().__init__()
        
        self.cnn_encoder = cnn_encoder
        self.lstm_encoder = lstm_encoder
        self.lstm_decoder = lstm_decoder
        
        assert lstm_encoder.hidden_dim == lstm_decoder.hidden_dim,             "Hidden dimensions of encoder and decoder must be equal!"
        assert lstm_encoder.num_layers == lstm_decoder.num_layers,             "Encoder and decoder must have equal number of layers!"
        assert lstm_encoder.batch_size == lstm_decoder.batch_size,             "Encoder and decoder must have equal batch size!"
        
    def forward(self, inputs, seq_len, teacher_forcing_ratio, labels=None, lab_len=None, 
    labels10=None, max_label_len=None, mode_type="train", beam_width=1):
        ###########################################
        ##### Encoder #############################
        ###########################################
        #### CNN
        #Forward pass, backward pass, optimize
        in_lstm, seq_len_cnn = self.cnn_encoder(inputs, seq_len)
        in_lstm = torch.transpose(in_lstm, 1, 2) # input for LSTM batch_size x seq_length x input_size

        #### LSTM
        if (seq_len_cnn <= 0.).sum().item() > 0: # remove neagtive samples            
            negative_idx = seq_len_cnn > 0
            seq_len_cnn = seq_len_cnn[negative_idx]
            in_lstm = in_lstm[negative_idx, : , :] # [batch, seq_len, input_dim]
             
            lab_len = lab_len[negative_idx]
            labels = labels[negative_idx, :]
            labels10 = labels10[negative_idx, :, :]
        encoder_hidden, ecoder_output = self.lstm_encoder(in_lstm, seq_len_cnn)

        ###########################################
        ##### Sorting #############################
        ###########################################
        #### sort by decreasing target length
        sorted_len_target, sorted_idx_target = lab_len.sort(0, descending=True)
        sorted_hiddens1 = encoder_hidden[0][:, sorted_idx_target, :]
        sorted_hiddens2 = encoder_hidden[1][:, sorted_idx_target, :]
        sorted_hiddens = (sorted_hiddens1, sorted_hiddens2)
        sorted_encoder_output = ecoder_output[sorted_idx_target, :, :]
        # sort labels so that they match with order in batch
        labels_sorted = labels[sorted_idx_target, :]#.gather(0, sorted_idx_lab.long())
        labels10_sorted = labels10[sorted_idx_target, :, :] # batch, out_size, seq_len

        ###########################################
        ##### Decoder #############################
        ###########################################
        #### LSTM
        random_value = random.random()
        out_decoder, decoder_hidden = self.lstm_decoder(
        sorted_encoder_output, seq_len_cnn, max_label_len, labels10_sorted, 
        teacher_forcing_ratio, sorted_hiddens, 
        sorted_len_target, labels_sorted, random_value, mode_type, beam_width) # seq_length x batch_size x out_size   
        return out_decoder, labels_sorted, sorted_len_target


# In[ ]:


def get_train_loader_trainVal(tensor_x, sig_len, tensor_y, label_len, label10, batch_size, shuffle=True):
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y, sig_len, label_len, label10) # create your datset
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, 
                                               num_workers=0, pin_memory=False, shuffle=shuffle) # create your dataloader
    return(train_loader)
def get_train_loader(tensor_x, sig_len, tensor_y, label_len, label10, read_idx, batch_size, shuffle=False):
    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y, sig_len, label_len, label10, read_idx) # create your datset
    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size, 
                                               num_workers=0, pin_memory=False, shuffle=shuffle) # create your dataloader
    return(train_loader)
def convert_to_string(pred, target, target_lengths):
    import editdistance
    vocab = {0: "A", 1: "C", 2: "G", 3: "T", 4: "<EOS>", 5: "<PAD>"}
    editd = 0
    num_chars = 0
    for idx, length in enumerate(target_lengths):
        length = int(length.item())
        seq = pred[idx]
        seq_target = target[idx]
        encoded_pred = []
        for p in seq:
            if p == 4:
                break
            encoded_pred.append(vocab[int(p.item())])
        encoded_pred = ''.join(encoded_pred)
        encoded_target = ''.join([vocab[int(x.item())] for x in seq_target[0:length]])
        result = editdistance.eval(encoded_pred, encoded_target)
        editd += result
        num_chars += len(encoded_target)
    return editd, num_chars
def trainNet(model, train_ds, optimizer, criterion, clipping_value=None, val_ds=None, 
             test_ds=None, batch_size=256, n_epochs=500, teacher_forcing_ratio=0.5, reduced_TF=True,
             make_validation=1000, mode="train", shuffle=True, patience = 25, 
             file_name="model", earlyStopping=False, writer="", editD=True, reduce_lr=False):

    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("gradient clipping=", clipping_value)
    print("teacher forcing ratio=", teacher_forcing_ratio)
    print("shuffle=", shuffle)
    
    if val_ds is not None:
        input_x_val = val_ds[0]
        input_y_val = val_ds[1]
        input_y10_val = val_ds[2]
        signal_len_val = val_ds[3]
        label_len_val = val_ds[4]
        read_val = val_ds[5]

    input_x = train_ds[0]
    input_y = train_ds[1]
    input_y10 = train_ds[2]
    signal_len = train_ds[3]
    label_len = train_ds[4]
    read_train = train_ds[5]

    #Get training data
    train_loader = get_train_loader_trainVal(input_x, signal_len, 
                                    input_y, label_len, 
                                    input_y10, batch_size=batch_size, shuffle=True)
    if val_ds != None:
        val_loader = get_train_loader_trainVal(input_x_val, signal_len_val, 
                                  input_y_val, label_len_val, 
                                  input_y10_val, batch_size=batch_size, shuffle=True)
        if earlyStopping:
            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.01, name=file_name, relative=True, decrease_lr_scheduler=reduce_lr)
    
    dict_activations_in = {}
    dict_activations_forget = {}
    dict_activations_cell = {}
    dict_activations_out = {}
    dict_activations_in_decoder = {}
    dict_activations_forget_decoder = {}
    dict_activations_cell_decoder = {}
    dict_activations_out_decoder = {}
    
    dict_training_loss = {}
    dict_validation_loss = {}
    dict_training_acc = {}
    dict_validation_acc = {}
    dict_training_editd = {}
    dict_validation_editd = {}

    dict_training_loss2 = {}
    dict_validation_loss2 = {}
    dict_training_acc2 = {}
    dict_validation_acc2 = {}
    dict_training_editd2 = {}
    dict_validation_editd2 = {}

    dict_weights = {}
    dict_gradients = {}
    
    running_loss_train = 0.0
    running_loss_val = 0.0
    
    running_acc_train = 0.0
    running_acc_val = 0.0
    running_editd_train = 0.0
    running_editd_val = 0.0

    updates = 0

    heatmap_g = None
    heatmap_w = None
    heatmap_g_b = None
    heatmap_w_b = None
    counter_updates_teacherForcing = 0
    old_ed = 0
    if reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #Loop for n_epochs
    for epoch in range(n_epochs):
        if earlyStopping and early_stopping.early_stop: # break epoch loop
            print("Early stopping")
            break
            
        model.train()

        epoch_loss = 0
        epoch_acc = 0
        epoch_loss_val = 0
        epoch_acc_val = 0
        epoch_editd_val = 0
        epoch_editd = 0

        print("=" * 30)
        print("epoch {}/{}".format(epoch+1, n_epochs))
        print("=" * 30)

        total_train_loss = 0
        loss_iteration = []
        acc_iteration = []
        editd_iteration = []
               
        for iteration, data in enumerate(train_loader):
            model.train()
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            batch_x = data[0]
            batch_y = data[1]
            seq_len = data[2]
            lab_len = data[3]
            batch_y10 = data[4]
                
            #Wrap them in a Variable object
            inputs, labels, labels10 = Variable(batch_x, requires_grad=False), Variable(batch_y, requires_grad=False), Variable(batch_y10, requires_grad=False) # batch_size x out_size x seq_length 

            output, sorted_labels, sorted_labels_len = model(inputs, seq_len, teacher_forcing_ratio, labels,
                                          lab_len, labels10, labels.size(1), mode)
            
            output = torch.transpose(output, 0, 1).contiguous() # input for LSTM seq_length x out_size
            # Calculate cross entropy loss
            # output = (seq*batch, out dim), target = (seq*batch)
            # Target nicht one-hot encoden 
            reshaped_output = output.view(-1, output.size(2))
            reshaped_sorted_labels = sorted_labels.view(-1)
            notpadded_index = reshaped_sorted_labels != 5 # indices of not padded elements
            loss = criterion(reshaped_output, reshaped_sorted_labels.long()) 
            loss_iteration.append(loss.item())
            epoch_loss += loss.item()
            running_loss_train += loss.item()

            acc = (reshaped_output[notpadded_index, :].argmax(1) == 
                   reshaped_sorted_labels[notpadded_index]
                  ).sum().item() / reshaped_sorted_labels[notpadded_index].size(0)
            epoch_acc += acc
            running_acc_train += acc
            acc_iteration.append(acc)

            #if editD:
            #    if updates % make_validation == 0:
            #        ed = np.mean(np.array(convert_to_string(output.argmax(2), sorted_labels, sorted_labels_len)))
            #        ed2 = ed
            #    else:
            #        ed = 0
            #        ed2 = old_ed
            #
            #    old_ed = ed2
            #    epoch_editd += ed
            #    running_editd_train += ed
            #    editd_iteration.append(ed2)
            #    print("edit distance= {0:.4f}".format((epoch_editd / float(iteration + 1))))

            if updates % make_validation == 0:
                print("=" * 30)
                print("batch {} in epoch {}/{}".format(iteration+1, epoch+1, n_epochs))
                print("=" * 30)
                print("loss= {0:.4f}".format(epoch_loss / float(iteration + 1)))
                print("acc= {0:.4f} %".format((epoch_acc / float(iteration + 1)) * 100))
                if reduce_lr:
                    print("lr= " +  str(optimizer.param_groups[0]['lr']))
                print("teacher forcing ratio= {}".format(teacher_forcing_ratio), ", update= ", updates, ", half of updates= ", int((len(train_loader) * n_epochs)*0.5))

            # Backward pass
            loss.backward()
                    
            #clipping_value = 1 #arbitrary number of your choosing
            if clipping_value != "None":
                nn.utils.clip_grad_norm_(model.parameters(), int(clipping_value))
            
            # Update encoder and decoder
            optimizer.step()

            
            if (val_ds != None) and (updates % make_validation == 0): # or (updates == n_epochs-1)):
                
                if reduced_TF:
                #if updates > int((len(train_loader) * n_epochs)*0.5) and teacher_forcing_ratio >= 0.5: 
                    if (running_acc_train / float(make_validation)) > 0.35 and teacher_forcing_ratio >= 0.25: 
                        # if we have reached half of the updates
                        teacher_forcing_ratio = teacher_forcing_ratio * 0.95
                else:
                    teacher_forcing_ratio

            # Evaluation on the validation set
                val_losses = []
                val_acc = []
                val_editd = []

                model.eval()
                total_ed = 0
                total_num_chars = 0

                with torch.no_grad():
                    for iteration_val, data_val in enumerate(val_loader):
                        batch_x_val = data_val[0]
                        batch_y_val = data_val[1]
                        seq_len_val = data_val[2]
                        lab_len_val = data_val[3]
                        batch_y10_val = data_val[4]

                        inputs_val, labels_val, labels10_val = Variable(batch_x_val, requires_grad=False), Variable(batch_y_val, requires_grad=False), Variable(batch_y10_val, requires_grad=False) 
                        # batch_size x out_size x seq_length                     
                        
                        output_val, sorted_labels_val, sorted_labels_len_val = model(inputs_val, seq_len_val, 
                                                              0, labels_val, lab_len_val, 
                                                              labels10_val, labels_val.size(1), mode)

                        output_val = torch.transpose(output_val, 0, 1).contiguous() # input for LSTM seq_length x out_size
                        # Calculate cross entropy loss
                        # output = (seq*batch, out dim), target = (seq*batch)
                        # Target nicht one-hot encoden 
                        reshaped_output_val = output_val.view(-1, output_val.size(2))
                        reshaped_sorted_labels_val = sorted_labels_val.view(-1)
                        notpadded_index_val = reshaped_sorted_labels_val != 5 # indices of not padded elements
                        loss_val = criterion(reshaped_output_val, reshaped_sorted_labels_val.long()) 
                        val_losses.append(loss_val.item())
                        epoch_loss_val += loss_val.item()
                        running_loss_val += loss_val.item()
                        acc_val = (reshaped_output_val[notpadded_index_val, :].argmax(1) == 
                                   reshaped_sorted_labels_val[notpadded_index_val]
                                  ).sum().item() / reshaped_sorted_labels_val[notpadded_index_val].size(0)
                        epoch_acc_val += acc_val
                        running_acc_val += acc_val
                        val_acc.append(acc_val)
                        if editD:
                            ed_val, num_char_ref = convert_to_string(output_val.argmax(2), sorted_labels_val, sorted_labels_len_val)
                            epoch_editd_val += ed_val
                            running_editd_val += ed_val
                            val_editd.append(ed_val)
                            total_ed += ed_val
                            total_num_chars += num_char_ref

                    if editD:
                        cer = float(total_ed) / total_num_chars
                    if updates == 0:
                        writer.add_scalar('Loss/train', np.mean(loss_iteration), updates)
                        writer.add_scalar('Loss/validation', np.mean(val_losses), updates)
                        writer.add_scalar('Accuracy/train', np.mean(acc_iteration), updates)
                        writer.add_scalar('Accuracy/validation', np.mean(val_acc), updates)
                        if editD:
                            #writer.add_scalar('Edit Distance/train', running_editd_train, updates)
                            writer.add_scalar('Edit Distance/validation', cer, updates)
                            #dict_training_editd2[updates] = running_editd_train
                            dict_validation_editd2[updates] = cer

                        dict_training_loss2[updates] = np.mean(loss_iteration)
                        dict_training_acc2[updates] = np.mean(val_losses)
                        dict_validation_loss2[updates] = np.mean(acc_iteration)
                        dict_validation_acc2[updates] = np.mean(val_acc)
                        
                    else:
                        writer.add_scalar('Loss/train', np.mean(loss_iteration), updates)
                        writer.add_scalar('Loss/validation', np.mean(val_losses), updates)
                        writer.add_scalar('Accuracy/train', np.mean(acc_iteration), updates)
                        writer.add_scalar('Accuracy/validation', np.mean(val_acc), updates)
                        if editD:
                            #writer.add_scalar('Edit Distance/train', running_editd_train, updates)
                            writer.add_scalar('Edit Distance/validation', cer, updates)
                            #dict_training_editd2[updates] = running_editd_train #/ float(make_validation)
                            dict_validation_editd2[updates] = cer

                        dict_training_loss2[updates] = np.mean(loss_iteration)
                        dict_training_acc2[updates] = np.mean(val_losses)
                        dict_validation_loss2[updates] = np.mean(acc_iteration)
                        dict_validation_acc2[updates] = np.mean(val_acc)
                  
                    valid_loss = running_loss_val / float(iteration_val + 1)
                    running_loss_train = 0.0
                    running_loss_val = 0.0
                    running_acc_train = 0.0
                    running_acc_val = 0.0
                    running_editd_train = 0.0
                    running_editd_val = 0.0

                    
                    print("=" * 100)
                    print("Epoch: {}/{}...".format(epoch+1, n_epochs),
                      "Loss: {:.6f}...".format(epoch_loss / float(iteration + 1)),
                      "Accuarcy: {:.6f}...".format((epoch_acc / float(iteration + 1)) * 100),
                      "Val Loss: {:.6f}...".format(epoch_loss_val / float(iteration_val + 1)),
                      "Val Accuracy: {:.6f}%...".format((epoch_acc_val / float(iteration_val + 1)) * 100))
                    print("=" * 100)
                    dict_validation_loss[epoch] = val_losses
                    dict_validation_acc[epoch] = val_acc
                    if editD:
                        dict_validation_editd[epoch] = val_editd
                    # early_stopping needs the validation loss to check if it has decresed, 
                    # and if it has, it will make a checkpoint of the current model
                    if earlyStopping:
                        early_stopping(np.mean(val_losses), model, optimizer, updates)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                    if reduce_lr:
                        scheduler.step(loss_val)

            updates +=1

        dict_training_loss[epoch] = loss_iteration
        dict_training_acc[epoch] = acc_iteration
        if editD:
            dict_training_editd[epoch] = editd_iteration
    writer.close()
    if earlyStopping:
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint["model"])

    return ([[dict_training_loss, dict_validation_loss, dict_training_acc, dict_validation_acc, dict_training_editd, dict_validation_editd],
    [dict_training_loss2, dict_validation_loss2, dict_training_acc2, dict_validation_acc2, dict_training_editd2, dict_validation_editd2], 
    [dict_activations_in, dict_activations_forget, dict_activations_cell, dict_activations_out],
    [dict_activations_in_decoder, dict_activations_forget_decoder, dict_activations_cell_decoder, dict_activations_out_decoder],
    [dict_weights, dict_gradients]])
    
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights_orthogonal_lstm(m):
    for name, param in m.named_parameters():
        if "weight" in name and len(list(param.data.size())) > 1:
            nn.init.orthogonal_(param.data)

def plot_error_accuarcy(input, pdf=None, steps=50, validation=True, editD=True):
    sns.set(font_scale=1)
    loss_train, loss_val, acc_train, acc_val, editd_train, editd_val = input[0], input[1], input[2], input[3], input[4], input[5]
    fig = plt.figure(figsize=(18,2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(list(loss_train.keys()), list(loss_train.values()), label="training error")
    if validation:
        ax.plot(list(loss_val.keys()), list(loss_val.values()), label="validation error")

    plt.xlabel("Updates")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error vs. updates")
    
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(list(acc_train.keys()), [v*100 for v in list(acc_train.values())], label="training accuracy")
    if validation:
        ax.plot(list(acc_val.keys()), [v*100 for v in list(acc_val.values())], label="validation accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Accuracy in %")
    plt.legend()
    plt.title("Accuracy vs. updates")

    if editD:
        ax = fig.add_subplot(1, 3, 3)
        ax.plot(list(editd_train.keys()), list(editd_train.values()), label="training edit distance")
        if validation:
            ax.plot(list(editd_val.keys()), list(editd_val.values()), label="validation edit distance")
        plt.xlabel("Updates")
        plt.ylabel("Normalized Edit Distance")
        plt.legend()
        plt.title("Edit Distance vs. updates")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

def plot_error_accuarcy_iterations_train(input, pdf=None, editD=True):
    sns.set(font_scale=1)
    loss_train, acc_train, editd_train = input[0], input[2], input[4]
    fig = plt.figure(figsize=(18,2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(np.arange(0, len(np.concatenate(np.array(list(loss_train.values()))))), 
            np.concatenate(np.array(list(loss_train.values()))), label="training error")
    plt.xlabel("Updates")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error vs. updates from trainings set")
    
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(np.arange(0, len(np.concatenate(np.array(list(acc_train.values()))))), 
            [v*100 for v in np.concatenate(np.array(list(acc_train.values())))], label="training accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Accuracy in %")
    plt.legend()
    plt.title("Accuracy vs. updates from trainings set")

    if editD:
        ax = fig.add_subplot(1, 3, 3)
        ax.plot(np.arange(0, len(np.concatenate(np.array(list(editd_train.values()))))), 
            np.concatenate(np.array(list(editd_train.values()))), label="training edit distance")
        plt.xlabel("Updates")
        plt.ylabel("Normalized Edit Distance")
        plt.legend()
        plt.title("Edit Distance vs. updates from trainings set")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")
def plot_error_accuarcy_iterations_val(input, pdf=None, editD=True):
    sns.set(font_scale=1)
    loss_val, acc_val, editd_val =  input[1], input[3], input[5]
    fig = plt.figure(figsize=(18,2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(np.arange(0, len(np.concatenate(np.array(list(loss_val.values()))))), 
            np.concatenate(np.array(list(loss_val.values()))), label="validation error")
    plt.xlabel("Updates")
    plt.ylabel("Error")
    plt.legend()
    plt.title("Error vs. updates from validation set")
    
    ax = fig.add_subplot(1, 3, 2) 
    ax.plot(np.arange(0, len(np.concatenate(np.array(list(acc_val.values()))))), 
            [v*100 for v in np.concatenate(np.array(list(acc_val.values())))], label="validation accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Accuracy in %")
    plt.legend()
    plt.title("Accuracy vs. updates from validation set")

    if editD:
        ax = fig.add_subplot(1, 3, 3)
        ax.plot(np.arange(0, len(np.concatenate(np.array(list(editd_val.values()))))), 
            np.concatenate(np.array(list(editd_val.values()))), label="validation edit distance")
        plt.xlabel("Updates")
        plt.ylabel("Normalized Edit Distance")
        plt.legend()
        plt.title("Edit distance vs. updates from validation set")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")
def plot_activations(input, pdf=None, print_epoch=50000, title=""):
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(13,4))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.5)
    i = 0
    for p, label in zip(input, ["input gate", "forget gate", "cell activation", "out gate"]):
        i += 1
        ax = fig.add_subplot(2, 2, i)
        for epoch in p.keys():
            if epoch % print_epoch == 0 or epoch == max(p.keys()):
                x = np.arange(0, len(p[epoch].detach().numpy()))
                # this locator puts ticks at regular intervals
                if epoch == 0:
                    ax.plot(np.arange(0, len(p[epoch].detach().numpy())), 
                    p[epoch].detach().numpy(), label="update {}".format(epoch), color="#000000", alpha=0.8)
                else:
                    ax.plot(np.arange(0, len(p[epoch].detach().numpy())), 
                    p[epoch].detach().numpy(), label="update {}".format(epoch))
            plt.xlabel("Time Steps")
            plt.ylabel("Activation")
        if i == 1:
            plt.legend(bbox_to_anchor=(1.05, 1.05))
        plt.title("{} {}".format(label, title))
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

def plot_heatmap(dict_weights,save_files_path, filename, split_LSTMbiases = False):
    x_len = list(dict_weights.keys())
    y_len = list(dict_weights[x_len[0]].keys())
    input = np.zeros((len(y_len), len(x_len)))
    input_grad = np.zeros((len(y_len), len(x_len)))
    if split_LSTMbiases:
        y_len_biases = []
        for name in y_len:
            if "bias" in name and "lstm" in name and "linear" not in name:
                for b in ["input", "forget", "cell", "output"]:
                    y_len_biases.append(name +  "." + b)
        input_biases = np.zeros((len(y_len_biases), len(x_len)))
        input_grad_biases = np.zeros((len(y_len_biases), len(x_len)))

    for idx, u in enumerate(x_len):
        idx_b = 0
        matrix_param = dict_weights[u]
        for idx_p, p in enumerate(y_len):
            if len(matrix_param[p].shape) > 2: # and matrix_param[p].shape[1] == 1: # (256, 1, 11)
                m = matrix_param[p].reshape((matrix_param[p].shape[0], -1))
            else:
                m = matrix_param[p]
            input[idx_p, idx] = np.linalg.norm(m, ord=2)
            if split_LSTMbiases and "bias" in p and "lstm" in p and "linear" not in p:
                n = matrix_param[p].shape[0]
                # input gate
                start, end = 0, n//4
                input_biases[idx_b, idx] = np.linalg.norm(m[start: end], ord=2)
                # forget gate
                start, end = n//4, n//2
                input_biases[idx_b+1, idx] = np.linalg.norm(m[start: end], ord=2)
                # cell gate
                start, end = n//2, n//2 + n//4
                input_biases[idx_b+2, idx] = np.linalg.norm(m[start: end], ord=2)
                # output gate
                start, end = n//2 + n//4, n
                input_biases[idx_b+3, idx] = np.linalg.norm(m[start: end], ord=2)
                idx_b += 4
    
    y_len = ["\n".join([".".join([x.split(".")[0], x.split(".")[1]]), x.split(".")[2]]) for x in y_len]             
    df = pd.DataFrame(input, index=y_len, columns=x_len)
    print(df.head())
    sns.set(font_scale=0.4)
    svm = sns.heatmap(df, linewidths=0.0, edgecolor="none")
    figure = svm.get_figure()
    figure.savefig(save_files_path + "/heatmap_{}.pdf".format(filename))
    plt.clf()

    if split_LSTMbiases:
        y_len_biases = ["\n".join([".".join([x.split(".")[0], x.split(".")[3]]), x.split(".")[2]]) for x in y_len_biases]       
        df_b = pd.DataFrame(input_biases, index=y_len_biases, columns=x_len)
        print(df_b.head())
        sns.set(font_scale=0.4)
        svm = sns.heatmap(df_b, linewidths=0.0, edgecolor="none")
        figure2 = svm.get_figure()   
        figure2.savefig(save_files_path + "/heatmap_{}_biases.pdf".format(filename))
        plt.clf()

def bestPerformance2File(input, fname, editD=True):
    loss_train, loss_val, acc_train, acc_val, editd_train, editd_val = input[0], input[1], input[2], input[3], input[4], input[5]
    max_idx_train = max(acc_train, key=lambda k: acc_train[k])
    max_idx_val = max(acc_val, key=lambda k: acc_val[k])
    f = open(fname, "w")

    if editD:
        f.write("best performances on trainings set\n")
        f.write("trainings acc\tvalidation acc\ttrainings loss\tvalidation loss\ttrainings edit distance\tvalidation edit distance\tupdate\n") 
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(acc_train[max_idx_train], acc_val[max_idx_train], loss_train[max_idx_train], loss_val[max_idx_train], editd_train[max_idx_train], editd_val[max_idx_train], max_idx_train))
        f.write("\nbest performances on validation set\n")
        f.write("trainings acc\tvalidation acc\ttrainings loss\tvalidation loss\ttrainings edit distance\tvalidation edit distance\tupdate\n") 
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(acc_train[max_idx_val], acc_val[max_idx_val], loss_train[max_idx_val], loss_val[max_idx_val], editd_train[max_idx_val], editd_val[max_idx_val], max_idx_val))
    else:
        f.write("best performances on trainings set\n")
        f.write("trainings acc\tvalidation acc\ttrainings loss\tvalidation loss\tupdate\n") 
        f.write("{}\t{}\t{}\t{}\t{}\n".format(acc_train[max_idx_train], acc_val[max_idx_train], loss_train[max_idx_train], loss_val[max_idx_train], max_idx_train))
        f.write("\nbest performances on validation set\n")
        f.write("trainings acc\tvalidation acc\ttrainings loss\tvalidation loss\tupdate\n") 
        f.write("{}\t{}\t{}\t{}\t{}\n".format(acc_train[max_idx_val], acc_val[max_idx_val], loss_train[max_idx_val], loss_val[max_idx_val], max_idx_val))

    f.close()

def basecalling(argv):
    parser = make_argparser()
    args = parser.parse_args(argv[1:])
    infile = args.input
    fname = args.output
    port = args.gpu_port
    SEED = args.set_seed
    batch = args.batch_size
    epochs = args.epochs
    make_validation = args.make_validation
    teacher = args.tf_ratio
    reduced_TF = args.reduced_tf
    earlyStopping = args.early_stop
    patience_earlyStop = args.patience
    weight_decay = args.weight_decay #0.01 #0.01
    clipping_value = args.gradient_clip

    # LSTM
    hidden = args.hidden_units #256
    forget_bias = args.forget_bias_encoder
    forget_bias_decoder = args.forget_bias_decoder
    num_layers = args.lstm_layers
    bidir = args.bi_lstm
    attention = args.attention
    dropout = args.dropout
    # CNN
    input_bias_cnn = args.input_bias_cnn
    strides = args.strides
    kernel = args.kernel
    cnn_out = args.channel_number #256
    pooling_type = args.pooling_type #"average"
    n_layers_cnn = args.cnn_layers
    batch_normalization = args.batch_norm
    dropout_on = args.dropout_cnn
    dropout_input = args.dropout_input
    dropout_probability = args.drop_prob
    call = args.call
    lr = args.learning_rate
    editD = args.editD

    sgd = False
    out_classes = 5
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Load data 
    dict_classes = {0: "A", 1: "C", 2: "G", 3: "T", 4: "<EOF>", 5: "<PAD>"} # A=0, C=1, G=2, T=3, EOF=4
    script_dir = os.path.dirname(os.path.realpath('__file__')) # script directory
    file_out = script_dir + "/" + infile
    print(file_out)
    with open(file_out, 'rb') as handle:
        read_data = pickle.load(handle)
    save_files_path = script_dir + '/training_result_{}/'.format(fname)
    writer = SummaryWriter(script_dir + '/training_result_{}'.format(fname))
    train_set = read_data[0]
    val_set = read_data[1]
    
    train_set = [train_set[0].cuda(port), train_set[1].cuda(port), train_set[2].cuda(port), 
                train_set[3].cuda(port), train_set[4].cuda(port), train_set[5].cuda(port)]
    val_set = [val_set[0].cuda(port), val_set[1].cuda(port), val_set[2].cuda(port), 
                val_set[3].cuda(port), val_set[4].cuda(port), val_set[5].cuda(port)]
    print("train: ", train_set[0].size(), train_set[1].size(), train_set[2].size(), 
          train_set[3].size(), train_set[4].size(), train_set[5].size())
    print("validation: ", val_set[0].size(), val_set[1].size(), val_set[2].size(), 
          val_set[3].size(), val_set[4].size(), val_set[5].size())
                # [batch size] is typically chosen between 1 and a few hundreds, e.g. [batch size] = 32 is a good default value
    CNN = SimpleCNN(input_channel=1, output_channel=[cnn_out,cnn_out, cnn_out] , kernel_size=kernel, 
                    stride=strides, padding=[0,0,0], pooling=pooling_type, layers=n_layers_cnn,
                    batch_norm=batch_normalization, 
                    dropout = dropout_on, dropout_p = dropout_probability, dropout_input = dropout_input, input_bias_cnn=input_bias_cnn)
    out_channels = CNN.output_channel[n_layers_cnn-1]
    lstm = LSTMencoder(input_dim=out_channels, hidden_dim = hidden, 
                 batch_size=batch, output_dim=hidden, 
                 num_layers=num_layers, own_cell_encoder=False, bidirectional = bidir, port=port, dropout=dropout)
    lstm_dec = LSTMdecoder(input_dim=out_classes, hidden_dim = hidden, 
                 batch_size=batch, output_dim=out_classes, 
                 num_layers=num_layers, own_cell_decoder=False, bidirectional = bidir, port=port, attention=attention, dropout=dropout)
    
    model12 = Seq2Seq(cnn_encoder = CNN, lstm_encoder = lstm, lstm_decoder = lstm_dec)#.cuda(port)
    model12.apply(init_weights_orthogonal_lstm)
    
    for name, param in model12.named_parameters():
        if "bias" in name:
            if forget_bias != "None" and "lstm_encoder" in name:
                print(name,param.data.size())
                n = param.size(0)
                # forget gate
                start, end = n//4, n//2 # ordering ingate, forgetgate, cellgate, outgate
                param.data[start:end].fill_(float(int(forget_bias)))
                print(start, end)
                # ingate
                start, end = 0, n//4 # ordering ingate, forgetgate, cellgate, outgate
                param.data[start:end].fill_(0.)
                print(start, end)
                # cellgate, outgate
                start, end = n//2, n # ordering ingate, forgetgate, cellgate, outgate
                param.data[start:end].fill_(0.)
                print(start, end)
            if forget_bias_decoder != "None" and "lstm_decoder" in name and "linear" not in name:
                print(name,param.data.size())
                n = param.size(0)
                # forget gate
                start, end = n//4, n//2 # ordering ingate, forgetgate, cellgate, outgate
                param.data[start:end].fill_(float(int(forget_bias_decoder)))
                print(start, end)
                # ingate
                start, end = 0, n//4 # ordering ingate, forgetgate, cellgate, outgate
                param.data[start:end].fill_(0.)
                print(start, end)
                # cellgate, outgate
                start, end = n//2, n # ordering ingate, forgetgate, cellgate, outgate
                param.data[start:end].fill_(0.)
                print(start, end)
    model12 = model12.cuda(port)
    print(model12, next(model12.parameters()).is_cuda)
    if sgd:
        optimizer = optim.SGD(model12.parameters(), lr=lr, momentum=0)
    else:
        optimizer = optim.Adam(model12.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.999))
    criterion = torch.nn.NLLLoss(ignore_index=5)#.cuda(port)
    print(f'The model has {count_parameters(model12):,} trainable parameters')
    f = open(save_files_path + "{}_sig_length.txt".format(fname), "w")
    f.write(infile + " \n")
    f.write("Training: \nSignal:\n")
    f.write("{}\ttrue signal length: {}\n".format(train_set[0].size(), train_set[3]))
    f.write("\nTarget:\n")
    f.write("{}\ttrue target length: {}\n".format(train_set[1].size(), train_set[4]))
    f.write("\nRead idx:\n")
    f.write("{}\n\n".format(train_set[5]))
    
    f.write("Validation: \nSignal:\n")
    f.write("{}\ttrue signal length: {}\n".format(val_set[0].size(), val_set[3]))
    f.write("\nTarget:\n")
    f.write("{}\ttrue target length: {}\n".format(val_set[1].size(), val_set[4]))
    f.write("\nRead idx:\n")
    f.write("{}\n\n".format(val_set[5]))
    f.write("Model:\n")
    f.write(str(model12))
    f.write("\nThe model has {:,} trainable parameters\n".format(count_parameters(model12)))
    f.write("hyperparameters:\n")
    f.write("epochs={}, batch size={}, earlyStopping={}, patience={}, weight decay={}, clipping value={}, lr={}\n"
        .format(epochs, batch, earlyStopping, patience_earlyStop, weight_decay, clipping_value, lr))
    f.write("TF={}, reduced TF ratio={}\n".format(teacher, reduced_TF))
    f.write("forget gate bias encoder={}, forget gate bias decoder={}"
        .format(forget_bias, forget_bias_decoder))
    f.close()
    
    # with 10 reads, kernel size = 11
    start = time.time()
    out12 = trainNet(
        model12, train_ds = train_set, optimizer=optimizer, 
        criterion=criterion, clipping_value=clipping_value, 
        val_ds = val_set, 
        batch_size=batch, n_epochs=epochs, teacher_forcing_ratio=teacher, 
        make_validation=make_validation, file_name=save_files_path + "{}_checkpoint.pt".format(fname), 
        earlyStopping=earlyStopping, patience=patience_earlyStop, writer=writer, editD=editD)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("=" * 100)
    checkpoint = { 
    'updates': out12[-1],
    'model': model12.state_dict(),
    'optimizer': optimizer._optimizer.state_dict()}
    
    torch.save(checkpoint, save_files_path + '{}.pt'.format(fname))
    #np.savez_compressed(save_files_path + '{}_weightsGradients.npz'.format(fname), weights=out12[4][0], gradients=out12[4][1])
    pickle.dump(out12, open(save_files_path + "{}.p".format(fname), "wb" ))
    with PdfPages(save_files_path + "{}.pdf".format(fname)) as pdf:
        plot_error_accuarcy(out12[1], pdf, editD=editD)
        bestPerformance2File(out12[1], save_files_path + "best_performances_{}.txt".format(fname), editD=editD)
        plot_error_accuarcy_iterations_train(out12[0], pdf, editD=editD)
        plot_error_accuarcy_iterations_val(out12[0], pdf, editD=editD)
        print("Training took: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
if __name__ == '__main__':
    sys.exit(basecalling(sys.argv))


