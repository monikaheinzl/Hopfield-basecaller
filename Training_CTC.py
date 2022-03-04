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

sys.path.insert(0, '/hopfield-layers/')
from modules.transformer import HopfieldEncoderLayer, HopfieldDecoderLayer
from modules import Hopfield, HopfieldPooling

sys.path.insert(0, '/basecaller-modules')
from read_config import read_configuration_file
from cnn import SimpleCNN, BasicBlock, SimpleCNN_res
from cnn import outputLen_Conv, outputLen_AvgPool, outputLen_MaxPool
from hopfield_encoder_nosqrt import Embedder, PositionalEncoding, Encoder

from hopfield_decoder import Decoder
from early_stopping import EarlyStopping
from lr_scheduler2 import NoamOpt
from plot_performance import plot_error_accuarcy, plot_error_accuarcy_iterations_train, plot_error_accuarcy_iterations_val, plot_activations, plot_heatmap, bestPerformance2File
from plot_softmax2 import calculate_k_patterns, plot_softmax

from ctcdecode import CTCBeamDecoder

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

    parser.add_argument('-g', '--gpu_port', default="None",
                        help="Port on GPU mode")
    parser.add_argument('-s', '--set_seed', type=int, default=1234,
                        help="Set seed")

    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help="Batch size")
    parser.add_argument('-e', '--epochs', type=int, default=500,
                        help="Number of epochs")
    parser.add_argument('-v', '--make_validation', type=int, default=1000,
                        help="Make every n updates evaluation on the validation set")

    parser.add_argument('-max_w', '--max_window_size', type=int, default=1000,
                        help="Maximum window size")
    parser.add_argument('-max_t', '--max_target_size', type=int, default=200,
                        help="Maximum target size")

    # CNN arguments
    parser.add_argument("--input_bias_cnn", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)
    parser.add_argument('-c', '--channel_number', nargs='+', type=int, default=[256, 256, 256],
                        help="Number of output channels in Encoder-CNN")
    parser.add_argument('-l', '--cnn_layers', type=int, default=2,
                        help="Number of layers in Encoder-CNN")
    parser.add_argument('--pooling_type', default="None",
                        help="Pooling type in Encoder-CNN")
    parser.add_argument('--strides', nargs='+', type=int, default=[1, 2, 1],
                        help="Strides in Encoder-CNN")
    parser.add_argument('--kernel', nargs='+', type=int, default=[11, 11, 11],
                        help="Kernel sizes in Encoder-CNN")
    parser.add_argument('--padding', nargs='+', type=int, default=[0, 0, 0],
                        help="Padding in Encoder-CNN")
    parser.add_argument("--dropout_cnn", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--dropout_input", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--drop_prob', type=float, default=Range(0.0, 1.0),
                        help="Dropout probability Encoder-CNN")
    parser.add_argument("--batch_norm", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--src_emb', default="cnn",
                        help="Embedding type of input. Options: 'cnn', 'residual_blocks', 'hopfield_pooling'")
    parser.add_argument('--nhead_embedding', type=int, default=6,
                        help="number of heads in the multiheadattention models")
    #parser.add_argument("--res_layer", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    # LSTM arguments
    parser.add_argument("--input_bias_hopfield", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)
    parser.add_argument('-u', '--hidden_units', type=int, default=256,
                        help="Number of hidden units in the Transformer")
    parser.add_argument('--dff', type=int, default=1024,
                        help="Number of hidden units in the Feed-Forward Layer of the Transformer")
    parser.add_argument('--lstm_layers', type=int, default=5,
                        help="Number of layers in the Transformer")
    parser.add_argument('--nhead', type=int, default=6,
                        help="number of heads in the multiheadattention models")
    parser.add_argument('--drop_transf', type=float, default=Range(0.0, 1.0),
                        help="Dropout probability Transformer")
    parser.add_argument('--dropout_pos', type=float, default=Range(0.0, 1.0),
                        help="Positional Dropout probability")
    parser.add_argument('--scaling', default="None",
                        help="Gradient clipping")
    parser.add_argument("--pattern_projection_as_connected", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--normalize_stored_pattern", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--normalize_stored_pattern_affine", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--normalize_state_pattern", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--normalize_state_pattern_affine", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--normalize_pattern_projection", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--normalize_pattern_projection_affine", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    parser.add_argument("--stored_pattern_as_static", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--state_pattern_as_static", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--pattern_projection_as_static", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    parser.add_argument("--out_norm", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    parser.add_argument('--weight_decay', type=float, default=0,
                        help="Weight decay")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--decrease_lr", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--xavier_init", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help="Weight decay")

    parser.add_argument('--betas', nargs='+', default=["0.9", "0.98"], help="Betas in optimizer")
    parser.add_argument('--eps', default="1e-8",help="Eps in optimizer")
    parser.add_argument('--opt', default="adam", help="optimizer used")
    parser.add_argument('--gradient_clip', default="None", help="Gradient clipping")

    # early stopping
    parser.add_argument("--early_stop", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--patience', type=int, default=25,
                        help="Patience in early stopping")

    parser.add_argument("--editD", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--plot_weights", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument("--continue_training", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    parser.add_argument('--model_file', help="File path to model file.")
    parser.add_argument('--config_file', default="None", help="Path to config file")

    return parser

# Network
# -----------
# * CNN-Encoder
# * LSTM-Encoder
# * LSTM-Decoder

class Transformer(nn.Module):
    def __init__(self, cnn_encoder, ntoken, d_model, nhead, nhid, dff, nlayers, dropout=0.5, dropout_pos=0.5, 
                 max_len=250, max_len_trg=250, port=1, pattern_projection_as_connected=False, scaling=None, 
                 normalize_stored_pattern=False, normalize_stored_pattern_affine=False,
                 normalize_state_pattern=False, normalize_state_pattern_affine=False,
                 normalize_pattern_projection=False, normalize_pattern_projection_affine=False, input_bias_hopfield=True, out_norm=False):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.d_model = d_model
        self.out_norm = out_norm
        hopfield_self_src = Hopfield(input_size=d_model, hidden_size=nhid, num_heads=nhead, 
            batch_first=False, scaling=scaling, dropout=dropout,  pattern_projection_as_connected=pattern_projection_as_connected, 
            disable_out_projection=False,
            normalize_stored_pattern=normalize_stored_pattern,
            normalize_stored_pattern_affine= normalize_stored_pattern_affine,
            normalize_state_pattern=normalize_state_pattern,
            normalize_state_pattern_affine=normalize_state_pattern_affine,
            normalize_pattern_projection=normalize_pattern_projection,
            normalize_pattern_projection_affine=normalize_pattern_projection_affine,
            stored_pattern_as_static=False, state_pattern_as_static=False, 
            pattern_projection_as_static=False, input_bias=input_bias_hopfield)
        self.pos_enc_encoder = PositionalEncoding(d_model, dropout_pos, max_len) #, 115)
        self.encoder = Encoder(hopfield_self_src, d_model, nhead, nhid, dff, nlayers, dropout, port)
        if self.out_norm:
            self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(hopfield_self_src.output_size, ntoken) # 7

        self.port = port
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf')).to(self.port)
        mask = Variable(mask)
        return mask
    def forward(self, src, seq_len, update=None):
        src = src.detach()
        seq_len = seq_len.detach()
        src, seq_len_cnn = self.cnn_encoder(src, seq_len)
        src = src.transpose(1, 2).transpose(0, 1)
        src = self.pos_enc_encoder(src)
        output, src_mask = self.encoder(src, seq_len_cnn, src_mask=None) #, update=update)

        #if update == 0 or update == 55000 or update == 150000 or update == 250000:
        #    fig = self.encoder.tf_enc.softmax_heads #.cpu() #calculate_k_patterns(self.encoder.tf_enc.softmax_heads)
        #else:
        fig = None
            
        if self.out_norm:
            output = self.layer_norm(output)

        output = self.fc(output)
        output = F.log_softmax(output, dim=2)
        return output, seq_len_cnn, fig

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
def convert_to_string(pred, pred_len, target, target_lengths, vocab):
    import editdistance
    best_len = pred_len[:, 0]
    best_beam = pred[:, 0]
    editd = 0
    num_chars = 0
    for idx in range(len(best_len)):
        target_len = int(target_lengths[idx].item())
        prediction_len = int(best_len[idx].item())
        seq = best_beam[idx]
        seq_target = target[idx]
        encoded_pred = ''.join([vocab[int(x.item())] for x in seq[0:prediction_len]])
        encoded_target = ''.join([vocab[int(x.item())] for x in seq_target[0:target_len]])
        max_len = float(max([len(encoded_pred), len(encoded_target)]))
        #if max_len == 0:
        #    result = np.nan
        #elif editdistance.eval(encoded_pred, encoded_target) == 0:
        #    result = 0
        #else:
        result = editdistance.eval(encoded_pred, encoded_target) #/max_len
        #editd.append(result)
        editd += result
        num_chars += len(encoded_target)
    return editd, num_chars
def trainNet(model, train_ds, optimizer, criterion, clipping_value=None, val_ds=None, 
             test_ds=None, batch_size=256, n_epochs=500,
             make_validation=1000, mode="train", shuffle=True, patience = 25, 
             file_name="model", earlyStopping=False, writer="", 
             device=0, editD=True, decrease_lr=True, src_emb="cnn", plot_weights=True, last_checkpoint=None, file_path=None):

    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("gradient clipping=", clipping_value)
    print("shuffle=", shuffle)
    print("device=", device)

    
    #print("training data set=", train_ds[0].size(), train_ds[1].size())
    if val_ds is not None:
        input_x_val = val_ds[0]
        input_y_val = val_ds[1]
        input_y10_val = val_ds[2]
        signal_len_val = val_ds[3]
        label_len_val = val_ds[4]
        read_val = val_ds[5]

    input_x = train_ds[0]
    input_y = train_ds[1]
    print("classes", np.unique(input_y.cpu()))
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
            early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.01, name=file_name, relative=True, decrease_lr_scheduler=decrease_lr)
    
    dict_activations_in, dict_activations_forget, dict_activations_cell, dict_activations_out = {}, {}, {}, {}
    dict_activations_in_decoder, dict_activations_forget_decoder, dict_activations_cell_decoder, dict_activations_out_decoder = {}, {}, {}, {}    
    dict_training_loss, dict_validation_loss, dict_training_acc, dict_validation_acc, dict_training_editd, dict_validation_editd = {}, {}, {}, {}, {}, {}
    dict_training_loss2, dict_validation_loss2, dict_training_acc2, dict_validation_acc2, dict_training_editd2, dict_validation_editd2 = {}, {}, {}, {}, {}, {}
    dict_weights, dict_gradients = {}, {}
    
    running_loss_train, running_loss_val, running_acc_train, running_acc_val, running_editd_train, running_editd_val= 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if last_checkpoint is not None:
        updates = last_checkpoint #+ 1
    else:
        updates = 0

    updates_newTraining = 0

    heatmap_g = None
    heatmap_w = None
    heatmap_g_b = None
    heatmap_w_b = None
    counter_updates_teacherForcing = 0
    old_ed = 0
    #Loop for n_epochs
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.01, gamma=0.95)

    for epoch in range(n_epochs):
        if earlyStopping and early_stopping.early_stop: # break epoch loop
            print("Early stopping")
            break
            
        model.train()

        epoch_loss, epoch_acc, epoch_loss_val, epoch_acc_val, epoch_editd_val, epoch_editd = 0, 0, 0, 0, 0, 0
        
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
            #batch_read = data[5]
            #batch_y[batch_y == 4.] = 5
            #lab_len = lab_len - 1
                
            #Wrap them in a Variable object
            inputs, labels, labels10 = Variable(batch_x, requires_grad=False), Variable(batch_y, requires_grad=False), Variable(batch_y10, requires_grad=False) # batch_size x out_size x seq_length 
            
            # input:
            # batch x channel x length
            seq_len = inputs.squeeze(1).ne(-10).sum(1).float()
            lab_len = labels.ne(5).sum(1)

            if str(device) == "cpu":
                labels = labels.cpu()
                labels = labels.type(torch.LongTensor)
            else:
                labels = labels.type(torch.cuda.LongTensor)
            if updates == 0:
                print("labels", labels.size())


            output, output_len, figure_softmax = model(src=inputs, seq_len=seq_len, update=updates) #, src_mask=None, trg_mask=None)
            #output = output.contiguous()

            concat_label = torch.flatten(labels)
            concat_label = concat_label[concat_label.lt(5)]

            #loss = criterion(output, labels.long(), output_len.long(), lab_len.long()) #reshaped_output, reshaped_sorted_labels.long())    
            #loss = F.ctc_loss(output, labels.long(), output_len.long(), lab_len.long(), blank=4, reduction='sum')
            loss = F.ctc_loss(output, concat_label, output_len.long(), lab_len.long(), blank=0, reduction='sum', zero_infinity=True)

            # Backward pass
            loss.backward()
                    
            #clipping_value = 1 #arbitrary number of your choosing
            if clipping_value != "None":
                nn.utils.clip_grad_norm_(model.parameters(), float(clipping_value))
            
            # Update encoder and decoder
            optimizer.step()


            loss_iteration.append(loss.item() / inputs.size(0)) #.detach().cpu().item()) # detach.item
            epoch_loss += (loss.item() / inputs.size(0)) #.item()
            running_loss_train += (loss.item() / inputs.size(0)) #.item()

            #output = output.transpose(0, 1).cpu()#.contiguous()
            ## b x t x classess
            #ctc_decoder = CTCBeamDecoder("ACGT- ", model_path=None,
            #    alpha=0,
            #    beta=0,
            #    cutoff_top_n=6, #0
            #    cutoff_prob=1.0,
            #    beam_width=3,
            #    num_processes=1,
            #    blank_id=4,
            #    log_probs_input=True)
            #beam_results, beam_scores, timesteps, out_lens = ctc_decoder.decode(output, output_len)
            #best_beam = beam_results[:, 0]
            #if best_beam.size(1) < labels.size(1):
            #    pad = torch.ones(best_beam.size(0), abs(best_beam.size(1) - labels.size(1)))
            #    pad[:, :] = 5
            #    best_beam = torch.cat((best_beam, pad), 1)
#
            #reshaped_output = best_beam[:, :labels.size(1)].reshape(-1)
            #reshaped_sorted_labels = labels.view(-1).cpu()
            #notpadded_index = reshaped_sorted_labels != 5
            #acc = (reshaped_output[notpadded_index] == reshaped_sorted_labels[notpadded_index]).sum().item() / reshaped_sorted_labels[notpadded_index].size(0)
#
            #epoch_acc += acc
            #running_acc_train += acc
            #acc_iteration.append(acc) # acc
#
            #if editD:
            #    dict_classes = {0: "A", 1: "C", 2: "G", 3: "T", 4: "-", 5: " "}
            #    if updates % make_validation == 0:
            #        ed = np.mean(np.array(convert_to_string(beam_results, out_lens, labels, lab_len, dict_classes)))
            #        ed2 = ed
            #    else:
            #        ed = 0
            #        ed2 = old_ed
            #
            #    old_ed = ed2
            #    epoch_editd += ed
            #    running_editd_train += ed
            #    editd_iteration.append(ed2) #ed2
                
            if updates % make_validation == 0:
                print("=" * 30)
                print("batch {} in epoch {}/{}".format(iteration+1, epoch+1, n_epochs))
                print("=" * 30)
                print("loss= {0:.4f}".format(epoch_loss / float(iteration + 1)))
                #print("acc= {0:.4f} %".format((epoch_acc / float(iteration + 1)) * 100))
                print("update= ", updates, ", half of updates= ", int((len(train_loader) * n_epochs)*0.5))
                if decrease_lr:
                    print("lr= " +  str(optimizer._optimizer.param_groups[0]['lr']))
                else:
                    print("lr= " +  str(optimizer.param_groups[0]['lr']))
                #if editD and (updates % make_validation == 0):
                #    print("edit distance= {0:.4f}".format((epoch_editd / float(iteration + 1))))

                #data_heads = []
                #if figure_softmax is not None:
                #    data_heads.append(figure_softmax)
                #    del figure_softmax
                #    set_size = data_heads[0][0].size(2)
                #    indata, indata_pos = calculate_k_patterns(data_heads)
                #    plot = plot_softmax(indata, set_size=set_size)
                #    plot.savefig(file_path + "softmax_training_update{}.pdf".format(str(updates)))
                #    #plot2 = plot_position(indata_pos, indata, set_size=set_size)                    
                #    #plot2.savefig(file_path + "softmax_position_training_update{}.pdf".format(str(updates)))
                #    del data_heads[:]
                #    del data_heads
            
            

            if (val_ds != None) and (updates % make_validation == 0): # or updates == int((len(train_loader) * n_epochs))-1:  # or (updates == n_epochs-1)):
                val_losses = []
                val_acc = []
                val_editd = []
                
                # Evaluation on the validation set
                
                model.eval()
                data_heads_val = []
                samples_softmax = 0
                real_samples = 0
                total_ed = 0
                total_num_chars = 0

                with torch.no_grad():
                    print("number samples in val set= ", len(val_loader))
                    for iteration_val, data_val in enumerate(val_loader):
                        batch_x_val = data_val[0]
                        batch_y_val = data_val[1]
                        seq_len_val = data_val[2]
                        lab_len_val = data_val[3]
                        batch_y10_val = data_val[4]
                        #batch_read_val = data_val[5]
                        #batch_y_val[batch_y_val == 4.] = 5
                        #lab_len_val = lab_len_val - 1
                        #optimizer.zero_grad()
                        inputs_val, labels_val, labels10_val = Variable(batch_x_val, requires_grad=False), Variable(batch_y_val, requires_grad=False), Variable(batch_y10_val, requires_grad=False) 
                        # batch_size x out_size x seq_length                     

                        seq_len_val = inputs_val.squeeze(1).ne(-10).sum(1).float()
                        lab_len_val = labels_val.ne(5).sum(1)  
                    
                        if str(device) == "cpu":
                            labels_val = labels_val.cpu()
                            labels_val = labels_val.type(torch.LongTensor)
                        else:
                            labels_val = labels_val.type(torch.cuda.LongTensor)
                        #trg_val = torch.cat((torch.Tensor([6]).to(device).repeat(labels_val.size(0), 1), labels_val.float()), 1).type(torch.cuda.LongTensor)
                        #labels_val = labels_val.type(torch.cuda.LongTensor)
                        

                        if iteration_val == 0 or iteration_val % 10 == 0: # get softmax of heads only for every second sample, otherwise too memory intesive
                            updates_val = updates - 1
                            samples_softmax += int(inputs_val.size(0))
                            if samples_softmax <= 2592: # calculate softmax over 162 samples*32 = 5184 (0-161) afterwards stop
                                real_samples += int(inputs_val.size(0))
                                updates_val = updates
                        else: 
                            updates_val = updates - 1

                        output_val, output_val_len, figure_softmax = model(src=inputs_val, seq_len=seq_len_val, update=updates_val)
                        #output_val = output_val.contiguous()
                        
                        concat_label = torch.flatten(labels_val)
                        concat_label = concat_label[concat_label.lt(5)]


                        #loss_val = criterion(output_val, labels_val.long(), output_val_len.long(), lab_len_val.long()) #reshaped_output, reshaped_sorted_labels.long())    
                        #loss_val = F.ctc_loss(output_val, labels_val.long(), output_val_len.long(), lab_len_val.long(), blank=4, reduction='sum')
                        loss_val = F.ctc_loss(output_val, concat_label, output_val_len.long(), lab_len_val, blank=0, reduction='sum', zero_infinity=True)

                        if loss_val.isinf():
                            print("min seq len", min(seq_len_val))
                            print("min seq len after cnn", min(output_val_len))
                            print("min target len", min(lab_len_val))

                        val_losses.append(loss_val.item() / inputs_val.size(0)) #.detach().cpu().item()) # detach.item
                        epoch_loss_val += (loss_val.item() / inputs_val.size(0)) #.item()
                        running_loss_val += (loss_val.item() / inputs_val.size(0)) #.item()

                        output_val = output_val.transpose(0, 1).cpu()#.contiguous()
                        # b x t x classess
                        ctc_decoder = CTCBeamDecoder("-ACGT ", model_path=None,
                            alpha=0,
                            beta=0,
                            cutoff_top_n=6, #0
                            cutoff_prob=1.0,
                            beam_width=3,
                            num_processes=16,
                            blank_id=0,
                            log_probs_input=True)
                        beam_results, beam_scores, timesteps, out_lens = ctc_decoder.decode(output_val, output_val_len)
                        best_beam = beam_results[:, 0]
                        #if best_beam.size(1) < labels_val.size(1):
                        #    pad = torch.ones(best_beam.size(0), abs(best_beam.size(1) - labels_val.size(1)))
                        #    pad[:, :] = 5
                        #    best_beam = torch.cat((best_beam, pad), 1)
                            
                        reshaped_output = best_beam[:, :labels_val.size(1)].reshape(-1)
                        reshaped_sorted_labels = labels_val.view(-1).cpu()
                        notpadded_index = reshaped_sorted_labels != 5
                        acc_val = (reshaped_output[notpadded_index] == reshaped_sorted_labels[notpadded_index]).sum().item() / reshaped_sorted_labels[notpadded_index].size(0)
                        
                        if figure_softmax is not None:
                            data_heads_val.append(figure_softmax)
                            del figure_softmax

                        epoch_acc_val += acc_val
                        running_acc_val += acc_val
                        val_acc.append(acc_val) # acc_val
                        #print("=" * 30)
                        #print("Validation: Loss = {0:.4f}, Accuracy = ".format(loss_val.item()) + str(acc_val*100) + " %")
                        #print("=" * 30)
                       	if editD:
                       	    dict_classes = {0: "-", 1: "A", 2: "C", 3: "G", 4: "T", 5: " "}
                            #ed_val = np.mean(np.array(convert_to_string(beam_results, out_lens, labels_val, lab_len_val, dict_classes)))
                            ed_val, num_char_ref = convert_to_string(beam_results, out_lens, labels_val, lab_len_val, dict_classes)
                            epoch_editd_val += ed_val
                            running_editd_val += ed_val
                            val_editd.append(ed_val) # ed val
                            total_ed += ed_val
                            total_num_chars += num_char_ref


                    #if len(data_heads_val) > 0:
                    #    print("nr of batches in softmax plots", len(data_heads_val), "nr of samples", real_samples)
                    #    set_size = data_heads_val[0][0].size(2)
                    #    indata, indata_pos = calculate_k_patterns(data_heads_val)
                    #    plot = plot_softmax(indata, set_size=set_size)
                    #    plot.savefig(file_path + "softmax_validation_update{}.pdf".format(str(updates)))
                    #    #plot2 = plot_position(indata_pos, indata, set_size=set_size)                    
                    #    #plot2.savefig(file_path + "softmax_position_validation_update{}.pdf".format(str(updates)))
                    #    del data_heads_val[:]
                    #    del data_heads_val


                    if editD:
                        cer = float(total_ed) / total_num_chars
                    if updates == 0 or updates_newTraining == 0:
                        writer.add_scalar('Loss/train', np.mean(loss_iteration), updates)
                        writer.add_scalar('Loss/validation', np.mean(val_losses), updates)
                        #writer.add_scalar('Accuracy/train', running_acc_train, updates)
                        writer.add_scalar('Accuracy/validation', np.mean(val_acc), updates)
                        if editD:
                            #writer.add_scalar('Edit Distance/train', running_editd_train, updates)
                            writer.add_scalar('Edit Distance/validation', cer, updates)
                            dict_training_editd2[updates] = running_editd_train
                            dict_validation_editd2[updates] = (running_editd_val / float(iteration_val + 1))
                        dict_training_loss2[updates] = running_loss_train
                        dict_training_acc2[updates] = running_acc_train
                        dict_validation_loss2[updates] = (running_loss_val / float(iteration_val + 1))
                        dict_validation_acc2[updates] = (running_acc_val / float(iteration_val + 1))
                        
                    else:
                        writer.add_scalar('Loss/train', np.mean(loss_iteration), updates)
                        writer.add_scalar('Loss/validation', np.mean(val_losses), updates)
                        #writer.add_scalar('Accuracy/train', running_acc_train / float(make_validation), updates)
                        writer.add_scalar('Accuracy/validation', np.mean(val_acc), updates)
                       	if editD:
                            #writer.add_scalar('Edit Distance/train', running_editd_train, updates) #/ float(make_validation), updates)
                            writer.add_scalar('Edit Distance/validation', cer, updates)
                            dict_training_editd2[updates] = running_editd_train #/ float(make_validation)
                       	    dict_validation_editd2[updates] = (running_editd_val / float(iteration_val + 1))
                        dict_training_loss2[updates] = (running_loss_train / float(make_validation))
                        dict_training_acc2[updates] = (running_acc_train / float(make_validation))
                        dict_validation_loss2[updates] = (running_loss_val / float(iteration_val + 1))
                        dict_validation_acc2[updates] = (running_acc_val / float(iteration_val + 1))
                        

                    valid_loss = running_loss_val / float(iteration_val + 1)
                    running_loss_train = 0.0
                    running_loss_val = 0.0
                    running_acc_train = 0.0
                    running_acc_val = 0.0
                    running_editd_train = 0.0
                    running_editd_val = 0.0

                    
                    print("=" * 100)
                    print("Epoch: {}/{}...".format(epoch+1, n_epochs),
                      "Loss: {:.6f}...".format(np.mean(loss_iteration)),
                      "Accuarcy: {:.6f}%...".format(dict_training_acc2[updates] * 100),
                      "Val Loss: {:.6f}...".format(np.mean(val_losses)),
                      "Val Accuracy: {:.6f}%...".format(np.mean(val_acc) * 100))
                    if editD:
                        print("Val edit distance: {:.6f}...".format(cer))
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
                    #scheduler.step()

            updates +=1
            updates_newTraining += 1

        dict_training_loss[epoch] = loss_iteration
        dict_training_acc[epoch] = acc_iteration
        if editD:
            dict_training_editd[epoch] = editd_iteration

    writer.close()
    if earlyStopping:
        checkpoint = torch.load(file_name)
        model.load_state_dict(checkpoint["model"])
        #model.load_state_dict(torch.load(file_name))

    return ([[dict_training_loss, dict_validation_loss, dict_training_acc, dict_validation_acc, dict_training_editd, dict_validation_editd],
    [dict_training_loss2, dict_validation_loss2, dict_training_acc2, dict_validation_acc2, dict_training_editd2, dict_validation_editd2], 
    [dict_activations_in, dict_activations_forget, dict_activations_cell, dict_activations_out],
    [dict_activations_in_decoder, dict_activations_forget_decoder, dict_activations_cell_decoder, dict_activations_out_decoder],
    [dict_weights, dict_gradients], updates])

def decode_prediction(input, dict_classes):
    return([dict_classes[idx] for idx in input if idx != 5])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_param_xavier(m):
    for name, param in m.named_parameters():
        if param.data.dim() > 1:
            nn.init.xavier_uniform_(param.data)

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

    max_window_size = args.max_window_size
    max_target_size = args.max_target_size
    earlyStopping = args.early_stop
    patience_earlyStop = args.patience

    weight_decay = args.weight_decay #0.01 #0.01
    xavier_init = args.xavier_init
    
    clipping_value = args.gradient_clip

    # LSTM
    input_bias_hopfield = args.input_bias_hopfield
    hidden = args.hidden_units #256
    dff = args.dff
    num_layers = args.lstm_layers
    nhead = args.nhead
    drop_transf = args.drop_transf
    dropout_pos = args.dropout_pos
    scaling=args.scaling
    if scaling == "None":
        scaling = None
    else:
        scaling = 1 / math.sqrt(hidden) * float(scaling)
    pattern_projection_as_connected = args.pattern_projection_as_connected
    normalize_stored_pattern=args.normalize_stored_pattern
    normalize_stored_pattern_affine=args.normalize_stored_pattern_affine
    normalize_state_pattern = args.normalize_state_pattern
    normalize_state_pattern_affine=args.normalize_state_pattern_affine
    normalize_pattern_projection = args.normalize_pattern_projection
    normalize_pattern_projection_affine = args.normalize_pattern_projection_affine
    stored_pattern_as_static = args.stored_pattern_as_static
    state_pattern_as_static = args.state_pattern_as_static
    pattern_projection_as_static = args.pattern_projection_as_static
    out_norm = args.out_norm
    # CNN
    input_bias_cnn = args.input_bias_cnn
    strides = args.strides
    kernel = args.kernel
    padding = args.padding
    cnn_out = args.channel_number #256
    pooling_type = args.pooling_type #"average"
    n_layers_cnn = args.cnn_layers
    batch_normalization = args.batch_norm
    dropout_on = args.dropout_cnn
    dropout_input = args.dropout_input
    dropout_probability = args.drop_prob
    src_emb=args.src_emb
    nhead_embedding=args.nhead_embedding

    lr = args.learning_rate
    decrease_lr = args.decrease_lr
    warmup_steps = args.warmup_steps
    opt = args.opt
    betas = args.betas
    eps = args.eps

    editD = args.editD
    plot_weights = args.plot_weights
    continue_training = args.continue_training
    model_file = args.model_file
    config_file = args.config_file


    sgd = False
    out_classes = 5 + 1 # + pad token
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if port == "gpu":
        device = torch.device("cuda")
    elif port == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(port))

    if continue_training:
        if config_file != "None":
            with open(config_file, "r") as f:
                dict_param = json.load(f)
            src_emb, n_layers_cnn, cnn_out, kernel, strides, batch_normalization, pooling_type, dropout_on, dropout_input, dropout_probability = read_configuration_file(dict_param, "embedding")
            out_classes, num_layers, hidden, nhead, dff, drop_transf, dropout_pos, max_window_size, max_target_size, scaling, connected, nstored_pattern, nstored_pattern_affine, nstate_pattern, nstate_pattern_affine, npattern_projection, npattern_projection_affine = read_configuration_file(dict_param, "transformer")

    if scaling == "None" or scaling is None:
        scaling = None
    else:
        scaling = float(scaling)

    print("script on " + str(device))
    print("scaling hopfield =", scaling)

    # Load data 
    dict_classes = {0: "A", 1: "C", 2: "G", 3: "T", 4: "-", 5: "<PAD>"} # A=0, C=1, G=2, T=3, EOF=4
    script_dir = os.path.dirname(os.path.realpath('__file__')) # script directory
    
    file_out = script_dir + "/" + infile
    print(file_out)
    with open(file_out, 'rb') as handle:
        read_data = pickle.load(handle) #, map_location=torch.device('cuda:0'))
    
    writer = SummaryWriter(script_dir + '/training_result_{}'.format(fname))
    save_files_path = script_dir + '/training_result_{}/'.format(fname)
    train_set = read_data[0]
    val_set = read_data[1]
    
    train_set = [train_set[0].to(device), train_set[1].to(device), train_set[2].to(device), 
                train_set[3].to(device), train_set[4].to(device), train_set[5].to(device)]
    val_set = [val_set[0].to(device), val_set[1].to(device), val_set[2].to(device), 
                val_set[3].to(device), val_set[4].to(device), val_set[5].to(device)]
    print("train: ", train_set[0].size(), train_set[1].size(), train_set[2].size(), 
          train_set[3].size(), train_set[4].size(), train_set[5].size())
    print("validation: ", val_set[0].size(), val_set[1].size(), val_set[2].size(), 
          val_set[3].size(), val_set[4].size(), val_set[5].size())
    
    if src_emb == "residual_blocks":
        kernel_branch1 = 1
        kernel_branch2 = kernel[0]
        padding = padding[0]
        cnn_out = cnn_out[0]
        print(kernel_branch1, kernel_branch2)
        CNN_layers = []
        for l in range(n_layers_cnn):
            if l == 0: # first layer: channel_in = 1
                res_blocks = BasicBlock(input_channel=1, output_channel=cnn_out, kernel_size_branch1=kernel_branch1, kernel_size_branch2=kernel_branch2,
                            stride=strides, padding=padding, batch_norm=batch_normalization, input_bias_cnn=input_bias_cnn)
            else:
                res_blocks = BasicBlock(input_channel=cnn_out, output_channel=cnn_out, kernel_size_branch1=kernel_branch1, kernel_size_branch2=kernel_branch2,
                            stride=strides, padding=padding, batch_norm=batch_normalization, input_bias_cnn=input_bias_cnn)
            CNN_layers.append(res_blocks)
        
        CNN = SimpleCNN_res(res_layer=CNN_layers, layers = n_layers_cnn, dropout = dropout_on, dropout_p = dropout_probability, dropout_input = dropout_input)
        print(res_blocks.output_channel)
        out_channels = res_blocks.output_channel
    elif src_emb == "cnn":
        CNN = SimpleCNN(input_channel=1, output_channel=cnn_out, kernel_size=kernel, 
                    stride=strides, padding=padding, pooling=pooling_type, layers=n_layers_cnn,
                    batch_norm=batch_normalization, 
                    dropout = dropout_on, dropout_p = dropout_probability, dropout_input = dropout_input, input_bias_cnn=input_bias_cnn)
        out_channels = CNN.output_channel[n_layers_cnn-1]
        print(out_channels)

    model12 = Transformer(CNN, out_classes, out_channels, nhead, hidden, dff,
                          num_layers, drop_transf, dropout_pos=dropout_pos, max_len=max_window_size, max_len_trg=max_target_size, 
                          port=device, scaling=scaling, 
                          pattern_projection_as_connected=pattern_projection_as_connected,
                          normalize_stored_pattern=normalize_stored_pattern, normalize_stored_pattern_affine=normalize_stored_pattern_affine,
                          normalize_state_pattern=normalize_state_pattern, normalize_state_pattern_affine=normalize_state_pattern_affine,
                          normalize_pattern_projection=normalize_pattern_projection, normalize_pattern_projection_affine=normalize_pattern_projection_affine, input_bias_hopfield=input_bias_hopfield, out_norm=out_norm)#.to(device)
    if xavier_init:
        model12.apply(init_param_xavier)
    model12 = model12.to(device)

    print(model12, next(model12.parameters()).is_cuda)

    if all(b == "None" for b in betas) and eps == "None":
    	betas = ["0.9", "0.999"]
    	eps = "1e-8"
    elif eps == "None" and all(b != "None" for b in betas):
    	betas = list(map(float, betas))
    	eps = "1e-8"
    elif eps != "None" and all(b == "None" for b in betas):
    	betas = ["0.9", "0.999"]
    elif all(b != "None" for b in betas) and eps != "None":
    	betas = list(map(float, betas))

    if opt == "adam":
    	optimizer = optim.Adam(model12.parameters(), betas=tuple(betas), eps=float(eps), lr=lr, weight_decay=weight_decay, amsgrad=False)
    elif opt == "sgd":
    	optimizer = optim.SGD(model12.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == "RMSprop":
    	optimizer = optim.RMSprop(model12.parameters(), eps=float(eps), lr=lr, weight_decay=weight_decay)

    #criterion = torch.nn.NLLLoss(ignore_index=5)#.to(device)
    criterion = torch.nn.CTCLoss(blank=0, reduction='sum')

    if decrease_lr:
        optimizer = NoamOpt(model12.d_model, warmup_steps, optimizer)
    else:
        optimizer = optimizer
    print(f'The model has {count_parameters(model12):,} trainable parameters')

    last_checkpoint = None
    if continue_training:
        checkpoint = torch.load(model_file)
        model12.load_state_dict(checkpoint["model"])
        print("=> loading checkpoint '{}'".format(model_file))
        optimizer._optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer._step = checkpoint['updates']
        #optimizer.warmup = 1
        last_checkpoint = checkpoint['updates']
        print("=> loaded checkpoint {}".format(checkpoint['updates']))

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
    f.write("epochs={}, batch size={}, earlyStopping={}, patience={}, clipping value={}, scaling={}, nheads={}\n"
        .format(epochs, batch, earlyStopping, patience_earlyStop, clipping_value, scaling, nhead))
    f.write("initialisation: xavier_init={}\n".format(xavier_init))
    f.write("optimizer: {}, betas={}, eps={}, lr={}, weight decay={}; decrease_lr={}, warmup={}\n".format(opt, betas, eps, lr, weight_decay, decrease_lr, warmup_steps))
    f.close()

    dict_param = {}
    dict_param["embedding"] = {"embedding_type": src_emb, "layer": n_layers_cnn, "channel": cnn_out, "kernel": kernel, 
                                "stride": strides, "batch_norm": batch_normalization, "pooling": pooling_type, "dropout": dropout_on,
                                "input_prob": dropout_input, "output_prob": dropout_probability}
    dict_param["transformer"] = {"out_classes": out_classes, "layer": num_layers, "hidden_units": hidden, "heads": nhead, "dim_ff": dff, "dropout": drop_transf, "dropout_pos": dropout_pos,
                                "max_window_size": max_window_size, "max_target_size": max_target_size, "beta": scaling, "pattern_projection_as_connected": pattern_projection_as_connected,
                                "normalize_stored_pattern": normalize_stored_pattern, "normalize_stored_pattern_affine": normalize_stored_pattern_affine, 
                                "normalize_state_pattern": normalize_state_pattern, "normalize_state_pattern_affine": normalize_state_pattern_affine,
                                "normalize_pattern_projection": normalize_pattern_projection, "normalize_pattern_projection_affine": normalize_pattern_projection_affine}   
    # write parameter to json file
    with open(save_files_path + "model.json", "w") as f:
        json.dump(dict_param, f)
    
    # with 10 reads, kernel size = 11
    start = time.time()
    out12 = trainNet(
        model12, train_ds = train_set, optimizer=optimizer, 
        criterion=criterion, clipping_value=clipping_value, 
        val_ds = val_set, 
        batch_size=batch, n_epochs=epochs,
        make_validation=make_validation, file_name=save_files_path + "{}_checkpoint.pt".format(fname), 
        earlyStopping=earlyStopping, patience=patience_earlyStop, writer=writer, device=device, 
        editD=editD, decrease_lr=decrease_lr, src_emb=src_emb, plot_weights=plot_weights, last_checkpoint=last_checkpoint, file_path=save_files_path)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("=" * 100)

    checkpoint = { 
        'updates': out12[-1],
        'model': model12.state_dict(),
        'optimizer': optimizer._optimizer.state_dict()}
    torch.save(checkpoint, save_files_path + '{}.pt'.format(fname))
    #torch.save(model12.state_dict(), save_files_path + '{}.pt'.format(fname))
    pickle.dump(out12, open(save_files_path + "{}.p".format(fname), "wb" ))
    
    with PdfPages(save_files_path + "{}.pdf".format(fname)) as pdf:
        plot_error_accuarcy(out12[1], pdf, editD=editD)
        bestPerformance2File(out12[1], save_files_path + "best_performances_{}.txt".format(fname), editD=editD)
        plot_error_accuarcy_iterations_train(out12[0], pdf, editD=editD)
        plot_error_accuarcy_iterations_val(out12[0], pdf, editD=editD)

    if plot_weights:
        np.savez_compressed(save_files_path + '{}_weightsGradients.npz'.format(fname), weights=out12[4][0], gradients=out12[4][1])
        plot_heatmap(out12[4][0], save_files_path, filename="weights", split_LSTMbiases=False)
       	plot_heatmap(out12[4][1], save_files_path, filename="gradients", split_LSTMbiases=False)

    print("Training took: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
if __name__ == '__main__':
    sys.exit(basecalling(sys.argv))

