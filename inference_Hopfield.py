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

# beam search decoder was used from https://github.com/dreamgonfly/Transformer-pytorch and partially adapted
# In[ ]:


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
import json


sys.path.insert(0, '/hopfield-layers/')
from modules.transformer import HopfieldEncoderLayer, HopfieldDecoderLayer
from modules import Hopfield, HopfieldPooling
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.insert(0, '/basecaller-modules')
from read_config import read_configuration_file
from cnn import SimpleCNN, BasicBlock, SimpleCNN_res
from cnn import outputLen_Conv, outputLen_AvgPool, outputLen_MaxPool
from hopfield_encoder_nosqrt import Embedder, PositionalEncoding, Encoder

from hopfield_decoder import Decoder
from early_stopping import EarlyStopping
from lr_scheduler import NoamOpt
from beam_adapted_storeSoftmax import Beam
from read_fast5_to_dict import read_fast5_into_windows

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
    parser.add_argument('-m', '--model', required = True,
                        help="File path to model file.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output folder name")
    parser.add_argument('-ou', '--output_name', required = True,
                        help="Output folder name")

    parser.add_argument("--validation", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)


    parser.add_argument('-config', '--config_file', default="None",
                        help="Path to config file")

    parser.add_argument('-b', '--beam_size', type=int, default=3,
                        help="Beam size")

    parser.add_argument('-g', '--gpu_port', default="None",
                        help="Port on GPU mode")
    parser.add_argument('-s', '--set_seed', type=int, default=1234,
                        help="Set seed")

    parser.add_argument('-max_w', '--max_window_size', type=int, default=1000,
                        help="Maximum window size")
    parser.add_argument('-max_t', '--max_target_size', type=int, default=200,
                        help="Maximum target size")

    # CNN arguments
    parser.add_argument("--input_bias_cnn", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)
    parser.add_argument('-c', '--channel_number', nargs='+', type=int, default=[256, 256, 256],
                        help="Number of output channels in Encoder-CNN")
    parser.add_argument('-l', '--cnn_layers', type=int, default=1,
                        help="Number of layers in Encoder-CNN")
    parser.add_argument('--pooling_type', default="None",
                        help="Pooling type in Encoder-CNN")
    parser.add_argument('--strides', nargs='+', type=int, default=[1, 1, 1],
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
    parser.add_argument('--nhead_embedding', type=int, default=1,
                        help="number of heads in the multiheadattention models")
    #parser.add_argument("--res_layer", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

    # Hopfield arguments
    parser.add_argument("--input_bias_hopfield", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True)
    parser.add_argument('-u', '--hidden_units', type=int, default=256,
                        help="Number of hidden units in the Transformer")
    parser.add_argument('--dff', type=int, default=256,
                        help="Number of hidden units in the Feed-Forward Layer of the Transformer")
    parser.add_argument('--lstm_layers', type=int, default=1,
                        help="Number of layers in the Transformer")
    parser.add_argument('--nhead', type=int, default=1,
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
                 normalize_pattern_projection=False, normalize_pattern_projection_affine=False, input_bias_hopfield=True):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.d_model = d_model
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
        hopfield_self_target = Hopfield(input_size=d_model, hidden_size=nhid, num_heads=nhead, 
            batch_first=False, scaling=scaling, dropout=dropout, pattern_projection_as_connected=pattern_projection_as_connected, 
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
        self.pos_enc_decoder = PositionalEncoding(d_model, dropout_pos, max_len_trg) #, max_len_trg)
        self.embed_target = nn.Embedding(7, d_model, padding_idx=5) # input 7=<SOS>ACTG<EOF>PADDING, 0-5
        self.encoder = Encoder(hopfield_self_src, d_model, nhead, nhid, dff, nlayers, dropout, port)
        self.decoder = Decoder(hopfield_self_target, hopfield_self_src, d_model, nhead, nhid, dff, nlayers, dropout, port)
        self.fc = nn.Linear(hopfield_self_target.output_size, ntoken) # ntoken
        #self.fc = nn.Linear(d_model, ntoken)
        self.port = port
    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf')).to(self.port)
        mask = Variable(mask)
        return mask
    def forward(self, src, trg, seq_len, trg_len, src_emb="cnn"):
        #if src_emb == "cnn" or src_emb == "residual_blocks":
        src = src.detach()
        seq_len = seq_len.detach()
        trg = trg.detach()
        trg_len = trg_len.detach()

        src, seq_len_cnn = self.cnn_encoder(src, seq_len)
        src = src.transpose(1, 2).transpose(0, 1)
        trg_mask = ((trg == 5) | (trg == 4))
        trg_mask = Variable(trg_mask)

        nopeak_mask = self._generate_square_subsequent_mask(trg.size(1))
        
        trg = self.embed_target(trg).transpose(0, 1)
        src = self.pos_enc_encoder(src)
        trg = self.pos_enc_decoder(trg)

        e_outputs, src_mask = self.encoder(src, seq_len_cnn, src_mask=None, src_emb=src_emb)
        output = self.decoder(target=trg, encoder_output=e_outputs, encoder_output_mask=src_mask, target_mask=trg_mask, nopeak_mask=nopeak_mask)
        output = output.transpose(0, 1)
        output = self.fc(output)
        output = F.log_softmax(output, dim=2)
        return output

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

def decode_prediction(input, dict_classes):
    return([dict_classes[idx] for idx in input if idx != 5])

def inference(model, test_ds, dict_classes, mode="inference", beam_size=1, device=0, src_emb="cnn", fname=None, max_target_size=200, max_window_size = 2048, validation=True, output_name="all"):
    #Print all of the hyperparameters of the training iteration:
    # Evaluation on the validation set
    val_losses = []
    val_acc = []
    basecalled_sequences = dict()
    epoch_loss_val = 0
    epoch_acc_val = 0
    edit_d = []
    old_read_id = None
    counter_read = 0

    out_name = output_name #"{}_uncorrected_beam{}_all".format(typeset, beam_size)
    if not os.path.exists(fname + '/{}'.format(out_name)):
        os.makedirs(fname + '/{}'.format(out_name))

    if not os.path.exists(fname + '/{}_softmax'.format(out_name)):
        os.makedirs(fname + '/{}_softmax'.format(out_name))
    if not os.path.exists(fname + '/{}_windows'.format(out_name)):
        os.makedirs(fname + '/{}_windows'.format(out_name))
    read_i = 0
    for file in os.listdir(test_ds):
        print(test_ds, file)
        test_set = read_fast5_into_windows(test_ds + "/" + file, max_window_size, read_i)

        f2 = open(fname + '/{}/{}.fasta'.format(out_name, file.split(".fast5")[0]), "w")
        f2.write(">" + file.split(".fast5")[0] + "\n")

        f = open(fname + '/{}_windows/{}.fasta'.format(out_name, file.split(".fast5")[0]), "w")
        
        
        test_set = [torch.from_numpy(test_set[0]).to(device), torch.from_numpy(test_set[1]).to(device), torch.from_numpy(test_set[2]).to(device), 
                    torch.from_numpy(test_set[3]).to(device), torch.from_numpy(test_set[4]).to(device), torch.from_numpy(test_set[5]).to(device)]
        print("test: ", test_set[0].size(), test_set[1].size(), test_set[2].size(), 
              test_set[3].size(), test_set[4].size(), test_set[5].size())

        print("===== INFERENCE =====")        
        input_x = test_set[0]
        input_y = test_set[1]
        input_y10 = test_set[2]
        signal_len = test_set[3]
        label_len = test_set[4]
        read_index = test_set[5]

        print(signal_len)
    
        #Get training data
        test_loader = get_train_loader(input_x, signal_len, 
                                        input_y, label_len, 
                                        input_y10, read_index, batch_size=1, shuffle=False)

        dict_softmax = {}
        
        read_window = []
        model.eval()
        with torch.no_grad():
            for iteration_val, data_val in enumerate(test_loader):
                batch_x_val = data_val[0]
                batch_y_val = data_val[1]
                seq_len_val = data_val[2]
                lab_len_val = data_val[3]
                batch_y10_val = data_val[4]
                batch_read = data_val[5]
    
                print("=" * 30)
                print("sample {}/{}".format(iteration_val+1, len(test_loader)))
                print("read id = ", int(batch_read.item()))
                print("read = ", read_i, " from ", len(os.listdir(test_ds)))
                #print("signal")
                print("=" * 30)
    
                #if int(batch_read.item()) < 109:
                #    continue

                if seq_len_val.item() <= 0: # if window has length 0 --> skip
                    continue
    
                batch_x_val = Variable(batch_x_val, requires_grad=False)#.long() # batch_size x out_size x seq_length 
                #print("batch_x_val= ", batch_x_val.size())
                ######## ENCODER ###########
                ############################
                if src_emb == "cnn":
                    encoded_src, seq_len_encoded = model.cnn_encoder(batch_x_val, seq_len_val)
                    encoded_src = encoded_src.transpose(1, 2).transpose(0, 1)

                if seq_len_encoded.item() <= 0:
                    continue

                #print("encoded_src", encoded_src.size()) #, "src_non_peak", src_non_peak.size())
                memory, memory_mask = model.encoder(model.pos_enc_encoder(encoded_src), seq_len_encoded, src_mask = None)
    
                # Repeat beam_size times
                num_candidates = beam_size
                #memory_beam = memory.detach().repeat(1, beam_size, 1)  # (seq_len, beam_size, hidden_size)
                #memory_mask_beam = memory_mask.detach().repeat(beam_size, 1)
                # max_target_size*1/4
                beam = Beam(beam_size=beam_size, min_length=10, n_top=num_candidates, start_token_id=6, end_token_id=4)
                
                # for comparison normal greedy decoding
                trg_greedy = [6, ] # SOS token
    
                ######## DECODER ###########
                ############################
                for i in range(max_target_size): #itertools.count():
                    # BEAM SEARCH
                    trg = beam.get_current_state()  # (beam_size, seq_len)
                    trg_tensor = Variable(torch.LongTensor(trg).to(device), requires_grad=False)#.unsqueeze(0)
                    trg_tensor = model.embed_target(trg_tensor).transpose(0,1)
                    trg_tensor = model.pos_enc_decoder(trg_tensor)
                    if trg_tensor.size(1) < beam_size: # if beam size > classes
                        # select all classes in this time step and adapt memory size to correct beam size
                        cur_beam_size = trg_tensor.size(1)
                        memory_beam = memory.detach().repeat(1, cur_beam_size, 1)  # (seq_len, beam_size, hidden_size)
                        memory_mask_beam = memory_mask.detach().repeat(cur_beam_size, 1)
                    else: # expand memory to beam size
                        memory_beam = memory.detach().repeat(1, beam_size, 1)  # (seq_len, beam_size, hidden_size)
                        memory_mask_beam = memory_mask.detach().repeat(beam_size, 1)
    
                    output = model.decoder(trg_tensor, memory_beam, encoder_output_mask=memory_mask_beam, target_mask=None, nopeak_mask=None).transpose(0, 1)
                    output = output[:, -1, :].unsqueeze(1) # size beam x seq_len x hidden --> last time step size: beam x hidden --> unsqueeze to size beam x seq_len x hidden
                    output = model.fc(output)
                    output = F.log_softmax(output, dim=2).contiguous()

                    output = output.view(-1, output.size(2))
                    
                    beam.advance(output)
    
                    # for comparison normal greedy decoding
                    #trg_tensor_greedy = Variable(torch.LongTensor(trg_greedy).unsqueeze(0).to(device), requires_grad=False)
                    #trg_tensor_greedy = model.embed_target(trg_tensor_greedy).transpose(0,1)
                    #trg_tensor_greedy = model.pos_enc_decoder(trg_tensor_greedy)
                    #output_greedy = model.decoder(trg_tensor_greedy, memory, encoder_output_mask=memory_mask, target_mask=None, nopeak_mask=None).transpose(0, 1)
                    #output_greedy = model.fc(output_greedy)
                    #output_greedy = F.log_softmax(output_greedy, dim=2).contiguous()
                    #output_greedy = output_greedy.view(-1, output_greedy.size(2))
                    #out_token_greedy = output_greedy.argmax(1)[-1].item()
                    #trg_greedy.append(out_token_greedy)
                    #if out_token_greedy == 4:
                    #    break
    
                    # break loop if top candidate (candidate with highest probability) finished and all candidates finished
                    if beam.done():
                        break
    
                
                # BEAM SEARCH
                ## retrieve n candidate sequences and their sequence plus score
                num_candidates = 1 # only top beam
                scores, ks = beam.sort_finished(minimum=num_candidates)
                hypothesises = []
                softmaxes_summed = []
                softmaxes = []
                for i, (times, k) in enumerate(ks[:num_candidates]):
                    hypothesis, softmax_summed, softmax = beam.get_hypothesis(times, k)
                    hypothesises.append((hypothesis, scores[i]))
                    softmaxes_summed.append(softmax_summed)
                    softmaxes.append(softmax)
                hypothesises = [([token.item() for token in h[0]], h[1]) for h in hypothesises]
                hs = [(decode_prediction(h[0], dict_classes), h[1]) for h in hypothesises]
                best_candidate = "".join(hs[0][0][:-1])
                read_window.append(best_candidate)
                f.write(">window{}\n".format(iteration_val))
                f.write("".join(hs[0][0]) + "\n")                       
                old_read_id = int(batch_read.item())
                counter_read += 1
    
                #print(softmaxes_summed[0], len(softmaxes_summed[0]), softmaxes[0], len(softmaxes[0]), len(best_candidate), best_candidate)
                dict_softmax[iteration_val] = (softmaxes_summed[0], softmaxes[0])
                #sys.exit()

                #if iteration_val > 2:
                #    break
        with open(fname + '/{}_softmax/{}.pickle'.format(out_name, file.split(".")[0]), 'wb') as handle:
            pickle.dump(dict_softmax, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        read_i += 1
        f2.write("".join(read_window)+ "\n")
        f2.close()
        f.close()
        #sys.exit()
    
    #print("mean edit distance = ", np.nanmean(np.asarray(edit_d)))
    return (basecalled_sequences)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def basecalling(argv):
    parser = make_argparser()
    args = parser.parse_args(argv[1:])
    
    infile = args.input
    modelfile = args.model
    fname = args.output
    port = args.gpu_port
    SEED = args.set_seed

    config_file = args.config_file
    beam_size = args.beam_size
    max_window_size = args.max_window_size
    max_target_size = args.max_target_size
    validation = args.validation
    output_name = args.output_name

    # LSTM
    input_bias_hopfield = args.input_bias_hopfield
    hidden = args.hidden_units #256
    dff = args.dff
    num_layers = args.lstm_layers
    nhead = args.nhead
    drop_transf = args.drop_transf
    scaling=args.scaling
    connected = args.pattern_projection_as_connected
    nstored_pattern=args.normalize_stored_pattern
    nstored_pattern_affine=args.normalize_stored_pattern_affine
    nstate_pattern = args.normalize_state_pattern
    nstate_pattern_affine=args.normalize_state_pattern_affine
    npattern_projection = args.normalize_pattern_projection
    npattern_projection_affine = args.normalize_pattern_projection_affine
    stored_pattern_as_static = args.stored_pattern_as_static
    state_pattern_as_static = args.state_pattern_as_static
    pattern_projection_as_static = args.pattern_projection_as_static
    # CNN
    input_bias_cnn = args.input_bias_cnn
    strides = args.strides
    kernel = args.kernel
    cnn_out = args.channel_number #256
    padding = args.padding
    pooling_type = args.pooling_type #"average"
    n_layers_cnn = args.cnn_layers
    batch_normalization = args.batch_norm
    dropout_on = args.dropout_cnn
    dropout_input = args.dropout_input
    dropout_probability = args.drop_prob
    src_emb=args.src_emb
    nhead_embedding=args.nhead_embedding

    out_classes = 5 #+ 2
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    if scaling == "None":
        scaling = None
    else:
        scaling = 1 / math.sqrt(hidden) * float(scaling)

    if port == "gpu":
        device = torch.device("cuda")
    elif port == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(port))

    if config_file != "None":
        with open(config_file, "r") as f:
            dict_param = json.load(f)
        SEED, src_emb, n_layers_cnn, cnn_out, kernel, padding, strides, batch_normalization, pooling_type, dropout_on, dropout_input, dropout_probability, input_bias_cnn = read_configuration_file(dict_param, "embedding")
        out_classes, num_layers, hidden, nhead, dff, drop_transf, dropout_pos, max_window_size, max_target_size, scaling, connected, nstored_pattern, nstored_pattern_affine, nstate_pattern, nstate_pattern_affine, npattern_projection, npattern_projection_affine, input_bias_hopfield = read_configuration_file(dict_param, "transformer")

    if scaling == "None" or scaling is None:
        scaling = None
    else:
        scaling = float(scaling)
    	
    print("script on " + str(device))
    print("scaling hopfield =", scaling)

    # Load data 
    script_dir = os.path.dirname(os.path.realpath('__file__')) # script directory    
    file_out = infile
    print(file_out)    
    
    if src_emb == "residual_blocks":
        kernel_branch1 = 1
        kernel_branch2 = kernel[0]
        cnn_out = cnn_out[0]
        CNN_layers = []
        for l in range(n_layers_cnn):
            if l == 0: # first layer: channel_in = 1
                res_blocks = BasicBlock(input_channel=1, output_channel=cnn_out, kernel_size_branch1=kernel_branch1, kernel_size_branch2=kernel_branch2,
                            stride=strides, padding=0, output_dim=out_classes, batch_norm=batch_normalization)
            else:
                res_blocks = BasicBlock(input_channel=cnn_out, output_channel=cnn_out, kernel_size_branch1=kernel_branch1, kernel_size_branch2=kernel_branch2,
                            stride=strides, padding=0, output_dim=out_classes, batch_norm=batch_normalization)
            CNN_layers.append(res_blocks)
        
        CNN = SimpleCNN_res(res_layer=CNN_layers, layers = n_layers_cnn, dropout = dropout_on, dropout_p = dropout_probability, dropout_input = dropout_input)
        out_channels = res_blocks.output_channel
    elif src_emb == "cnn":
        # [batch size] is typically chosen between 1 and a few hundreds, e.g. [batch size] = 32 is a good default value
        CNN = SimpleCNN(input_channel=1, output_channel=cnn_out, kernel_size=kernel, 
                    stride=strides, padding=[0,0,0], pooling=pooling_type, layers=n_layers_cnn,
                    batch_norm=batch_normalization, 
                    dropout = dropout_on, dropout_p = dropout_probability, dropout_input = dropout_input, input_bias_cnn=input_bias_cnn)
        out_channels = CNN.output_channel[n_layers_cnn-1]

    model12 = Transformer(CNN, out_classes, out_channels, nhead, hidden, dff,
                          num_layers, drop_transf, dropout_pos=dropout_pos, max_len=max_window_size, max_len_trg=max_target_size, 
                          port=device, scaling=scaling, pattern_projection_as_connected=connected,
                          normalize_stored_pattern=nstored_pattern, normalize_stored_pattern_affine=nstored_pattern_affine,
                          normalize_state_pattern=nstate_pattern, normalize_state_pattern_affine=nstate_pattern_affine,
                          normalize_pattern_projection=npattern_projection, normalize_pattern_projection_affine=npattern_projection_affine, input_bias_hopfield=input_bias_hopfield)#.to(device)
    print(model12, next(model12.parameters()).is_cuda)
    checkpoint = torch.load(modelfile,map_location='cpu')
    model12.load_state_dict(checkpoint["model"])
    model12 = model12.to(device)
    print(f'The model has {count_parameters(model12):,} trainable parameters')    
    dict_classes = {0: "A", 1: "C", 2: "G", 3: "T", 4: "<EOS>", 5: "<PAD>", 6: "<SOS>"}

    start = time.time()
    basecalled = inference(model12, file_out, dict_classes, mode="inference", beam_size=beam_size, device=device, src_emb=src_emb, fname=fname, max_target_size=max_target_size, max_window_size=max_window_size, validation=validation, output_name=output_name)
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("=" * 100)
        
if __name__ == '__main__':
    sys.exit(basecalling(sys.argv))


