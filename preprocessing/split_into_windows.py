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
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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
import itertools 
from statsmodels import robust

plt.switch_backend('agg')

script_dir = os.path.dirname(os.path.realpath('__file__')) # script directory
file_out = "/system/user/Documents/data.pickle"
print(file_out)
with open(file_out, 'rb') as handle:
    read_data = pickle.load(handle)

# .npz file containing the class labels after clustering the data into training, validation and test set
fname = '/system/user/Documents/labels.npz'
loaded = np.load(fname, allow_pickle=True)
train_set_names = loaded["readNames_cluster0"]
val_set_names = loaded["readNames_cluster1"]
test_set_names = loaded["readNames_cluster2"]

def split_into_windows(read_data, window_size=500, port=1, mode_type="train"):
    readNames = np.array(list(read_data.keys()))
    signals_notPadded = [read_data[k][1] for k in readNames]
    signals_len = [len(read_data[k][1]) for k in readNames]
    labels = [read_data[k][2][:, 2] for k in readNames]
    start = [read_data[k][3] for k in readNames]
    end = [read_data[k][4] for k in readNames]
    segment_len = [abs(np.array(read_data[k][3]) - np.array(read_data[k][4])) + 1 for k in readNames]
    label_len = [len(read_data[k][2]) for k in readNames]

    print("max signal length = ", max(signals_len))
    print("max label length= ", max(label_len))
    print(len(signals_notPadded))
    
    read_idx = []
    seq_length_window = []
    target_length_window = []
    target_list = []
    target10_list = []
    segments_start_list = []
    segments_end_list = []
    count_reads = 0
    dict_read_id_name = {}
    for read_i in range(len(readNames)):
        dict_read_id_name[read_i] = readNames[read_i]
        signals_i = signals_notPadded[read_i]
        target_i = labels[read_i]
        target_i_encoded = torch.Tensor(LabelEncoder().fit_transform(target_i.ravel()).reshape(*target_i.shape))
        # transform to binary
        target_i_encoded_10 = torch.Tensor(OneHotEncoder(categories='auto').
                                           fit_transform(target_i_encoded.reshape(-1, 1)).toarray())
        start_i = torch.Tensor(start[read_i])
        end_i = torch.Tensor(end[read_i])    
        segment_len_i = torch.Tensor(segment_len[read_i])    
        median_segment_len = torch.median(segment_len_i).item()
        indices_seg = torch.arange(0, len(target_i), 1)
        data_window = torch.full((len(signals_i), window_size), -10)

        print("idx", read_i, len(signals_i), len(target_i))

        old_idx = 0
        idx = 0
        counter = 0
        while idx < len(signals_i):
            if idx != 0:
                if mode_type == "train":
                    #print(end_i[idx_segments].size())
                    if end_segments.size(0) > 0:
                        idx = int((end_segments[-1] + 1).item()) #step_size + old_idx
                    else:
                        idx = window_size + old_idx
                else:
                    idx = window_size + old_idx
            if idx + window_size > len(signals_i):
                window_end = len(signals_i)
            else:
                window_end = idx + window_size
            
            data = torch.from_numpy(np.array(signals_i)[idx:window_end].reshape(1, -1))
            # extract target
            # first signal has index idx, last signal: idx+window_size-1
            # retrieve all labels where segment start >= idx & segment end <= idx+window_size-1
            idx_segments = torch.squeeze((start_i >= idx) & (end_i <= window_end))
            idx_indices = indices_seg[idx_segments]
            target = target_i_encoded[idx_segments]
            target10 = target_i_encoded_10[idx_segments, :]

            end_segments = end_i[idx_segments]
            start_segments = start_i[idx_segments]

            if target.size(0) == 0:
                if  mode_type == "train" and window_end <= int(end_i[-1]): # not break yet if end of sequence is reached, just skip middle ones
                    old_idx = idx
                    idx += 1
                    continue
                else:
                    break # stop if no window with target available, last segment end is shorter than sequence length
            
            starts = start_segments
            ends = end_segments
            if idx > 0:
                starts = starts - start_segments[0]
                ends = ends - start_segments[0]
    
            if mode_type == "train":
                # difference between window end and last segment end
                # if not 0 --> not all signals per nucleotide are in the current window
                # if less than 90% of the missing values bases on the median are present -> set signal to 0
                # if more than 90% --> add nucleotide of next segment to target
                partial_segment_of_nucleotide = abs(window_end - end_segments[-1] - 1).item()
                if int(partial_segment_of_nucleotide) > 0:
                    data = data[:, 0:data.size(1) - int(partial_segment_of_nucleotide)]

                partial_segment_of_nucleotide_start = abs(idx - int(start_segments[0].item()))
                if int(partial_segment_of_nucleotide_start) > 0:
                    data = data[:, partial_segment_of_nucleotide_start:]

                # add EOF token
                target = torch.cat((target, torch.Tensor([4])))
                # add column for EOF token
                target10 = torch.cat((target10, torch.zeros(target10.size(0), 1)), dim=1)
                # add row with encoding of EOF token
                target10 = torch.cat((target10,  torch.Tensor([[0, 0, 0, 0, 1]])), dim=0)
                #print("after padding", target)

            if mode_type == "train" and data.size(1) < 1024: # otherwise length gets negative with large stride
                old_idx = idx
                idx += 1
                continue

            seq_length_window.append(data.size(1))
            read_idx.append(read_i)
            segments_start_list.append(starts)
            segments_end_list.append(ends)
            data_window[counter, 0:data.size(1)] = data
            target_list.append(target)
            target_length_window.append(len(target))
            target10_list.append(target10)
            
            old_idx = idx
            idx += 1    
            counter += 1

        #data_window = torch.unsqueeze(data_window[data_window.sum(dim=1) != 0, :].type(torch.FloatTensor), 1)
        data_window = torch.unsqueeze(data_window[:counter, :].type(torch.FloatTensor), 1)  
        
        if count_reads != 0:
            input_signals_padded_window = torch.cat((all_data_window, data_window), 0)
            all_data_window = input_signals_padded_window
        else:
            all_data_window = data_window
        count_reads += 1
    # pad target
    input_y_bases = []
    target_len_not_padded_window = target_length_window

    starts_padded = []
    ends_padded = []

    #print(target_len_not_padded_window, max(target_len_not_padded_window))
    padded_y_bases = np.zeros((len(target_list), max(target_len_not_padded_window)))

    if mode_type == "train":
        padded_starts = np.zeros((len(target_list), max(target_len_not_padded_window)-1)) # excl eos token
        padded_ends = np.zeros((len(target_list), max(target_len_not_padded_window)-1))
    else:
        padded_starts = np.zeros((len(target_list), max(target_len_not_padded_window))) # excl eos token
        padded_ends = np.zeros((len(target_list), max(target_len_not_padded_window)))
    for idx_lab, lab in enumerate(target_list):
        if len(lab) < max(target_len_not_padded_window):
            padded5 = np.repeat(5, abs(len(lab) - max(target_len_not_padded_window)))
            new_lab = np.append(lab, padded5)

            if mode_type == "train":
                padded_neg = np.repeat(-1, abs(len(segments_start_list[idx_lab]) - max(target_len_not_padded_window)+1))
                padded_neg_ends = np.repeat(-1, abs(len(segments_end_list[idx_lab]) - max(target_len_not_padded_window)+1))

            else:
                padded_neg = np.repeat(-1, abs(len(segments_start_list[idx_lab]) - max(target_len_not_padded_window)))
                padded_neg_ends = np.repeat(-1, abs(len(segments_end_list[idx_lab]) - max(target_len_not_padded_window)))
            new_lab_start = np.append(segments_start_list[idx_lab], padded_neg)
            new_lab_end = np.append(segments_end_list[idx_lab], padded_neg_ends)
        else:
            new_lab = lab
            new_lab_start = segments_start_list[idx_lab]
            new_lab_end = segments_end_list[idx_lab]
        padded_y_bases[idx_lab, :] = new_lab
        padded_starts[idx_lab, :] = new_lab_start
        padded_ends[idx_lab, :] = new_lab_end
    
    input_y_bases = torch.from_numpy(padded_y_bases)#.cuda(port).long()
    input_y_bases_oneHot = torch.transpose(torch.transpose(nn.utils.rnn.pad_sequence(target10_list), 0, 1), 1, 2)#.cuda(port)
    print("input signal size", all_data_window.size())
    print("target size", input_y_bases.size(), input_y_bases_oneHot.size())
    print("min target len", min(target_length_window))
    print("min seq len", min(seq_length_window))
    true_label_len = torch.Tensor(target_length_window)#.cuda(port)
    signals_len = torch.Tensor(seq_length_window)#.cuda(port)
    read_idx = torch.Tensor(read_idx)#.cuda(port)
    segments_start_list = torch.from_numpy(padded_starts)
    segments_end_list = torch.from_numpy(padded_ends)
    return(all_data_window, input_y_bases, 
           input_y_bases_oneHot, signals_len, true_label_len, read_idx, dict_read_id_name, segments_start_list, segments_end_list) #.cuda(port))

print("train_set", len(train_set_names), "val_set", len(val_set_names), "test_set", len(test_set_names))

subdict_train = {x: read_data[x] for x in train_set_names}
subdict_val = {x: read_data[x] for x in val_set_names}
subdict_test = {x: read_data[x] for x in test_set_names}

print(len(subdict_train.keys()), len(subdict_val.keys()), len(subdict_test.keys()))

signals_len = [len(subdict_train[k][1]) for k in list(subdict_train.keys())]
labels_len = [len(subdict_train[k][2][:, 2]) for k in list(subdict_train.keys())]

signals_train = [len(subdict_train[k][1]) for k in list(subdict_train.keys())]
signals_val= [len(subdict_val[k][1]) for k in list(subdict_val.keys())]
signals_test = [len(subdict_test[k][1]) for k in list(subdict_test.keys())]

target_train = [len(subdict_train[k][2][:, 2])for k in list(subdict_train.keys())]
target_val= [len(subdict_val[k][2][:, 2]) for k in list(subdict_val.keys())]
target_test = [len(subdict_test[k][2][:, 2]) for k in list(subdict_test.keys())]

window_size = 2048 

train_x, train_y, train_y_10, seq_len_train, true_label_len_train, read_idx_train, dict_read_id_name_train, starts_train, ends_train = split_into_windows(
    subdict_train, window_size=window_size, port=1, mode_type="train")#

val_x, val_y, val_y_10, seq_len_val, true_label_len_val, read_idx_val, dict_read_id_name_val, starts_val, ends_val = split_into_windows(
    subdict_val, window_size=window_size,  port=1, mode_type="train")

test_x, test_y, test_y_10, seq_len_test, true_label_len_test, read_idx_test, dict_read_id_name_test, starts_test, ends_test = split_into_windows(
    subdict_test, window_size=window_size, port=1, mode_type="inference")

# save to file
file_out = "/system/user/Documents/TrainValidationTestset.pickle"
print(file_out)
save_to_file = [[train_x, train_y, train_y_10, seq_len_train, true_label_len_train, read_idx_train, starts_train, ends_train], 
               [val_x, val_y, val_y_10, seq_len_val, true_label_len_val, read_idx_val, starts_val, ends_val],
               [test_x, test_y, test_y_10, seq_len_test, true_label_len_test, read_idx_test, starts_test, ends_test]]
with open(file_out, 'wb') as handle:
    pickle.dump(save_to_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


