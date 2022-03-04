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

import sys
import numpy as np
import argparse
import os
import h5py
import random
import pickle
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


def read_fast5_into_windows(fast5_fn, window_size, read_i):
    # Open file
        #try:
        fast5_data = h5py.File(fast5_fn, 'r')
        #except IOError:
        #    raise IOError('Error opening file. Likely a corrupted file.')
    
        # Get raw data
        try:
            raw_attr = fast5_data['Raw/Reads/']
            read_name = list(raw_attr.keys())[0]
            raw_dat = raw_attr[read_name + '/Signal'][()]
        except:
            raise RuntimeError(
                'Raw data is not stored in Raw/Reads/Read_[read#] so ' +
                'new segments cannot be identified.')

        # Reading extra information
        #corr_start_rel_to_raw = corr_attrs['read_start_rel_to_raw']  #
        #event_starts = corr_data['start'] + corr_start_rel_to_raw
        #event_lengths = corr_data['length']
        #event_bases = corr_data['base']

        #start = [int(s) - int(event_starts[0]) for s in event_starts]
        #end = [int(s) - int(event_starts[0]) - 1 for s in event_lengths]
        #starts = corr_data['start']

        # dummy data
        #if read_name in list(out_dict.keys()):
        #   read_name = read_name + "_1"
        start = [None for s in range(1000)]
        end = [None for s in range(1000)]
        event_bases = [None for s in range(1000)]
        starts = [None for s in range(1000)]

        name = fast5_fn.split(".fast5")[0]
        print(read_i, name)

        signals_i = (raw_dat - np.mean(raw_dat)) / np.float(np.std(raw_dat))
        fast5_data.close()

        ######### make windows

        read_idx = []
        seq_length_window = []
        count_reads = 0        
        data_window = torch.full((len(signals_i), window_size), -10, dtype=torch.float)

        print("idx", read_i, len(signals_i))
        old_idx = 0
        idx = 0
        counter = 0
        while idx < len(signals_i):
            if idx != 0:
                idx = window_size + old_idx

            if idx + window_size > len(signals_i):
                window_end = len(signals_i)
            else:
                window_end = idx + window_size
            

            data = torch.from_numpy(np.array(signals_i)[idx:window_end].reshape(1, -1))
    
            if data.size(1) == 0:
                break

            seq_length_window.append(data.size(1))
            read_idx.append(read_i)
            data_window[counter, 0:data.size(1)] = data
            old_idx = idx
            idx += 1    
            counter += 1


        #data_window = torch.unsqueeze(data_window[data_window.sum(dim=1) != 0, :].type(torch.FloatTensor), 1)  
        all_data_window = torch.unsqueeze(data_window[:counter, :], 1) #.type(torch.FloatTensor)
        
        ###########################
        ##### dummy variables #####
        input_y_bases = np.zeros((all_data_window.size(0), window_size))
        input_y_bases_oneHot = np.zeros((all_data_window.size(0),window_size, 5))
        segments_start_list = np.zeros((all_data_window.size(0), window_size)) # excl eos token
        segments_end_list = np.zeros((all_data_window.size(0), window_size))
        true_label_len = np.zeros((all_data_window.size(0), window_size))
        ###########################
        seq_length_window = np.array(seq_length_window)
        read_idx = np.array(read_idx)
        all_data_window = all_data_window.detach().cpu().numpy()
        print("input signal size", all_data_window.shape)

        #input_y_bases = torch.from_numpy(input_y_bases)#.cuda(port).long()
        #input_y_bases_oneHot = torch.from_numpy(input_y_bases_oneHot)
        #true_label_len = torch.from_numpy(true_label_len)#.cuda(port)
        #seq_length_window = torch.Tensor(seq_length_window)#.cuda(port)
        #read_idx = torch.Tensor(read_idx)#.cuda(port)
        #segments_start_list = torch.from_numpy(segments_start_list)
        #segments_end_list = torch.from_numpy(segments_end_list)
        print(all_data_window)
        return ([all_data_window, input_y_bases, input_y_bases_oneHot, seq_length_window, true_label_len, read_idx, segments_start_list, segments_end_list])

        #del save_to_file[:]
        #del save_to_file
        del fast5_data
        del signals_i

