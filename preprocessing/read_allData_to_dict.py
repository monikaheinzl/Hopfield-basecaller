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

script_dir = os.path.dirname(os.path.realpath('__file__')) # script directory
rel_path = "/system/user/Documents/data/" # data directory
allReads_names = os.listdir(rel_path)
unique_fileNames = [np.unique(np.array([i.split(".")[0] for i in allReads_names]))][0]
print(unique_fileNames)
print("nr. of reads= ", len(allReads_names))
read_len_lst = []
nrReadsTemplate = 0
input_x = {}
for fname in unique_fileNames: 
    if "input" not in fname:
        print(fname)
        file_signal = rel_path + "/"  + fname + ".signal"
        file_label = rel_path + "/"  + fname + ".label"
        signal = np.genfromtxt(file_signal, delimiter=" ")
        norm_signal = (signal - np.mean(signal)) / np.float(np.std(signal))
        label = np.genfromtxt(file_label, delimiter=" ", dtype='str')        
        start = [int(s) for s in label[:, 0]]
        end = [int(s) for s in label[:, 1]]
        input_x[fname] = (signal, norm_signal, label, start, end)

readNames = list(input_x.keys())
signals_len = [len(input_x[k][0]) for k in readNames]#.cuda(port)
label_len = [len(input_x[k][1]) for k in readNames]
print("max signal length = ", max(signals_len))
print("max label length= ", max(label_len))

# save to pickle file
file_out = "/system/user/Documents/data.pickle"
print(file_out)
with open(file_out, 'wb') as handle:
    pickle.dump(input_x, handle, protocol=pickle.HIGHEST_PROTOCOL)





