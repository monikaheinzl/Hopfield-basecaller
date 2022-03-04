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
import copy
from typing import Optional, Any
from torch.nn import TransformerEncoder, TransformerEncoderLayer

sys.path.insert(0, '/system/user/heinzl/hopfield-layers/')
from modules.transformer import HopfieldEncoderLayer, HopfieldDecoderLayer
from modules import Hopfield, HopfieldPooling

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=250, port=1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.port = port
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)#.to(self.port)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)#.to(self.port)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))#.to(self.port)
        pe[:, 0::2] = torch.sin(position * div_term)#.to(self.port)
        pe[:, 1::2] = torch.cos(position * div_term)#.to(self.port)
        pe = pe.unsqueeze(0).transpose(0, 1)#.to(self.port)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0), :], requires_grad=False)
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, hopfield, ninp, nhead, nhid, dff, nlayers, dropout=0.5, port=1):
        super(Encoder, self).__init__()
        self.port = port
        encoder_layers = HopfieldEncoderLayer(hopfield, dff, dropout)
        self.tf_enc = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src, seq_len_cnn, src_mask = None, src_emb = None):
        src_mask_padding = torch.zeros((src.size(1), src.size(0)), dtype=torch.bool).to(self.port) #(src == 0.).squeeze(2)
        for idx, length_cnn in enumerate(seq_len_cnn):
            if length_cnn.item() < 0:
                length_cnn = torch.Tensor([0])
            src_mask_padding[idx, int(length_cnn.item()):] = torch.ones(int(abs(src.size(0) - length_cnn.item())), dtype=torch.bool)
        src_mask_padding = Variable(src_mask_padding)
        output = self.tf_enc(src=src, mask = src_mask, src_key_padding_mask = src_mask_padding)
        return output, src_mask_padding
