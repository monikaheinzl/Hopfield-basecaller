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

sys.path.insert(0, '/system/user/heinzl/hopfield-layers/')
from modules.transformer import HopfieldEncoderLayer, HopfieldDecoderLayer
from modules import Hopfield, HopfieldPooling


class Decoder(nn.Module):
    def __init__(self, hopfield_self, hopfield_cross, ninp, nhead, nhid, dff, nlayers, dropout=0.5, port=1):
        super(Decoder, self).__init__()
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        #hopfield = Hopfield(input_size=ninp, hidden_size=nhid, num_heads=nhead, batch_first=False)
        decoder_layers = HopfieldDecoderLayer(hopfield_self, hopfield_cross, dff, dropout)
        self.tf_dec = TransformerDecoder(decoder_layers, nlayers)

    def forward(self, target, encoder_output, encoder_output_mask=None, target_mask=None, nopeak_mask=None):
        output = self.tf_dec(tgt=target, memory=encoder_output, 
            tgt_mask=nopeak_mask, tgt_key_padding_mask=target_mask, 
            memory_key_padding_mask=encoder_output_mask)
        return output