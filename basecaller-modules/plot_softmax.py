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

def calculate_k_patterns(input):
	plt.clf()
	layers = len(input[0].keys())
	sns.set(font_scale=0.5)
	fig = plt.figure(figsize=(10,1*layers)) #figsize=(15,15)) #figsize=(18,2)
	#fig.subplots_adjust(left=0, right=0.5, bottom=0, top=1) #, hspace=0.2, wspace=0.2)
	#fig.subplots_adjust(left=0.4, right=0.5) #, bottom=0, top=1) #, hspace=0.2, wspace=0.2)

	# b x heads x n x n
	heads = input[0][0].size(1)
	set_size = input[0][0].size(2)
	k_patterns = np.zeros(shape=(layers,heads))
	plot_i = 1
	processed_input = {}
	for batch_i in input:
		for layers_i in reversed(range(layers)):
			processed_input[layers_i] = {}
			for head_i in range(heads):
				softmax = batch_i[layers_i][:, head_i, :, :]
				processed_input[layers_i][head_i] = [] 
				#ks = []
				for sample_i in softmax:
					sorted_softmax = torch.sort(sample_i.cpu(), descending=True)[0]
					cumsum = torch.cumsum(sorted_softmax, dim=1).numpy()
					index90 = np.argmax(cumsum >= 0.9, axis=1) + 1 #np.where(cumsum >= 0.9) # first index where cumsum > 90
					processed_input[layers_i][head_i].extend(index90)

	del input[:]
	del input
	for layers_i in reversed(range(layers)):
		#print(layers_i)
		head = 1
		for head_i in range(heads):
			softmax = processed_input[layers_i][head_i]
			#print(layers_i, head_i, np.median(softmax))
			ax = fig.add_subplot(layers, heads, plot_i)
			#parts = ax.violinplot(ks, vert=False, showextrema=False, showmedians=False) #, positions=np.arange(0, input[0].size(2)))
			#parts = sns.violinplot(ks, linewidth=0.5, color="#777575") #, orient="v") #, showextrema=False, showmedians=False) #, positions=np.arange(0, input[0].size(2)))
			parts = sns.violinplot(x=softmax, linewidth=0.5, color="#777575") #, orient="v") #, showextrema=False, showmedians=False) #, positions=np.arange(0, input[0].size(2)))
			ax.set_xlim(0, set_size)
			spaces=50
			#max_tick=round(input[0].size(2)/50)*50 + 1
			ax.set_xticks(np.arange(0, set_size, spaces))
			ax.set(xticklabels=[]) 
			if head == 1:
				ax.set_ylabel("Layer {}".format(str(layers_i + 1)))
			if layers_i == 0: # first layer
				ax.set_xticklabels(np.arange(0, set_size, spaces), rotation = 45)
				ax.set_xlabel("k patterns")
			if layers_i == layers - 1: # last layer
				ax.set_title("Head {}".format(str(head)))
			#ax.text(N*0.4, 0.8, s=str(int(np.median(ks))), fontweight='bold')
			ax.text(set_size*0.4, 0.3, s=str(int(np.median(softmax))), fontweight='bold')
			#ax.set_yticks([])
			#for pc in parts['bodies']:
		#		pc.set_facecolor('#A5A3A3') # grey
		#		pc.set_edgecolor('black')
	#			pc.set_alpha(1)
			if 1 / 2 * set_size < np.median(softmax):
				color = "#F4B8B5" # red

			elif 1 / 8 * set_size < np.median(softmax) and np.median(softmax) <= 1 / 2 * set_size:
				#color = "#D9A252" # orange
				color = "#FFE3C8" # orange
			elif 1 / 32 * set_size < np.median(softmax) and np.median(softmax) <= 1 / 8 * set_size:
				#color = "#7DC159" # green
				color = "#B9EBB3" # green
			elif np.median(softmax) <= 1 / 32 * set_size:
				#color = "#4AB7E6" # blue
				color = "#ABD8E4" # blue
			ax.set_facecolor(color)

			head += 1
			plot_i += 1
	#fig.tight_layout()
	plt.tight_layout()
	plt.show()
	return(fig)


def plot_softmax_head(input):
	print()