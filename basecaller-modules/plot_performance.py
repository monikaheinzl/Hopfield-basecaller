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
            if matrix_param[p] is None:
                continue
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

    y_len = [".".join(x.split(".")[1:]) \
                         if "tf" in x  \
                        else ".".join(x.split("."))
                        for x in y_len]
    df = pd.DataFrame(input, index=y_len, columns=x_len)
    print(df.head())
    sns.set(font_scale=0.2)
    svm = sns.heatmap(df, linewidths=0.0, edgecolor="none")
    figure = svm.get_figure()
    figure.savefig(save_files_path + "/heatmap_{}.pdf".format(filename))
    plt.clf()

    if split_LSTMbiases:
        y_len_biases = [".".join([x.split(".")[0], x.split(".")[1]]) for x in y_len_biases]           
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
