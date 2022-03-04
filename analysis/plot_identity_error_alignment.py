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

from __future__ import division

import argparse
import itertools
import json
import operator
import os
import re
import sys
import pickle
import math
from distutils.util import strtobool

import numpy as np
import pysam
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from polyleven import levenshtein
from Bio import SeqIO
import seaborn as sns
import pandas as pd
from scipy import stats

###### Usage
#python plot_identity_error_alignment_normUnal.py -i basecaller1/norm_unaligned_assembly_polished basecaller2/norm_unaligned_assembly_polished basecaller3/norm_unaligned_assembly_polished -l basecaller1 basecaller2 basecaller3 -o outfolder -p appendix_outputname
#

def safe_div(x, y):
    if y == 0:
        return None
    return x / y

plt.rcParams["patch.force_edgecolor"] = False 

def plot_error_identity(df, pdf=None):
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(13,2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 2, 1)
    sns.barplot(x="basecaller", hue="genome", y="error", data=df, linewidth=0, ax=ax)
    plt.xlabel("Basecallers")
    plt.ylabel("Error")
    plt.title("Error rate of aligned reads to reference genome")
    ax.get_legend().remove()
    
    ax = fig.add_subplot(1, 2, 2)
    sns.barplot(x="basecaller", hue="genome", y="identity", data=df, linewidth=0, ax=ax)
    plt.xlabel("Basecallers")
    plt.ylabel("Identity")
    plt.title("Identity rate of aligned reads to reference genome")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

def plot_match_mismatch_indels(df, pdf=None, stacked=True):
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(10,5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.2)

    if stacked:
        ax = fig.add_subplot(2, 2, 1)
        ax2 = ax.twiny()
        #sns.barplot(x="basecaller", hue="genome", y="match", data=df, linewidth=0, ax=ax)
        #plt.xlabel("Basecallers")
        #plt.ylabel("%Matches")
        #plt.title("Matches")
        df0 = df[['basecaller', 'genome', 'mismatch', 'deletion', 'insertion', 'unaligned']]
        cols = df0.columns
        u, idx = np.unique(df.basecaller.tolist(), return_index=True)
        order = u[np.argsort(idx)] #[u[index] for index in sorted(idx)]
        df0['basecaller'] = pd.Categorical(df0.basecaller, categories=order, ordered=True) # ['f', 'a', 'w', 'h']  # prevent sorting 
        df0.set_index(['basecaller', 'genome'], inplace=True)
        colors = plt.cm.Paired.colors
        df1 = df0.unstack(level=-1) # unstack the 'Context' column
        (df1['mismatch']+df1['deletion']+df1['insertion']+df1['unaligned']).plot(kind='bar', color=[colors[1], colors[0]], rot=0, ax=ax, linewidth=0)
        print(df1['mismatch']+df1['deletion']+df1['insertion']+df1['unaligned'])
        (df1['deletion']+df1['insertion']+df1['unaligned']).plot(kind='bar', color=[colors[3], colors[2]], rot=0, ax=ax, linewidth=0)
        (df1['insertion']+df1['unaligned']).plot(kind='bar', color=[colors[5], colors[4]], rot=0, ax=ax, linewidth=0)
        df1['unaligned'].plot(kind='bar', color=[colors[7], colors[6]], rot=0, ax=ax, linewidth=0)
        #legend_labels = [f'{val} ({context})' for val, context in df1.columns]
        ticks = []
        for r in range(df.shape[0]//2):
            ticks.append(r - 0.25)
            ticks.append(r + 0.05)
        ax.set_xticks(ticks)
        ax.set_xticklabels(['lambda', 'ecoli'] * (df.shape[0]//2), rotation=45, fontsize=8) 
        ax.grid(axis="x")
        legend_labels = []
        labels = ["mismatch", "", "deletion", "", "insertion", "", "unaligned", ""]
        #for val in labels:
       #    if val in legend_labels:
       #        legend_labels.append("")
       #    else:
       #        legend_labels.append(val)
        #legend_labels = [f'{val} ({context})' for val, context in df1.columns]3
        ax.legend(labels, bbox_to_anchor=(-0.08, 1.2), loc=2, borderaxespad=0., ncol=4, fontsize=10) #(1.05, 1)
        ax.set_ylabel("mean error in %")
        ax.set_xlabel("species")
        ax.set_yscale('log') #,base=20)

        #ax.text(0.02, -0.2, '      '.join(order), transform=ax.transAxes, fontsize=11) #horizontalalignment='center', verticalalignment='center'
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks([0.02, 1, 2, 3])
        ax2.set_xticklabels(order, fontsize=10)
        ax.xaxis.set_ticks_position('none') 
        #ax2.xaxis.set_ticks_position('none')
        ax2.grid(axis="x")

        #ax.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.ylabel("Proportion of errors")
    else:
        ax = fig.add_subplot(2, 2, 1)
        sns.barplot(x="basecaller", hue="genome", y="mismatch", data=df, linewidth=0, ax=ax)
        plt.xlabel("Basecallers")
        plt.ylabel("Mismatches")
        plt.title("Mismatches")
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xticks(fontsize=8)

        #ax._legend.remove()

        ax = fig.add_subplot(2, 2, 3)
        sns.barplot(x="basecaller", hue="genome", y="deletion", data=df, linewidth=0, ax=ax)
        plt.xlabel("Basecallers")
        plt.ylabel("Deletion")
        plt.title("Deletion")
        ax.get_legend().remove()
        plt.xticks(fontsize=8)

        #ax._legend.remove()

        ax = fig.add_subplot(2, 2, 4)
        sns.barplot(x="basecaller", hue="genome", y="insertion", data=df, linewidth=0, ax=ax)
        plt.xlabel("Basecallers")
        plt.ylabel("Insertion")
        plt.title("Insertion")
        ax.get_legend().remove()
        plt.xticks(fontsize=8)

        #ax._legend.remove()
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

def plot_boxplot(data, labels, pdf=None, title="relative read length", ylabel="read length / reference length in %", reference=None):
    sns.set(font_scale=1)
    fig = plt.figure(figsize=(6,2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=2, hspace=0.2, wspace=0.2)
    ax = fig.add_subplot(1, 1, 1)
    box = plt.boxplot(data, patch_artist=True)
    ticks = np.arange(1, len(labels)+1)
    plt.xticks(ticks, labels, rotation=45, ha="right")

    
    plt.ylabel(ylabel)
    plt.xlabel("Basecaller")
    plt.title(title)
    #plt.yscale('log') #,base=20)
    if reference is not None:
        plt.axhline(reference, c='r')

    colors = len(labels[-3:]) * ['#EAEAF2'] + 3* ["#88888C"]
    #colors2 = len(labels[-3:]) * ['#DD8655'] + 3* ["#181819"]

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        #med.set_facecolor(color2)

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
        plt.close("all")

def make_argparser():
    parser = argparse.ArgumentParser(description='Prints summary about alignment of basecalled reads.')
    parser.add_argument('-i', '--fastq', nargs="*",
                        help='FASTA/Q files with basecalled reads.')
    parser.add_argument('-l', '--labels', nargs="*",
                        help='list of labels. same order as list with fastq/a files')
    parser.add_argument('-o', '--out',
                        help='out path.')
    parser.add_argument('-p', '--prefix', default="basecalled",
                        help='out path.')
    parser.add_argument('--stacked', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=True,
                        help='stack error rates in plot.')
    return parser

def median_abs_dev(x):
    return(stats.median_absolute_deviation(x))

def report_errors(argv):
    parser = make_argparser()
    args = parser.parse_args(argv[1:])
    fastq = args.fastq
    basecallers = args.labels
    out = args.out
    prefix = args.prefix
    stacked = args.stacked
    
    with PdfPages(out + "/{}_error_alignment_rates.pdf".format(prefix)) as pdf:
        lambd = []
        ecoli = []
        df = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned', 'identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])
        df_std = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned','identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])
        df_all = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned', 'identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])
        df_all_std = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned','identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])


        df_median = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned', 'identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])
        df_std_median = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned','identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])
        df_all_median = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned', 'identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])
        df_all_std_median = pd.DataFrame(columns=['basecaller', 'genome', 'match', 'mismatch', 'deletion', 'insertion', 'unaligned','identity', 'error', 'mqual', 'relative read length', 'aligned \% of read'])

        i = 0
        rel_lengths = []
        qual_list = []
        error_list = []
        iden_list = []
        al_list = []
        m_list = []
        mm_list = []
        d_list = []
        i_list = []
        unal_list = []
        for directory, basecaller in zip(fastq, basecallers):
            lengths = []
            rel_l = []
            qual = []
            error = []
            iden = []
            al = []
            m = []
            mm = []
            d = []
            ins = []
            unal = []

            for name in os.listdir(directory):
                if name.endswith('alignment.txt'): # or name.endswith('alignment.txt'):
                    gen = name.split("_")[0]
                    data = pd.read_csv(directory + "/" + name, sep="\t", header=0)
                    mean = data.iloc[:, 4:].mean(axis = 0).values.tolist()
                    std = data.iloc[:, 4:].std(axis = 0).values.tolist()

                    median = data.iloc[:, 4:].median(axis = 0).values.tolist()
                    mad = stats.median_absolute_deviation(data.iloc[:, 4:], axis = 0)

                    l = [basecaller, gen]
                    l2= [basecaller, gen]
                    l.extend(mean)
                    df.loc[i] = l
                    df_all.loc[i] = l
                    l2.extend(std)
                    df_std.loc[i] = l2
                    df_all_std.loc[i] = l2

                    l_2 = [basecaller, gen]
                    l2_2= [basecaller, gen]
                    l_2.extend(median)
                    df_median.loc[i] = l_2
                    df_all_median.loc[i] = l_2
                    l2_2.extend(mad)
                    df_std_median.loc[i] = l2_2
                    df_all_std_median.loc[i] = l2_2



                    relative = data.loc[:, "relative read length"].values.tolist()
                    #relative = [i for i in relative if i != 0]
                    rel_l.extend(relative)
                    qual.extend(data.loc[:, "mqual"].values.tolist())
                    error.extend(data.loc[:, "error"].values.tolist())
                    iden.extend(data.loc[:, "identity"].values.tolist())
                    aligned = data.loc[:, "aligned percentage of reads"].values.tolist()
                    #unaligned = [i for i in unaligned if i != 0]
                    al.extend(aligned)
                    m.extend(data.loc[:, "match"].values.tolist())
                    mm.extend(data.loc[:, "mismatch"].values.tolist())
                    d.extend(data.loc[:, "deletion"].values.tolist())
                    ins.extend(data.loc[:, "insertion"].values.tolist())
                    unal.extend(data.loc[:, "unaligned"].values.tolist())
                    i += 1

            rel_lengths.append(rel_l)
            qual_list.append(qual)
            error_list.append(error)
            iden_list.append(iden)
            al_list.append(al)
            m_list.append(m)
            mm_list.append(mm)
            d_list.append(d)
            i_list.append(ins)
            unal_list.append(unal)

        for bi, b in enumerate(basecallers): 
            df_all.loc[len(df)+bi] = [b, "all", np.mean(np.array(m_list)[bi]), np.mean(np.array(mm_list)[bi]), np.mean(np.array(d_list)[bi]), \
            np.mean(np.array(i_list)[bi]), np.mean(np.array(unal_list)[bi]),  np.mean(np.array(iden_list)[bi]), np.mean(np.array(error_list)[bi]), \
            np.mean(np.array(qual_list)[bi]), np.mean(np.array(rel_lengths)[bi]), np.mean(np.array(al_list)[bi])]
            df_all_std.loc[len(df)+bi] = [b, "all", np.std(np.array(m_list)[bi]), np.std(np.array(mm_list)[bi]), np.std(np.array(d_list)[bi]), \
            np.std(np.array(i_list)[bi]), np.std(np.array(unal_list)[bi]),  np.std(np.array(iden_list)[bi]), np.std(np.array(error_list)[bi]), \
            np.std(np.array(qual_list)[bi]), np.std(np.array(rel_lengths)[bi]), np.std(np.array(al_list)[bi])]


            df_all_median.loc[len(df)+bi] = [b, "all", np.median(np.array(m_list)[bi]), np.median(np.array(mm_list)[bi]), np.median(np.array(d_list)[bi]), \
            np.median(np.array(i_list)[bi]), np.median(np.array(unal_list)[bi]),  np.median(np.array(iden_list)[bi]), np.median(np.array(error_list)[bi]), \
            np.median(np.array(qual_list)[bi]), np.median(np.array(rel_lengths)[bi]), np.median(np.array(al_list)[bi])]
            df_all_std_median.loc[len(df)+bi] = [b, "all", median_abs_dev(np.array(m_list)[bi]), median_abs_dev(np.array(mm_list)[bi]), median_abs_dev(np.array(d_list)[bi]), \
            median_abs_dev(np.array(i_list)[bi]), median_abs_dev(np.array(unal_list)[bi]),  median_abs_dev(np.array(iden_list)[bi]), median_abs_dev(np.array(error_list)[bi]), \
            median_abs_dev(np.array(qual_list)[bi]), median_abs_dev(np.array(rel_lengths)[bi]), median_abs_dev(np.array(al_list)[bi])]
        

        print(df_median)

        with pd.ExcelWriter('{}/error_alignment_rates.xlsx'.format(out)) as writer:  
            df_all.to_excel(writer, index = False, sheet_name='mean')
            df_all_std.to_excel(writer, index = False, sheet_name='standard deviation')
            df_all_median.to_excel(writer, index = False, sheet_name='median')
            df_all_std_median.to_excel(writer, index = False, sheet_name='median absolute deviation')
        plot_error_identity(df, pdf=pdf)
        plot_match_mismatch_indels(df, pdf=pdf, stacked=stacked)

        plot_boxplot(rel_lengths, basecallers, reference=100, pdf=pdf) #, title="relative read length", ylabel="read length / reference length")
        #plot_boxplot(qual_list, basecallers, reference=None, pdf=pdf, title="mean quality scores", ylabel="mean quality scores")
        plot_boxplot(error_list, basecallers, reference=None, pdf=pdf, title="error scores", ylabel="mismatches+indels / alignment + unalignment length") #, title="relative read length", ylabel="read length / reference length")
        plot_boxplot(iden_list, basecallers, reference=None, pdf=pdf, title="identity scores", ylabel="matches / alignment + unalignment length")
        plot_boxplot(unal_list, basecallers, reference=None, pdf=pdf, title="aligned percentage of reads", ylabel="alignment length / read length")

if __name__ == '__main__':
    sys.exit(report_errors(sys.argv))







