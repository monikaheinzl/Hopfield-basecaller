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

import numpy as np
import pysam
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from Bio import SeqIO
import pandas as pd

def safe_div(x, y):
    if y == 0:
        return None
    return x / y

def make_argparser():
    parser = argparse.ArgumentParser(description='Prints summary about alignment of basecalled reads.')
    parser.add_argument('-i', '--alignFile',
                        help='SAM or PAF file with aligned reads.')
    parser.add_argument('-f', '--fastq',
                        help='FASTA/Q file with basecalled reads.')
    parser.add_argument('-o1', '--out_reads',
                        help='TXT file with error rates of each read.')
    parser.add_argument('-o2', '--out_all',
                        help='TXT file with median error rates of all reads.')
    parser.add_argument('-p', '--out_pickle',
                        help='TXT file with error rates of each read.')
    parser.add_argument('-n', '--file_names',
                        help='TXT file with error rates of each read.')
    return parser


def report_errors(argv):
    parser = make_argparser()
    args = parser.parse_args(argv[1:])
    file = args.alignFile
    raw = args.fastq
    outfile_reads = args.out_reads
    outfile_all = args.out_all
    outpickle = args.out_pickle
    file_names = args.file_names
    read_count_mapped = 0
    read_count_unmapped = 0
    
    dict_reads = {}
    pattern = re.compile(r'(:[0-9]+|\*[a-z][a-z]|[=\+\-][A-Za-z]+)')

    if raw.endswith('.fastq'):
        file_type = "fastq"  
    elif raw.endswith('.fasta'):
        file_type = "fasta"

    if file_names != str(None):
        read_names = np.loadtxt(file_names, dtype=str)    
    
    raw_reads_name = []
    raw_reads_seq = {}    
    for record in SeqIO.parse(raw, file_type):
        raw_reads_name.append(record.id)
        raw_reads_seq[record.id] = record.seq

    aligned_reads_name = []
    
    with open(outfile_reads, "w") as out_read:
        out_read.write("Query name\tread length\talignment length\talignment+unaligned length\tmatch\tmismatch\tdeletion\tinsertion\tunaligned\tidentity\terror\tmqual\trelative read length\taligned percentage of reads\n")
        with open(file, 'rt') as paf:
            for line in paf:
                paf_parts = line.strip().split('\t')
                if ".fast5" in paf_parts[0]:
                    paf_parts[0] = paf_parts[0].split(".")[0]
                # if less than half of read algined --> skip
                #if float(paf_parts[10]) < int(paf_parts[1]) / 2 :
                #    continue
                if paf_parts[0] not in aligned_reads_name:
                    aligned_reads_name.append(paf_parts[0])
                cigar = paf_parts[-1]
                it = pattern.finditer(cigar)
                match = 0
                mismatch = 0
                insertion = 0
                deletion = 0
                stats = {"len": 0, "allen": 0, "allen + unaligned": 0, "match": 0, "mismatch": 0, "deletion": 0, "insertion": 0, "unaligned": 0, "identity": 0, "error": 0, "mqual": 0, "rel_length": 0, "aligned": 0}
                stats["len"] = paf_parts[1]
                stats["mqual"] = int(paf_parts[11])
                for matces in it:
                    if re.search(r":[0-9]+", matces.group()): # match
                        length = matces.group().split(":")[-1]
                        match += int(length)
                        #print(matces, length)
                    elif re.search(r"\*[a-z][a-z]", matces.group()): #mismatch
                        mismatch += 1
                        #print(matces, length)
                    elif re.search(r"\+[A-Za-z]+", matces.group()): #insertion
                        length = matces.group().split("+")[-1]
                        #print(matces.group().split("+"))
                        insertion += len(length)
                        #print(matces, len(length))
                    elif re.search(r"\-[A-Za-z]+", matces.group()): #deletion
                        length = matces.group().split("-")[-1]
                        #print(matces.group().split("+"))
                        deletion += len(length)
                        #print(matces, len(length))
                
                al_len = paf_parts[10]
                unaligned_bases = int(stats["len"]) - match - mismatch - insertion
                normalise = int(al_len) + unaligned_bases
                stats["allen"] = al_len
                stats["allen + unaligned"] = normalise
                stats["match"] = safe_div(match, float(normalise))
                stats["mismatch"] = safe_div(mismatch, float(normalise))
                stats["deletion"] = safe_div(deletion, float(normalise))
                stats["insertion"] = safe_div(insertion, float(normalise))
                stats["unaligned"] = safe_div(unaligned_bases, float(normalise))
                stats["identity"] = safe_div(int(paf_parts[9]), float(normalise))
                stats["error"] = stats["mismatch"] + stats["deletion"] + stats["insertion"] + stats["unaligned"]
                read_start = int(paf_parts[2])
                read_end = int(paf_parts[3])
                ref_start = int(paf_parts[7])
                ref_end = int(paf_parts[8])
                stats["rel_length"] = 100 * ((read_end - read_start) / float((ref_end - ref_start)))
                stats["aligned"] = 100 * (int(al_len) / float(paf_parts[1]))
                #print(match, mismatch, deletion, stats["identity"], stats["error"])
                read_count_mapped += 1
                if paf_parts[0] not in list(dict_reads.keys()):
                    dict_reads[paf_parts[0]] = stats
                    out_read.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    paf_parts[0], stats["len"], stats["allen"], stats["allen + unaligned"], stats["match"], stats["mismatch"],
                    stats["deletion"], stats["insertion"], stats["unaligned"], stats["identity"], stats["error"], stats["mqual"], stats["rel_length"], stats["aligned"]))
                    i = 1
                #else:
                #    i += 1
                #    dict_reads[paf_parts[0] + "#" + str(i)] = stats
                #    out_read.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                #    paf_parts[0] + "#" + str(i), stats["len"], stats["allen"], stats["match"], stats["mismatch"],
                #    stats["deletion"], stats["insertion"], stats["identity"], stats["error"], stats["mqual"], stats["rel_length"], stats["aligned"]))
        missing = list(set(raw_reads_name) - set(list(dict_reads.keys())))

        for ni, n in enumerate(missing):
            out_read.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        n, len(raw_reads_seq[n]), 0, len(raw_reads_seq[n]), 0, 0, 0, 0, 1, 0, 1, 0, 0, 0))

        if file_names != str(None):
            missing = list(set(read_names.tolist()) - set(raw_reads_name) )
            print("#filtered before bascalling", len(missing))
            for ni, n in enumerate(missing):
                out_read.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        n, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0))
    
    with open(outpickle, 'wb') as handle:
        pickle.dump(dict_reads, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    def avrg(lst): 
        return sum(lst) / len(lst)

    m = []
    mm = []
    ins = []
    dele = []
    iden = []
    err = []
    rel = []
    q = []
    un = []
    unal = []
    with open(outfile_all, "w") as out_all:
        out_all.write("match\tmismatch\tdeletion\tinsertion\tunaligned\tidentity\terror\tmqual\trelative read length\taligned percentage of reads\n")
        data = pd.read_csv(outfile_reads, sep="\t", header=0)
        mean = data.iloc[:, 4:].mean(axis = 0).values.astype(str).tolist()
        median = data.iloc[:, 4:].median(axis = 0).values.astype(str).tolist()
        unaligned = len(raw_reads_name) - len(aligned_reads_name)
        out_all.write("\t".join(mean) + "\n\n")
        out_all.write("\t".join(median) + "\n")
        #out_all.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(avrg(m), avrg(mm), avrg(ins), avrg(dele),  avrg(unal), avrg(iden), avrg(err), avrg(q), avrg(rel), avrg(un)))

        out_all.write("#total raw reads\t{}\n".format(len(raw_reads_name)))
        out_all.write("#mapped reads\t{}\n".format(len(aligned_reads_name)))
        out_all.write("#unmapped reads\t{}\n".format(len(raw_reads_name) - len(aligned_reads_name)))
        if file_names != str(None):
            out_all.write("#filtered before bascalling\t{}\n".format(len(missing)))
    
    print("#total raw reads= ", len(raw_reads_name))
    print("#mapped reads= ", len(aligned_reads_name))
    print("#unmapped reads= ", len(raw_reads_name) - len(aligned_reads_name))
    

if __name__ == '__main__':
    sys.exit(report_errors(sys.argv))

