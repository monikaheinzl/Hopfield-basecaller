#!/usr/bin/env python3
"""
Copyright 2019 Ryan Wick (rrwick@gmail.com)
https://github.com/rrwick/Basecalling-comparison

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. This program is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. You should
have received a copy of the GNU General Public License along with this program. If not, see
<http://www.gnu.org/licenses/>.

This script takes MUMmer SNPs as input (via stdin) and outputs a summary of the assembly's errors.
Use it like this:

prefix=ref_vs_assembly
ref_fasta=ref.fasta
assembly_fasta=assembly.fasta
ref_contig=chromosome
assembly_contig=tig00000001

nucmer --prefix="$prefix" "$ref_fasta" "$assembly_fasta"
delta-filter -r -q "$prefix".delta > "$prefix".filter
rm "$prefix".delta
show-snps -ClrTH -x5 "$prefix".filter | python3 error_summary.py "$ref_contig" "$assembly_contig"
"""

import collections
import math
import sys


homo_lengths_to_count = [3, 4, 5, 6, 7, 8]


def main():
    r_contig = str(sys.argv[1])
    a_contig = str(sys.argv[2])

    r_length, alignment_length = None, None
    snp_count = 0
    homo_ins_count, non_homo_ins_count = 0, 0
    homo_del_count, non_homo_del_count = 0, 0
    homo_ins_counts = collections.defaultdict(int)
    homo_del_counts = collections.defaultdict(int)
    sub_count = 0
    dcm_count, no_motif_count = 0, 0

    error_type_counts = collections.defaultdict(int)
    error_type_counts_mean = collections.defaultdict(int)
    
    old_r_contig = None
    old_a_contig = None

    n_contig = 1
    contig_length = 0
    reference_length = 0
    for line in sys.stdin:
        parts = line.strip().split('\t')

        if str(sys.argv[1]) != "None" or str(sys.argv[2]) != "None":
            if parts[12] != r_contig or parts[13] != a_contig:
                continue   
        else:
            r_contig = parts[12]
            a_contig = parts[13]

        if r_length is None:
            r_length = int(parts[7])
            alignment_length = r_length
        snp_count += 1
        r_base = parts[1]
        a_base = parts[2]
        ref_seq = parts[8]


        error_type = get_error_type(r_base, a_base, ref_seq)

        if old_a_contig is not None and old_r_contig is not None and (old_a_contig != a_contig or old_r_contig != r_contig): # new contig
            for error_type in ['dcm', 'homo del', 'homo ins', 'other del', 'other ins', 'sub']:

                error_type_counts_mean[error_type] += (error_type_counts[error_type] / r_length)
                error_type_counts[error_type] = 0 # reset to zero
            r_length = int(parts[7])
            contig_length += r_length
            reference_length += int(parts[6])
            n_contig += 1 # count number of contigs
        else:
            error_type_counts[error_type] += 1

        old_a_contig = a_contig
        old_r_contig = r_contig
    #print("contig length", contig_length)
    #print("ref length", reference_length)
    #print(n_contig)
    print('{}'.format('\t'.join(['dcm', 'homo del', 'homo ins', 'other del', 'other ins', 'sub'])))
    for error_type in ['dcm', 'homo del', 'homo ins', 'other del', 'other ins', 'sub']:
        

        if n_contig == 1: # only contig --> divide by length
            if error_type_counts[error_type] == 0:
                rate = error_type_counts[error_type]
            else:
                rate = error_type_counts[error_type] / r_length
        else: # take mean error rates of all contigs
            error_type_counts_mean[error_type] += (error_type_counts[error_type] / r_length)   
            
            rate = error_type_counts_mean[error_type] #/ n_contig
            #print("#contigs", n_contig, error_type, error_type_counts_mean[error_type])
        end_char = '\n' if error_type == 'sub' else '\t'
        print('{:.7f}'.format(rate), end=end_char)


def get_deletion_homopolymer_length(seq):
    seq_len = len(seq)
    middle_i = seq_len // 2
    base = seq[middle_i]
    start = middle_i
    while start > 0:
        if seq[start-1] == base:
            start -= 1
        else:
            break
    end = middle_i+1
    while end < seq_len:
        if seq[end] == base:
            end += 1
        else:
            break
    homopolymer = seq[start:end].replace('.', '')
    return len(homopolymer)


def get_insertion_homopolymer_length(seq):
    seq_len = len(seq)
    middle_i = seq_len // 2
    base_1 = seq[middle_i-1]
    base_2 = seq[middle_i+1]
    if base_1 != base_2:
        return 1
    start = middle_i
    while start > 0:
        if seq[start-1] == base_1:
            start -= 1
        else:
            break
    end = middle_i+1
    while end < seq_len:
        if seq[end] == base_1:
            end += 1
        else:
            break
    homopolymer = seq[start:end].replace('.', '')
    return len(homopolymer)


def error_in_dcm_motif(seq):
    seq = seq[1:-1]
    seq = seq.replace('.', '')
    return ('CCAGG' in seq) or ('CCTGG' in seq)


def get_error_type(r_base, a_base, ref_seq):
    if error_in_dcm_motif(ref_seq):
        return 'dcm'
    if a_base == '.' and get_deletion_homopolymer_length(ref_seq) >= 3:
        return 'homo del'
    if r_base == '.' and get_insertion_homopolymer_length(ref_seq) >= 3:
        return 'homo ins'
    if a_base == '.':  # deletion
        return 'other del'
    if r_base == '.':  # insertion
        return 'other ins'
    return 'sub'


if __name__ == '__main__':
    main()
