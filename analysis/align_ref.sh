#!/bin/bash
:'
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
'

filename_lst="ecoli Lambda"
threads=20

file_appendix="fasta"

modelname="final_model"
fastqdir="/training_result_"$modelname"/assembly_beam3"
result_dir="/training_result_"$modelname"/assembly_beam3"

out_name="final"
for filename in $filename_lst
do
	echo ${filename}
	if [ ${filename} == "ecoli" ]; then
		ref_file='ecoli.fasta'
		
	elif [ ${filename} == "Lambda" ]; then
		ref_file='lambda.fasta'
	fi

	mkdir $result_dir/norm_unaligned
	reference_dir="references/"$ref_file 
	read_alignment=$result_dir/${filename}"_"$out_name"_alignment.paf"
    out1=$result_dir"/norm_unaligned/"${filename}"_"$out_name"_stats_read_alignment.txt"
    out2=$result_dir"/norm_unaligned/"${filename}"_"$out_name"_stats_read_alignment_mean.txt"
    stats_pickle=$result_dir"/norm_unaligned/"${filename}"_"$out_name"_stats_read_alignment.pickle"

	echo "reads alignment: minimap2..."
	printf "\n"
	/system/user/minimap2/minimap2 -cx map-ont -t $threads --cs $reference_dir $result_dir/${filename}.$file_appendix > $read_alignment
	python sam_report.py -i $read_alignment -f $result_dir/${filename}.$file_appendix -o1 $out1 -o2 $out2 -p $stats_pickle -n "None"
done