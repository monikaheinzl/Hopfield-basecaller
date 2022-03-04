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

filename_lst="ecoli Lambda" # Lambda"
threads=20

file_appendix="fasta"

modelname="final_model"
fastqdir="/training_result_"$modelname"/assembly_beam3"
result_dir="/training_result_"$modelname"/assembly_beam3"

for filename in $filename_lst
do
	echo ${filename}
	if [ ${filename} == "ecoli" ]; then
		ref_file='ecoli.fasta'
		genome_size=4600k
	elif [ ${filename} == "Lambda" ]; then
		ref_file='lambda.fasta'
		genome_size=48k
	fi

	echo $fastqdir

	mkdir $result_dir/norm_unaligned_assembly2
	reference_dir="references/"$ref_file #"reference.fasta"
	read_alignment=$result_dir/${filename}"_"$out_name"_alignment.paf"
    out1=$result_dir"/norm_unaligned_assembly2/"${filename}"_"$out_name"_stats_read_alignment.txt"
    out2=$result_dir"/norm_unaligned_assembly2/"${filename}"_"$out_name"_stats_read_alignment_mean.txt"
    stats_pickle=$result_dir"/norm_unaligned_assembly2/"${filename}"_"$out_name"_stats_read_alignment.pickle"

    echo $result_dir 
	echo "assemble reads: flye..."
	printf "\n"
	mkdir $result_dir/assembly_${filename}
	flye --nano-raw $result_dir/${filename}.$file_appendix -o $result_dir/assembly_${filename}/assembly -t $threads --meta --genome-size $genome_size #--asm-coverage 50

	python quast-2.2/quast.py $result_dir/assembly_${filename}/assembly/assembly.fasta \
        -R $reference_dir \
        -o $result_dir/assembly_${filename}/quast

	echo "chop assembly into pieces"
	printf "\n"
	python chop_up_assembly.py $result_dir/assembly_${filename}/assembly/assembly.fasta 10000 > $result_dir/assembly_${filename}/assembly_piece.fasta

	echo "reads alignment: minimap2..."
	printf "\n"
	/system/user/minimap2/minimap2 -x asm5 -t $threads --cs $reference_dir $result_dir/assembly_${filename}/assembly_piece.fasta > $result_dir/assembly_${filename}/assembly_piece.paf
	python sam_report.py -i $result_dir/assembly_${filename}/assembly_piece.paf -f $result_dir/assembly_${filename}/assembly_piece.fasta -o1 $out1 -o2 $out2 -p $stats_pickle -n "None"
done