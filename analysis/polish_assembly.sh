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

out_name="final"

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

	mkdir $result_dir/norm_unaligned_assembly_polished
	reference_dir="references/"$ref_file #"reference.fasta"
	read_alignment=$result_dir/${filename}"_"$out_name"_alignment.paf"
    out1=$result_dir"/norm_unaligned_assembly_polished/"${filename}"_stats_assembly_alignment.txt"
    out2=$result_dir"/norm_unaligned_assembly_polished/"${filename}"_stats_assembly_alignment_mean.txt"
    out3=$result_dir"/norm_unaligned_assembly_polished/"${filename}"_mean_error_polished_assembly_sum.txt"
    stats_pickle=$result_dir"/norm_unaligned_assembly_polished/"${filename}"_stats_assembly_alignment.pickle"

    echo $result_dir
    ######## already done ###############
	##echo "assembly reads: flye..."
	##printf "\n"
    ##flye --nano-raw $result_dir/${filename}.$file_appendix -o $result_dir/assembly_polishing_${filename}/assembly -t $threads --asm-coverage 50 --genome-size $genome_size
    ######## already done ###############

	echo "polish genomes with 10 rounds: rebaler..."
	printf "\n"
	mkdir $result_dir/assembly_polishing_${filename}
	mkdir $result_dir/assembly_polishing_${filename}/rebaler
	Rebaler/rebaler-runner.py --threads $threads $result_dir/assembly_${filename}/assembly/assembly.fasta $result_dir/${filename}.$file_appendix > $result_dir/assembly_polishing_${filename}/rebaler/rebaler.fasta

	echo "construct consensus sequence: medaka..." # medake v1.2.0
	printf "\n"
    mkdir $result_dir/assembly_polishing_${filename}/medaka
    ## GPU
    TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES="0" medaka_consensus -b 50 -f -i $result_dir/${filename}.$file_appendix -d $result_dir/assembly_polishing_${filename}/rebaler/rebaler.fasta -o $result_dir/assembly_polishing_${filename}/medaka -t $threads -m r941_min_high_g360  # or -m r941_min_high, as appropriate

	# analyse polished assembly
	dnadiff -p dnadiff $reference_dir $result_dir/assembly_polishing_${filename}/medaka/consensus.fasta
	echo "analyze consensus sequence: mummer4..." 
	printf "\n"
	nucmer --prefix=$result_dir/assembly_polishing_${filename}/nucmer_${filename} $reference_dir $result_dir/assembly_polishing_${filename}/medaka/consensus.fasta --threads=1
    delta-filter -r -q $result_dir/assembly_polishing_${filename}/nucmer_${filename}.delta > $result_dir/assembly_polishing_${filename}/delta_${filename}.filter
    n_contigs=$(($(cat $result_dir/assembly_polishing_${filename}/medaka/consensus.fasta|wc -l) / 2)) #|bc) 
    show-snps -ClrTH -x5 $result_dir/assembly_polishing_${filename}/delta_${filename}.filter | python /analysis/error_summary.py "None" "None" > $out3
    fi

    echo "chop assembly into pieces"
	printf "\n"
	python chop_up_assembly.py $result_dir/assembly_polishing_${filename}/medaka/consensus.fasta 10000 > $result_dir/assembly_polishing_${filename}/medaka/consensus_piece.fasta

	echo "reads alignment: minimap2..."
	printf "\n"
	/system/user/minimap2/minimap2 -x asm5 -t $threads --cs $reference_dir $result_dir/assembly_polishing_${filename}/medaka/consensus_piece.fasta > $result_dir/assembly_polishing_${filename}/medaka/consensus_piece.paf
	python sam_report.py -i $result_dir/assembly_polishing_${filename}/medaka/consensus_piece.paf -f $result_dir/assembly_polishing_${filename}/medaka/consensus_piece.fasta -o1 $out1 -o2 $out2 -p $stats_pickle -n "None"

    echo "finished..." # medake v1.2.0
	printf "\n"
done