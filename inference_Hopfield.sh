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

infile="/system/user/Documents/raw_fast5_folder"
modelfolder="training_result_final_model/final_model.pt"
configfile="training_result_final_model/model.json"
outfolder="training_result_final_model"

port=1 
seed=14455
beam_size=3 

output_name="result_basecalling"

max_window_size=2048
max_target_size=500 

validation=False

# CNN
channels=256
c1=$channels 
c2=$channels
c3=$channels

cnn_layers=3
pooling="None"
batch_norm=True 
src_emb="cnn" 

# strides
s1=2 
s2=2
s3=2

# kernel
k1=11 
k2=3
k3=11 

#padding
p1=0 
p2=0
p3=0 

input_bias_cnn=False

####### Hopfield
dff=1024 
hidden_units=64
lstm_layers=5
nhead=8

# Param Hopfield
scaling="None"
pattern_projection_as_connected=False
normalize_stored_pattern=True
normalize_stored_pattern_affine=True
normalize_state_pattern=True
normalize_state_pattern_affine=True
normalize_pattern_projection=True
normalize_pattern_projection_affine=True
static=False
input_bias_hopfield=False

python inference_Hopfield.py -i $infile -m $modelfolder -o $outfolder -config $configfile -b $beam_size -g $port -s $seed --max_window_size $max_window_size --max_target_size $max_target_size \
												--input_bias_cnn $input_bias_cnn --channel_number $c1 $c2 $c3 --padding $p1 $p2 $p3 -l $cnn_layers --pooling_type $pooling --strides $s1 $s2 $s3 --kernel $k1 $k2 $k3 \
												--src_emb $src_emb --batch_norm $batch_norm \
												--input_bias_hopfield $input_bias_hopfield -u $hidden_units --dff $dff --lstm_layers $lstm_layers --nhead $nhead\
												--scaling $scaling --pattern_projection_as_connected $pattern_projection_as_connected \
												--normalize_stored_pattern $normalize_stored_pattern --normalize_stored_pattern_affine $normalize_stored_pattern_affine \
												--normalize_state_pattern $normalize_state_pattern --normalize_state_pattern_affine $normalize_state_pattern_affine \
												--normalize_pattern_projection $normalize_pattern_projection --normalize_pattern_projection_affine $normalize_pattern_projection_affine \
												--stored_pattern_as_static $static --state_pattern_as_static $static --pattern_projection_as_static $static --validation $validation --output_name $output_name


