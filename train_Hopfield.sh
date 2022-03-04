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

infile="/system/user/Documents/TrainValidationTestset.pickle"

port=3 
seed=14455
batch_size=32 
nepochs=250
make_val=5000 

max_window_size=2048
max_target_size=500

####### CNN
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

# dropout
dropout_cnn=False #False
dropout_input=False
drop_prob=0.0

input_bias_cnn=False

####### Hopfield
dff=1024 
hidden_units=64
lstm_layers=5
dropout_pos=0.1
drop_transf=0.1
nhead=8

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

####### Other
weight_decay=0.0
lr=0.0001
decrease_lr=True
warmup_steps=10000
gradient_clip=1 

# early stopping
early_stop=True
patience=30
editD=True
plot_weights=False

# if you have already a trained model and want to continue training this model
continue_training=False
modelname="model_continue"
model_file="/system/user/Documents/training_result_"$modelname"_checkpoint.pt"
config_file="/system/user/Documents/training_result_"$modelname"/model.json"


outfolder="final_model_CTC"
python Training_Hopfield.py -i $infile -o $outfolder -g $port -s $seed -b $batch_size -e $nepochs -v $make_val --max_window_size $max_window_size --max_target_size $max_target_size \
												--input_bias_cnn $input_bias_cnn --channel_number $c1 $c2 $c3 --padding $p1 $p2 $p3 -l $cnn_layers --pooling_type $pooling --strides $s1 $s2 $s3 --kernel $k1 $k2 $k3 \
												--dropout_cnn $dropout_cnn --dropout_input $dropout_input --drop_prob $drop_prob --src_emb $src_emb --batch_norm $batch_norm \
												--input_bias_hopfield $input_bias_hopfield -u $hidden_units --dff $dff --lstm_layers $lstm_layers --nhead $nhead --drop_transf $drop_transf --dropout_pos $dropout_pos \
												--scaling $scaling --pattern_projection_as_connected $pattern_projection_as_connected \
												--normalize_stored_pattern $normalize_stored_pattern --normalize_stored_pattern_affine $normalize_stored_pattern_affine \
												--normalize_state_pattern $normalize_state_pattern --normalize_state_pattern_affine $normalize_state_pattern_affine \
												--normalize_pattern_projection $normalize_pattern_projection --normalize_pattern_projection_affine $normalize_pattern_projection_affine \
												--stored_pattern_as_static $static --state_pattern_as_static $static --pattern_projection_as_static $static \
												--weight_decay $weight_decay --learning_rate $lr --decrease_lr $decrease_lr --warmup_steps $warmup_steps --gradient_clip $gradient_clip \
												--early_stop $early_stop --patience $patience --editD $editD --plot_weights $plot_weights \
												--continue_training $continue_training --model_file $model_file --config_file $config_file
