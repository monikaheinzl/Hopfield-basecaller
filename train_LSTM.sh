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

port=0
seed=13824583
batch_size=32

max_window_size=2048
nepochs=250
make_val=5000

####### CNN
channels=256
cnn_layers=3
pooling="None" #"average"
batch_norm=True #False
# strides
s1=1
s2=2
s3=2
# kernel
k1=11
k2=11
k3=11
# dropout
dropout_cnn=False #False
dropout_input=False
drop_prob=0
input_bias_cnn=False

####### LSTM
hidden_units=256
lstm_layers=3
forget_bias_encoder=None
forget_bias_decoder=None
bi_lstm=True # True
attention=True
dropout=0.1

####### Other
# teacher forcing
reduced_tf=True # True
tf_ratio=1

weight_decay=0
lr=0.0001
gradient_clip='None'

# early stopping
early_stop=True
patience=30
editD=True

# inference
call=False

outfolder="final_model_LSTM"
python Training_LSTM.py -i $infile -o $outfolder -g $port -s $seed -b $batch_size -e $nepochs -v $make_val \
								-c $channels -l $cnn_layers --pooling_type $pooling --strides $s1 $s2 $s3 --kernel $k1 $k2 $k3 \
								--dropout_cnn $dropout_cnn --dropout_input $dropout_input --drop_prob $drop_prob --batch_norm $batch_norm \
								-u $hidden_units --lstm_layers $lstm_layers --forget_bias_encoder $forget_bias_encoder --forget_bias_decoder $forget_bias_decoder --bi_lstm $bi_lstm \
								--reduced_tf $reduced_tf --tf_ratio $tf_ratio --weight_decay $weight_decay --learning_rate $lr --gradient_clip $gradient_clip \
								--early_stop $early_stop --patience $patience --call $call --editD $editD --attention $attention --dropout $dropout --input_bias_cnn $input_bias_cnn

