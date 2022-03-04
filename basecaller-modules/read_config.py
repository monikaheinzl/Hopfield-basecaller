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

def read_configuration_file(dict_param, type_m="embedding"):
	param_emb = dict_param["embedding"]
	param_tf = dict_param["transformer"]
	if type_m == "embedding":
		return param_emb["SEED"], param_emb["embedding_type"], param_emb["layer"], param_emb["channel"], param_emb["kernel"], param_emb["padding"], param_emb["stride"], \
			param_emb["batch_norm"], param_emb["pooling"], param_emb["dropout"], param_emb["input_prob"], param_emb["output_prob"], param_emb["input_bias_cnn"]
	elif type_m == "transformer":
		return param_tf["out_classes"], param_tf["layer"], param_tf["hidden_units"], param_tf["heads"], param_tf["dim_ff"], param_tf["dropout"], param_tf["dropout_pos"],\
			param_tf["max_window_size"], param_tf["max_target_size"], param_tf["beta"], param_tf["pattern_projection_as_connected"], \
			param_tf["normalize_stored_pattern"], param_tf["normalize_stored_pattern_affine"], \
			param_tf["normalize_state_pattern"], param_tf["normalize_state_pattern_affine"], \
			param_tf["normalize_pattern_projection"], param_tf["normalize_pattern_projection_affine"], param_tf["input_bias_hopfield"]
