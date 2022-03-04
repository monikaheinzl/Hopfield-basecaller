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


import torch


class Beam:

    def __init__(self, beam_size=3, min_length=0, n_top=1, start_token_id=6, end_token_id=4, batch_size=1):
        self.beam_size = beam_size
        self.min_length = min_length
        self.batch_size = batch_size

        self.end_token_id = end_token_id
        self.top_sentence_ended = False

        self.prev_ks = []
        self.next_ys = [torch.LongTensor(beam_size).fill_(start_token_id)] # remove padding
        self.next_softmax = [torch.Tensor([0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1])] # remove padding
        #self.next_ys_softmax = [] 
        self.next_softmax_pred = [torch.Tensor([0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,1])]

        self.current_scores = torch.FloatTensor(beam_size).zero_()
        #self.all_scores = []

        # Time and k pair for finished.
        self.finished = []
        self.n_top = n_top
        #self.bestpath = [[]]*beam_size
        self.bestpath = [[start_token_id]]*beam_size


    #def advance(self, next_log_probs, current_attention):
    def advance(self, next_log_probs):
        # next_probs : beam_size X vocab_size
        vocabulary_size = next_log_probs.size(1)
        # current length of decoded sequence
        current_length = len(self.next_ys)
        #print("current_length", current_length, "min_length", self.min_length)

        # if current lenght smaller than minimum length --> set eos token to a very small value
        # which prevents that the eos is not selected in the current time step
        if current_length < self.min_length:
            for beam_index in range(len(next_log_probs)):
                next_log_probs[beam_index][self.end_token_id] = -10000e10

        #print(next_log_probs, len(self.prev_ks))
        #print("next_log_probs", next_log_probs)
        #print("current scores", self.current_scores)
        #print(self.current_scores.unsqueeze(1).expand_as(next_log_probs))
        # sum previous scores
        if len(self.prev_ks) > 0: # after first time step
            # expand current stored scores (seq_len=1, classes=5) to size of current probabilities (beam_size, classes=5) 
            # and add output probabilites to stored scores
            beam_scores = next_log_probs + self.current_scores.unsqueeze(1).expand_as(next_log_probs)
            # Don't let EOS have children.
            # if last stored class == eos token --> set to very small value
            # prevents that eos token is selected for next time step and sequence is extended
            last_y = self.next_ys[-1]
            for beam_index in range(last_y.size(0)):
                if last_y[beam_index] == self.end_token_id:
                    beam_scores[beam_index] = -1e10 # -1e20 raises error when executing
        else: # first time step, squeeze output to size (seq_len=1, classes=5)
            beam_scores = next_log_probs[0]
        #print("next_log_probs", next_log_probs)
        #print("beam scores", beam_scores)
        flat_next_log_probs = next_log_probs.view(-1)
        flat_beam_scores = beam_scores.view(-1)
        #print("flat_beam_scores", flat_beam_scores)

        # select top beam_size scores
        # top_score_ids = index of top k scores
        # if beam size > number of classes (usually in first time step) --> select all classes
        #print("self.beam_size", self.beam_size, "available beams", flat_beam_scores.size(0))
        if self.beam_size > flat_beam_scores.size(0):
            top_scores, top_score_ids = flat_beam_scores.topk(k=flat_beam_scores.size(0), dim=0, largest=True, sorted=True)
        # else --> topk classes
        else:
            top_scores, top_score_ids = flat_beam_scores.topk(k=self.beam_size, dim=0, largest=True, sorted=True)
        #print("top scores", top_scores, top_score_ids)

        # update scores with new ones (old + current scores)
        self.current_scores = top_scores
        #print("current_scores", self.current_scores)
        #self.all_scores.append(self.current_scores)

        # vocabulary_size = number of classes = 5 (A,G,C,T,<EOS>)
        # top_score_ids is flattened ((beam, vocabulary_size)  --> (beam x vocabulary_size, 1)), so calculate which word and beam each score came from
        # e.g. beam = 3, id = 9 // 5 = 1 --> from second beam (0, 1, 2)
        # id = 10 // 5 = 2 --> from third beam (0, 1, 2)
        # id = 0 // 5 = 0 --> from first beam (0, 1, 2)
        prev_k = top_score_ids // vocabulary_size  # (beam_size, )
        # (top_score_ids - prev_k * vocabulary_size) is the index of the best vocab in each Beam
        # e.g. id = 10, beam_id = 2
        # 10 - 2*5 = 0 = id in beam 3 (10 in all three beams)
        # 9 - 1*5 = 4 = id in beam 2 (9 in all three beams)
        # 0 - 0*5 = 0 = id in beam 1
        next_y = top_score_ids - prev_k * vocabulary_size  # (beam_size, )
        next_y_softmax = prev_k * vocabulary_size 
        #print("prev_k", prev_k, "next_y", next_y)
        # attach most probable class
        self.prev_ks.append(prev_k)
        # attach pointer of candidate with most probable class
        self.next_ys.append(next_y)
        #self.next_ys_softmax.append(top_score_ids)
        #print(next_y.size(), next_y,top_score_ids, flat_beam_scores.size())
        self.next_softmax.append(flat_beam_scores)
        self.next_softmax_pred.append(flat_next_log_probs)
        #print(self.next_softmax)

        bstpath = []
        for bst_i in range(len(prev_k)): # for beam size
            # add current token to bestpath
            # beam index = prev_k[bst_i].item()
            bstpath.append(self.bestpath[prev_k[bst_i].item()]+[next_y[bst_i].item()])
        self.bestpath = bstpath

        for beam_index, last_token_id in enumerate(next_y):
            if last_token_id == self.end_token_id: #if eos token = 4 reached
                # add scores to a list where the finished ones are stored (list of scores, length of sequence, beam_index)
                self.finished.append((self.current_scores[beam_index], len(self.next_ys) - 1, beam_index))

        if next_y[0] == self.end_token_id: # break loop, eos toke reached for top hypothesis (=hypothesis with highest probability)
            self.top_sentence_ended = True

    def get_current_state(self):
        "Get the outputs for the current timestep."
        #return self.next_ys[-1]
        return self.bestpath

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def done(self):
        # break loop if top candidate (candidate with highest probability) finished
        # and all candidates finished
        return self.top_sentence_ended and len(self.finished) >= self.n_top 

    def get_hypothesis(self, timestep, k):
        #print(self.next_softmax)
        #sys.exit()
        # do for one candidate sequence k
        hypothesis = []
        softmaxes = []
        softmaxes_pred = []
        #print("cand", k)
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1): # start with last time step
            #print(j, len(self.prev_ks[:timestep]) - 1)
            # get predicted classes, j+1 --> skip sos token
            # next_ys = (sequence length, beam_size), k = candidate 
            #print(self.next_ys[j + 1][k])
            hypothesis.append(self.next_ys[j + 1][k])
            
            if k == 0:
                k_start = k
                #k_end = k_start7
            elif k == 1:
                k_start = k + 7 - 1
                #k_end = k_start + 7
            elif k == 2:
                k_start = k + (2*7) - 2
            #print(self.next_ys[0][k], self.next_softmax[0], len(self.next_ys), len(self.next_softmax), j+1, k)
            #print(len(self.next_softmax[j + 1]), self.next_softmax[j + 1])
            #softmaxes.append( self.next_softmax[j + 1][k_start:k_start + 7])
            #softmaxes_pred.append( self.next_softmax_pred[j + 1][k_start:k_start + 7])

            if len(self.next_softmax[j + 1]) == 7:
                softmaxes.append( self.next_softmax[j + 1].cpu().tolist())
            elif len(self.next_softmax_pred[j + 1]) == 7:
                softmaxes_pred.append( self.next_softmax_pred[j + 1].cpu().tolist())
            else:
                softmaxes.append( self.next_softmax[j + 1][k_start:k_start + 7].cpu().tolist())
                softmaxes_pred.append( self.next_softmax_pred[j + 1][k_start:k_start + 7].cpu().tolist())

            k = self.prev_ks[j][k]
        #sys.exit()
        #print(softmaxes)
        return [hypothesis[::-1], softmaxes[::-1],softmaxes_pred[::-1]]

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # if no eos token finished is empty --> add minimum outputs to it
            # Add from beam until we have minimum outputs.
            # Get minimum candidates from beam search
            while len(self.finished) < minimum:
                s = self.current_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        # sort candidates in descanding order (candidate with highest prob first outputted)
        self.finished = sorted(self.finished, key=lambda a: a[0], reverse=True)
        #print("finished", self.finished)
        # get scores of all candidates
        scores = [sc for sc, _, _ in self.finished]
        # get length and index of candidates
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks