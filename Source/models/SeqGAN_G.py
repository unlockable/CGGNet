# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_G.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config as cfg
import torch.jit as jit
from torch import nn, Tensor
from typing import Tuple, Optional, List
from torch.jit import script_method
from torch import Tensor
from torch.nn import Parameter, ModuleList


from models.generator import LSTMGenerator

class SeqGAN_G(nn.Module):    
    def __init__(self, embedding_dim, num_layers, hidden_dim, vocab_size, max_seq_len, padding_idx, word2idx_dict, batch_size, gpu=False):
        super(SeqGAN_G, self).__init__()
        self.name = 'seqgan'

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu
        
        self.num_layers = num_layers

        self.temperature = 1.0

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=0.5)
        
        self.word2idx_dict = word2idx_dict
        self.scripted_custom_lstm = jit.script(self.custom_lstm)

        self.lstm_default = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.contract_version = None

        self.variable = False
        
        solidity_version_str = ["pragma", "solidity"]
    
        self.batch_one_hot_v = torch.zeros(batch_size, self.vocab_size, dtype=torch.float)
        self.batch_version_v = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float)
        self.declared_flag = False
        self.version_flag = False
        self.last_pair_tokens = torch.zeros(batch_size, dtype=torch.int)
        self.init_params()
        self.seqlen = 0
   
    
    def get_version_in_sentence(self, batch_codes) :
        if batch_codes.shape[1] < 4 :  # <sos> pragma solidity
            return
        
        # check if the first two columns of batch_code match with solidity_version_str_inds
        mask = (self.solidity_version_str_inds_tensor[0] == batch_codes[:, 1]) & (self.solidity_version_str_inds_tensor[1] == batch_codes[:, 2])
        
        # create a result tensor of the same length filled with 0
        result = torch.zeros_like(batch_codes[:, 3])
        
        # assign batch_code[:, 3] where the mask is True, leave it as 0 otherwise
        result[mask] = batch_codes[:, 3][mask]
        
        return result[mask]


    def forward(self, inp, hidden, train_step=False, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        '''
        if train_step == True:
            versions = self.get_version_in_sentence(inp)
            self.batch_version_v = self.embeddings(versions)
        
        if train_step == False and self.seqlen == 3 :
            self.batch_version_v = emb
        '''
        
        if train_step :            
            self.check_solidity_types_in_sentence(inp)        
        else :
            if len(inp.shape) < 2:
                inp = inp.unsqueeze(1)
                emb = emb.unsqueeze(1)
            self.check_solidity_types_in_words(inp[:, -1:])  
        #pragma solidity
        #test = self.get_paired_tokens(inp)
        
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim
        
        out, hidden = self.lstm_default(emb, hidden)
        #out, hidden = self.lstm_multi_layer(emb,self.batch_one_hot_v, hidden) 이거 최종
        #out, hidden = self.lstm_multi_layer2(emb, hidden)
        #out, hidden = self.scripted_custom_lstm(emb, self.batch_one_hot_v, hidden)  # out: batch_size * seq_len * hidden_dim
        #out, hidden = self.scripted_custom_lstm(emb, hidden)  
        #out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        
        
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)
        
        #마지막 word에서 zero로 초기화를 시켜줘야함
        self.seqlen += 1
        
        if need_hidden:
            return pred, hidden
        else:
            return pred
        
    def init_variable(self, size):    
        self.batch_one_hot_v = torch.zeros(size, self.vocab_size, dtype=torch.float)
        self.seqlen = 0
        return
            

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        print("sampling")
        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, train_step=False, need_hidden=True)
                    
                #out, hidden = self.forward(inp, hidden, init_variable=False, need_hidden=True)  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
            self.init_variable(inp.shape[0])
        samples = samples[:num_samples]

        return samples
    
    def vulnerable_sample(self, num_samples, batch_size, sentence, start, end, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        samples = sentence * batch_size

        return samples


    
    def init_hidden_with_version(self, batch_size, versions):        
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)


        # Ensure nationality_indices is a tensor and is on the same device as the model
        version_indices = versions.clone().detach()
        
        # Get the embeddings for the batch
        version_vector = self.embeddings(version_indices)
        
        # Initialize all layers of h with the feature vectors
        for layer in range(self.num_layers):
            h[layer, :, :] = version_vector

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c

        
    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        
        loss = -torch.sum(pred * reward)

        return loss
    '''
    def masked_batchPGLoss(self, inp, target, reward, reward_threshold=1.0):
        """
        Computes the policy gradient loss with focus on high-reward tokens, while still maintaining a gradient path for low-reward tokens.
        :param inp: Tensor of shape (batch_size x seq_len), inp should be target with the start symbol prepended
        :param target: Tensor of shape (batch_size x seq_len), the target sequence tokens
        :param reward: Tensor of shape (batch_size x seq_len), discriminator reward for each token in each sentence
        :return: Scalar, the policy loss
        """
        
        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)
        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()
        pred = torch.sum(out * target_onehot, dim=-1)
        
        # Scale low-reward tokens
        scaling_factor = 1e-6
        scaled_reward = reward + scaling_factor * (1 - reward)
        
        # Compute loss considering the scaling factor
        loss = -torch.sum(pred * scaled_reward)
        
        return loss
    '''
    def masked_batchPGLoss(self, inp, target, reward, reward_threshold=1.0):
        """
        Returns a policy gradient loss
    
        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size x seq_len (discriminator reward for each token in each sentence)
        :return loss: policy loss
        """
    
        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)
    
        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
    
        # Center the rewards around their mean
        mean_reward = torch.mean(reward)
        centered_reward = reward - mean_reward
    
        # Compute the loss using the centered rewards
        loss = -torch.sum(pred * centered_reward)
    
        return loss

            
    def init_hidden(self, batch_size=cfg.batch_size):        
        
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c
        
    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

