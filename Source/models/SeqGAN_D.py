# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : SeqGAN_D.py
# @Time         : Created at 2019-04-25
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

from models.discriminator import CNNDiscriminator
from models.generator import LSTMGenerator

import torch
import torch.nn.functional as F

dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


class SeqGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(SeqGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                       dropout)




class SeqGAN_D_LSTM(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SeqGAN_D_LSTM, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)

    def get_pred(self, input, target):
        pred = self.forward(input, self.init_hidden(input.size(0)))
        target_onehot = F.one_hot(target.view(-1), self.vocab_size).float()
        pred = torch.sum(pred * target_onehot, dim=-1)
        return pred
