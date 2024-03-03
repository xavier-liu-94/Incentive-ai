import torch
from torch import nn
from ._modules import *


class SequenceFeature(nn.Module):

    def __init__(self, n_layers, head_num, dim_mid, dim_x, ff_hiden_dim, dropout=0.1):

        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            SequenceMappingLayer(dim_x, dim_mid, head_num, ff_hiden_dim, dropout=dropout)
            for _ in range(n_layers)])

    # x_seq_in -> batch, n_x, dim_x
    # x_seq_mask -> batch, n_x(1), n_x
    # return -> batch, n_x, dim_x
    def forward(self, x_seq_in, x_seq_mask):

        enc_output = self.layer_norm(x_seq_in)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, x_mask=x_seq_mask)

        return enc_output


class ConditionSequenceFeature(nn.Module):

    def __init__(self, n_layers, head_num, dim_mid, dim_x, dim_condition, ff_hidden_dim, dropout=0.1):

        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            ConditionSequenceMappingLayer(dim_x, dim_condition, dim_mid, head_num, ff_hidden_dim, dropout=dropout)
            for _ in range(n_layers)])
        
    # x_seq ->  batch, n_x, dim_x
    # x_seq_mask -> batch, n_x(1), n_x
    # condition_seq -> batch, n_y, dim_y
    # condition_seq_mask -> batch, n_x(1), n_y
    # return -> batch, n_x, dim_x
    def forward(self, x_seq, x_seq_mask, condition_seq, condition_seq_mask):
        # -- Forward
        dec_output = self.layer_norm(x_seq)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output, condition_seq, x_mask=x_seq_mask, condition_mask=condition_seq_mask)

        return dec_output