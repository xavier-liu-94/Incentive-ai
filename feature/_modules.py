import torch.nn as nn
import torch.nn.functional as F
import torch
from ._contextual_mapping import *


class SequenceMappingLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, dim_x, dim_mid, head_num, hidden_dim, dropout=0.1):
        super(SequenceMappingLayer, self).__init__()
        self.slf_attn = NormalizedContextualMapping(dim_x, dim_x, dim_mid, head_num, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_x, hidden_dim, dropout=dropout)

    # x -> batch, seq_len, dim_x
    # x_mask -> batch, seq_len(1), seq_len
    # return -> batch, seq_len, dim_x
    def forward(self, x, x_mask=None):
        enc_output = self.slf_attn(x, x, mask=x_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class ConditionSequenceMappingLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, dim_x, dim_condition, dim_mid, head_num, hidden_dim, dropout=0.1):
        super(ConditionSequenceMappingLayer, self).__init__()
        self.slf_mapping = NormalizedContextualMapping(dim_x, dim_x, dim_mid, head_num, dropout=dropout)
        self.relation_mapping = NormalizedContextualMapping(dim_x, dim_condition, dim_mid, head_num, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_x, hidden_dim, dropout=dropout)
    
    # x -> batch, seq_len, dim_x
    # condition -> batch, condition_seq_len, dim_y
    # x_mask -> batch, seq_len, seq_len
    # condition_mask -> batch, seq_len(1), condition_seq_len
    # return -> batch, seq_len, dim_x
    def forward(self, x, condition, x_mask=None, condition_mask=None):
        dec_output = self.slf_mapping(x, x, mask=x_mask)
        dec_output = self.relation_mapping(x, condition, mask=condition_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class NormalizedContextualMapping(nn.Module):

    def __init__(self, dim_x, dim_context, dim_mid, head_num, dropout=0.1):
        super().__init__()

        self.n_head = head_num

        self.context_map = ContextualMapping(
            dim_x=dim_x, 
            dim_context=dim_context, 
            dim_mid=dim_mid, 
            head_num=head_num, 
            pe_mapping_factory=lambda: LinearSoftmax(dim_x, dim_context, dim_mid, head_num, temperature=dim_mid ** 0.5,attn_dropout=dropout))
        
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)


    def forward(self, x, context, mask=None):
        # x -> batch, n_x, dim_x
        # context -> batch, n_y, dim_y
        # mask -> batch, n_x, n_y
        # return -> batch, n_x, dim_x
        q = self.context_map(x, context, mask)
        return self.layer_norm(q)


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class GradNorm(nn.Module):

    def __init__(self, moving_value=1e-4) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))
        self.register_full_backward_hook(self.backward)
        self.moving_value = moving_value

    def forward(x):
        return x
    
    @staticmethod
    def backward(module, grad_input, grad_output):
        mean, std = torch.std_mean(grad_output)
        module.scale = (1-module.moving_value) * module.scalar + module.moving_value * std
        module.bias = (1-module.moving_value) * module.bias + module.moving_value * mean
        return (module.scale * grad_output[0] + module.bias, )
