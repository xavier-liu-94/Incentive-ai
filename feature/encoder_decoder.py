import torch
from torch import nn
from .embedding import *
from .module import *


class Encoder(nn.Module):

    def __init__(self, x_values_num, n_layers, head_num, dim_mid,
            dim_x, ff_hiden_dim, pad_idx, dropout=0.1, n_position=200):

        super().__init__()
        self.src_word_emb = nn.Embedding(x_values_num, dim_x, padding_idx=pad_idx)
        self.position_enc = SettledPositionalEncoding(dim_x, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            SequenceMappingLayer(dim_x, dim_mid, head_num, ff_hiden_dim, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)

    # x_seq -> batch, n_x
    # x_seq_mask -> batch, 1, n_x
    # return -> batch, n_x, dim_x
    def forward(self, x_seq, x_seq_mask):

        enc_output = self.src_word_emb(x_seq)
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, x_mask=x_seq_mask)

        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(self, x_values_num, n_layers, head_num, dim_mid,
            dim_x, dim_condition, ff_hidden_dim, pad_idx, n_position=200, dropout=0.1):

        super().__init__()
        self.trg_word_emb = nn.Embedding(x_values_num, dim_x, padding_idx=pad_idx)
        self.position_enc = SettledPositionalEncoding(dim_x, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            ConditionSequenceMappingLayer(dim_x, dim_condition, dim_mid, head_num, ff_hidden_dim, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)
        self.trg_word_prj = nn.Linear(dim_x, x_values_num, bias=False)

    # x_seq -> batch, n_x
    # x_seq_mask -> batch, n_y, n_x
    # condition_seq -> batch, n_y, dim_y
    # condition_seq_mask -> batch, 1, n_y
    # return -> batch, n_x, dim_x
    def forward(self, x_seq, x_seq_mask, condition_seq, condition_seq_mask):
        # -- Forward
        dec_output = self.trg_word_emb(x_seq)
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output, condition_seq, x_mask=x_seq_mask, condition_mask=condition_seq_mask)

        output = self.trg_word_prj(dec_output)

        return output


def get_encoder(values_num, seq_max_len, pad_idx) -> Encoder:
    return Encoder(
        x_values_num=values_num,
        n_layers=6,
        head_num=8,
        dim_mid=64,
        dim_x=512,
        ff_hiden_dim=512,
        pad_idx=pad_idx,
        n_position=seq_max_len)


def get_decode(values_num, seq_max_len, pad_idx) -> Decoder:
    return Decoder(
        x_values_num=values_num,
        n_layers=6,
        head_num=8,
        dim_mid=64,
        dim_x=512,
        dim_condition=512,
        ff_hiden_dim=512,
        pad_idx=pad_idx,
        n_position=seq_max_len,
    )
