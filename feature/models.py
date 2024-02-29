import torch
from torch import nn
from .embeddings import *
from .modules import *


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
                dec_output, condition_seq, x_mask=x_seq_mask, dec_mask=condition_seq_mask)

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


class SequenceFeature(nn.Module):

    def __init__(self, n_layers, head_num, dim_mid, dim_x, ff_hiden_dim, dropout=0.1):

        super().__init__()
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            SequenceMappingLayer(dim_x, dim_mid, head_num, ff_hiden_dim, dropout=dropout)
            for _ in range(n_layers)])

    # x_seq_in -> batch, n_x, dim_x
    # x_seq_mask -> batch, 1 or n_x, n_x
    # return -> batch, n_x, dim_x
    def forward(self, x_seq_in, x_seq_mask):

        enc_output = self.layer_norm(x_seq_in)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, x_mask=x_seq_mask)

        return enc_output


class CascadeSeqClassifier(torch.nn.Module):

    def __init__(self, num_class, input_dims, sub_seq_len, pad_idx, dim_x=512) -> None:
        super().__init__()
        self.emb = ContinousSeqEmbedding(input_dims, dim_x, 200, pad_idx)
        self.trans1 = SequenceFeature(6, 8, 64, dim_x, 2048)
        self.fc = torch.nn.Linear(200, 1)
        self.trans2 = SequenceFeature(6, 8, 64, dim_x, 2048)
        self.trg_word_prj = nn.Linear(512, num_class, bias=False)
        self.sub_seq_len = sub_seq_len
        self.pad_idx = pad_idx

    # x -> batch, len, num_embeddings
    # x_mask -> batch, 1 or n_x, n_x
    def forward(self, x, x_mask):

        batch, n, num_embeddings = x.size()
        append_size = n % self.sub_seq_len
        if append_size > 0:
            x = torch.cat([x, self.pad_idx * torch.ones([batch, self.sub_seq_len-append_size, num_embeddings])], dim=-2)
            x_mask = torch.cat([x_mask, self.pad_idx * torch.ones([batch, 1, self.sub_seq_len-append_size])], dim=-1)
            n = n + self.sub_seq_len - append_size
        
        num_sub_seq = n // self.sub_seq_len

        x = x.reshape([batch * num_sub_seq, self.sub_seq_len, num_embeddings])
        x_sub_mask = x_mask.reshape([batch * num_sub_seq, 1, self.sub_seq_len])

        # batch * num_sub_seq, sub_seq_len, num_embeddings -> batch * num_sub_seq, sub_seq_len, dim_x
        seq_feature = self.emb(x)

        first_out = self.trans1(seq_feature, x_sub_mask)

        second_input = torch.squeeze(self.fc(first_out.reshape([batch, num_sub_seq, self.sub_seq_len, -1]).permute([0,1,3,2])), dim=-1)
        second_mask = torch.sum(x_sub_mask.reshape([batch, 1, num_sub_seq, self.sub_seq_len]), dim=-1) > 0

        # batch, num_sub_seq, dim_x -> batch, num_sub_seq, dim_x
        second_output = self.trans2(second_input, second_mask)

        return self.trg_word_prj(torch.mean(second_output, dim=1))

