import torch
from torch import nn
from .embedding import *
from .module import *
from .sequence_feature import *


class SequenceClassifier(torch.nn.Module):

    def __init__(self, num_class, input_dims, max_seq_len, pad_idx, 
        dim_x=512, layers=6, head_num=8, dim_head_hidden=64, ff_hiden_dim=2048, 
        emb_factory=ContinousSeqEmbedding) -> None:
        super().__init__()
        self.emb = emb_factory(input_dims=input_dims, dim_x=dim_x, n_position=max_seq_len, pad_idx=pad_idx)
        self.trans = SequenceFeature(layers, head_num, dim_head_hidden, dim_x, ff_hiden_dim)
        self.trg_word_prj = nn.Linear(dim_x, num_class, bias=False)
        self.pad_idx = pad_idx
    
    # x -> batch, len. OR batch, len, num_embeddings.
    # x_mask -> batch, len(1), len
    def forward(self, x, x_mask):
        seq_feature = self.emb(x)

        out = self.trans(seq_feature, x_mask)

        return self.trg_word_prj(torch.mean(out, dim=1))


class CascadeSeqClassifier(torch.nn.Module):

    def __init__(self, num_class, input_dims, sub_seq_len, pad_idx, 
        dim_x=512, layers=6, head_num=8, dim_head_hidden=64, ff_hiden_dim=2048, 
        emb_factory=ContinousSeqEmbedding) -> None:
        super().__init__()
        self.emb = emb_factory(input_dims=input_dims, dim_x=dim_x, n_position=sub_seq_len, pad_idx=pad_idx)
        self.trans1 = SequenceFeature(layers, head_num, dim_head_hidden, dim_x, ff_hiden_dim)
        self.fc = torch.nn.Linear(sub_seq_len, 1)
        self.trans2 = SequenceFeature(layers, head_num, dim_head_hidden, dim_x, ff_hiden_dim)
        self.trg_word_prj = nn.Linear(dim_x, num_class, bias=False)
        self.sub_seq_len = sub_seq_len
        self.pad_idx = pad_idx

    # x -> batch, len. OR batch, len, num_fields.
    # x_mask -> batch, len(1), len
    def forward(self, x, x_mask):
        
        input_size = x.size()
        if len(input_size) == 2:
            batch, n = input_size
            adding_part = []
        else:
            batch, n, num_embeddings = input_size
            adding_part = [num_embeddings]

        append_size = n % self.sub_seq_len
        if append_size > 0:
            x = torch.cat([x, self.pad_idx * torch.ones([batch, self.sub_seq_len-append_size] + adding_part)], dim=-2)
            x_mask = torch.cat([x_mask, self.pad_idx * torch.ones([batch, 1, self.sub_seq_len-append_size])], dim=-1)
            n = n + self.sub_seq_len - append_size
        
        num_sub_seq = n // self.sub_seq_len

        x = x.reshape([batch * num_sub_seq, self.sub_seq_len] + adding_part)
        x_sub_mask = x_mask.reshape([batch * num_sub_seq, 1, self.sub_seq_len])

        # batch * num_sub_seq, sub_seq_len, {num_embeddings} -> batch * num_sub_seq, sub_seq_len, dim_x
        seq_feature = self.emb(x)

        first_out = self.trans1(seq_feature, x_sub_mask)

        second_input = torch.squeeze(self.fc(first_out.reshape([batch, num_sub_seq, self.sub_seq_len, -1]).permute([0,1,3,2])), dim=-1)
        second_mask = torch.sum(x_sub_mask.reshape([batch, 1, num_sub_seq, self.sub_seq_len]), dim=-1) > 0

        # batch, num_sub_seq, dim_x -> batch, num_sub_seq, dim_x
        second_output = self.trans2(second_input, second_mask)

        return self.trg_word_prj(torch.mean(second_output, dim=1))
