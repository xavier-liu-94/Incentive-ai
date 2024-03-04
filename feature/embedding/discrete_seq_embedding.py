import torch
from torch import nn
import numpy as np


class DiscreteSeqEmbedding(nn.Module):

    def __init__(self, value_num, output_dim, pad_idx) -> None:
        super().__init__()
        self.src_word_emb = nn.Embedding(value_num, output_dim, padding_idx=pad_idx)
    
    def forward(self, x):
        # batch, max_len -> batch, max_len, output_dim
        return self.src_word_emb(x)