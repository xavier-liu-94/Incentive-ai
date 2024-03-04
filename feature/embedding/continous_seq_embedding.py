import torch
from torch import nn
import numpy as np


class ContinousSeqEmbedding(nn.Module):
    def __init__(self, num_embeddings, output_dim, n_position, pad_idx) -> None:
        super().__init__()
        self.w = torch.nn.parameter.Parameter(torch.empty((num_embeddings, output_dim)))
        torch.nn.init.normal_(self.w)
        self.pad_idx = pad_idx
    
    # x: batch, len, num_embeddings
    def forward(self, x: torch.Tensor):

        # is_pad: batch, len, num_embeddings
        is_pad = 1*(x!=self.pad_idx)

        # batch, len, num_embeddings, output_dim
        values = is_pad.unsqueeze(-1) * x.unsqueeze(-1) * self.w.unsqueeze(0).unsqueeze(0)

        # batch, len, output_dim
        return torch.sum(values, dim=-2)