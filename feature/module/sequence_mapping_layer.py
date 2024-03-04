import torch.nn as nn
from .normalized_contextual_mapping import NormalizedContextualMapping
from .positionwise_feed_forward import PositionwiseFeedForward


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
