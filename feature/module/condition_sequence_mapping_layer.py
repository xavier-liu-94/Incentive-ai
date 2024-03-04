import torch.nn as nn
from .normalized_contextual_mapping import NormalizedContextualMapping
from .positionwise_feed_forward import PositionwiseFeedForward


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
