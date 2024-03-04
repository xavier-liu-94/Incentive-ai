import torch.nn as nn
from ._contextual_mapping import *


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
