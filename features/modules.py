import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from .contextual_mapping import *


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, dim_x, dim_mid, head_num, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = NormalizedContextualMapping(dim_x, dim_x, dim_mid, head_num, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_x, hidden_dim, dropout=dropout)

    def forward(self, x, x_mask=None):
        enc_output = self.slf_attn(x, x, mask=x_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, dim_x, dim_condition, dim_mid, head_num, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_mapping = NormalizedContextualMapping(dim_x, dim_x, dim_mid, head_num, dropout=dropout)
        self.relation_mapping = NormalizedContextualMapping(dim_x, dim_condition, dim_mid, head_num, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(dim_x, hidden_dim, dropout=dropout)

    def forward(self, x, y, x_mask=None, dec_mask=None):
        dec_output = self.slf_mapping(x, x, mask=x_mask)
        dec_output = self.relation_mapping(x, y, mask=dec_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output


class NormalizedContextualMapping(nn.Module):
    ''' Same as Multi-Head Attention module '''

    def __init__(self, dim_x, dim_condition, dim_mid, head_num, dropout=0.1):
        super().__init__()

        self.n_head = head_num

        self.context_map = ContextualMapping(
            dim_x=dim_x, 
            dim_condition=dim_condition, 
            dim_mid=dim_mid, 
            head_num=head_num, 
            pe_mapping_factory=lambda: LinearSoftmax(dim_x, dim_condition, dim_mid, head_num, temperature=dim_mid ** 0.5,attn_dropout=dropout))
        
        self.layer_norm = nn.LayerNorm(dim_x, eps=1e-6)


    def forward(self, x, condition, mask=None):
        # x -> batch, n_x, dim_x
        # condition -> batch, n_y, dim_y
        # mask -> batch, n_x, n_y
        # return -> batch, n_x, dim_x
        q = self.context_map(x, condition, mask)
        return self.layer_norm(q)


class SettledPositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(SettledPositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    encod_block = SettledPositionalEncoding(d_hid=48, n_position=96)
    pe = encod_block.pos_table.squeeze().T.cpu().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    pos = ax.imshow(pe, cmap="RdGy", extent=(1, pe.shape[1] + 1, pe.shape[0] + 1, 1))
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Hidden dimension")
    ax.set_title("Positional encoding over hidden dimensions")
    ax.set_xticks([1] + [i * 10 for i in range(1, 1 + pe.shape[1] // 10)])
    ax.set_yticks([1] + [i * 10 for i in range(1, 1 + pe.shape[0] // 10)])
    plt.show()