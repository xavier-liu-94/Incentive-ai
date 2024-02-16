import torch
from torch import nn
import numpy as np


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


class DiscreteSeqEmbedding(nn.Module):

    def __init__(self, value_num, output_dim, n_position, pad_idx) -> None:
        super().__init__()
        self.src_word_emb = nn.Embedding(value_num, output_dim, padding_idx=pad_idx)
        self.position_enc = SettledPositionalEncoding(output_dim, n_position=n_position)
    
    def forward(self, x):
        # batch, max_len -> batch, max_len, output_dim
        return self.position_enc(self.src_word_emb(x))


class ContinousSeqEmbedding(nn.Module):
    def __init__(self, output_dim, n_position, pad_idx) -> None:
        super().__init__()
        self.src_word_emb = nn.Embedding(2, output_dim, padding_idx=pad_idx)
        self.position_enc = SettledPositionalEncoding(output_dim, n_position=n_position)
        self.pad_idx = pad_idx
    
    def forward(self, x: torch.Tensor):
        is_pad = x==self.pad_idx
        
        # batch, max_len -> batch, max_len, output_dim
        return self.position_enc(self.src_word_emb(x))


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
