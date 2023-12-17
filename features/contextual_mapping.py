import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Callable
from abc import abstractmethod


# Permute equivalent mapping capture relation of x and y
class PEMappingWithMask(nn.Module):

    @abstractmethod
    def forward(self, x, y, mask):
        # x -> batch, n_x, dim_x
        # y -> batch, n_y, dim_y
        # mask -> batch, n_x, n_y. 0 means ignroed
        # return -> batch, head_num, n_x, n_y
        raise NotImplementedError


class ContextualMapping(nn.Module):

    def __init__(
            self, 
            dim_x: int, 
            dim_y: int, 
            dim_mid: int,
            head_num: int,
            pe_mapping_factory: Callable[[],PEMappingWithMask],
            dropout_ratio: float=0.1
            ) -> None:
        super().__init__()
        self.head_num = head_num
        self.dim_mid = dim_mid
        self.head_num = head_num

        # Permute equivalent mapping
        self.pe = pe_mapping_factory()

        # value tranfer
        self.vt = nn.Linear(dim_y, dim_mid*head_num, bias=False)

        # output tranfer. Theoratically, it is part of value
        self.fc = nn.Linear(dim_mid*head_num, dim_x, bias=False)

        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
        # x -> batch, n_x, dim_x
        # y -> batch, n_y, dim_y
        # mask -> batch, n_x, n_y
        # return -> batch, n_x, dim_x
        n_x = x.size(1)
        n_y = y.size(1)

        # batch, head_num, n_x, n_y
        relation = self.pe(x, y, mask)

        # batch, n_y, head_num, dim_mid
        value = self.vt(y).view(-1, n_y, self.head_num, self.dim_mid)

        # for matmul(a, b), torch will do matmul for the last two dimention and deeming the rest dimention as match index

        # multi_head_output -> batch, head_num, n_x, dim_mid
        multi_head_output = torch.matmul(relation, value.permute(0,2,1,3))

        result = self.dropout(self.fc(
            multi_head_output.permute(0, 2, 1, 3).contiguous().view(-1, n_x, self.head_num*self.dim_mid)
            )) + x
            
        return result


class LinearSoftmax(PEMappingWithMask):

    def __init__(self, 
                 dim_x: int, 
                 dim_y: int, 
                 dim_mid: int,
                 head_num: int,
                 temperature: float,
                 attn_dropout: float=0.1
                 ) -> None:
        super().__init__()
        self.linear_x = nn.Linear(dim_x, dim_mid*head_num, bias=False)
        self.linear_y = nn.Linear(dim_y, dim_mid*head_num, bias=False)
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.dim_mid = dim_mid
        self.head_num = head_num

    def forward(self, x, y, mask=None):
        # x -> batch, n_x, dim_x
        # y -> batch, n_y, dim_y
        # mask -> batch, n_x, n_y. 0 means ignroed

        n_x = x.size(1)
        n_y = y.size(1)

        # batch, n_x, head_num, dim_mid
        multi_head_x = self.linear_x(x).view(-1, n_x, self.head_num, self.dim_mid)

        # batch, n_y, head_num, dim_mid
        multi_head_y = self.linear_y(y).view(-1, n_y, self.head_num, self.dim_mid)

        # batch, head_num, n_x, n_y
        similarity = torch.matmul(multi_head_x.permute(0,2,1,3), multi_head_y.permute(0,2,3,1))

        if mask is not None:
            mask = mask.unsqueeze(1)
            similarity = similarity.masked_fill(mask==0, -1e9)

        return self.dropout(F.softmax(similarity, dim=-1))

    
