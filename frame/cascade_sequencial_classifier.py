import torch
import torch.nn.functional as F
from .utils import *
from typing import *


class CascadeSequencialClassifier():

    def __init__(
        self,
        feature_model_1: torch.nn.Module, # backward normalized
        seq_pad_idx: int,
        sub_seq_len: int,
        feature_model_2: torch.nn.Module, 
        device: str
        ) -> None:

        self.first_model = feature_model_1
        self.second_model = feature_model_2
        self.seq_pad_idx = seq_pad_idx
        self.sub_seq_len = sub_seq_len
        self.device = device

        self.opt = torch.optim.Adam([x for x in feature_model_1.parameters()]+[x for x in feature_model_2.parameters()], 1e-4)

        self.state = "train"

        feature_model_1.to(device)
        feature_model_2.to(device)
        feature_model_1.train()
        feature_model_2.train()
    
    def train(
        self, 
        batch_seq: torch.Tensor,    # batch, n1.
        probabilities: torch.Tensor, # batch, prob of class
        ) -> None:
        batch, n = batch_seq.size()
        append_size = n % self.sub_seq_len
        if append_size > 0:
            batch_seq = torch.cat([batch_seq, self.seq_pad_idx * torch.ones([batch, self.sub_seq_len-append_size])])
            n = n + self.sub_seq_len - append_size
        num_sub_seq = n // self.sub_seq_len

        first_input = batch_seq.reshape([batch * num_sub_seq, self.sub_seq_len])

        # batch * num_sub_seq, sub_seq_len -> batch * num_sub_seq, hidden_class_num
        first_output = self.first_model(first_input)

        second_input = F.softmax(first_output.reshape([batch, num_sub_seq, -1]), dim=-1)

        # batch, num_sub_seq, hidden_class_num -> batch, class_num
        second_output = self.second_model(second_input)

        loss = F.cross_entropy(second_output, probabilities)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        print(loss.detach().cpu().numpy())