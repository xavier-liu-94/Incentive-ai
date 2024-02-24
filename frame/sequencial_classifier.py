import torch
import torch.nn.functional as F
from .utils import *
from typing import *


class SequencialClassifier():

    def __init__(
        self,
        classify_model: torch.nn.Module, # backward normalized
        seq_pad_idx: int,
        device: str
        ) -> None:

        self.model = classify_model
        self.seq_pad_idx = seq_pad_idx
        self.device = device

        self.opt = torch.optim.Adam(classify_model.parameters(), 1e-4)

        self.state = "train"

        classify_model.to(device)
        classify_model.train()
    
    def train(
        self, 
        batch_seq: torch.Tensor,    # batch, n1.
        probabilities: torch.Tensor, # batch, prob of class
        ) -> None:

        batch_mask = get_seq_mask(batch_seq, self.seq_pad_idx)
        output = self.model(batch_seq, batch_mask)

        loss = F.cross_entropy(output, probabilities)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        print(loss.detach().cpu().numpy())