import torch
from torch import functional as F
from .utils import *


class SequencialClassifier():

    def __init__(
        self,
        predict_model: torch.nn.Module, 
        seq_pad_idx: int,
        context_model: torch.nn.Module, 
        context_pad_idx: int,
        device) -> None:
        self.predict_model = predict_model
        self.seq_pad_idx = seq_pad_idx
        self.context_model = context_model
        self.context_pad_idx = context_pad_idx
        self.opt = torch.optim.Adam([x for x in predict_model.parameters()]+[x for x in context_model.parameters()], 1e-4)
        self.predict_model.to(device)
        self.context_model.to(device)
        self.predict_model.train()
        self.context_model.train()
        self.state = "train"
        self.device = device
    
    