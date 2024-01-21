import torch
from torch import functional as F
from utils import *


class NextSequencialPredictor():

    def __init__(
        self, 
        predict_model: torch.nn.Module, 
        seq_pad_idx: int,
        context_model: torch.nn.Module, 
        context_pad_idx: int,
        device
        ) -> None:

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

    def train(
        self,
        batch_seq: torch.Tensor,    # batch, n1. Usually the first token of sequence is a start token
        batch_context: torch.Tensor,    # batch, n2
        ):
        if self.state != "train":
            self.predict_model.train()
            self.context_model.train()
            self.state = "train"

        batch_seq.to(self.device)
        batch_context.to(self.device)
        _, n1 = batch_seq.size()
        seq_input, seq_trg = batch_seq[:, :-1], batch_seq[:, 1:].contiguous().view(-1)
        seq_mask = (get_seq_mask(seq_input, self.seq_pad_idx) & get_triu_mask(n1-1).to(self.device))
        
        context_mask = get_seq_mask(batch_context, self.context_pad_idx)

        encoded_batch_context = self.context_model(batch_context, context_mask)

        batch_preds = self.predict_model(seq_input, seq_mask, encoded_batch_context, context_mask)

        flated_batch_preds = batch_preds.view(-1, batch_preds.size(2))

        loss = F.cross_entropy(flated_batch_preds, seq_trg, ignore_index=self.seq_pad_idx, reduction='sum')

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        n_correct, n_word = seq_eval(flated_batch_preds, seq_trg, self.seq_pad_idx)

        return loss, n_correct, n_word
        
    def eval(
        self, 
        batch_seq: torch.Tensor,    # batch, n1. Usually the first token of sequence is a start token
        batch_context: torch.Tensor    # batch, n2
        ):
        if self.state != "eval":
            self.predict_model.eval()
            self.context_model.eval()
            self.state = "eval"

        batch_seq.to(self.device)
        batch_context.to(self.device)
        _, n1 = batch_seq.size()
        
        seq_input, seq_trg = batch_seq[:, :-1], batch_seq[:, 1:].contiguous().view(-1)
        seq_mask = get_seq_mask(seq_input, self.seq_pad_idx) & get_triu_mask(n1-1).to(self.device)
        
        context_mask = get_seq_mask(batch_context, self.context_pad_idx)

        encoded_batch_context = self.context_model(batch_context, context_mask)

        batch_preds = self.predict_model(seq_input, seq_mask, encoded_batch_context, context_mask)

        flated_batch_preds = batch_preds.view(-1, batch_preds.size(2))

        loss = F.cross_entropy(flated_batch_preds, seq_trg, ignore_index=self.seq_pad_idx, reduction='sum')
        n_correct, n_word = seq_eval(flated_batch_preds, seq_trg, self.seq_pad_idx)

        return loss, n_correct, n_word