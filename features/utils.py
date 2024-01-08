import torch

def get_seq_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    # seq -> batch, n
    # return -> batch, 1, n | bool
    return (seq != pad_idx).unsqueeze(-2)

def get_triu_mask(len_s: int, device=None) -> torch.Tensor:
    # return -> 1, len_s, len_s | bool
    return (1 - torch.triu(torch.ones((1, len_s, len_s), device=device), diagonal=1)).bool() 

def flat_batch_target(trg: torch.Tensor) -> torch.Tensor:
    # trg -> batch, n
    # return -> batch * (n-1)
    return trg[:, 1:].contiguous().view(-1)

def seq_eval(pred: torch.Tensor, trg: torch.Tensor, trg_pad_idx: int):
    non_pad_mask = trg.ne(trg_pad_idx)
    n_correct = pred.eq(trg).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word