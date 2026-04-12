import numpy as np
import torch


def compute_pos_weight(samples):
    Y = np.stack([s["label"] for s in samples], axis=0) 

    pos = Y.sum(axis=0)
    neg = len(Y) - pos

    pos_weight = neg / np.clip(pos, 1, None)

    return torch.tensor(pos_weight, dtype=torch.float32)