import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

IMG_SIZE = 224
TRAIN_BS = 32
VAL_BS = 32
EPOCHS = 100
LR = 3e-5
WEIGHT_DECAY = 1e-2
VAL_SIZE = 0.15
SEED = 42