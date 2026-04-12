import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLS = ["N", "D", "G", "C", "A", "H", "M", "O"]

IMG_SIZE = 224
TRAIN_BS = 16
VAL_BS = 32
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15
SEED = 42