import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_COLS = ["Normal", "Diabetes", "Glaucoma", "Cataract", "AMD", "Hypertension", "Myopia", "Other"]

IMG_SIZE = 224
TRAIN_BS = 16
VAL_BS = 32
EPOCHS = 100
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.15
SEED = 42