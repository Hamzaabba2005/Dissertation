import os, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_COLS = ["N","D","G","C","A","H","M","O"]
BASE_PATH = "/Users/hamzaabba/Documents/Dissertation/data"
CSV_PATH = os.path.join(BASE_PATH, "full_df.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "preprocessed_images")

print("CSV exists:", os.path.exists(CSV_PATH))
print("Images dir exists:", os.path.exists(IMAGES_DIR))

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
print("Device:", DEVICE)


df = pd.read_csv(CSV_PATH)
df.head()


def pick_images_dir(base_path, df):
    candidates = []
    
    # Add all subdirectories automatically
    for name in os.listdir(base_path):
        p = os.path.join(base_path, name)
        if os.path.isdir(p):
            candidates.append(p)

    # Test some filenames from CSV
    test_names = []
    for i in range(min(20, len(df))):
        test_names.append(str(df.iloc[i]["Left-Fundus"]))
        test_names.append(str(df.iloc[i]["Right-Fundus"]))

    best = None
    best_hits = -1
    
    for cand in candidates:
        hits = sum(os.path.exists(os.path.join(cand, nm)) for nm in test_names)
        if hits > best_hits:
            best = cand
            best_hits = hits

    return best, best_hits

IMAGES_DIR, hits = pick_images_dir(BASE_PATH, df)
print("Selected IMAGES_DIR:", IMAGES_DIR)
print("Hits:", hits)


def build_samples_existing(df, images_dir):
    samples = []
    missing = 0
    
    for _, r in df.iterrows():
        y = r[LABEL_COLS].to_numpy(dtype=np.float32)
        
        for col in ["Left-Fundus", "Right-Fundus"]:
            fname = str(r[col])
            path = os.path.join(images_dir, fname)
            
            if os.path.exists(path):
                samples.append((path, y))
            else:
                missing += 1

    return samples, missing

samples, missing = build_samples_existing(df, IMAGES_DIR)
print("Total usable samples:", len(samples))
print("Missing images:", missing)


from sklearn.model_selection import train_test_split

train_samples, val_samples = train_test_split(samples, test_size=0.15, random_state=42, shuffle=True)

train_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class ODIRDataset(Dataset):
    def __init__(self, samples, tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)
        y = torch.tensor(y, dtype=torch.float32)
        return img, y

train_ds = ODIRDataset(train_samples, train_tfm)
val_ds = ODIRDataset(val_samples, val_tfm)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

len(train_ds), len(val_ds)

def build_model(num_labels=8):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_labels)
    return m

model = build_model(len(LABEL_COLS)).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)


@torch.no_grad()
def multilabel_accuracy(logits, y, threshold=0.5):
    # logits: [B, C], y: [B, C] float in {0,1}
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).to(y.dtype)
    return (preds == y).float().mean().item()

def run_epoch(model, loader, criterion, optimizer=None, train=True, device="cuda"):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float()   # ensure float for BCEWithLogitsLoss

        if y.ndim != 2:
            raise ValueError(f"Expected y shape [B, C], got {tuple(y.shape)}")
        
        if train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)

        if logits.shape != y.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != y shape {tuple(y.shape)}")

        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += multilabel_accuracy(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

for epoch in range(1, 20):
    tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer, train=True, device=DEVICE)
    va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None, train=False, device=DEVICE)
    print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")


checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "resnet18_odir_multilabel.pt"))
print("Saved.")