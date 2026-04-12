import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ODIRDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample["path"]).convert("RGB")
        img = self.transform(img)

        y = torch.tensor(sample["label"], dtype=torch.float32)

        return img, y


def create_dataloaders(
    train_samples,
    val_samples,
    train_transform,
    val_transform,
    test_samples=None,
    train_batch_size=64,
    val_batch_size=128,
    num_workers=4
):
    train_ds = ODIRDataset(train_samples, train_transform)
    val_ds = ODIRDataset(val_samples,   val_transform)

    train_loader = DataLoader(
        train_ds, 
        batch_size=train_batch_size, 
        shuffle=True,  
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,   
        batch_size=val_batch_size,   
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True
    )

    if test_samples is not None:
        test_ds = ODIRDataset(test_samples, val_transform)
        test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
        return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

    return train_ds, val_ds, train_loader, val_loader