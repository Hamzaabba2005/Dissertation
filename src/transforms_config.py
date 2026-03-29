from torchvision import transforms


def get_transforms(img_size):

    train_tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(img_size, scale=(0.95, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.03,
            hue=0.01
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tfm, val_tfm