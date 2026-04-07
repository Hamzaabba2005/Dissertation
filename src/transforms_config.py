from torchvision import transforms


def get_transforms(img_size):

    train_tfm = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # slightly larger before crop
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),  # much more aggressive zoom
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),              # safe for fundus
        transforms.RandomRotation(degrees=30),              # fundus is rotation-invariant
        transforms.ColorJitter(
            brightness=0.3,                                 # up from 0.1
            contrast=0.3,                                   # up from 0.1
            saturation=0.2,                                 # up from 0.03
            hue=0.02,                                       # keep small — hue shifts can mislead
        ),
        transforms.RandomGrayscale(p=0.05),                # occasional grayscale, helps generalisation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.15),                            # erase small patches only
            ratio=(0.3, 3.3),
            value=0                                        # fill with black (near fundus background)
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