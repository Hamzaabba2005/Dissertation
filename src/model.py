import torch.nn as nn


class ODIRModel(nn.Module):
    def __init__(self, backbone, num_labels=8, dropout=0.3):
        super().__init__()

        self.backbone = backbone

        # handle common torchvision classifier layers
        if hasattr(self.backbone, "fc"):  # ResNet
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_labels)
            )

        elif hasattr(self.backbone, "classifier"):  # DenseNet / EfficientNet
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_labels)
            )

        else:
            raise ValueError("Unsupported backbone architecture")

    def forward(self, x):
        return self.backbone(x)