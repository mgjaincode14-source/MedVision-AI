import torch
import torch.nn as nn
from torchvision import models

class MedVisionDenseNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(MedVisionDenseNet, self).__init__()

        try:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        except AttributeError:
            self.densenet = models.densenet121(pretrained=pretrained)

        original_conv1 = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        if pretrained:
            self.densenet.features.conv0.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)

        num_ftrs = self.densenet.classifier.in_features

        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet(x)

if __name__ == "__main__":
    model = MedVisionDenseNet()
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(output)
