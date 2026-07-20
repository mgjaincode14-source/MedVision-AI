# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torchvision import models

class MedVisionDenseNet(nn.Module):
    """
    A Deep Learning classifier built on DenseNet121.
    Pre-trained on ImageNet. The final classification head is modified to output 
    14 separate probabilities for 14 different lung diseases.
    """
    def __init__(self, num_classes=14, pretrained=True):
        super(MedVisionDenseNet, self).__init__()
        
        # 1. Load the base DenseNet121 model
        try:
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
        except AttributeError:
            self.densenet = models.densenet121(pretrained=pretrained)
        
        # 2. DenseNet expects 3-channel (RGB) images by default.
        # Our X-rays are 1-channel (Grayscale).
        # We replace the first convolutional layer to accept 1 channel instead of 3.
        # We copy the weights from the original first layer (averaging them across RGB) to retain pre-training benefits.
        original_conv1 = self.densenet.features.conv0
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        if pretrained:
            self.densenet.features.conv0.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
            
        # 3. Replace the final classifier with our CUSTOM DEEP HEAD
        num_ftrs = self.densenet.classifier.in_features
        
        self.densenet.classifier = nn.Sequential(
            # First custom fully-connected layer
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.4), # Prevent overfitting by randomly zeroing 40% of neurons
            
            # Second custom layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            # Final output layer for 14 diseases
            nn.Linear(256, num_classes),
            nn.Sigmoid() # Sigmoid required for multi-label classification
        )
        
    def forward(self, x):
        return self.densenet(x)

if __name__ == "__main__":
    # Quick test to ensure the architecture works
    model = MedVisionDenseNet()
    dummy_input = torch.randn(2, 1, 224, 224) # Batch size 2, 1 channel, 224x224
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}") # Expected: [2, 14]
    print(output)
