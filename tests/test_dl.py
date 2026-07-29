import os
import sys
import torch
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "dl"))

from network import MedVisionDenseNet
from data_loader import XrayDataset

def test_medvision_densenet_init():
    # Test model initializes without pretrained weights to save time/bandwidth in tests
    model = MedVisionDenseNet(num_classes=14, pretrained=False)
    assert model is not None
    
    # Check if the first layer accepts 1 channel
    assert model.densenet.features.conv0.in_channels == 1

def test_medvision_densenet_forward():
    model = MedVisionDenseNet(num_classes=14, pretrained=False)
    # Create dummy batch of 2 grayscale images (2, 1, 224, 224)
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model(dummy_input)
    
    # Output should be (batch_size, num_classes)
    assert output.shape == (2, 14)
    
    # Since we use Sigmoid, all outputs should be bounded between 0 and 1
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)

def test_dataset_length():
    csv_path = os.path.join(PROJECT_ROOT, "data", "sample_labels.csv")
    img_dir = os.path.join(PROJECT_ROOT, "data", "images")
    
    # Only test if data exists (helpful for CI where data might not be downloaded)
    if os.path.exists(csv_path):
        dataset = XrayDataset(csv_path, img_dir)
        assert len(dataset) > 0
