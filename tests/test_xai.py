import os
import sys
import torch
import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "dl"))
sys.path.append(os.path.join(PROJECT_ROOT, "phase3_xai"))

from network import MedVisionDenseNet
from gradcam import GradCAM

def test_gradcam_output_shape():
    # Setup dummy model and target layer
    model = MedVisionDenseNet(num_classes=14, pretrained=False)
    model.eval()
    target_layer = model.densenet.features.denseblock4.denselayer16.conv2
    
    grad_cam = GradCAM(model, target_layer)
    
    # Dummy input tensor (1, 1, 224, 224)
    input_tensor = torch.randn(1, 1, 224, 224)
    input_tensor.requires_grad = True
    
    # Target class 0 (Atelectasis)
    heatmap = grad_cam(input_tensor, target_class=0)
    
    # The heatmap should be a 2D numpy array of size 224x224
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)

def test_gradcam_normalization():
    model = MedVisionDenseNet(num_classes=14, pretrained=False)
    model.eval()
    target_layer = model.densenet.features.denseblock4.denselayer16.conv2
    
    grad_cam = GradCAM(model, target_layer)
    
    input_tensor = torch.randn(1, 1, 224, 224)
    input_tensor.requires_grad = True
    
    heatmap = grad_cam(input_tensor, target_class=0)
    
    # Heatmap should be normalized between 0.0 and 1.0
    assert heatmap.max() <= 1.0
    assert heatmap.min() >= 0.0
