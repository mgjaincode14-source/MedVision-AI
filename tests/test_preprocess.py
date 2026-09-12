import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from preprocess import load_image, apply_clahe, resize_image, normalize_image, preprocess_pipeline

import cv2

def test_load_image(tmp_path):
    # Create a dummy image file
    img_path = str(tmp_path / "dummy.png")
    dummy_img = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    
    img = load_image(img_path)
    assert isinstance(img, np.ndarray)
    assert len(img.shape) == 2 # Grayscale

def test_load_image_not_found():
    with pytest.raises(FileNotFoundError):
        load_image("fake_path.png")

def test_resize_image():
    # Create a dummy 500x500 image
    dummy_img = np.zeros((500, 500), dtype=np.uint8)
    resized = resize_image(dummy_img, target_size=(224, 224))
    assert resized.shape == (224, 224)

def test_normalize_image():
    # Create a dummy image with max pixel value 255
    dummy_img = np.full((10, 10), 255, dtype=np.uint8)
    normalized = normalize_image(dummy_img)
    assert normalized.max() <= 1.0
    assert normalized.min() >= 0.0

def test_preprocess_pipeline(tmp_path):
    img_path = str(tmp_path / "dummy.png")
    dummy_img = np.zeros((500, 500), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    
    results = preprocess_pipeline(img_path, target_size=(224, 224))
    
    assert 'original' in results
    assert 'enhanced' in results
    assert 'resized' in results
    assert 'normalized' in results
    
    assert results['normalized'].shape == (224, 224)
    assert results['normalized'].max() <= 1.0
