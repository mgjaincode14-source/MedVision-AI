import os
import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode the image file: {image_path}")
    return image

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

def resize_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0

def preprocess_pipeline(image_path: str, target_size: tuple = (224, 224)) -> dict:
    original = load_image(image_path)
    enhanced = apply_clahe(original)
    resized = resize_image(enhanced, target_size)
    normalized = normalize_image(resized)
    return {
        'original': original,
        'enhanced': enhanced,
        'resized': resized,
        'normalized': normalized
    }
