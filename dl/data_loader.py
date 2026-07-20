import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys

# Ensure we can import from Phase 1 src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
from preprocess import preprocess_pipeline

class XrayDataset(Dataset):
    """
    PyTorch Dataset for Chest X-rays.
    Reads the CSV, parses the Finding Labels into a 14-class multi-label target,
    and processes the images using Phase 1 preprocessing.
    """
    def __init__(self, csv_path, img_dir, target_size=(224, 224)):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.target_size = target_size
        
        # The 14 diseases we are predicting
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 1. Use Phase 1 to preprocess the image
        # pipeline_out contains 'original', 'enhanced', 'resized', 'normalized'
        pipeline_out = preprocess_pipeline(img_path, self.target_size)
        img_normalized = pipeline_out['normalized'] # shape (224, 224)
        
        # Convert to PyTorch Tensor. 
        # PyTorch expects shape (Channels, Height, Width)
        # Since it's grayscale, add a channel dimension: (1, 224, 224)
        img_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)
        
        # 2. Extract multi-labels
        finding_labels = row['Finding Labels']
        label_vector = torch.zeros(len(self.classes), dtype=torch.float32)
        
        # Example label: "Cardiomegaly|Effusion" -> split by "|"
        if "No Finding" not in finding_labels:
            diseases = finding_labels.split("|")
            for disease in diseases:
                if disease in self.classes:
                    idx_class = self.classes.index(disease)
                    label_vector[idx_class] = 1.0
                    
        return img_tensor, label_vector

def get_dataloader(csv_path, img_dir, batch_size=32, shuffle=True, num_workers=0):
    """
    Helper function to create a PyTorch DataLoader
    """
    dataset = XrayDataset(csv_path, img_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
