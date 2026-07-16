import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from preprocess import preprocess_pipeline

class XrayDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, img_dir, batch_size=32, target_size=(224, 224), shuffle=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        
        self.labels_list = [
            "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
        ]
        
        self._prepare_data()
        self.on_epoch_end()

    def _prepare_data(self):
        self.image_paths = []
        self.labels = []
        
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row["Image Index"])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                
                label_vector = np.zeros(len(self.labels_list), dtype=np.float32)
                row_labels = str(row["Finding Labels"]).split("|")
                for label in row_labels:
                    if label in self.labels_list:
                        idx = self.labels_list.index(label)
                        label_vector[idx] = 1.0
                self.labels.append(label_vector)
                
        self.image_paths = np.array(self.image_paths)
        self.labels = np.array(self.labels)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = self.image_paths[indexes]
        batch_labels = self.labels[indexes]
        
        X = np.empty((self.batch_size, *self.target_size, 1), dtype=np.float32)
        for i, path in enumerate(batch_paths):
            pipeline_out = preprocess_pipeline(path, self.target_size)
            X[i] = np.expand_dims(pipeline_out['normalized'], axis=-1)
            
        return X, batch_labels

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
