import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from network import MedVisionDenseNet

def train_model():
    print("Initializing Phase 2: PyTorch Deep Learning Training Pipeline...")
    
    # 1. Setup paths
    # Get the project root directory (since we are in phase2_dl/)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(PROJECT_ROOT, "data", "sample_labels.csv")
    img_dir = os.path.join(PROJECT_ROOT, "data", "images")
    
    if not os.path.exists(csv_path) or not os.path.exists(img_dir):
        print(f"Error: Dataset not found at {csv_path} or {img_dir}.")
        print("Please ensure the dataset is properly extracted.")
        return

    # 2. Hyperparameters
    BATCH_SIZE = 8
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # 3. Initialize Data Loader
    print(f"Loading data from: {csv_path}")
    dataloader = get_dataloader(csv_path, img_dir, batch_size=BATCH_SIZE)
    print(f"Total batches per epoch: {len(dataloader)}")
    
    # 4. Initialize Model, Loss Function, and Optimizer
    print("Loading DenseNet121 architecture...")
    model = MedVisionDenseNet(num_classes=14, pretrained=True).to(DEVICE)
    
    # Binary Cross Entropy Loss is used for Multi-Label Classification
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    print("\nStarting Training...")
    model.train() # Set model to training mode
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        for i, (images, labels) in enumerate(dataloader):
            # Move data to the selected device (CPU or GPU)
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 5 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.4f}")
                running_loss = 0.0
                
            # For this tutorial/sample, we can break early so it doesn't take hours
            if i == 15: # Stop after 15 batches per epoch just to verify it runs
                print("Stopping early for verification purposes...")
                break
                
    print("\nTraining Verification Complete!")
    
    # Save the model
    os.makedirs(os.path.join(PROJECT_ROOT, "output", "models"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "output", "models", "medvision_densenet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train_model()
