import os
import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_dataloader
from network import MedVisionDenseNet

def train_model():
    print("Initializing Phase 2: PyTorch Deep Learning Training Pipeline...")

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(PROJECT_ROOT, "data", "sample_labels.csv")
    img_dir = os.path.join(PROJECT_ROOT, "data", "images")

    if not os.path.exists(csv_path) or not os.path.exists(img_dir):
        print(f"Error: Dataset not found at {csv_path} or {img_dir}.")
        print("Please ensure the dataset is properly extracted.")
        return

    BATCH_SIZE = 8
    EPOCHS = 2
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print(f"Loading data from: {csv_path}")
    dataloader = get_dataloader(csv_path, img_dir, batch_size=BATCH_SIZE)
    print(f"Total batches per epoch: {len(dataloader)}")

    print("Loading DenseNet121 architecture...")
    model = MedVisionDenseNet(num_classes=14, pretrained=True).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting Training...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.4f}")
                running_loss = 0.0

            if i == 15:
                print("Stopping early for verification purposes...")
                break

    print("\nTraining Verification Complete!")

    os.makedirs(os.path.join(PROJECT_ROOT, "output", "models"), exist_ok=True)
    save_path = os.path.join(PROJECT_ROOT, "output", "models", "medvision_densenet.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train_model()
