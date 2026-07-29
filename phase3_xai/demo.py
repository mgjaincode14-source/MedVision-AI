import os
import sys
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))
sys.path.append(os.path.join(PROJECT_ROOT, "dl"))

from preprocess import preprocess_pipeline
from network import MedVisionDenseNet
from gradcam import GradCAM
from visualize import apply_heatmap

def run_gradcam_demo():
    print("Initializing Phase 3: Grad-CAM Explainable AI...")

    img_path = os.path.join(PROJECT_ROOT, "data", "images", "00000061_025.png")
    model_path = os.path.join(PROJECT_ROOT, "output", "models", "medvision_densenet.pth")
    output_dir = os.path.join(PROJECT_ROOT, "output", "gradcam")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(img_path) or not os.path.exists(model_path):
        print(f"Error: Missing image or trained model.")
        print(f"Ensure Phase 2 has completed successfully and saved the model to {model_path}")
        return

    classes = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
    ]

    print("Loading PyTorch Model...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedVisionDenseNet(num_classes=14, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    target_layer = model.densenet.features.denseblock4.denselayer16.conv2
    grad_cam = GradCAM(model, target_layer)

    print(f"Preprocessing Image: {os.path.basename(img_path)}...")
    pipeline_out = preprocess_pipeline(img_path, target_size=(224, 224))
    img_normalized = pipeline_out['normalized']

    input_tensor = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad = True

    print("Running prediction to find Top 3 diseases...")
    with torch.no_grad():
        output_probs = model(input_tensor)[0]

    top3_probs, top3_indices = torch.topk(output_probs, 3)

    print("Generating Heatmaps for Top 3 predictions...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()

    orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_img = cv2.resize(orig_img, (224, 224))
    axes[0].imshow(orig_img, cmap='gray')
    axes[0].set_title("Original X-Ray", fontsize=14)
    axes[0].axis('off')

    for i in range(3):
        idx = top3_indices[i].item()
        prob = top3_probs[i].item()
        class_name = classes[idx]

        heatmap = grad_cam(input_tensor, idx)

        blended_bgr = apply_heatmap(img_path, heatmap, alpha=0.5)
        blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)

        ax = axes[i+1]
        ax.imshow(blended_rgb)
        ax.set_title(f"{class_name} ({prob*100:.1f}%)", fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "gradcam_summary.png")
    plt.savefig(summary_path, dpi=300)
    plt.close()

    print("\nPhase 3 Demo Complete!")
    print(f"Summary visualization saved to: {summary_path}")

if __name__ == "__main__":
    run_gradcam_demo()
