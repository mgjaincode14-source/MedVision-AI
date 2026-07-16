import os
import matplotlib.pyplot as plt
from preprocess import preprocess_pipeline

def run_tests():
    image_path = "data/images/00000061_025.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image not found at: {image_path}. Please check your dataset installation.")
        
    results = preprocess_pipeline(image_path)
    print(f"Original: {results['original'].shape}, range: [{results['original'].min()}, {results['original'].max()}]")
    print(f"Enhanced: {results['enhanced'].shape}, range: [{results['enhanced'].min()}, {results['enhanced'].max()}]")
    print(f"Resized: {results['resized'].shape}")
    print(f"Normalized: {results['normalized'].shape}, range: [{results['normalized'].min():.4f}, {results['normalized'].max():.4f}]")
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "preprocessing_comparison.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(results['original'], cmap='gray')
    axes[0].set_title(f"Original\nShape: {results['original'].shape}")
    axes[0].axis('off')
    axes[1].imshow(results['enhanced'], cmap='gray')
    axes[1].set_title("CLAHE Enhanced")
    axes[1].axis('off')
    axes[2].imshow(results['normalized'], cmap='gray')
    axes[2].set_title(f"Normalized & Resized\nShape: {results['resized'].shape}")
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")

if __name__ == "__main__":
    run_tests()
