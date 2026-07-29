import cv2
import numpy as np

def apply_heatmap(original_img_path, heatmap_array, alpha=0.5, colormap=cv2.COLORMAP_JET):
    img = cv2.imread(original_img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {original_img_path}")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.resize(img, (224, 224))

    heatmap_uint8 = np.uint8(255 * heatmap_array)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    blended = cv2.addWeighted(heatmap_colored, alpha, img, 1 - alpha, 0)

    return blended
