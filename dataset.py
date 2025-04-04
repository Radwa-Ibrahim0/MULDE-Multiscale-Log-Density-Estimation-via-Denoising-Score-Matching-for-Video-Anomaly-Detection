import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

def get_dataset(dataset_path, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    frame_paths = []
    video_folders = os.listdir(dataset_path)  # Get list of videos (e.g., "01_001", "01_002", ...)
    
    for video_folder in video_folders:
        frame_dir = os.path.join(dataset_path, video_folder, "frames")
        if os.path.exists(frame_dir):
            frames = sorted(os.listdir(frame_dir))  # Sort to maintain sequence order
            frame_paths.extend([os.path.join(frame_dir, frame) for frame in frames])

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize all images to (64, 64)
        transforms.ToTensor()  # Convert images to tensors
    ])

    images = []
    for frame_path in frame_paths:
        img = Image.open(frame_path).convert("RGB")  # Load image
        img = transform(img)  # Apply transformations
        images.append(img)

    images = torch.stack(images)  # Convert list of tensors to a single tensor batch

    print(f"Loaded {len(images)} frames from {len(video_folders)} videos.")
    
    return images  # Return the dataset as a tensor batch

# Example usage
dataset_path = "/kaggle/input/shanghaitech-anomaly-detection/dataset/mp"
images = get_dataset(dataset_path)
