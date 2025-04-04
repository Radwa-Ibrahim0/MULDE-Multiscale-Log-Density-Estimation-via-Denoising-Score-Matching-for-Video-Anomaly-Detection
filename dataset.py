import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def get_dataset(dataset_path, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    frame_paths = []
    video_folders = os.listdir(dataset_path)  # Get list of videos (e.g., "01_001", "01_002", ...)
    print("video folders: ")
    print(video_folders)
    print("ana abl loop get dataset")
    print(dataset_path)
    for video_folder in video_folders:
        frame_dir = os.path.join(dataset_path, video_folder, "frames")
        print(frame_dir)
        if os.path.exists(frame_dir):
            frames = sorted(os.listdir(frame_dir))  # Sort to maintain sequence order
            frame_paths.extend([os.path.join(frame_dir, frame) for frame in frames])

    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize all images to (64, 64)
        transforms.ToTensor()  # Convert images to tensors
    ])
    print("ana abl tany loop get dataset")
    images = []
    for frame_path in tqdm(frame_paths, desc="Processing frames"):
        img = Image.open(frame_path).convert("RGB")  # Load image
        img = transform(img)  # Apply transformations
        images.append(img)

    images = torch.stack(images)  # Convert list of tensors to a single tensor batch

    print(f"Loaded {len(images)} frames from {len(video_folders)} videos.")
    
    return images  # Return the dataset as a tensor batch

def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy

# Example usage
