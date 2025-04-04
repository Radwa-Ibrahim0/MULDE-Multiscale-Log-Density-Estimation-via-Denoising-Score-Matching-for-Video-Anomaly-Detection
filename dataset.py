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
    video_folders = os.listdir(dataset_path)
    for video_folder in video_folders:
        frame_dir = os.path.join(dataset_path, video_folder, "frames")
        if os.path.exists(frame_dir):
            frames = sorted(os.listdir(frame_dir))
            frame_paths.extend([os.path.join(frame_dir, frame) for frame in frames])

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    images = []
    for frame_path in tqdm(frame_paths, desc="Processing frames"):
        img = Image.open(frame_path).convert("RGB")
        img = transform(img)
        images.append(img)

    images = torch.stack(images)

    # Dummy labels & basic splitting (e.g., 80% train, 20% test)
    split_idx = int(0.8 * len(images))
    data_train = images[:split_idx]
    labels_train = np.zeros(len(data_train)).astype(np.uint8)  # Dummy labels for training

    data_test = images[split_idx:]
    labels_test = np.random.randint(0, 2, size=len(data_test)).astype(np.uint8)  # Simulated labels

    id_to_type = {0: "normal", 1: "anomaly"}

    print(f"Loaded {len(images)} frames total: {len(data_train)} train, {len(data_test)} test.")

    return data_train, labels_train, data_test, labels_test, id_to_type

def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy

# Example usage
