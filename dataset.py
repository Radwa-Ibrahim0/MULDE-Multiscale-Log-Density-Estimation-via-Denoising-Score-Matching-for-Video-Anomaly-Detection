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

    def load_frames_from_folder(folder_path):
        frame_paths = []
        video_folders = sorted(os.listdir(folder_path))
        for video_folder in video_folders:
            video_path = os.path.join(folder_path, video_folder)
            if os.path.isdir(video_path):
                frames = sorted(os.listdir(video_path))
                frame_paths.extend([os.path.join(video_path, frame) for frame in frames])
        return frame_paths

    # Load training and testing frames separately
    train_frame_paths = load_frames_from_folder(os.path.join(dataset_path, 'training', 'frames'))
    test_frame_paths = load_frames_from_folder(os.path.join(dataset_path, 'testing', 'frames'))

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Load training images
    train_images = []
    for path in tqdm(train_frame_paths, desc="Processing training frames"):
        img = Image.open(path).convert("RGB")
        img = transform(img)
        train_images.append(img)
    data_train = torch.stack(train_images)
    labels_train = np.zeros(len(data_train)).astype(np.uint8)  # Dummy labels for training

    # Load testing images
    test_images = []
    for path in tqdm(test_frame_paths, desc="Processing testing frames"):
        img = Image.open(path).convert("RGB")
        img = transform(img)
        test_images.append(img)
    data_test = torch.stack(test_images)
    labels_test = np.random.randint(0, 2, size=len(data_test)).astype(np.uint8)  # Simulated labels

    id_to_type = {0: "normal", 1: "anomaly"}

    print(f"Loaded {len(data_train)} training frames, {len(data_test)} testing frames.")

    return data_train, labels_train, data_test, labels_test, id_to_type

def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy

# Example usage:
# dataset_path = "path/to/ped2-ds"
# get_dataset(dataset_path)
