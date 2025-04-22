import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class FullImageDataset(Dataset):
    """Dataset for loading full images from a directory."""
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.endswith(".png")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Load an image and apply transformations."""
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img