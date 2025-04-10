# utils/dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class PairedImageDatasetWithMask(Dataset):
    def __init__(self, root_dir, transform=None):
        self.fg_dir = os.path.join(root_dir, 'bg1k_masks')
        self.full_dir = os.path.join(root_dir, 'bg1k_imgs')
        self.transform = transform

        self.ids = [f.split('_')[0] for f in os.listdir(self.fg_dir) if f.endswith('_mask.png')]
        self.ids = sorted(self.ids, key=lambda x: int(x))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        fg_path = os.path.join(self.fg_dir, f"{img_id}_mask.png")
        full_path = os.path.join(self.full_dir, f"{img_id}.png")

        fg = Image.open(fg_path).convert("RGBA")
        full = Image.open(full_path).convert("RGB")

        # Extract alpha channel as mask (1 for foreground, 0 for background)
        alpha = fg.split()[-1].convert("L")  # shape (H, W)
        alpha_np = np.array(alpha)  # 转 numpy
        mask = torch.tensor((alpha_np > 10).astype(np.float32))  # 生成 0/1 前景 mask

        if self.transform:
            full = self.transform(full)
            fg = self.transform(fg.convert("RGB"))
            # 将 mask 转为 PIL Image 才能正确调用 transform
            from torchvision.transforms.functional import to_pil_image
            mask = self.transform(to_pil_image(mask))

        return fg, full, mask