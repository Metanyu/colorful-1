import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from skimage import color


class ColorizationDataset(Dataset):
    def __init__(self, root, extensions=('.jpg', '.jpeg', '.png')):
        self.root = Path(root)
        self.images = [p for p in self.root.rglob('*') if p.suffix.lower() in extensions]

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        lab = color.rgb2lab(np.array(img))
        return torch.FloatTensor(np.transpose(lab, (2, 0, 1))), 0

    def __len__(self):
        return len(self.images)
