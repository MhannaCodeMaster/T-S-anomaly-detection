from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MVTecDataset(object):
    def __init__(self, image_list, transform=None):
        self.image_list = image_list
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.ToTensor()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        return [Image.open(p).convert('RGB') for p in self.image_list]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.transform(self.dataset[idx])
        return self.image_list[idx], image

class PatchDataset(Dataset):
    def __init__(self, df, tf, label: str, crops_root):
        self.paths =[str(Path(crops_root) / p) for p in df["patch_path"].tolist()]
        self.parent_ids = df["parent_id"].tolist()
        self.label = 0 if label == "ok" else 1
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(self.label, dtype=torch.long)
        return x, y, self.parent_ids[i], path
