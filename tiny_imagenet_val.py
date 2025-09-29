# tiny_imagenet_val.py
import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetVal(Dataset):
    """
    It reads val/val_annotations.txt, which maps each validation image filename to a WordNet ID (WNID). 
    Using provided class_to_idx dict (WNID → integer), it builds a list of (image_path, label_idx) pairs 
    pointing to files in val/images/.

    Tiny-ImageNet validation dataset:
    - images in: val/images/*.JPEG
    - labels in: val/val_annotations.txt (filename -> wnid)
    """
    def __init__(self, val_dir, class_to_idx, transform=None):
        self.transform = transform
        self.samples = []

        anno_path = os.path.join(val_dir, "val_annotations.txt")
        with open(anno_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                filename, wnid = parts[0], parts[1]
                if wnid not in class_to_idx:
                    # Shouldn't happen for Tiny-ImageNet, but guard anyway
                    continue
                label_idx = class_to_idx[wnid]
                img_path = os.path.join(val_dir, "images", filename)
                self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
