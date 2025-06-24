# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from PIL import Image as PILImage
import blobfile as bf
from torch.utils.data import Dataset


def _list_image_files_recursively(data_dir):
    """List all image files in a directory recursively. Adapted from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageFolderDataset(Dataset):
    def __init__(self, folder, transform, conditional=False):
        super().__init__()
        self.folder = folder
        self.transform = transform

        self.local_images = _list_image_files_recursively(folder)
        self.local_class_names = (
            ["None"] * len(self.local_images)
            if not conditional
            else [bf.basename(path).split("_")[0] for path in self.local_images]
        )
        self.class_names = list(sorted(set(self.local_class_names)))
        self.class_name_to_id = {x: i for i, x in enumerate(self.class_names)}
        self.local_classes = [self.class_name_to_id[x] for x in self.local_class_names]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = PILImage.open(f)
            pil_image.load()

        arr = self.transform(pil_image)

        label = self.local_classes[idx]
        return arr, label
