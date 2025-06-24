from torch.utils.data import Dataset
import os
import cv2
import torch
import pandas as pd
from common.constants import ColumnNames
from pipeline.transforms.lesion_cropper import LesionCropper
from config import SEGMENTATION_MODEL_CHECKPOINT_PATH
from typing import Tuple, List, Optional

class ClassificationDataset(Dataset):
    """
    Dataset for training or validation.
    Loads images and corresponding labels from a DataFrame.
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[object] = None) -> None:
        self.dataframe = dataframe     # Keep original index; no reset_index()
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int, str]:
        # Fetch the row by label (index)
        sample = self.dataframe.loc[index]
        image_path = sample[ColumnNames.IMAGE_PATH]

        # Validate image path
        if not isinstance(image_path, str):
            raise ValueError(f"[ClassificationDataset] index={index}: Invalid path {image_path!r}")

        # Load and convert image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply optional transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Return image and associated labels
        return (
            image,
            sample[ColumnNames.TARGET],
            sample[ColumnNames.SKIN_TONE]
        )

    def __len__(self) -> int:
        return len(self.dataframe)


class ClassificationTestDataset(Dataset):
    """
    Inference-only dataset.
    Loads images from a list of paths and returns (image_tensor, filename).
    """

    def __init__(self, image_paths: List[str], transform: Optional[object] = None) -> None:
        self.image_paths = image_paths
        self.transform = transform
        self.lesion_cropper = LesionCropper(SEGMENTATION_MODEL_CHECKPOINT_PATH)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        # Fetch image path
        image_path = self.image_paths[index]

        # Load and convert image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"[ClassificationTestDataset] Failed to read {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply lesion cropping
        image = self.lesion_cropper.crop(image)

        # Apply optional transformations
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
        else:
            # Fallback: convert to PyTorch tensor manually
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

        # Extract filename for reference
        filename = os.path.basename(image_path)

        return image_tensor, filename
    
    def __len__(self) -> int:
        return len(self.image_paths)
