from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from common.constants import ColumnNames

class SegmentationDataset(Dataset):
    """
    Dataset that loads images and their corresponding masks
    from file paths provided in a DataFrame.
    Expects the DataFrame to have 'image_path' and 'mask_path' columns.
    """

    def __init__(self, dataframe: pd.DataFrame, transform: Optional[object] = None) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Get the data row
        sample = self.dataframe.iloc[index]

        # Extract image and mask paths
        image_path = sample[ColumnNames.IMAGE_PATH]
        mask_path = sample[ColumnNames.MASK_PATH]

        # Read the image and convert from BGR to RGB
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask in grayscale mode
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")

        # Apply transformations if available
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
    
    def __len__(self) -> int:
        return len(self.dataframe)
