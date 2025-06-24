import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from common.constants import ColumnNames
from config import *

class SegmentationPreprocessor:
    """
    Handles preprocessing of segmentation data. Resizes and pads images and their
    corresponding masks to a uniform square shape. Stores processed versions and updates metadata.
    """

    def __init__(self, size: int = 512, output_folder: str = PREPROCESSED_MASKS_FOLDER, use_cache: bool = True):
        # Set desired output size and folder path
        self.output_size = size
        self.output_folder = output_folder
        self.use_cache = use_cache

    def process_pair(self, record: Dict) -> Dict:
        """
        Process a single image and mask pair. If caching is enabled and files already exist,
        skip processing. Otherwise, resize and pad both image and mask, save them, and
        update the record with new file paths.

        Args:
            record (Dict): Dictionary containing metadata, including image and mask paths.

        Returns:
            Dict: Updated dictionary with paths to the processed image and mask.
        """

        # Extract original paths
        original_image_path = record[ColumnNames.IMAGE_PATH]
        original_mask_path = record[ColumnNames.MASK_PATH]

        # Get filenames only
        image_filename = os.path.basename(original_image_path)
        mask_filename = os.path.basename(original_mask_path)

        # Define paths for processed output
        processed_image_path = os.path.join(self.output_folder, image_filename)
        processed_mask_path = os.path.join(self.output_folder, mask_filename)

        # Preserve original paths in record if not already present
        if pd.isna(record.get(ColumnNames.ORIGINAL_IMAGE_PATH)):
            record[ColumnNames.ORIGINAL_IMAGE_PATH] = original_image_path

        if pd.isna(record.get(ColumnNames.ORIGINAL_MASK_PATH)):
            record[ColumnNames.ORIGINAL_MASK_PATH] = original_mask_path

        # If using cache and processed files already exist, skip processing
        if self.use_cache and os.path.exists(processed_image_path) and os.path.exists(processed_mask_path):
            record[ColumnNames.IMAGE_PATH] = processed_image_path
            record[ColumnNames.MASK_PATH] = processed_mask_path
            return record

        # Load and validate the image
        image = cv2.imread(original_image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {original_image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load and validate the mask
        mask = cv2.imread(original_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {original_mask_path}")

        # Resize and pad image and mask
        processed_image, processed_mask = self._resize_and_pad(image, mask)

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Save the processed image and mask
        cv2.imwrite(processed_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(processed_mask_path, processed_mask)

        # Update record with processed paths
        record[ColumnNames.IMAGE_PATH] = processed_image_path
        record[ColumnNames.MASK_PATH] = processed_mask_path
        
        return record

    def _resize_and_pad(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resizes image and mask to fit within a square of `output_size`, preserving aspect ratio,
        and adds padding to reach exact square dimensions.

        Args:
            image (np.ndarray): RGB image.
            mask (np.ndarray): Grayscale mask.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of padded image and mask.
        """
        
        # Determine scaling factor to fit within square
        height, width = image.shape[:2]
        scale = self.output_size / max(width, height)
        new_width, new_height = int(round(width * scale)), int(round(height * scale))

        # Resize image and mask using appropriate interpolation
        image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Compute padding amounts for both axes
        pad_x = self.output_size - new_width
        pad_y = self.output_size - new_height
        pad_left, pad_right = pad_x // 2, pad_x - (pad_x // 2)
        pad_top, pad_bottom = pad_y // 2, pad_y - (pad_y // 2)

        # Pad image with black pixels and mask with zero
        image_padded = cv2.copyMakeBorder(
            image_resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        mask_padded = cv2.copyMakeBorder(
            mask_resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

        return image_padded, mask_padded
