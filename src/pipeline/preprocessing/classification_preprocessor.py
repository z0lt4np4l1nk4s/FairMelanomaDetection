import os
import cv2
import pandas as pd
from pipeline.transforms.lesion_cropper import LesionCropper
from pipeline.transforms.hair_remover import HairRemover
from pipeline.classifiers.skin_tone_classifier import SkinToneClassifier
from common.constants import ColumnNames, Constants
from config import SEGMENTATION_MODEL_CHECKPOINT_PATH, PREPROCESSED_FOLDER
import numpy as np

class ClassificationPreprocessor:
    """
    A class to preprocess images for a machine learning pipeline.
    It includes methods for lesion cropping, hair removal, skin tone classification, and resizing images.
    """

    def __init__(self, 
            segmentation_model_path: str = SEGMENTATION_MODEL_CHECKPOINT_PATH,
            output_size: int = 512,
            output_folder: str = PREPROCESSED_FOLDER,
            use_cache: bool = True
        ):
        # Initialize lesion cropper and skin tone classifier
        self.lesion_cropper = LesionCropper(segmentation_model_path)
        self.skin_tone_classifier = SkinToneClassifier(segmentation_model_path)
        self.output_size = output_size
        self.output_folder = output_folder
        self.use_cache = use_cache

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def process_image(
        self, 
        record: dict, 
        is_fitzpatrick=False
    ) -> dict:
        """
        Loads an image using its path from 'record', applies preprocessing (cropping, resizing),
        saves the processed image, and updates the record with the new path.
        
        Args:
            record (dict): Dictionary containing the key 'image_path' with full path to the image.
            output_size (int): Size (width and height) for output images.
            output_folder (str): Destination folder to save processed images.
            use_cache (bool): If True, uses cached preprocessed images if available.
        
        Returns:
            dict: Updated record with modified 'image_path' and metadata.
        """

        # Get original image path
        original_image_path = record[ColumnNames.IMAGE_PATH]

        # Extract the image filename
        image_filename = os.path.basename(original_image_path)

        # Build the full output path
        processed_image_path = f"{self.output_folder}/{image_filename}"

        # Load and prepare the image
        image = cv2.imread(original_image_path)
        if image is None:
            raise ValueError(f"Image not found: {original_image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Classify skin tone if not already present
        if pd.isna(record.get(ColumnNames.SKIN_TONE)):
            record[ColumnNames.SKIN_TONE] = (
                Constants.DARK_SKIN_TONE if is_fitzpatrick else self.skin_tone_classifier.classify_skin_tone(image)
            )

        # Set group label based on target and skin tone
        record[ColumnNames.GROUP] = f"{record[ColumnNames.TARGET]}_{record[ColumnNames.SKIN_TONE]}"

        # Save original image path if missing
        if pd.isna(record.get(ColumnNames.ORIGINAL_IMAGE_PATH)):
            record[ColumnNames.ORIGINAL_IMAGE_PATH] = original_image_path

        # If cache is enabled and processed image already exists, skip processing
        if self.use_cache and os.path.exists(processed_image_path):
            record[ColumnNames.IMAGE_PATH] = processed_image_path
            return record
        
        if is_fitzpatrick:
            # Resize with padding
            h, w, _ = image.shape
            scale = min(self.output_size / h, self.output_size / w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            canvas = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
            top, left = (self.output_size - new_h) // 2, (self.output_size - new_w) // 2
            canvas[top:top + new_h, left:left + new_w] = resized
            image_to_save = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        else:
            # Crop, remove hair, resize
            image = self.lesion_cropper.crop(image)
            # image = hairRemover.remove_hair(image)
            image = cv2.resize(image, (self.output_size, self.output_size))
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert image back to BGR for saving
        cv2.imwrite(processed_image_path, image_to_save)

        # Update the record with new processed image path
        record[ColumnNames.IMAGE_PATH] = processed_image_path

        return record
