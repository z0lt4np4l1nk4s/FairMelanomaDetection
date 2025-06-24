import cv2
import numpy as np
import torch
from common.constants import Constants
import matplotlib.pyplot as plt
from config import DEVICE
from segmentation_model.model import SegmentationModel

class SkinToneClassifier:
    def __init__(
        self,
        segmentation_model_path: str,
        segmentation_threshold: float = 0.4,
        target_size: int = 512,
    ):
        # Set up lesion segmentation model
        self.device = torch.device(DEVICE)
        self.segmentation_model = SegmentationModel().to(self.device)
        self.segmentation_model.load_state_dict(torch.load(segmentation_model_path, map_location=self.device))
        self.segmentation_model.eval()

        self.segmentation_threshold = segmentation_threshold
        self.target_size = target_size

    def classify_skin_tone(self, input_image: np.ndarray) -> int:
        """
        Classifies skin tone based on brightness of non-lesion skin pixels.
        """

        # Get lesion mask and invert it to find non-lesion (skin) regions
        lesion_mask = self._get_lesion_mask(input_image)
        non_lesion_skin_mask = cv2.bitwise_not(lesion_mask)

        # Remove tiny noise in the non-lesion mask
        clean_kernel = np.ones((7, 7), np.uint8)
        non_lesion_skin_mask = cv2.morphologyEx(non_lesion_skin_mask, cv2.MORPH_OPEN, clean_kernel, iterations=1)

        # Extract skin pixels using the mask
        skin_pixels = input_image[np.where(non_lesion_skin_mask != 0)]
        if skin_pixels.size == 0:
            return Constants.UNKNOWN_SKIN_TONE

        # Filter out extremely dark or bright pixels (likely noise)
        brightness_values = np.mean(skin_pixels, axis=1)
        filtered_skin_pixels = skin_pixels[(brightness_values > 30) & (brightness_values < 250)]
        if filtered_skin_pixels.size == 0:
            return Constants.UNKNOWN_SKIN_TONE

        # Use median brightness for robustness
        median_brightness = np.median(np.mean(filtered_skin_pixels, axis=1))
        return self._map_brightness_to_skin_tone(median_brightness)

    def visualize_skin_detection(self, input_image: np.ndarray) -> None:
        """
        Visualizes the original image, lesion mask, clean skin mask, and final detected skin with tone classification.
        """

        # Compute lesion mask and general skin mask
        lesion_mask = self._get_lesion_mask(input_image)
        general_skin_mask = self._detect_skin_mask(input_image)

        # Clean skin mask by removing lesion regions
        skin_mask_without_lesion = cv2.bitwise_and(general_skin_mask, cv2.bitwise_not(lesion_mask))

        # Classify the skin tone
        skin_tone_category = self.classify_skin_tone(input_image)

        # Mask the skin regions from the original image
        isolated_skin = cv2.bitwise_and(input_image, input_image, mask=skin_mask_without_lesion)

        # Plot the results
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        ax = axes.ravel()

        ax[0].imshow(input_image)
        ax[0].set_title("Original Image")
        ax[0].axis("off")

        ax[1].imshow(lesion_mask, cmap='gray')
        ax[1].set_title("Lesion Mask")
        ax[1].axis("off")

        ax[2].imshow(skin_mask_without_lesion, cmap='gray')
        ax[2].set_title("Skin Mask (excluding lesion)")
        ax[2].axis("off")

        ax[3].imshow(cv2.cvtColor(isolated_skin, cv2.COLOR_BGR2RGB))
        ax[3].set_title(f"Detected Skin\nTone: {skin_tone_category}")
        ax[3].axis("off")

        plt.tight_layout()
        plt.show()

    def _map_brightness_to_skin_tone(self, brightness: float) -> int:
        """
        Maps median brightness value to a skin tone category.
        """

        if brightness < 96:
            return Constants.DARK_SKIN_TONE
        elif brightness < 128:
            return Constants.MEDIUM_SKIN_TONE
        elif brightness < 160:
            return Constants.MEDIUM_LIGHT_SKIN_TONE
        elif brightness < 192:
            return Constants.LIGHT_SKIN_TONE
        else:
            return Constants.VERY_LIGHT_SKIN_TONE

    def _detect_skin_mask(self, input_image: np.ndarray) -> np.ndarray:
        """
        Detects general skin regions based on Cr channel thresholding (Otsu's method).
        """

        # Convert to YCrCb color space
        ycrcb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2YCrCb)
        cr_channel = ycrcb_image[:, :, 1]

        # Apply Gaussian blur and Otsu's thresholding
        blurred_cr = cv2.GaussianBlur(cr_channel, (5, 5), 0)
        _, skin_mask = cv2.threshold(
            blurred_cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Clean up the mask with morphological operations
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
        return skin_mask

    def _get_lesion_mask(self, input_image: np.ndarray) -> np.ndarray:
        """
        Predicts the lesion mask using the segmentation model.
        Returns a binary mask resized back to the original image size.
        """

        original_height, original_width = input_image.shape[:2]

        # Resize input for model inference
        resized_input = cv2.resize(input_image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        input_tensor = (torch.from_numpy(resized_input.astype(np.float32) / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(self.device))

        # Get lesion probability map
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)
            probability_map = torch.sigmoid(output)[0, 0].cpu().numpy()

        # Threshold to binary mask and resize back
        binary_mask = (probability_map >= self.segmentation_threshold).astype(np.uint8) * 255
        lesion_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        return lesion_mask
