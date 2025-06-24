import cv2
import torch
import numpy as np
from segmentation_model.model import SegmentationModel
from config import DEVICE
import threading

_thread_local = threading.local()

class LesionCropper:
    """
    Cropper that uses a segmentation model to identify and crop around lesions in medical images.
    Ensures lesions aren't over-cropped, preserving context and discarding very small artifacts.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.25,
        target_size: int = 512,
        min_area_pct: float = 5e-5,
        margin_px: int = 45,
        dilation_iter: int = 4,
        max_mask_pct: float = 0.8,
    ) -> None:
        # Load pretrained segmentation model
        self.device = torch.device(DEVICE)
        self.model = SegmentationModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Set parameters for processing
        self.threshold = threshold  # Mask probability threshold
        self.target_size = target_size  # Final crop size
        self.min_area = min_area_pct * (target_size ** 2)  # Minimum mask area to keep
        self.margin_px = margin_px  # Padding margin around lesion box
        self.dilation_iter = dilation_iter  # Dilation iterations to expand mask
        self.max_mask_pct = max_mask_pct  # If lesion covers too much image, skip cropping

        # Initialize gamma correction LUT
        inv_gamma = 1.0 / 1.2
        self.gamma_lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)

    def crop(self, input_image: np.ndarray) -> np.ndarray:
        """
        Crop the image around the detected lesion region.

        Args:
            input_image (np.ndarray): Original RGB image as NumPy array.

        Returns:
            np.ndarray: Cropped and resized lesion image.
        """

        # Preprocess image: square padding, enhancement, resizing
        preprocessed_image = self._preprocess(input_image)

        # Predict lesion mask using the segmentation model
        with torch.no_grad():
            image_tensor = (torch.from_numpy(preprocessed_image.astype(np.float32) / 255.0)
                            .permute(2, 0, 1).unsqueeze(0).to(self.device))
            prob_map = torch.sigmoid(self.model(image_tensor))[0, 0].cpu().numpy()

        # Apply threshold to get binary mask and clean it with morphological opening
        binary_mask = (prob_map >= self.threshold).astype(np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Filter connected components by minimum area
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
        filtered_mask = np.zeros_like(binary_mask)
        for i in range(1, num_components):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                filtered_mask[labels == i] = 1

        # Determine whether to use mask or fallback to full image
        if filtered_mask.sum() == 0:
            final_mask = np.ones_like(binary_mask)  # Fallback: use full image
        else:
            # Compute bounding box area of the mask
            ys, xs = np.where(filtered_mask)
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            bbox_area = (y2 - y1) * (x2 - x1)
            if bbox_area / binary_mask.size > self.max_mask_pct:
                final_mask = np.ones_like(binary_mask)  # Too large, fallback
            else:
                final_mask = filtered_mask

        # Optionally dilate the mask to include context
        if self.dilation_iter:
            final_mask = cv2.dilate(final_mask, np.ones((3, 3), np.uint8), iterations=self.dilation_iter)

        # Crop image using bounding box of the final mask
        ys, xs = np.where(final_mask)

        y1 = max(ys.min() - self.margin_px, 0)
        y2 = min(ys.max() + self.margin_px, preprocessed_image.shape[0])
        x1 = max(xs.min() - self.margin_px, 0)
        x2 = min(xs.max() + self.margin_px, preprocessed_image.shape[1])

        cropped_image = preprocessed_image[y1:y2, x1:x2]

        # Final resize and return
        cropped_image = self._pad_to_square(cropped_image)
        return cv2.resize(cropped_image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Pad image to square, apply CLAHE, gamma correction, and resize.
        """
        padded = self._pad_to_square(img)
        lab_img = cv2.cvtColor(padded, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab_img)

        clahe = self._get_clahe()
        l_clahe = clahe.apply(l)

        lab_enhanced = cv2.merge((l_clahe, a, b))
        rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        gamma_corrected = cv2.LUT(rgb_enhanced, self.gamma_lut)

        return cv2.resize(gamma_corrected, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)


    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        """
        Pad image to make it square by adding black borders.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Square padded image.
        """

        height, width = img.shape[:2]

        if height == width:
            return img
        
        pad_total = abs(height - width)
        pad1, pad2 = pad_total // 2, pad_total - (pad_total // 2)

        if height > width:
            return cv2.copyMakeBorder(img, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            return cv2.copyMakeBorder(img, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    def _get_clahe(self):
        if not hasattr(_thread_local, "clahe"):
            _thread_local.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return _thread_local.clahe
