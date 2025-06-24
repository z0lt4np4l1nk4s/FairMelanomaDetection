import cv2
import matplotlib.pyplot as plt
import numpy as np

class HairRemover:
    def __init__(self) -> None:
        # Initialize the object with no hair percentage calculated yet
        self._hair_percentage = None

    def remove_hair(self, image: np.ndarray) -> np.ndarray:
        """
        Removes hair-like artifacts from an input image.

        Args:
            image (np.ndarray): Input RGB image as a numpy array.

        Returns:
            np.ndarray: Image with hair artifacts removed.
        """

        # Convert the input image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Create a rectangular kernel for morphological operations
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        # Apply the blackhat morphological operation to highlight hair-like structures
        blackhat_image = cv2.morphologyEx(grayscale_image, cv2.MORPH_BLACKHAT, morph_kernel)

        # Threshold the blackhat image to create a binary mask of the hair regions
        _, hair_mask = cv2.threshold(blackhat_image, 24, 255, cv2.THRESH_BINARY)

        # Use inpainting to remove hair based on the mask
        inpainted_image = cv2.inpaint(image, hair_mask, 1, cv2.INPAINT_TELEA)

        # Calculate the percentage of the image covered by detected hair
        total_pixels = image.shape[0] * image.shape[1]
        hair_pixels = cv2.countNonZero(hair_mask)
        hair_fraction = hair_pixels / total_pixels if total_pixels != 0 else 0
        self._hair_percentage = hair_fraction * 100

        return inpainted_image

    def display_result(self, image_input: np.ndarray) -> None:
        """
        Processes the input image using remove_hair() and displays the original
        and processed images side-by-side along with the detected hair percentage.

        Args:
            image_input (np.ndarray): Input image as a numpy array.
        """

        # Process the image to remove hair
        processed_image = self.remove_hair(image_input)

        # Convert images from BGR to RGB for proper color display with matplotlib
        original_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Display original and processed images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(processed_rgb)
        axes[1].set_title(f"Processed Image\nHair Area: {self._hair_percentage:.2f}%")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
