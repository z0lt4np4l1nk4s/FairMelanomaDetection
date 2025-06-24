import cv2
import numpy as np
import torch
import random
from typing import List, Tuple
from config import SEGMENTATION_IMAGE_SIZE

class _ResizePadSquare:
    """
    Resizes an image and mask while maintaining aspect ratio,
    then pads to a square of the target size.
    """

    def __init__(self, target_size: int) -> None:
        self.target_size = target_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        scale = self.target_size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))

        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        pad_x = self.target_size - new_w
        pad_y = self.target_size - new_h
        pad_left, pad_right = pad_x // 2, pad_x - (pad_x // 2)
        pad_top, pad_bottom = pad_y // 2, pad_y - (pad_y // 2)

        image_padded = cv2.copyMakeBorder(image_resized, pad_top, pad_bottom, pad_left, pad_right,
                                          borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right,
                                         borderType=cv2.BORDER_CONSTANT, value=0)

        return image_padded, mask_padded

class _RandomHorizontalFlip:
    """
    Randomly horizontally flips the image and mask.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask

class _RandomVerticalFlip:
    """
    Randomly vertically flips the image and mask.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask

class _RandomRotation:
    """
    Applies a random rotation to the image and mask.
    """

    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        angle = random.uniform(-self.degrees, self.degrees)
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, mask

class _ColorJitter:
    """
    Randomly adjusts brightness and contrast.
    """

    def __init__(self, brightness: float = 0.0, contrast: float = 0.0) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.brightness > 0:
            factor = 1 + random.uniform(-self.brightness, self.brightness)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        if self.contrast > 0:
            factor = 1 + random.uniform(-self.contrast, self.contrast)
            mean = np.mean(image, axis=(0, 1), keepdims=True)
            image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)

        return image, mask

class _ToTensor:
    """
    Converts image and mask numpy arrays to torch tensors.
    """

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask = mask.astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return image_tensor, mask_tensor

class _Compose:
    """
    Composes multiple transformations together.
    """

    def __init__(self, transforms_list: List[object]) -> None:
        self.transforms = transforms_list

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

class _RandomAffineCV:
    """
    Applies random affine transformations: rotation, translation, scaling, and shearing.
    """

    def __init__(self, degrees: float = 45, translate: Tuple[float, float] = (0.2, 0.2),
                 scale_range: Tuple[float, float] = (0.7, 1.3), shear: float = 20) -> None:
        self.degrees = degrees
        self.translate = translate
        self.scale_range = scale_range
        self.shear = shear

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        tx = random.uniform(-self.translate[0] * w, self.translate[0] * w)
        ty = random.uniform(-self.translate[1] * h, self.translate[1] * h)
        shear_rad = np.deg2rad(random.uniform(-self.shear, self.shear))

        M = cv2.getRotationMatrix2D(center, angle, scale)
        S = np.array([[1, np.tan(shear_rad)], [0, 1]], dtype=np.float32)
        M[:, :2] = np.dot(M[:, :2], S)
        M[:, 2] += (tx, ty)

        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return image, mask

class _RandomGaussianBlur:
    """
    Applies Gaussian blur to the image randomly.
    """

    def __init__(self, p: float = 0.3, kernel_size: Tuple[int, int] = (5, 5)) -> None:
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, self.kernel_size, 0)
        return image, mask

class _RandomSyntheticHair:
    """
    Draws synthetic hairs (lines) on the image.
    """

    def __init__(self, p: float = 0.5, hair_count: Tuple[int, int] = (5, 15),
                 thickness_range: Tuple[int, int] = (1, 3)) -> None:
        self.p = p
        self.hair_count = hair_count
        self.thickness_range = thickness_range

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() >= self.p:
            return image, mask

        h, w = image.shape[:2]
        for _ in range(random.randint(*self.hair_count)):
            num_points = random.randint(2, 4)
            points = np.array([[random.randint(0, w-1), random.randint(0, h-1)] for _ in range(num_points)], np.int32)
            color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
            thickness = random.randint(*self.thickness_range)
            cv2.polylines(image, [points.reshape((-1, 1, 2))], isClosed=False, color=color, thickness=thickness)

        return image, mask

class _CLAHETransform:
    """
    Applies CLAHE (local contrast enhancement) to the image.
    """

    def __init__(self, clipLimit: float = 2.0, tileGridSize: Tuple[int, int] = (8, 8)) -> None:
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=self.tileGridSize)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return image, mask
    
class _Resize:
    """
    Resizes both image and mask to the exact target size without preserving aspect ratio.
    """

    def __init__(self, target_size: int) -> None:
        self.target_size = target_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image_resized = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        return image_resized, mask_resized

# Segmentation training transformations
segmentation_train_transformation = _Compose([
    _CLAHETransform(),
    _RandomHorizontalFlip(),
    _RandomVerticalFlip(p=0.3),
    _RandomAffineCV(degrees=15, translate=(0.05, 0.05), scale_range=(0.9, 1.1), shear=5),
    _RandomGaussianBlur(p=0.1, kernel_size=(3, 3)),
    _ColorJitter(brightness=0.15, contrast=0.15),
    _RandomSyntheticHair(),
    _Resize(target_size=SEGMENTATION_IMAGE_SIZE),
    _ToTensor()
])

# Segmentation validation transformations
segmentation_val_transformation = _Compose([
    _Resize(target_size=SEGMENTATION_IMAGE_SIZE),
    _ToTensor()
])