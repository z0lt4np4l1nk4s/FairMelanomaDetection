import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMAGE_SIZE

# Classification training transformations
classification_train_transformation = A.Compose([
    A.Transpose(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(std_range=(0.2, 0.44)),
    ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.0),
        A.ElasticTransform(alpha=3),
    ], p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-15, 15), p=0.6),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.CoarseDropout(
        num_holes_range=(1, 5),
        hole_height_range=(int(IMAGE_SIZE * 0.1875), int(IMAGE_SIZE * 0.1875)),
        hole_width_range=(int(IMAGE_SIZE * 0.1875), int(IMAGE_SIZE * 0.1875)),
        p=0.7
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Classification validation transformations
classification_val_transformation = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])