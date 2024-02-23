import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

TRANSFORMER = Compose([
    A.RandomResizedCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.ShiftScaleRotate(p=0.5),
    # A.HueSaturationValue(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(0.1, 0.2, p=0.8),
        # A.RandomGamma(p=0.5),
    ], p=1.0),
    # A.OneOf([
    #     A.Blur(p=0.1),
    #     A.GaussianBlur(p=0.1),
    #     A.MotionBlur(p=0.1),
    # ], p=0.1),
    # A.OneOf([
    #     A.GaussNoise(p=0.1),
    #     A.ISONoise(p=0.1),
    #     A.GridDropout(ratio=0.5, p=0.2),
    #     A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
    # ], p=0.2),
    A.ImageCompression(75, 100, p=0.5),
    A.ToFloat(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=1
    ),
    ToTensorV2(),
])
