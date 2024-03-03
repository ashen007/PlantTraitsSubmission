import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

TRANSFORMER = Compose([
    A.RandomSizedCrop((int(.75 * 512), 512), 288, 288, 1.0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(0.1, 0.2, p=1.0),
    A.ShiftScaleRotate(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.OneOf([
        A.Blur(p=0.1),
        A.GaussianBlur(p=0.1),
        A.MotionBlur(p=0.1),
    ], p=0.1),
    A.OneOf([
        A.GaussNoise(p=0.1),
        A.ISONoise(p=0.1),
        A.GridDropout(ratio=0.5, p=0.2),
        A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
    ], p=0.2),
    A.ToFloat(),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=1
    ),
    ToTensorV2(),
])

TEST_TRANSFORMER = Compose([A.Resize(288, 288),
                            A.ToFloat(),
                            A.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=1
                            ),
                            ToTensorV2(),
                            ])
