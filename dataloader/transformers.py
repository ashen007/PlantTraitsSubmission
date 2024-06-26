import cv2
import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

TRANSFORMER = Compose([A.RandomResizedCrop(224, 224),
                       # A.Resize(512, 512),
                       A.HorizontalFlip(p=0.5),
                       A.RandomBrightnessContrast(0.1, 0.2, p=1.0),
                       A.ShiftScaleRotate(p=0.5),
                       A.HueSaturationValue(p=0.5),
                       A.OneOf([
                           A.Blur(p=0.1),
                           A.GaussianBlur(p=0.1),
                           A.MotionBlur(p=0.1),
                       ], p=0.1),
                       A.OneOf([A.GaussNoise(p=0.1),
                                A.ISONoise(p=0.1),
                                A.GridDropout(ratio=0.5, p=0.2),
                                A.CoarseDropout(max_holes=16,
                                                min_holes=8,
                                                max_height=16,
                                                max_width=16,
                                                min_height=8,
                                                min_width=8,
                                                p=0.2)
                                ], p=0.2),
                       A.ToFloat(),
                       A.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225],
                                   max_pixel_value=1
                                   ),
                       ToTensorV2(),
                       ])

INPAINT_TRANSFORMER = Compose([A.Resize(224, 224),
                               A.ToFloat(),
                               A.Normalize(
                                   mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225],
                                   max_pixel_value=1
                               ),
                               ToTensorV2(),
                               ])

TEST_TRANSFORMER = Compose([A.Resize(512, 512),
                            A.ToFloat(),
                            A.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=1
                            ),
                            ToTensorV2(),
                            ])

TEST_TIME_TRANSFORMER = Compose([A.Resize(256, 256),
                                 A.RandomBrightnessContrast(0.1, 0.2, p=1.0),
                                 A.OneOf([A.Blur(p=0.1),
                                          A.GaussianBlur(p=0.1),
                                          A.MotionBlur(p=0.1),
                                          ], p=0.1),
                                 A.VerticalFlip(p=0.5),
                                 A.HorizontalFlip(p=0.5),
                                 A.ShiftScaleRotate(shift_limit=0.2,
                                                    scale_limit=0.2,
                                                    rotate_limit=20,
                                                    interpolation=cv2.INTER_LINEAR,
                                                    border_mode=cv2.BORDER_REFLECT_101,
                                                    p=1,
                                                    ),
                                 A.ToFloat(),
                                 A.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225],
                                     max_pixel_value=1
                                 ),
                                 ToTensorV2(), ])
