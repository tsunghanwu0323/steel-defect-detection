import pandas as pd
from sklearn.model_selection import train_test_split
from module.segmentation.data_generator import DataGenerator
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose, RandomGamma, Rotate,IAAAffine
)


class SegmentationPreprocessor(object):

    def __init__(self, config, logger, mode='train'):
        self.config = config
        self.logger = logger

        if mode == 'train':
            self._load_data()

    def _load_data(self):
        self.train_df = pd.read_csv(self.config['input_train_set'])
        mask_count_df = self.train_df.groupby('ImageId')["ClassId"].count().reset_index()\
            .rename(columns={"ClassId": "Num_ClassId"})
        self.mask_count_df = mask_count_df.sort_values('Num_ClassId', ascending=False, inplace=True)

        self.train_idx, self.val_idx = train_test_split(
            mask_count_df.index,
            random_state=2021,
            test_size=0.2
        )

    def generate_data(self):
        aug = Compose([
            Blur(p=0.2, blur_limit=2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            HorizontalFlip(p=0.5),
            Rotate(limit=5, p=0.3),
            VerticalFlip(p=0.5),
        ])

        train_generator = DataGenerator(
            self.train_idx,
            df=self.mask_count_df,
            target_df=self.train_df,
            batch_size=self.config['batch_size'],
            n_classes=self.config['n_classes'],
            aug=aug
        )

        val_generator = DataGenerator(
            self.val_idx,
            df=self.mask_count_df,
            target_df=self.train_df,
            batch_size=self.config['batch_size'],
            n_classes=self.config['n_classes']
        )

        return train_generator, val_generator
