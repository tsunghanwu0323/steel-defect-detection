from utils import rle2mask

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from albumentations import Compose

aug_null = Compose([])


class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, list_ids, df, target_df=None, mode='fit', base_path='data/train_images', batch_size=32,
                 dim=(256, 1600), n_channels=1, n_classes=4, random_state=2021, shuffle=True, aug=aug_null):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = base_path
        self.target_df = target_df
        self.list_IDs = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = random_state
        self.aug = aug

        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epochs"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]

        X = self.__generate_X(list_IDs_batch)

        if self.mode == 'fit':
            y = self.__generate_y(list_IDs_batch)

            for img in range(len(X)):
                augmented = self.aug(image=X[img], mask=y[img])
                X[img] = augmented['image']
                y[img] = augmented['mask']

            return X, y
        elif self.mode == 'predict':
            return X
        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict".')

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def __generate_X(self, list_IDs_batch):
        """Generates data containing batch_size samples"""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            img_path = f"{self.base_path}/{im_name}"
            img = self.__load_grayscale(img_path)

            # Store example
            X[i, ] = img

        return X

    def __generate_y(self, list_IDs_batch):
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=int)

        for i, ID in enumerate(list_IDs_batch):
            im_name = self.df['ImageId'].iloc[ID]
            image_df = self.target_df[self.target_df['ImageId'] == im_name].copy().reset_index()

            masks = np.zeros((*self.dim, self.n_channels))

            for j in range(len(image_df)):
                rle = image_df.loc[j,'EncodedPixels']
                cls =  image_df.loc[j,'ClassId']
                masks[:, :, cls-1] = rle2mask(rle, self.dim)

            y[i] = masks

        return y

    def __load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)

        return img

    def __load_rgb(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.

        return img