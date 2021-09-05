import os
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ClassificationPreprocessor(object):

    def __init__(self, config, logger, mode='train'):
        self.config = config
        self.logger = logger

        if mode == 'train':
            self._load_data()
            self._resave_image()
        elif mode == 'sub':
            self._load_sub_data()

    def _load_data(self):
        # load train data and split into train, test and val set
        train_df = pd.read_csv(self.config['input_train_set'])

        # combine defected and no-defect images
        train_img_path = self.config['train_image_path']

        train_image_list = os.listdir(train_img_path)

        image_defects = train_df['ImageId'].unique().tolist()
        image_no_defects = [item for item in train_image_list if item not in image_defects]

        # df_bin is the dataframe with label defect based on train_df
        df_bin = train_df.drop_duplicates(subset=['ImageId'], keep='last').copy()
        df_bin.loc[:, 'defect_label'] = 1

        df_no_defect_bin = pd.DataFrame(image_no_defects, columns=['ImageId'])
        df_no_defect_bin.loc[:, 'defect_label'] = 0

        frames = [df_bin[['ImageId', 'defect_label']], df_no_defect_bin]
        df = pd.concat(frames).reset_index(drop=True)

        # split train data to train, test and validation with 0.6 ,0.2, 0.2 ratio
        self.df_train, self.df_test = train_test_split(df, test_size=0.2, stratify=df['defect_label'], random_state=self.config['seed'])
        self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.25, stratify=self.df_train['defect_label'], random_state=self.config['seed'])

    def _load_sub_data(self):
        # load sample submission for further use
        submission_df = pd.read_csv(self.config['input_sub_set'])
        unique_submission_images = submission_df['ImageId'].unique()
        self.df_sub = pd.DataFrame(unique_submission_images, columns=['ImageId'])

    def _resave_image(self):

        def load_image(code, base, resize=True):
            path = f'{base}/{code}'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if resize:
                img = cv2.resize(img, (self.config['image_size'], self.config['image_size']))

            return img

        def validate_path(path):
            if not os.path.exists(path):
                os.makedirs(path)

        def process_img(df, path):
            # read image, resize and save
            for code in tqdm(df['ImageId']):
                img = load_image(code, base=self.config['train_image_path'])
                img_path = code.replace('.jpg', '')
                cv2.imwrite(f'{path}/{img_path}.png', img)

            df['ImageId'] = df['ImageId'].apply(lambda x: x.replace('.jpg', '.png'))
            return df

        train_path = self.config['temp_train_path']
        validate_path(train_path)
        val_path = self.config['temp_val_path']
        validate_path(val_path)
        test_path = self.config['temp_test_path']
        validate_path(test_path)

        self.df_train = process_img(self.df_train, train_path)
        self.df_val = process_img(self.df_val, val_path)
        self.df_test = process_img(self.df_test, test_path)

    def generate_data(self):
        # using original generator
        data_generator = ImageDataGenerator(
            zoom_range=0.1,
            fill_mode='constant',
            cval=0.,
            rotation_range=20,
            height_shift_range=0.05,
            width_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1/255.,
        )
        val_generator = ImageDataGenerator(
            rescale=1/255.
        )
        train_gen = data_generator.flow_from_dataframe(
            self.df_train,
            directory=self.config['temp_train_path'],
            x_col='ImageId',
            y_col='defect_label',
            class_mode='raw',
            target_size=(self.config['image_size'], self.config['image_size']),
            batch_size=self.config['batch_size']
        )
        val_gen = val_generator.flow_from_dataframe(
            self.df_val,
            directory=self.config['temp_val_path'],
            x_col='ImageId',
            y_col='defect_label',
            class_mode='raw',
            target_size=(self.config['image_size'], self.config['image_size']),
            batch_size=self.config['batch_size']
        )
        test_gen = val_generator.flow_from_dataframe(
            self.df_test,
            directory=self.config['temp_test_path'],
            x_col='ImageId',
            y_col='defect_label',
            class_mode='raw',
            target_size=(self.config['image_size'], self.config['image_size']),
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        return train_gen, val_gen, test_gen, self.df_train, self.df_val, self.df_test

    def generate_sub_data(self):
        submission_gen = ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
            self.df_sub,
            directory=self.config['sub_image_path'],
            x_col='ImageId',
            class_mode=None,
            target_size=(self.config['image_size'], self.config['image_size']),
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        return submission_gen, self.df_sub
