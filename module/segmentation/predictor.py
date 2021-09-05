import gc
from utils import build_rles
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from module.segmentation import DataGenerator
from tqdm import tqdm
import pandas as pd
import numpy as np


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


class Predictor(object):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def load_model(self):
        dependencies = {
            'dice_coef': dice_coef
        }
        model = load_model(self.config['save_model_path'], custom_objects=dependencies)

        return model

    def predict(self, model, df_defect, df_no_defect):
        df = []
        step = 300

        # Prediction for the predicted defected images
        for i in range(0, df_defect.shape[0], step):
            batch_idx = list(range(i, min(df_defect.shape[0], i + step)))

            test_generator = DataGenerator(
                batch_idx,
                df=df_defect,
                shuffle=False,
                mode='predict',
                base_path=self.config['test_image_path'],
                target_df=df_defect,
                batch_size=1,
                n_classes=4,
                n_channels=self.config['channels']
            )

            batch_pred_mask = model.predict(
                test_generator,
                verbose=1
            )

            for j, b in tqdm(enumerate(batch_idx)):
                filename = df_defect['ImageId'].iloc[b]

                data = {
                    'ImageId':  [filename, filename, filename, filename],
                    'ClassId': [1, 2, 3, 4],
                    'EncodedPixels': ['', '', '', '']
                }
                image_df = pd.DataFrame(data, columns=['ImageId','ClassId','EncodedPixels'])

                pred_masks = batch_pred_mask[j, ].round().astype(int)
                pred_rles = build_rles(pred_masks)
                for n in range(4):
                    image_df.loc[n, 'EncodedPixels'] = pred_rles[n]

                df.append(image_df)
            gc.collect()
        df = pd.concat(df)

        # Now, we combine results from the predicted masks with the rest of images
        # that our first classification model classified as having all 4 masks missing.
        tt = []
        for img in df_no_defect.index:
            image_df = pd.DataFrame(columns=['ImageId','ClassId','EncodedPixels'])
            for n in range(4):
                image_df.loc[n, 'EncodedPixels'] = np.nan
                image_df.loc[n, 'ClassId'] = n + 1
                image_df.loc[n, 'ImageId'] = df_no_defect.loc[img, "ImageId"]

            tt.append(image_df)

        tt = pd.concat(tt)

        # merge the defected predictions and no-defected
        final_submission_df = pd.concat([df, tt])

        final_submission_df["EncodedPixels"] = final_submission_df["EncodedPixels"].apply(lambda x: np.nan if x == '' else x)
        final_submission_df["ClassId"] = final_submission_df["ClassId"].astype(str)
        final_submission_df['ImageId_ClassId'] = final_submission_df['ImageId'] + "_" + final_submission_df["ClassId"]

        final_submission_df[['ImageId_ClassId', 'EncodedPixels']].to_csv(self.config['submission_file_path'], index=False)
        print("Model: ", self.config['model_name'])
        print("Generated submission file: ", self.config['submission_file_path'])


