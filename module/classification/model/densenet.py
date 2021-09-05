import math
from datetime import *
from tensorflow.keras import Sequential
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


class DenseNet(object):

    def __init__(self, config):
        super(DenseNet, self).__init__()
        self.config = config
        self.model = self._build()

    def _build(self):
        densenet = DenseNet121(
            include_top=False,
            input_shape=(256, 256, 3),
            weights='imagenet'
        )
        model = Sequential()
        model.add(densenet)
        model.add(GlobalAveragePooling2D())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # The output of the model
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=Nadam(),
            metrics=['accuracy']
        )

        return model

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def _compute_steps_per_epoch(self, num_data):
        # We take the ceiling because we do not drop the remainder of the batch
        return math.ceil(1. * num_data / self.config['batch_size'])

    def fit(self, train_gen, val_gen, df_train, df_val):
        steps_per_epoch = self._compute_steps_per_epoch(df_train.shape[0])
        validation_steps = self._compute_steps_per_epoch(df_val.shape[0])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=50)
        mc = ModelCheckpoint(
            self.config['trained_model_path'],
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto'
        )
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=validation_steps,
            epochs=self.config['epochs'],
            callbacks=[es, mc]
        )

        return history

    def predict(self, test_gen, df_test, sub):
        if not sub:
            y_test = df_test['defect_label']
            step_size = self._compute_steps_per_epoch(y_test.shape[0])
        else:
            step_size = self._compute_steps_per_epoch(df_test.shape[0])

        predictions = self.model.predict(
            test_gen,
            steps=step_size,
            verbose=1
        )
        y_pred = (predictions > 0.5).astype('int32')

        if not sub:
            report = classification_report(y_test, y_pred, target_names=['No Defect', 'Defect'])
            print(report)

            return y_test, y_pred
        else:
            return None, y_pred

    def save_model(self, path):
        filename = "densenet_model.h5"
        self.model.save(path + filename, save_format='h5')