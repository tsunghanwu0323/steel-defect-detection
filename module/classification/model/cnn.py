import math
from datetime import *
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report


class CNN(object):

    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.model = self._build()

    def _build(self):
        image_size = self.config['image_size']
        model = Sequential()

        # conv layer
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                         input_shape=(image_size, image_size, 3)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', ))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        model.add(Flatten())

        # fc1 layer
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(rate=0.5))

        # fc2 layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(rate=0.5))

        # The output of the model
        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
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
        filename = "severstal-classification-CNNbase" + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".h5"
        self.model.save(path + filename, save_format='h5')