import math
from datetime import datetime
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


class VGG(object):

    def __init__(self, config):
        super(VGG, self).__init__()
        self.config = config
        self.model = self._build()

    def _build(self):
        model = VGG16(include_top=False,
                      input_shape=(224, 224, 3),
                      weights='imagenet'
                      )

        for layer in model.layers:
            layer.trainable = False

        flat1 = Flatten()(model.layers[-1].output)
        den1 = Dense(256, activation='relu')(flat1)
        output = Dense(1, activation='sigmoid')(den1)

        model = Model(inputs=model.inputs, outputs=output)

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        return model

    def load_weights(self, weight_file_path):
        self.model.load_weights(weight_file_path)

    def _compute_steps_per_epoch(self, num_data):
        return math.ceil(1. * num_data / self.config['batch_size'])

    def fit(self, train_gen, val_gen, df_train, df_val):
        steps_per_epoch = self._compute_steps_per_epoch(df_train.shape[0])
        validation_steps = self._compute_steps_per_epoch(df_val.shape[0])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=50)
        mc = ModelCheckpoint(
            self.config['train_model_path'],
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
        filename = "severstal-classification-VGG" + "-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".h5"
        self.model.save(path + filename, save_format='h5')
