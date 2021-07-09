import os

from .predictor import Predictor
from .trainer import Trainer
from utils import history_plot, plot_cm, plot_roc_curve


class Classifier(object):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.trainer = None
        self.predictor = None
        self.y_pred = None

    def train(self, train_gen, val_gen, df_train, df_val):
        config = self.config['training']
        self.trainer = Trainer(config, self.logger)

        # Check model is already trained or not
        if not os.path.exists(config['trained_model_path']):
            print('Training...')
            model, history = self.trainer.train(train_gen, val_gen, df_train, df_val)
            history_plot(config['model_name'])

            return model, history
        else:
            print('Load trained model weights...')
            model = self.trainer.load_weights()

            return model, None

    def predict(self, model, test_gen, df_test, sub=False):
        config = self.config['predicting']
        self.predictor = Predictor(config, self.logger, model)

        if not sub:
            y_test, self.y_pred = self.predictor.predict(test_gen, df_test)
            plot_cm(y_test, self.y_pred)
            plot_roc_curve(y_test, self.y_pred)
        else:
            _, y_pred = self.predictor.predict(test_gen, df_test, sub=True)

    def save_result(self, df_sub):
        df_sub['defect_label'] = self.y_pred
        self.predictor.save_result(df_sub)
