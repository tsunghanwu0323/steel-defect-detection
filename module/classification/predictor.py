import os


class Predictor(object):

    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model

    def predict(self, test_gen, df_test, sub=False):
        y_test, y_pred = self.model.predict(test_gen, df_test, sub)
        return y_test, y_pred

    def save_result(self, result):
        self.model.save_model(self.config['save_model_path'])

        if not os.path.exists('./submission'):
            os.makedirs('./submission')
        result.to_csv('./submission/submission_df_classification_result.csv', index=False)
