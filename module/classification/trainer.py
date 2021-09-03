from module.classification.model import CNN, DenseNet, VGG


class Trainer:

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._create_model()

    def _create_model(self):
        if self.config['model_name'] == 'cnn':
            self.model = CNN(self.config)
        elif self.config['model_name'] == 'densenet':
            self.model = DenseNet(self.config)
        elif self.config['model_name'] == 'vgg':
            self.model = VGG(self.config)

    def train(self, train_gen, val_gen, df_train, df_val):
        history = self.model.fit(train_gen, val_gen, df_train, df_val)

        return self.model, history

    def load_weights(self):
        self.model.load_weights(self.config['trained_model_path'])

        return self.model



