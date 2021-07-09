import yaml
import argparse
import logging
import os
from datetime import datetime
from module import Preprocessor, Classifier
from utils import setup_gpu, history_plot, plot_cm, plot_roc_curve


log_folder = './logs/'
filename = '{:%m-%d-%Y}'.format(datetime.now())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process configuration')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()

    formatter = logging.Formatter('%(asctime)-15s %(message)s')
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger('global_logger')

    # create log folder if not exist
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)

    fileHandler = logging.FileHandler(log_folder + filename, 'w', 'utf-8')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    with open(args.config, 'r') as config_file:
        try:
            setup_gpu()
            print('------------------- Classification -------------------')
            print('Preprocessing...')
            config = yaml.safe_load(config_file)
            preprocessor = Preprocessor(config['preprocessing'], logger)
            train_gen, val_gen, test_gen, sub_gen, df_train, df_val, df_test, df_sub = preprocessor.generate_data()

            classifier = Classifier(config['classification'], logger)
            model, history = classifier.train(train_gen, val_gen, df_train, df_val)

            print('Predicting on part of test set...')
            classifier.predict(model, test_gen, df_test)

            print('Predicting on whole test set and create submission file...')
            classifier.predict(model, sub_gen, df_sub, sub=True)


        except yaml.YAMLError as error:
            logger.warning('Config file error: {}'.format(error))



