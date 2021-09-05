import yaml
import argparse
import logging
import os
from datetime import datetime
from module import ClassificationPreprocessor, Classifier
from module import segmentation
from utils import setup_gpu, history_plot, plot_cm, plot_roc_curve


log_folder = './logs/'
filename = '{:%m-%d-%Y}'.format(datetime.now())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process configuration')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, default='train')
    parser.add_argument('--log_level', type=str, default='INFO')
    args = parser.parse_args()
    mode = args.mode

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
            config = yaml.safe_load(config_file)
            if mode == 'train':
                print('------------------- Classification -------------------')
                print('Preprocessing...')
                preprocessor = ClassificationPreprocessor(config['preprocessing'], logger)
                train_gen, val_gen, test_gen, df_train, df_val, df_test = preprocessor.generate_data()

                classifier = Classifier(config['classification'], logger)
                print('Training...')
                model, history = classifier.train(train_gen, val_gen, df_train, df_val)

                print('Predicting on part of test set...')
                classifier.predict(model, test_gen, df_test)

                # print('Predicting on whole test set and create submission file...')
                # classifier.predict(model, sub_gen, df_sub, sub=True)
            elif mode == 'train_test':
                print('test')
            elif mode == 'sub':
                print('------------------- Classification -------------------')
                print('Preprocessing')
                preprocessor = ClassificationPreprocessor(config['preprocessing'], logger, mode='sub')
                sub_gen, df_sub = preprocessor.generate_sub_data()

                print('Load model weights...')
                classifier = Classifier(config['classification'], logger)
                model = classifier.load_weights()

                print('Predicting...')
                submit_test = classifier.predict(model, sub_gen, df_sub, mode='sub')
                df_sub['defect_label'] = (submit_test > 0.5).astype('int32')
                sub_df_defect = df_sub[df_sub['defect_label'] == 1].copy()
                sub_df_no_defect = df_sub[df_sub['defect_label'] == 0].copy()

                print('------------------- Segmentation -------------------')
                predictor = segmentation.Predictor(config['segmentation']['predicting'], logger)
                seg_model = predictor.load_model()
                result = predictor.predict(seg_model, sub_df_defect, sub_df_no_defect)
            else:
                print('mode can only be train, train_test or sub')

        except yaml.YAMLError as error:
            logger.warning('Config file error: {}'.format(error))
