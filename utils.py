import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(rle, input_shape):
    width, height = input_shape[:2]

    mask = np.zeros(width * height).astype(np.unit8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    starts -= 1
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return mask.reshape(height, width).T


def build_rles(masks):
    width, height, depth = masks.shape

    rles = [mask2rle(masks[:, :, i]) for i in range(depth)]

    return rles


def setup_gpu():
    # check and set up using GPU for training
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use first GPU and only use 1 GB GPU RAM
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            print('Run on GPU...')
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        print('Run on CPU...')


def history_plot(model_id, history):
    plt.figure(figsize=(14, 16))
    # plot loss
    plt.subplot(1, 2, 1)
    plt.title('Cross Entropy Loss - ' + model_id, fontsize=12)
    plt.plot(history.history['loss'], color='blue', label='train base')
    plt.plot(history.history['val_loss'], color='cyan', label='val base')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')

    # plot accrucy
    plt.subplot(1, 2, 2)
    plt.title('Classification Accuracy ' + model_id, fontsize=10)
    plt.plot(history.history['accuracy'], color='blue', label='train base')
    plt.plot(history.history['val_accuracy'], color='cyan', label='val base')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc='lower right')

    plt.show()


def plot_cm(y_true, y_pred, figsize=(6, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            # TP n TN
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            # FP
            elif i < j:
                annot[i, j] = '%.1f%%\n%d\nFP' % (p, c)
            # FN
            else:
                annot[i, j] = '%.1f%%\n%d\nFN' % (p, c)
    cm = pd.DataFrame(cm, index=['No Defect', 'Defect'], columns=['No Defect', 'Defect'])
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=1.4)
    sns.heatmap(cm, cmap='YlGnBu', annot=annot, fmt='', ax=ax, annot_kws={'fontsize': 12})
    plt.show()


def plot_roc_curve(y_true, y_pred):
    # calculate score
    auc = roc_auc_score(y_true, y_pred)
    # summarize score
    print('ROC AUC=%.3f' % (auc))

    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_true, y_pred)

    # plot roc curve
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='AUC = ' + str(auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
