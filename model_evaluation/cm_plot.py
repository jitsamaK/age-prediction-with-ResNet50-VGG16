import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def cm_plot(ytrue, ypred, normalize=False, cmap=plt.cm.Blues, label='default'):
    """
    The function is to plot a confusion matrix with and without normalization
        ytrue = list or array contains y true
        ypred = list or array contain y predict
        normalize = True or False
        label = 'default' or list of label
    """
    
    label_names = np.array(list(set(ytrue)))
    yhat = np.argmax(ypred,axis=1)
    cm = confusion_matrix(ytrue, yhat)
    
    if type(label) == str:
        axis_label = label_names
    else:
        axis_label = label
    
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix - % Over True Label'
    else:
        title = 'Non-normalized Confusion Matrix - Actual Number'
    
    fig = plt.figure()
    fig.set_size_inches(8, 6, forward=True)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, axis_label, rotation=90)
    plt.yticks(tick_marks, axis_label)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('ytrue')
    plt.xlabel('ypred')
    