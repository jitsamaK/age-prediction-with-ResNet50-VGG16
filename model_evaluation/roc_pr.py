from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from scipy import interp
import numpy as np

def roc_pr(ytrue, yprob, focus_label=1):
    """
    This function is to generate data for ploring roc and pr
    """
    roc = roc_curve(ytrue, yprob, pos_label=focus_label)
    pr = precision_recall_curve(ytrue, yprob, pos_label=focus_label)
    value = {'roc: fpr-tpr-thresholds': roc, 'pr: precision-recall': pr}
	
    return value


def prplt_avg(ytrue, ypred, yprob):    
    """
    This function is to create PR curve on average performance
    """
    ytrue_cat = to_categorical(ytrue, num_classes=len(set(list(set(ytrue))+list(set(ypred)))))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        precision[i], recall[i], _ = precision_recall_curve(ytrue_cat[:, i], yprob[:, i])
        average_precision[i] = average_precision_score(ytrue_cat[:, i], yprob[:, i])
    
    # # Compute micro-average PR curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(ytrue_cat.ravel(), yprob.ravel())
    average_precision["micro"] = average_precision_score(ytrue_cat, yprob, average="micro")
    
    # Visualization    
    plt.figure(figsize=(8,6))
    plt.plot(recall['micro'], precision['micro'], color='dodgerblue',
             label='Micro-averaged Precision: %0.4f' % average_precision["micro"])
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2  ,color='dodgerblue')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    plt.title('Average Precision Score - Micro-averaged')
    #print('Micro-averaged Precision: {0:0.4f}'.format(average_precision["micro"]))
    
    
    
def rocplt_avg(ytrue, ypred, yprob):  
    """
    This function is to create ROC curve on average performance
    """    
    ytrue_cat = to_categorical(ytrue, num_classes=len(set(list(set(ytrue))+list(set(ypred)))))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        fpr[i], tpr[i], _ = roc_curve(ytrue_cat[:, i], yprob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(ytrue_cat.ravel(), yprob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    plt.figure(figsize=(8,6))
    lw = 2
    plt.plot(fpr[2], tpr[2], color='dodgerblue',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc[2])
    plt.fill_between(fpr[2], tpr[2], alpha=0.2, color='dodgerblue')
    plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')              
    plt.title('Receiver Operating Characteristic (ROC) - Micro-averaged')            
              
    plt.legend(loc="lower right")
    plt.show()
    
    
def prplt_byclass(ytrue, ypred, yprob):
    # Compute Precision-Recall and plot curve
    ytrue_cat = to_categorical(ytrue, num_classes=len(set(list(set(ytrue))+list(set(ypred)))))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        precision[i], recall[i], _ = precision_recall_curve(ytrue_cat[:, i], yprob[:, i])
        average_precision[i] = average_precision_score(ytrue_cat[:, i], yprob[:, i])
        
     # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(ytrue_cat.ravel(), yprob.ravel())
    average_precision["micro"] = average_precision_score(ytrue_cat, yprob, average="micro")   

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.figure(figsize=(8,6))
    plt.plot(recall["micro"], precision["micro"], linestyle=':', lw=2,
             label='Micro-average Precision-recall curve (area = {0:0.4f})'
                   ''.format(average_precision["micro"]))
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2)

    
    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        plt.plot(recall[i], precision[i],
                 label='Precision-recall curve of class {0} (area = {1:0.4f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve to Multi-class')
    plt.legend(loc="lower right")
    plt.show()   
    
    
def rocplt_byclass(ytrue, ypred, yprob):
    # Compute macro-average ROC curve and ROC area
    ytrue_cat = to_categorical(ytrue, num_classes=len(set(list(set(ytrue))+list(set(ypred)))))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        fpr[i], tpr[i], _ = roc_curve(ytrue_cat[:, i], yprob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    fpr["micro"], tpr["micro"], _ = roc_curve(ytrue_cat.ravel(), yprob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(set(list(set(ytrue))+list(set(ypred)))))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(set(list(set(ytrue))+list(set(ypred))))

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8,6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)
    plt.fill_between(fpr["macro"], tpr["macro"], alpha=0.2)

    for i in range(len(set(list(set(ytrue))+list(set(ypred))))):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to Multi-class')
    plt.legend(loc="lower right")
    plt.show()