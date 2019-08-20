from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

def overall_eva(y_valid, ypred, yprob):
    """
    This function is to elimiate english stopwords.
    Each input is a list and the output will be a dictionary of all model evaluations.
    """
    accuracy = round(accuracy_score(y_valid, ypred), 3)
    precision = round(precision_score(y_valid, ypred), 3)
    recall = round(recall_score(y_valid, ypred), 3)
    F1 = round(f1_score(y_valid, ypred), 3)
    auc = round(roc_auc_score(y_valid, yprob),2)
    confusion_value = confusion_matrix(ypred, y_valid).ravel()
    average_precision = round(average_precision_score(y_valid, yprob), 3)
	
    evalution = {'accuracy': accuracy,
                 'precision': precision,
                 'recall': recall,
                 'F1': F1,
                 'auc': auc,
                 'tn, fn, fp, tp': confusion_value,
                 'average_precision_score': average_precision
                }
    return evalution