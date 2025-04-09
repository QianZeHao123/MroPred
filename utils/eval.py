# test this model with Precision, Recall, F1 score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve


def evaluate_model(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    # compute Youden's J statistic
    youden_j_stat = tpr - fpr
    best_idx = np.argmax(youden_j_stat)  # find the index of the best threshold
    best_threshold = thresholds[best_idx]  # get the best threshold
    threshold = best_threshold
    # Convert probabilities to binary predictions using the best threshold
    y_pred_binary = np.where(y_pred > threshold, 1, 0)
    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)
    auc = roc_auc_score(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    confusion = confusion_matrix(y_true, y_pred_binary)
    # return precision, recall, f1, accuracy, auc, average_precision
    return {
        "Best Threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "auc": auc,
        "average_precision": average_precision,
        "confusion_matrix": confusion,
    }
