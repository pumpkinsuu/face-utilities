"""
    https://github.com/davidsandberg
    facenet calculate_roc
"""
import numpy as np
from sklearn.model_selection import KFold


MIN_THRESHOLD = 0
MAX_THRESHOLD = 4


# Calculate TPR, FPR and Accuracy
def calculate_accuracy(predict_list, issame_list):
    tp = np.sum(np.logical_and(predict_list, issame_list))
    fp = np.sum(np.logical_and(predict_list, np.logical_not(issame_list)))
    tn = np.sum(np.logical_and(np.logical_not(predict_list), np.logical_not(issame_list)))
    fn = np.sum(np.logical_and(np.logical_not(predict_list), issame_list))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / len(predict_list)
    return tpr, fpr, acc


# Find best threshold by accuracy
def calculate_roc(dist_list, issame_list, n_fold=10):
    size = len(issame_list)

    max_threshold = MIN_THRESHOLD
    min_threshold = MAX_THRESHOLD
    thresholds = np.arange(max_threshold, min_threshold, 0.01)

    k_fold = KFold(n_splits=n_fold, shuffle=True)
    accuracy = 0
    tp_rate = 0
    fp_rate = 0
    indices = np.arange(size)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        acc_train = np.zeros((len(thresholds)))

        for threshold_idx, threshold in enumerate(thresholds):
            predict_list = np.less(dist_list, threshold)
            _, _, acc_train[threshold_idx] = calculate_accuracy(predict_list[train_set], issame_list[train_set])

        best_threshold_index = np.argmax(acc_train)

        predict_list = np.less(dist_list, thresholds[best_threshold_index])
        tpr, fpr, acc = calculate_accuracy(predict_list[test_set], issame_list[test_set])

        tp_rate += tpr
        fp_rate += fpr
        accuracy += acc

        if max_threshold < thresholds[best_threshold_index]:
            max_threshold = thresholds[best_threshold_index]
        if min_threshold > thresholds[best_threshold_index]:
            min_threshold = thresholds[best_threshold_index]

    tp_rate /= n_fold
    fp_rate /= n_fold
    accuracy /= n_fold

    return tp_rate, fp_rate, accuracy, max_threshold, min_threshold
