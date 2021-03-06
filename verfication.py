import numpy as np

from find_threshold import calculate_roc


# Random data for verification help_func
def random_data(dataset, distance):
    dist_list = []
    issame_list = []

    labels = list(dataset.keys())
    np.random.shuffle(labels)
    limit = int(len(labels) / 2)

    for i in range(limit):
        np.random.shuffle(dataset[labels[i]])
        np.random.shuffle(dataset[labels[i + limit]])
        n = int(len(dataset[labels[i]]) / 2)

        for j in range(n):
            dist_list.append(
                distance(
                    dataset[labels[i]][j],
                    dataset[labels[i]][j + n]
                )
            )
            issame_list.append(True)

            dist_list.append(
                distance(
                    dataset[labels[i]][j],
                    dataset[labels[i + limit]][j]
                )
            )
            issame_list.append(False)

    return np.array(dist_list), np.array(issame_list)


def test(dataset, distance, n_test=100, n_fold=10):
    """
    Verification help_func

    :param dataset: dataset contain face embed
    :param distance: euclidean or cosine
    :param n_test: number of help_func
    :param n_fold: number of fold
    :return accuracy, tpr, fpr, min_threshold, max_threshold
    """

    m_tpr = 0
    m_fpr = 0
    m_acc = 0
    m_max_tol = 0
    m_min_tol = 0

    for i in range(n_test):
        dist_list, issame_list = random_data(dataset, distance)

        tpr, fpr, accuracy, max_threshold, min_threshold = calculate_roc(dist_list, issame_list, n_fold)
        m_tpr += tpr / n_test
        m_fpr += fpr / n_test
        m_acc += accuracy / n_test
        m_max_tol += max_threshold / n_test
        m_min_tol += min_threshold / n_test

    return m_acc, m_tpr, m_fpr, m_min_tol, m_max_tol


def benchmark(path, models, n_test=100, n_fold=10):
    from scipy.spatial.distance import cosine, euclidean
    import pandas as pd
    from utilities import load_dataset

    df = pd.DataFrame(columns=[
        'model',
        'metric',
        'accuracy',
        'TPR',
        'FPR',
        'min threshold',
        'max threshold'
    ])

    for model in models:
        dataset = load_dataset(path, model)

        acc, tpr, fpr, min_tol, max_tol = test(dataset, euclidean, n_test, n_fold)
        df = df.append({
            'model': model.name,
            'metric': 'euclidean',
            'accuracy': acc,
            'TPR': tpr,
            'FPR': fpr,
            'min threshold': min_tol,
            'max threshold': max_tol
        }, ignore_index=True)
        acc, tpr, fpr, min_tol, max_tol = test(dataset, cosine, n_test, n_fold)
        df = df.append({
            'model': model.name,
            'metric': 'cosine',
            'accuracy': acc,
            'TPR': tpr,
            'FPR': fpr,
            'min threshold': min_tol,
            'max threshold': max_tol
        }, ignore_index=True)

    return df
