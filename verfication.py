import numpy as np

from help_func.find_threshold import calculate_roc


# Random data for verification help_func
def random_data(dataset, distance):
    dist_list = []
    issame_list = []

    ids = list(dataset.keys())
    np.random.shuffle(ids)
    limit = int(len(ids) / 2)

    for i in ids[:limit]:
        np.random.shuffle(dataset[i])
        np.random.shuffle(dataset[i + limit])
        n = int(len(dataset[i]) / 2)

        for j in range(n):
            dist_list.append(
                distance(
                    dataset[i][j],
                    dataset[i][j + n]
                )
            )
            issame_list.append(True)

            dist_list.append(
                distance(
                    dataset[i][j],
                    dataset[i + limit][j]
                )
            )
            issame_list.append(False)

    return dist_list, issame_list


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
    import time
    from utilities import load_dataset

    df = pd.DataFrame(columns=[
        'model',
        'metric',
        'accuracy',
        'TPR',
        'FPR',
        'min threshold',
        'max threshold',
        'embedding time'
    ])

    for model in models:
        t = time.time()
        dataset = load_dataset(path, model)
        t = time.time() - t

        acc, tpr, fpr, min_tol, max_tol = test(dataset, euclidean, n_test, n_fold)
        df = df.append({
            'model': model.name,
            'metric': 'euclidean',
            'accuracy': acc,
            'TPR': tpr,
            'FPR': fpr,
            'min threshold': min_tol,
            'max threshold': max_tol,
            'embedding time': t
        }, ignore_index=True)
        acc, tpr, fpr, min_tol, max_tol = test(dataset, cosine, n_test, n_fold)
        df = df.append({
            'model': model.name,
            'metric': 'cosine',
            'accuracy': acc,
            'TPR': tpr,
            'FPR': fpr,
            'min threshold': min_tol,
            'max threshold': max_tol,
            'embedding time': t
        }, ignore_index=True)

    return df
