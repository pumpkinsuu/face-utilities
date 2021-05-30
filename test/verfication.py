import numpy as np

from help_func.utilities import load_dataset
from help_func.find_threshold import calculate_roc


# Random data for verification help_func
def random_v_data(dataset, distance):
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


def verification(path, model, distance, normalize=False, n_test=100, n_fold=10):
    """
    Verification help_func

    :param path: path to dataset
    :param model: face model
    :param distance: euclidean or cosine
    :param normalize: normalize embed
    :param n_test: number of help_func
    :param n_fold: number of fold
    """
    dataset = load_dataset(path, model, normalize)

    m_tpr = 0
    m_fpr = 0
    m_accuracy = 0
    m_max_threshold = 0
    m_min_threshold = 0

    for i in range(n_test):
        dist_list, issame_list = random_v_data(dataset, distance)

        tpr, fpr, accuracy, max_threshold, min_threshold = calculate_roc(dist_list, issame_list, n_fold)
        m_tpr += tpr / n_test
        m_fpr += fpr / n_test
        m_accuracy += accuracy / n_test
        m_max_threshold += max_threshold / n_test
        m_min_threshold += min_threshold / n_test

    print(model.name)
    print(f'Accuracy: {m_accuracy: .3f}')
    print(f'True p rate: {m_tpr: .3f}')
    print(f'False p rate: {m_fpr: .3f}')
    print(f'Threshold: [{m_min_threshold: .3f} -> {m_max_threshold: .3f}]')
