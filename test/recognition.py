import numpy as np

from help_func.utilities import load_dataset
from help_func.find_threshold import calculate_roc


def random_r_data(dataset, method, metric, n_face=3, n_test=3):
    labels = list(dataset.keys())
    np.random.shuffle(labels)

    total = len(labels)
    n = int(total / 2)
    embed_sz = len(dataset[labels[0]][0])

    known_embeds = np.empty((total, n_face, embed_sz))
    known_labels = np.empty(total * n_test)
    test_embeds = np.empty((total * n_test * 2, embed_sz))

    for i in range(n):
        np.random.shuffle(dataset[labels[i]])
        np.random.shuffle(dataset[labels[i+n]])

        known_embeds[i] = dataset[labels[i]][:n_face]

        pos = i * n_test
        known_labels[pos:pos+n_test] = labels[i]
        test_embeds[pos:pos+n_test] = dataset[labels[i]][n_face:n_face+n_test]
        
        pos = (i + n) * n_test
        test_embeds[pos:pos+n_test] = dataset[labels[i+n]][:n_test]

    dist_list, idx_list = method(test_embeds, known_embeds, metric)
    issame_list = np.zeros(len(dist_list), dtype=bool)
    issame_list[:len(known_labels)] = known_labels[idx_list] == known_labels

    return dist_list, issame_list


def recognition(path, model, method, metric, normalize=False, n_face=3, n_face_t=3, n_test=100, n_fold=10):
    """
    Recognition help_func

    :param path: path to dataset
    :param model: face model
    :param method: mean_first, mean_after, min_after
    :param metric: euclidean or cosine
    :param normalize: normalize embed
    :param n_face: number of known faces
    :param n_face_t: number of help_func faces
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
        dist_list, issame_list = random_r_data(
            dataset=dataset,
            method=method,
            metric=metric,
            n_face=n_face,
            n_test=n_face_t
        )

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
