import numpy as np

from find_threshold import calculate_roc


def random_data(dataset, method, metric, n_face=3, n_test=3):
    labels = np.array(list(dataset.keys()))
    np.random.shuffle(labels)

    total = len(labels)
    n = int(total / 2)
    embed_sz = len(dataset[labels[0]][0])

    known_embeds = np.empty((n, n_face, embed_sz))
    known_sz = n * n_test
    known_labels = np.empty(known_sz, dtype='U25')
    test_embeds = np.empty((known_sz * 2, embed_sz))

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
    issame_list[:known_sz] = labels[idx_list[:known_sz]] == known_labels

    return dist_list, issame_list


def test(dataset, method, metric, n_face=3, n_face_t=3, n_test=100, n_fold=10):
    """
    Recognition help_func

    :param dataset: dataset contain face embed
    :param method: mean_first, mean_after, min_after
    :param metric: euclidean or cosine
    :param n_face: number of known faces
    :param n_face_t: number of help_func faces
    :param n_test: number of help_func
    :param n_fold: number of fold
    :return m_acc, m_tpr, m_fpr, m_min_tol, m_max_tol
    """

    m_tpr = 0
    m_fpr = 0
    m_acc = 0
    m_max_tol = 0
    m_min_tol = 0

    for i in range(n_test):
        dist_list, issame_list = random_data(
            dataset=dataset,
            method=method,
            metric=metric,
            n_face=n_face,
            n_test=n_face_t
        )

        tpr, fpr, accuracy, max_threshold, min_threshold = calculate_roc(dist_list, issame_list, n_fold)
        m_tpr += tpr / n_test
        m_fpr += fpr / n_test
        m_acc += accuracy / n_test
        m_max_tol += max_threshold / n_test
        m_min_tol += min_threshold / n_test

    return m_acc, m_tpr, m_fpr, m_min_tol, m_max_tol


def benchmark(path, models, n_face=3, n_face_t=3, n_test=100, n_fold=10):
    import pandas as pd
    from utilities import load_dataset
    from recognition_method import min_later, mean_later, mean_first

    methods = [mean_first, mean_later, min_later]

    df = pd.DataFrame(columns=[
        'model',
        'method',
        'metric',
        'accuracy',
        'TPR',
        'FPR',
        'min threshold',
        'max threshold'
    ])

    for model in models:
        dataset = load_dataset(path, model)

        for method in methods:
            acc, tpr, fpr, min_tol, max_tol = test(
                dataset=dataset,
                method=method,
                metric='euclidean',
                n_face=n_face,
                n_face_t=n_face_t,
                n_test=n_test,
                n_fold=n_fold
            )
            df = df.append({
                'model': model.name,
                'method': method.__name__,
                'metric': 'euclidean',
                'accuracy': acc,
                'TPR': tpr,
                'FPR': fpr,
                'min threshold': min_tol,
                'max threshold': max_tol
            }, ignore_index=True)
            acc, tpr, fpr, min_tol, max_tol = test(
                dataset=dataset,
                method=method,
                metric='cosine',
                n_face=n_face,
                n_face_t=n_face_t,
                n_test=n_test,
                n_fold=n_fold
            )
            df = df.append({
                'model': model.name,
                'method': method.__name__,
                'metric': 'cosine',
                'accuracy': acc,
                'TPR': tpr,
                'FPR': fpr,
                'min threshold': min_tol,
                'max threshold': max_tol
            }, ignore_index=True)

    return df
