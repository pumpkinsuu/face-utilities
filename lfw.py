from io import BytesIO
import numpy as np
from PIL import Image
import pickle

from find_threshold import calculate_roc


def load(path, model):
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')

    total = len(bins)
    embeds = np.empty((total, model.output))

    for i in range(total):
        img = Image.open(BytesIO(bins[i])).convert('RGB')
        embeds[i, ...] = model.embedding(img)
    return np.array(issame_list), embeds


def test(issame_list, embeds, distance, n_fold=10):
    """
    Verification LFW

    :param issame_list: is same list
    :param embeds: face embed
    :param distance: euclidean or cosine
    :param n_fold: number of fold
    """
    total = len(issame_list)
    dist_list = np.empty(total)

    embeds1 = embeds[0::2]
    embeds2 = embeds[1::2]
    for i in range(total):
        dist_list[i] = distance(embeds1[i], embeds2[i])

    tpr, fpr, acc, max_tol, min_tol = calculate_roc(dist_list, issame_list, n_fold)

    return acc, tpr, fpr, min_tol, max_tol


def benchmark(path, models, n_fold=10):
    from scipy.spatial.distance import cosine, euclidean
    import pandas as pd

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
        issame_list, embeds = load(path, model)

        acc, tpr, fpr, min_tol, max_tol = test(issame_list, embeds, euclidean, n_fold)
        df = df.append({
            'model': model.name,
            'metric': 'euclidean',
            'accuracy': acc,
            'TPR': tpr,
            'FPR': fpr,
            'min threshold': min_tol,
            'max threshold': max_tol
        }, ignore_index=True)
        acc, tpr, fpr, min_tol, max_tol = test(issame_list, embeds, cosine, n_fold)
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
