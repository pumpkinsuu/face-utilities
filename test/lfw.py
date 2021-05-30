from io import BytesIO
import numpy as np
from PIL import Image
import pickle

from help_func.find_threshold import calculate_roc


def lfw(path, model, distance, normalize=False, n_fold=10):
    """
    Verification help_func

    :param path: path to lfw bin
    :param model: face model
    :param distance: euclidean or cosine
    :param normalize: normalize embed
    :param n_fold: number of fold
    """
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')

    total = len(bins)
    dist_list = np.empty(total)
    embeds = np.empty((total, model.output))

    for i in range(total):
        img = Image.open(BytesIO(bins[i]))
        embeds[i, ...] = model.embedding(img, normalize)

    embeds1 = embeds[0::2]
    embeds2 = embeds[1::2]
    for i in range(total):
        dist_list[i] = distance(embeds1[i], embeds2[i])

    tpr, fpr, acc, max_tol, min_tol = calculate_roc(dist_list, issame_list, n_fold)

    print(model.name)
    print(f'Accuracy: {acc: .3f}')
    print(f'True p rate: {tpr: .3f}')
    print(f'False p rate: {fpr: .3f}')
    print(f'Threshold: [{min_tol: .3f} -> {max_tol: .3f}]')
