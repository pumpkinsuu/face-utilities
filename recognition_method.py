"""
    Face recognition methods
"""
from scipy.spatial.distance import cdist
import numpy as np


def mean_first(test_embeds, known_embeds, metric):
    """
    Mean all face of same person first

    :param test_embeds: numpy array
    :param known_embeds: numpy array
    :param metric: euclidean or cosine
    :return: dist list and index list
    """
    known_embeds = np.mean(known_embeds, axis=1)

    dists = cdist(test_embeds, known_embeds, metric)

    return np.min(dists, axis=1), np.argmin(dists, axis=1)


def mean_later(test_embeds, known_embeds, metric):
    """
    Mean distances of all face of same person after

    :param test_embeds: numpy array
    :param known_embeds: numpy array
    :param metric: euclidean or cosine
    :return: dist list and index list
    """
    shape = known_embeds.shape
    known_embeds = known_embeds.reshape(shape[0] * shape[1], shape[2])

    dists = cdist(test_embeds, known_embeds, metric) \
        .reshape((len(test_embeds), shape[0], shape[1])) \
        .mean(axis=2)

    return np.min(dists, axis=1), np.argmin(dists, axis=1)


def min_later(test_embeds, known_embeds, metric):
    """
    Min distances of all face of same person after

    :param test_embeds: numpy array
    :param known_embeds: numpy array
    :param metric: euclidean or cosine
    :return: dist list and index list
    """
    shape = known_embeds.shape
    known_embeds = known_embeds.reshape(shape[0] * shape[1], shape[2])

    dists = cdist(test_embeds, known_embeds, metric) \
        .reshape((len(test_embeds), shape[0], shape[1])) \
        .min(axis=2)

    return np.min(dists, axis=1), np.argmin(dists, axis=1)
