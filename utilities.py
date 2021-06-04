import os
from PIL import Image


def mk_dataset(detector, src, dst='dataset'):
    """
    Create face dataset

    :param detector: face detector
    :param src: path to images
    :param dst: path to dataset
    """
    if os.path.isdir(dst):
        raise Exception('Destination folder exist')
    os.makedirs(dst)

    folders = os.listdir(src)

    for folder in folders:
        for file in os.listdir(f'{src}/{folder}'):
            img = Image.open(f'{src}/{folder}/{file}')
            img = detector.extract(img)
            Image.fromarray(img).save(f'{dst}/{folder}_{file}')


def load_dataset(path, model):
    """
    Load face dataset

    :param path: path to dataset
    :param model: face models
    """
    files = os.listdir(path)
    dataset = {}

    for file in files:
        k = file.split('_')[0]
        if k not in dataset:
            dataset[k] = []

        img = Image.open(f'{path}/{file}')
        embed = model.embedding(img)
        dataset[k].append(embed)

    return dataset


def get_model(name):
    """
    Get face embedding model

    :param name: dlib, facenet or mobile
    :return: model
    """
    if name == 'dlib':
        from models.dlib import Model
        return Model()
    if name == 'facenet':
        from models.facenet import Model
        return Model('models/pb/facenet.pb')
    from models.mobile import Model
    return Model('models/pb/mobile.pb')


def get_detector():
    """
    Get face detector

    :return: face detector
    """
    from help_func.detector import Detector
    return Detector()
