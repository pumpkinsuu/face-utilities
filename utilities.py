import os
from PIL import Image


def mk_dataset(detector, src, dst='dataset', align=True):
    """
    Create face dataset

    :param detector: face detector
    :param src: path to images
    :param dst: path to dataset
    :param align: aligned face
    """
    if os.path.isdir(dst):
        raise Exception('Destination folder exist')
    os.makedirs(dst)

    folders = os.listdir(src)

    for folder in folders:
        for file in os.listdir(f'{src}/{folder}'):
            img = Image.open(f'{src}/{folder}/{file}').convert('RGB')
            img = detector.extract(img, align=align)
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

        img = Image.open(f'{path}/{file}').convert('RGB')
        embed = model.embedding(img)
        dataset[k].append(embed)

    return dataset
