import os
from PIL import Image


def mk_dataset(src, dst='dataset'):
    """
    Create face dataset

    :param src: path to images
    :param dst: path to dataset
    """
    if os.path.isdir(dst):
        raise Exception('Destination folder exist')
    os.makedirs(dst)

    from help_func.face_extract import Extractor
    extractor = Extractor()

    folders = os.listdir(src)

    for folder in folders:
        for file in os.listdir(f'{src}/{folder}'):
            img = Image.open(f'{src}/{folder}/{file}')
            img = extractor.extract(img)
            Image.fromarray(img).save(f'{dst}/{folder}_{file}')


def load_dataset(path, model, normalize=False):
    """
    Load face dataset

    :param path: path to dataset
    :param model: face model
    :param normalize: normalize embed
    """
    files = os.listdir(path)
    dataset = {}

    for file in files:
        k = file.split('_')[0]
        if k not in dataset:
            dataset[k] = []

        img = Image.open(f'{path}/{file}')
        embed = model.embedding(img, normalize)
        dataset[k].append(embed)

    return dataset
