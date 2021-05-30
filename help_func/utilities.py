import os
from PIL import Image


def extract_face(path, model):
    """
    Extract face from dataset

    :param path: path to dataset
    :param model:
    :return:
    """



def load_dataset(path, model):
    """
    Load face dataset

    :param path: path to dataset
    :param model: face model
    """
    folders = os.listdir(path)
    dataset = {k: [] for k in folders}

    for folder in folders:
        for file in os.listdir(f'{path}/{folder}'):
            img = Image.open(f'{path}/{folder}/{file}')

            embed = model.embedding(img)
            dataset[folder].append(embed)

    return dataset
