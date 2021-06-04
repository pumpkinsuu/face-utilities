"""
https://github.com/ageitgey/face_recognition
"""
import face_recognition as fr
import numpy as np
from PIL import Image
from models.utilities import l2_normalize


class Model:
    def __init__(self):
        self.name = 'Dlib'
        self.input = (150, 150)
        self.output = 128
        self.face_location = [(0, 150, 150, 0)]

    def preprocess(self, img: Image):
        _img = img.convert('RGB').resize(self.input, Image.ANTIALIAS)
        return np.array(_img, dtype='uint8')

    def embedding(self, img: Image):
        _img = self.preprocess(img)

        embed = fr.face_encodings(_img, self.face_location)[0]

        return l2_normalize(embed)
