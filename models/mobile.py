"""
https://github.com/sirius-ai/MobileFaceNet_TF
"""
import tensorflow as tf
import numpy as np
from PIL import Image

from models.utilities import load_pb, l2_normalize


class Model:
    def __init__(self, path):
        self.name = 'Mobilefacenet'
        self.input = (112, 112)
        self.output = 128
        self.graph = load_pb(path)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.tf_input = self.graph.get_tensor_by_name('img_inputs:0')
        self.tf_output = self.graph.get_tensor_by_name('embeddings:0')

    def preprocess(self, img: Image):
        _img = img.resize(self.input, Image.ANTIALIAS)
        _img = np.array(_img, dtype='uint8')
        _img = (_img - 127.5) * 0.0078125
        return _img

    def embedding(self, img: Image):
        _img = self.preprocess(img)

        embed = self.sess.run(self.tf_output, feed_dict={self.tf_input: [_img]})[0]

        return l2_normalize(embed)
