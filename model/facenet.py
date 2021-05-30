"""
https://github.com/davidsandberg/facenet
"""
import tensorflow as tf
import numpy as np
from PIL import Image

from model.utilities import load_pb


# Prewhiten image array
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


# Normalize embed array
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


class Model:
    def __init__(self, path):
        self.name = 'Facenet'
        self.input = (160, 160)
        self.output = 512
        self.graph = load_pb(path)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.tf_input = self.graph.get_tensor_by_name('input:0')
        self.tf_output = self.graph.get_tensor_by_name('embeddings:0')
        self.tf_placeholder = self.graph.get_tensor_by_name('phase_train:0')

    def preprocess(self, img: Image):
        _img = img.convert('RGB').resize(self.input, Image.ANTIALIAS)
        _img = np.array(_img, dtype='uint8')
        return prewhiten(_img)

    def embedding(self, img: Image):
        _img = self.preprocess(img)

        feed_dict = {self.tf_input: [_img], self.tf_placeholder: False}
        embed = self.sess.run(self.tf_output, feed_dict=feed_dict)[0]

        return l2_normalize(embed)
