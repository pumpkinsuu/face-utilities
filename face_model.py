import face_recognition as fr
import tensorflow as tf
import numpy as np
from PIL import Image


def load_pb(path):
    """
    Load tensorflow .pb file

    :param path: path to .pb file
    :return: graph
    """
    with tf.compat.v1.gfile.GFile(path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph


# Normalize embed array
def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


# Prewhiten image array
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


class Dlib:
    """
    https://github.com/ageitgey/face_recognition
    """
    def __init__(self):
        self.name = 'Dlib'
        self.input = (150, 150)
        self.output = 128
        self.face_location = [(0, 150, 150, 0)]

    def preprocess(self, img: Image):
        _img = img.resize(self.input, Image.ANTIALIAS)
        return np.array(_img, dtype='uint8')

    def embedding(self, img: Image):
        _img = self.preprocess(img)

        embed = fr.face_encodings(_img, self.face_location)[0]

        return l2_normalize(embed)


class Facenet:
    """
    https://github.com/davidsandberg/facenet
    """
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
        _img = img.resize(self.input, Image.ANTIALIAS)
        _img = np.array(_img, dtype='uint8')
        return prewhiten(_img)

    def embedding(self, img: Image):
        _img = self.preprocess(img)

        feed_dict = {self.tf_input: [_img], self.tf_placeholder: False}
        embed = self.sess.run(self.tf_output, feed_dict=feed_dict)[0]

        return l2_normalize(embed)


class Mobile:
    def __init__(self, path):
        """
        https://github.com/sirius-ai/MobileFaceNet_TF
        """
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
