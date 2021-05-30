import tensorflow as tf
import numpy as np


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
