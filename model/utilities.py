import tensorflow as tf


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
