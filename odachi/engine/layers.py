
import numpy as np
import tensorflow as tf


class GraphConv(tf.keras.layers.Layer):
    '''Tensorflow layer for performing Kipf-style graph convolutions'''
    def __init__(self, n,
                 num_feat=41,
                 num_atoms=130,
                 activation=tf.nn.elu,
                 knockdown = 0.1,
                 BATCH_SIZE=1):

        super(GraphConv, self).__init__(autocast=False)

        self.n = n
        self.num_feat = num_feat
        self.num_atoms = num_atoms
        self.activation = activation
        self.batch_size = BATCH_SIZE

        w_init = tf.random_normal_initializer()
        b_init = tf.zeros_initializer()

        self.w = tf.Variable(initial_value=w_init(shape=(self.batch_size,
                                                         self.num_feat,
                                                         self.num_feat),
                                                  dtype='float32'),
                             trainable=True,
                             name=self.n + '_w')

        self.b = tf.Variable(initial_value=b_init(shape=(self.batch_size,
                                                         self.num_atoms,
                                                         self.num_feat),
                                                  dtype='float32'),
                             trainable=True,
                             name=self.n + '_b')

        self.drop = lambda : tf.cast(tf.greater_equal(tf.random.uniform((self.batch_size,
                                                                         self.num_feat,
                                                                         self.num_feat)),
                                                      knockdown),
                                     tf.float32)

    @tf.function
    def call(self, inputs):
        A, X = inputs
        X = self.activation(A @ (X @ (self.drop() * self.w)) + self.b)
        return A, X


class ConvEmbed(tf.keras.Model):
    '''Convenience class of stacked convolutional layers'''
    def __init__(self,
                 num_feat=41,
                 num_atoms=130,
                 depth = 10,
                 knock = 0.2,
                 BATCH_SIZE=1):
        super(ConvEmbed, self).__init__()

        self.layer_stack = []

        for idx in range(depth):
            n = 'conv_' + str(idx)
            self.layer_stack.append(GraphConv(n, num_feat=num_feat, num_atoms=num_atoms, knockdown=knock, BATCH_SIZE=BATCH_SIZE))

    def call(self, I):
        for layer in self.layer_stack:
            I = layer(I)
        return I[1]


class ExtractVec(tf.keras.layers.Layer):
    '''Pulls comparison vector from convolved features matrix'''
    '''DEPRECIATED'''
    def __init__(self, feat=41):
        super(ExtractVec, self).__init__()
        self.feat = feat

    def call(self, inputs):
        X, (ax, ay) = inputs

        a = np.zeros((1, 130, 1))
        a[:, ax], a[:, ay] = 1, 1
        an = tf.squeeze(tf.cast(a, tf.bool), axis=0)
        an = tf.broadcast_to(an, [130, self.feat])

        X.set_shape((1, 130, self.feat))
        X = tf.boolean_mask(X, an, axis=1)
        X.set_shape((1, 2 * self.feat))

        return X


def conv_classifier():
    classifier = tf.keras.models.Sequential(name='odachi_classifier')
    classifier.add(tf.keras.layers.Dropout(0.5, input_shape=(82,)))

    classifier.add(tf.keras.layers.Dense(60, activation = tf.nn.elu))
    classifier.add(tf.keras.layers.Dropout(0.5))

    classifier.add(tf.keras.layers.Dense(40, activation = tf.nn.elu))
    classifier.add(tf.keras.layers.Dropout(0.5))

    classifier.add(tf.keras.layers.Dense(30, activation = tf.nn.elu))
    classifier.add(tf.keras.layers.Dropout(0.5))

    classifier.add(tf.keras.layers.Dense(20, activation = tf.nn.elu))
    classifier.add(tf.keras.layers.Dropout(0.5))

    classifier.add(tf.keras.layers.Dense(2, activation = 'softmax'))

    return classifier
