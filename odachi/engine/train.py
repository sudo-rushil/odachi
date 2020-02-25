
import os
import glob
import itertools
import numpy as np
import tensorflow as tf

from collections import deque
from odachi.engine.layers import ConvEmbed, conv_classifier


def map2retro(mapping):
    '''Convert mapping list into retrosynthetic similarity matrix.'''
    ret = np.logical_xor.outer(mapping, mapping)
    return np.logical_not(ret).astype(np.float32)


def train_batch(file):
    '''Use subbatch of Conv molecule to train the model on subset of possible atom combinations.'''
    loss = 0

    with tf.GradientTape() as tape:
        # Define subbatch size
        for _ in range(100):

            # Open Conv object:
            conv = pickle.load(file)

            # Perform DGCN embedding
            embed = odachi_embed([conv.adj_matrix, conv.atom_features])

            # Define possible indexes and targets
            idxs = list(itertools.combinations(np.arange(conv.num_atoms), 2))
            retro = map2retro(conv.mapping)[np.triu_indices(conv.num_atoms, 1)]
            targs = tf.keras.utils.to_categorical(retro, 2)
            vecs = deque(maxlen=len(idxs))

            # Classify retrosynthetic similarity on all possible combinations of feature vectors
            for aidx in idxs:
                vecs.append(tf.reshape(tf.gather(embed, aidx, axis=1), [1, 82]))

            vec = tf.concat(list(vecs), 0)
            out = classifier(vec)
            loss += loss_fn(targs, out)

    # Define model stack variables
    variables = odachi_embed.trainable_variables + \
                classifier.trainable_variables

    # Calculate and apply gradients
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))

    return loss


def train_file(filename, idx=None):
    '''Train model stack on all subbatches of Conv objects in a file - defines one 'batch'.'''
    loss_avg = tf.keras.metrics.Mean()
    file = open(filename, 'rb')

    for i in range(10):
        batch_loss = train_batch(file)
        loss_avg(batch_loss)
    file.close()
    # print(f'Epoch {epoch} Batch {idx} Loss {loss_avg.result().numpy():.10f} Time {time.time() - start:.4f}')

    return loss_avg.result().numpy()


def validate_single(conv, tries=10):
    '''Validate model performance on a single Conv.'''
    metrics = tf.keras.metrics.BinaryAccuracy()
    embed = odachi_embed([conv.adj_matrix, conv.atom_features])
    idxs = list(itertools.combinations(np.arange(conv.num_atoms), 2))
    tries = len(idxs) if len(idxs) < tries else tries

    for aidx in random.sample(idxs, tries):
        targ = tf.constant([[0, 1]], dtype=tf.float32) \
               if conv.mapping[aidx[0]] == conv.mapping[aidx[1]] \
               else tf.constant([[1, 0]], dtype=tf.float32)

        vec = odachi_ext([embed, aidx])
        out = classifier(vec)
        metrics.update_state(targ, out)

    return metrics.result().numpy()


def validate(filename, tries=10):
    '''Validate model performance on all Conv objects in a file.'''
    # print('\nValidating performance')
    val_acc_avg = tf.keras.metrics.Mean()
    file = open(filename, 'rb')

    while True:
        try:
            conv = pickle.load(file)

        except EOFError:
            break

        acc = validate_single(conv, tries)
        val_acc_avg(acc)

    file.close()
    # print(f'Epoch {epoch} Accuracy {val_acc_avg.result().numpy():.6f}')

    return val_acc_avg.result().numpy()


def train_epoch():
    '''Train model for a single epoch: iterate through all Conv files.'''
    convs = glob.glob('data/*')
    epoch_loss = tf.keras.metrics.Mean()
    valid_acc = tf.keras.metrics.Mean()

    for idx, convfile in enumerate(convs[:-5]):
        batch_loss = train_file(convfile, idx)
        epoch_loss(batch_loss)

    for convfile in convs[-5:]:
        val_acc = validate(convs[-1])
        valid_acc(val_acc)

    return epoch_loss.result().numpy(), valid_acc.result().numpy()


def save_model(idx):
    '''Save models for a given epoch number.'''
    odachi_embed.save_weights(f'odachi_embed_v1_e{idx}.h5')
    classifier.save(f'odachi_class_v1_e{idx}.h5')


if __name__ == '__main__':
    convs = glob.glob(os.path.join(os.path.dirname(os.path.realpath('__file__')), '/conv-data/*'))
    odachi_embed, classifier = ConvEmbed(4, 0.05), conv_classifier()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    EPOCHS = 10

    for epoch in range(EPOCHS):
        print(f'Beginning epoch {epoch}')

        epoch_loss, val_accs = train_epoch()
        print(f'Epoch {epoch} Training Loss {epoch_loss:.10f}')
        print(f'Validation Accuracy {val_accs:.4f}')
        save_model(epoch)
