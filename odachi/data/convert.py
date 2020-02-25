
import os
import uuid
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.linalg as la

from rdkit import Chem as chem
from rdkit.Chem import MolFromSmiles as ms

import odachi.data


def write_batch(start, name, data, batch_size=1000):
    '''Write batch of data to a pickle file of Conv objects.'''
    outfile = open(name, 'wb+')

    for idx in range(start, start+batch_size):
        conv = odachi.data.conv.Conv(data.Products[idx])
        conv.mapping = odachi.data.read.assign_clusters(idx, data)
        pickle.dump(conv, outfile)

    outfile.close()


def write_files():
    '''Converts all molecules in data to Conv objects for the model to train on.'''
    raw_data = odachi.data.read.read_data()
    data = odachi.data.read.process_data(raw_data)

    path = os.path.join(os.path.dirname(os.path.realpath('__file__')), '/conv-data/')
    os.mkdir(path)

    for i in range(data.shape[0] // 1000):
        write_batch(i * 1000, path + uuid.uuid4().hex, data)
