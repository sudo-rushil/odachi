
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.linalg as la

from rdkit import Chem as chem
from rdkit.Chem import rdmolops
from rdkit.Chem import MolFromSmiles as ms


class Conv(object):
    possible_atom_list = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg',
                          'Na', 'Br', 'Ca', 'Cu', 'Pd', 'I', 'Al']
    possible_valence_list = list(range(9))
    possible_deg_list = list(range(9))
    possible_hybridization_list = [chem.rdchem.HybridizationType.SP,
                                   chem.rdchem.HybridizationType.SP2,
                                   chem.rdchem.HybridizationType.SP3,
                                   chem.rdchem.HybridizationType.SP3D,
                                   chem.rdchem.HybridizationType.SP3D2]

    def __init__(self, smiles):
        '''Initialize Conv object from SMILES string'''
        self.smiles = smiles
        mol = ms(smiles)

        # Create atom features
        raw_nodes = [self._get_atom_features(a) for a in mol.GetAtoms()]
        atom_features = np.vstack(raw_nodes)
        self.num_atoms, self.num_feat = atom_features.shape
        pad = 130 - self.num_atoms

        # Pad atom features
        feat_paddings = tf.constant([[0, pad], [0,0]])
        self.atom_features = tf.pad(tf.constant(atom_features, dtype=tf.float32),
                                    feat_paddings, 'CONSTANT').numpy()

        # Pad and normalize adjacency matrix
        adj_paddings = tf.constant([[0, pad], [0,pad]])
        A = rdmolops.GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
        D = la.inv(la.sqrtm(np.diag(np.sum(A, axis=1))))
        A = np.matmul(np.matmul(D, A), D)

        self.adj_matrix = tf.pad(tf.constant(A, dtype=tf.float32),
                                 adj_paddings, 'CONSTANT').numpy()

    def _get_one_hot_unk(self, x, allowable_set):
        '''NOTE: Maps inputs not in the allowable set to the last element'''
        if x not in allowable_set:
            x = allowable_set[-1]

        return list(map(lambda s: x == s, allowable_set))

    def _get_atom_features(self, atom):
        results = (self._get_one_hot_unk(atom.GetSymbol(), self.possible_atom_list) +
                   self._get_one_hot_unk(atom.GetDegree(), self.possible_deg_list) +
                   self._get_one_hot_unk(atom.GetImplicitValence(), self.possible_valence_list) +
                   [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] +
                   self._get_one_hot_unk(atom.GetHybridization(), self.possible_hybridization_list) +
                   [atom.GetIsAromatic()])

        return np.array(results)
