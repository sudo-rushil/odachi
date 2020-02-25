
import os
import time
import itertools
import numpy as np
import tensorflow as tf

from collections import deque
from sklearn.cluster import KMeans

from rdkit import Chem as chem
from rdkit.Chem import rdDepictor, rdmolops
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles as ms

from odachi.data.read import mol_with_atom_index
from odachi.data.conv import Conv
from odachi.engine.layers import ConvEmbed

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


class _OdachiEngine(tf.keras.Model):
    def __init__(self, version=9, knock=0):
        super(OdachiEngine, self).__init__()

        self.embed = ConvEmbed(4, knock)
        self.embed.load_weights(os.path.join(MODEL_DIR, f'models/odachi_embed_v1_e{version}.h5'))

        self.classifier = tf.keras.models.load_model(os.path.join(MODEL_DIR, f'models/odachi_class_v1_e{version}.h5'), compile=False)

    def call(self, adj_matrix, atom_features, num_atoms):
        embed = self.embed([adj_matrix, atom_features])

        idxs = list(itertools.combinations(np.arange(num_atoms), 2))
        vecs = deque(maxlen=len(idxs))

        for idx in idxs:
            vecs.append(tf.reshape(tf.gather(embed, idx, axis=1), [1, 82]))

        vec = tf.concat(list(vecs), 0)

        return idxs, self.classifier(vec)


class Odachi:
    def __init__(self, knock=0):
        self.engines = [_OdachiEngine(i, knock) for i in range(10)]

    def _build_adj(self, conv, idxs, classes):
        similarity = np.eye(conv.num_atoms)

        for (i, j), cl in zip(idxs, classes):
            similarity[i, j] = similarity[j, i] = cl[-1]

        return similarity

    def _cluster(self, similarity, num_clusters=2):
        D = np.diag(similarity.sum(axis=1))
        L = D - similarity

        vals, vecs = np.linalg.eig(L)
        vecs = vecs[:, np.argsort(vals)]
        vals = vals[np.argsort(vals)]

        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(vecs[:,1:num_clusters])
        colors = kmeans.labels_

        return colors

    def _get_broken_bonds(self, mol, colors):
        bonds = []

        for idx in range(mol.GetNumBonds()):
            a = mol.GetBondWithIdx(idx).GetBeginAtomIdx()
            b = mol.GetBondWithIdx(idx).GetEndAtomIdx()
            if colors[a] != colors[b]:
                bonds.append(idx)

        return bonds

    def __call__(self, smiles, clusters = 2, version = 9):
        assert 0 <= version <= 9, "Version error: pick version between 0 and 9"

        start = time.time()

        imol, iconv = ms(smiles), Conv(smiles)

        similarity = self._build_adj(iconv, *self.engines[version](iconv.adj_matrix, iconv.atom_features, iconv.num_atoms))
        colors = self._cluster(similarity, clusters)
        bonds = self._get_broken_bonds(imol, colors)

        rdDepictor.Compute2DCoords(imol)
        drawer = rdMolDraw2D.MolDraw2DSVG(600,300)
        drawer.DrawMolecule(imol, highlightAtoms=[], highlightBonds=bonds)
        drawer.FinishDrawing()

        svg = drawer.GetDrawingText().replace('svg:','')
        # open(os.path.join(os.path.dirname(os.path.realpath('__file__')), f'{smiles}.svg'), 'w').write(svg)

        return {'smiles': smiles, 'svg': svg, 'bonds': bonds, 'time': time.time() - start}
