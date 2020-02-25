
import os
import argparse
import numpy as np
import tensorflow as tf

from odachi.engine.model import Odachi


def parse():
    parser = argparse.ArgumentParser(description='Run Odachi Retrosynthesis Engine on smiles string')
    parser.add_argument('smiles', type=str, help='a SMILES string for the molecule')
    parser.add_argument('--clusters', type=int, default=2,
                        help='number of disconnections')

    return parser


if __name__ == '__main__':
    odachi = Odachi()
    parser = parse()
    args = parser.parse_args()

    odachi(args.smiles, args.clusters)
