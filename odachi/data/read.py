
import os

import numpy as np
import pandas as pd

from rdkit import Chem as chem
from rdkit.Chem import MolFromSmiles as ms


def read_data():
    '''
    Reads reaction data files in /rxn-data/ and concatenates them into one.
    Output: pandas DataFrame containing reaction SMARTS and atom-labeled bond
        changes from the US Patent Office database.
    '''
    current_dir = os.path.dirname(os.path.realpath('__file__'))

    train_data_file = os.path.join(current_dir, 'odachi/data/rxn-data/train.txt')
    valid_data_file = os.path.join(current_dir, 'odachi/data/rxn-data/valid.txt')
    test_data_file = os.path.join(current_dir, 'odachi/data/rxn-data/test.txt')

    train = pd.read_csv(train_data_file, sep=' ', engine='python', header=None)
    valid = pd.read_csv(valid_data_file, sep=' ', engine='python', header=None)
    test = pd.read_csv(test_data_file, sep=' ', engine='python', header=None)

    return train.append(valid, ignore_index=True).append(test, ignore_index=True)


def mol_with_atom_index(mol):
    '''
    Convert atom indexes in RDKit mol from given to canonicalized values.
    Input: RDKit mol with custom or no atom indexes.
    Output: RDKit mol with canonicalized atom indexes.
    '''
    num_atoms = mol.GetNumAtoms()

    for idx in range(num_atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))

    return mol


def _rxn_to_mol(data):
    '''
    Parse reaction SMARTS for molecular SMILES of reactants and product for
    all entries in data.
    Input: DataFrame with reaction SMARTS in first column.
    Output: react: List of reactant SMILES
            prod: List of product SMILES
    '''
    react, prod = list(), list()

    for rxn in data[0]:
        r, p = rxn.split(">>")
        react.append(r)
        prod.append(p)

    return react, prod


def _bond_to_pair(data):
    '''
    Parse striings of changed bonds into tuples of bonds as pairs of atoms.
    Input: DataFrame with changed bond string in second column.
    Output: bond_list: List of list of changed bond tuples.
    '''
    bond_list = list()

    for bond_strings in data[1]:
        bonds = list()
        bond_str = bond_strings.split(';')

        for bond_str in bond_strings.split(';'):
            bond = tuple(map(int, bond_str.split('-')))
            bonds.append(bond)

        bond_list.append(bonds)

    return bond_list


def process_data(data):
    '''
    Combine operations of rxn_to_mol and bond_to_pair to create a new DataFrame.
    Input: Initial DataFrame.
    Output: Reprocessed DataFrame with changed bond tuples and separatated
        reactant and product SMILES.
    '''
    react, prod = _rxn_to_mol(data)
    bonds = _bond_to_pair(data)

    return pd.DataFrame(zip(*[bonds, prod, react]), columns=["Bonds", "Products", "Reactants"])


def _find_in_smiles(idx, reactants):
    '''
    Find atom label in list of reactant SMILES.
    Input:  idx: atom index.
            reactants: list of reactant SMILES.
    Output: reactant: SMILES of reactant that has the indicated atom.
    '''
    search = f':{idx}]'
    for reactant in reactants:
        if search in reactant:
            return reactant


def assign_clusters(idx, data):
    '''
    Takes as an input a data entry, and outputs a list with labels of what
    atoms in the product come from which reactants.
    Input:  idx: Entry index.
            data: Dataset after process_data operation.
    Output: mapping: List of correspondances between product atoms in reactants.
    '''
    reacts = data["Reactants"][idx].split('.')
    mol = ms(data["Products"][idx])
    atoms = mol.GetNumAtoms()
    mapping = list()

    for i in range(atoms):
        atom_str = mol.GetAtomWithIdx(i).GetProp('molAtomMapNumber')
        mapping.append(_find_in_smiles(atom_str, reacts))
        mol.GetAtomWithIdx(i).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(i).GetIdx()))

    distinct = list(sorted(set(mapping)))

    for j in range(atoms):
        mapping[j] = distinct.index(mapping[j])

    return mapping
