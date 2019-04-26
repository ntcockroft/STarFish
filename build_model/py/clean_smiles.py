#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Applies the molvs (https://github.com/mcs07/MolVS) standardization tool
to each dataset
	Runs RDKit stantize
	Runs RDKit removeHs
	Disconnects metals
	Corrects common drawing errors
	Ionization
	Recalculates stereochemistry

"""
import argparse
from rdkit import Chem
from molvs import Standardizer
import pandas as pd

def clean_smiles(smiles_df):
    """
    Helper function which runs the standardization tool on a list of smiles
    strings.

    Args:
        smiles_df: DataFrame which contains smiles strings in a column named
        "smiles"

    Returns:
        The original DataFrame, but with the smiles strings in the
        "smiles" column standardized and any rows which contained
        problematic smiles removed
    """
    standard = Standardizer(prefer_organic=True)
    for index, row in smiles_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            std_mol = standard.fragment_parent(mol, skip_standardize=False)
            smiles_df['smiles'][index] = Chem.MolToSmiles(std_mol)
        except:
            print("Error cleaning " + str(index) + " " +
                  str(row['smiles']))
            print(smiles_df.loc[index])
            smiles_df.drop(index, inplace=True)
    return smiles_df


def main():
    parser = argparse.ArgumentParser(description='Generate chemical \
                                     fingerprints from smiles strings')
    parser.add_argument('-S', '--smiles', action='store', nargs=1,
                        dest='smiles', help='File containg smiles strings to \
                        be standardized - should be in a column named \
                        "smiles" (.csv format)')
    parser.add_argument('-n', '--name', action='store', nargs=1,
                        dest='name', help='Name of output csv file to write')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())


    dataset = pd.read_csv(args['input'][0] + args['smiles'][0])
    dataset_standard = clean_smiles(dataset)
    dataset_standard.to_csv(args['output'][0] + args['name'][0], index=False)


if __name__ == "__main__":
    main()
