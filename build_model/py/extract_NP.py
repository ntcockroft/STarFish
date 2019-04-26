#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:ntcockroft

Extracts natural product compounds from binding data by comparing a list of
inchikeys and smiles strings of known natural products to a set of binding
data
"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Generate chemical \
                                     fingerprints from smiles strings')
    parser.add_argument('-s', '--smiles', action='store', nargs=1,
                        dest='smiles', help='File containg a list of smiles \
                        of natural products should be in a column named \
                        "smiles" (.csv format)')
    parser.add_argument('-k', '--inchikey', action='store', nargs=1,
                        dest='inchikey', help='File containg a list of \
                        inchikeys of natural products should be in a column \
                        named "inchikey" (.csv format)')
    parser.add_argument('-b', '--binding_data', action='store', nargs=1,
                        dest='binding', help='File containg compounds with \
                        both inchikey and smiles string representations along \
                        with their corresponding uniprot targets. Columns \
                        should be named "inchikey", "smiles", and "uniprot" \
                        respectively (.csv format)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    np_smiles = pd.read_csv(args['input'][0] + args['smiles'][0])
    np_inchikey = pd.read_csv(args['input'][0] + args['inchikey'][0])
    dataset = pd.read_csv(args['input'][0] + args['binding'][0])

    #Extract natural products from binding dataset
    np_smiles_bind = pd.merge(dataset, np_smiles, on='smiles')
    np_inchikey_bind = pd.merge(dataset, np_inchikey, on='inchikey')
    np_bind = pd.concat((np_smiles_bind, np_inchikey_bind), axis=0)

    #Drop any duplicates and account for improper inchikey-smiles pairs
    np_bind.drop_duplicates(inplace=True)
    np_bind.drop_duplicates(subset=['inchikey', 'uniprot'], inplace=True)
    np_bind.drop_duplicates(subset=['smiles', 'uniprot'], inplace=True)

    #Collect the now remaining synthetic data
    synth_inchikey = dataset[~dataset['inchikey'].isin(np_bind['inchikey'])]
    synth_bind = synth_inchikey[~synth_inchikey['smiles'] \
                                .isin(np_bind['smiles'])]

    #Drop any duplicates and account for improper inchikey-smiles pairs
    synth_bind.drop_duplicates(inplace=True)
    synth_bind.drop_duplicates(subset=['inchikey', 'uniprot'], inplace=True)
    synth_bind.drop_duplicates(subset=['smiles', 'uniprot'], inplace=True)

    #No longer need inchikeys, remove before writing output
    del np_bind['inchikey']
    del synth_bind['inchikey']

    np_bind.to_csv(args['output'][0] + '/BindingData_1uM_naturalProduct.csv',
                   index=False)
    synth_bind.to_csv(args['output'][0] + '/BindingData_1uM_synthetic.csv',
                      index=False)


if __name__ == "__main__":
    main()
