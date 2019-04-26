#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Converts sdf file to csv file with compounds represented as smiles and adds a
column with inchikeys. Keeps the information associated with each compound as
columns.
"""
import argparse
from rdkit.Chem import inchi
from rdkit.Chem import PandasTools

def main():
    parser = argparse.ArgumentParser(description='Convert compounds and \
                                     associated information from an sdf file \
                                     into a csv file and generate smiles')
    parser.add_argument('-S', '--sdf', action='store', nargs=1,
                        dest='sdf', help='File containing compounds \
                        (.sdf format)')
    parser.add_argument('-n', '--name', action='store', nargs=1,
                        dest='name', help='Name of output csv file to write')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())


    sdf_df = PandasTools.LoadSDF(args['input'][0] + args['sdf'][0],
                                 smilesName='smiles')
    sdf_df['inchikey'] = [inchi.MolToInchiKey(mol) for mol in sdf_df['ROMol']]
    sdf_df.to_csv(args['output'][0] + args['name'][0], index=False)


if __name__ == "__main__":
    main()
