#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Converts smiles strings to one of four specified fingerprint types
Fingerprint types: Avalon, Morgan ECFP4, Morgan FCFP4, MACCS

"""
import argparse
import csv
import time
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import MACCSkeys, MolFromSmiles
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Generate chemical \
                                     fingerprints from smiles strings')
    parser.add_argument('-S', '--smiles', action='store', nargs=1,
                        dest='smiles',
                        help='List of smiles strings to convert to chemical \
                        chemical fingerprint - should be in a column named \
                        "smiles" (.csv format)')
    parser.add_argument('-f', '--fingerprint', action='store', nargs='*',
                        dest='fingerprints', help='Desired fingerprint type \
                        (avalon, ecfp, fcfp, or maccs)')
    parser.add_argument('-n', '--name', action='store', nargs=1,
                        dest='name', help='Name of fingerprint csv file \
                        to write')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    for fptype in args['fingerprints']:
        data = pd.read_csv(args['input'][0] + args['smiles'][0],
                           usecols=['smiles'])
        ofile = args['output'][0] + args['name'][0]
        time_start = time.time()
        with open(ofile, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            for smiles in data.smiles.unique():
                mol = MolFromSmiles(smiles)
                try:
                    if fptype == 'avalon':
                        fp = GetAvalonFP(mol, nBits=2048)
                    elif fptype == 'ecfp':
                        fp = GetMorganFingerprintAsBitVect(mol, radius=2)
                    elif fptype == 'fcfp':
                        fp = GetMorganFingerprintAsBitVect(mol, radius=2,
                                                           useFeatures=True)
                    elif fptype == 'maccs':
                        fp = MACCSkeys.GenMACCSKeys(mol)

                    fp_bitstr = list(fp.ToBitString())
                    fp_bitstr.insert(0, smiles)
                    writer.writerow(fp_bitstr)
                except:
                    writer.writerow((smiles, "NA"))
                    print('Issue with conversion to ' + fptype +
                          ' fingerprint: ' +str(smiles))
        print('Done writing ' + fptype + ' fingerprints! Time elapsed: \
              {} seconds'.format(time.time()-time_start))

if __name__ == "__main__":
    main()
