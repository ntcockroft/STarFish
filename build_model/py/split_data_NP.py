#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: ntcockroft

Convert compound:target label pairs for natural compounds into the multilabel
format

'''
import argparse
import pandas as pd
from sklearn import preprocessing

def main():
    parser = argparse.ArgumentParser(description='Convert data pairs to \
                                     multilabel format')
    parser.add_argument('-d', '--data', action='store', nargs=1, dest='data',
                        help='SMILES strings for compounds and corresponding \
                        protein targets (.csv format)')
    parser.add_argument('-r', '--reference', action='store', nargs=1,
                        dest='ref', help='y data (protein target labels) of \
                        cross-validation data - used as a reference for \
                        protein target label ordering (.csv format)')
    parser.add_argument('-f', '--fingerprint', action='store', nargs=1,
                        dest='fp', help='File containing corresponding \
                        fingerprints of compounds - use full pathname')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'], help='Directory where \
                        input data files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'], help='Directory where \
                        output files should be written')
    args = vars(parser.parse_args())

    X = pd.read_csv(args['input'][0] + args['data'][0],
                    usecols=['smiles'])
    y = pd.read_csv(args['input'][0] + args['data'][0],
                    usecols=['smiles', 'uniprot'])
    y_ref = pd.read_csv(args['ref'][0])

    #Creates a dict of all possible uniprot ids, each with a different key
    mlb = preprocessing.MultiLabelBinarizer(classes=(list(set(y['uniprot']))))

    """
    We need to begin transforming the data we input into a form appropriate
    for a multi-label classification problem. Right now we just have 1 compound
    corresponding to 1 label. So we will have duplicate compound identifiers if
    a compound has more than 1 label
    """
    group_y = y.groupby('smiles')
    y = group_y.apply(lambda x: x['uniprot'].unique())

    #Drop duplicate compound records from X sets
    X = X.drop_duplicates(subset=['smiles'])

    """
    Removing the duplicate compound identifiers in the datasets altered
    their ordering. Therefore, we need to reconcile this to recover the
    proper feature:label associations. This ultimately produces the
    properly ordered compound:target label pairs in the multilabel format
    """
    y = y.to_frame()
    y.reset_index(level=0, inplace=True)
    y.columns = ['smiles', 'uniprot']
    y = y.sort_values(by=['smiles'])

    X = X.sort_values(by=['smiles'])
    X.reset_index(level=0, inplace=True)
    del X['index']

    #Convert labels from an array of strings to a one hot encoded matrix
    y = y.drop(columns=['smiles'])
    y = mlb.fit_transform(y['uniprot'])

    #Convert to dataframe and add corresponding column names - for book keeping
    y = pd.DataFrame(y, columns=mlb.classes)

    #Only keep targets that overlap with train/test sets
    y = y.drop(y.columns.difference(y_ref.columns), axis=1)

    #Natural product set may not contain all the targets used in training -
    #this will columns with negative data for those targets
    for column in y_ref.columns.difference(y.columns):
        y[column] = 0
    y = y[y_ref.columns]

    #May end up with compounds who now have no targets in the dataset so
    #these will be dropped
    y = y.loc[(y != 0).any(axis=1)]
    X = X.loc[y.index.values]

    for fp in args['fp']:
        fp_dict = pd.read_csv(fp, header=None)
        fp_dict.rename(columns={0:'smiles'}, inplace=True)

        #Add corresponding fingerprints for each smiles string
        X_merged = pd.merge(X, fp_dict, on='smiles')

        #Write set to file
        X_merged.to_csv(args['output'][0] +'/' + '/X_test.csv',
                        index=False)
        y.to_csv(args['output'][0] +'/' + '/y_test.csv',
                 index=False)

if __name__ == "__main__":
    main()
