#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Takes training and test data from a cross-validation fold and re-combines them
    Takes into account that a compound may be present in both the training and
    test sets (but corresponding to a different target label in each set - and
    therefore are two unique activity points)
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Re-combines training and \
                                     test fold data into a single dataset')
    parser.add_argument('-X', '--X_data', action='store', nargs=2, dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs=2, dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'], help='Directory where \
                        input data files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'], help='Directory where \
                        output files should be written')
    args = vars(parser.parse_args())

    #Sort the X and y dataset arguements passed to keep a consistent order
    args['X'].sort()
    args['y'].sort()

    #Collect and concatenate X data (fingerprints)
    X_0 = pd.read_csv(args['input'][0] + args['X'][0])
    X_1 = pd.read_csv(args['input'][0] + args['X'][1])
    X = pd.concat((X_0, X_1), axis=0)

    #Collect and concatenate y data (target labels)
    y_0 = pd.read_csv(args['input'][0] + args['y'][0])
    y_1 = pd.read_csv(args['input'][0] + args['y'][1])
    y = pd.concat((y_0, y_1), axis=0)

    #Concatenate X and y data (so labels can be sorted by compounds)
    target_list = list(y)
    dataset = pd.concat((X, y), axis=1)

    #Combine target labels for identical compounds
    y_new = dataset.groupby('smiles')[target_list].sum()

    #Get X data (fingerprints) and order them to match their labels
    X_new = X.drop_duplicates()
    X_new = X_new.set_index('smiles')
    X_new = X_new.loc[y_new.index]
    X_new = X_new.reset_index()

    X_new.to_csv(args['output'][0] + 'X_train.csv', index=False)
    y_new.to_csv(args['output'][0] + 'y_train.csv', index=False)

if __name__ == "__main__":
    main()
