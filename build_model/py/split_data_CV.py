#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: ntcockroft

Splits data for stratified 10 fold cross-validation and converts
compound:target pairs into a multilabel format

'''

import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing


def multilabel_cv(X, y, fp_list, output_dir, n=10):
    """
    Function to generate stratified cross validation datasets for training
    multilabel classifiers

    Args:
        X: Input features
        y: target labels
        fp_list: list of strings corresponding to differet chemical
                 fingerprints (e.g. ecfp, fcfp, avalon, maccs)
        output_dir: output directory to ouput folders containg training and
                    test sets for each fold
        n: number of cross-validation folds

    Returns:
        Writes training and test datasts for each fold. Generates a folder for
        each cross-validation fold and writes files for the training set input
        features, training set target labels, test set input features, and test
        set target labels in each
    """
    #Generate data subsets for cross validation
    skf = StratifiedKFold(n_splits=n)

    #Creates a dict of all possible uniprot ids, each with a different key
    mlb = preprocessing.MultiLabelBinarizer(classes=(list(set(y['uniprot']))))

    i = 1
    for train_index, test_index in skf.split(X, y['uniprot']):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        """
        We need to begin transforming the data we input into a form appropriate
        for a multi-label classification problem. Right now we just have 1
        compound corresponding to 1 label. So we will have duplicate compound
        identifiers if a compound has more than 1 label
        """
        group_y_train = y_train.groupby('smiles')
        group_y_test = y_test.groupby('smiles')
        y_train = group_y_train.apply(lambda x: x['uniprot'].unique())
        y_test = group_y_test.apply(lambda x: x['uniprot'].unique())

        #Drop duplicate compound records from X sets
        X_train = X_train.drop_duplicates(subset=['smiles'])
        X_test = X_test.drop_duplicates(subset=['smiles'])

        """
        Removing the duplicate compound identifiers in the datasets altered
        their ordering. Therefore, we need to reconcile this to recover the
        proper feature:label associations. This ultimately produces the
        properly ordered compound:target label pairs in the multilabel format
        """
        y_train = y_train.to_frame()
        y_train.reset_index(level=0, inplace=True)
        y_train.columns = ['smiles', 'uniprot']
        y_train = y_train.sort_values(by=['smiles'])

        y_test = y_test.to_frame()
        y_test.reset_index(level=0, inplace=True)
        y_test.columns = ['smiles', 'uniprot']
        y_test = y_test.sort_values(by=['smiles'])

        X_train = X_train.sort_values(by=['smiles'])
        X_test = X_test.sort_values(by=['smiles'])

        #Convert labels from an array of strings to a one hot encoded matrix
        y_train = y_train.drop(columns=['smiles'])
        y_train = mlb.fit_transform(y_train['uniprot'])

        y_test = y_test.drop(columns=['smiles'])
        y_test = mlb.fit_transform(y_test['uniprot'])

        #Convert to dataframe and add corresponding column names
        y_train = pd.DataFrame(y_train, columns=mlb.classes)
        y_test = pd.DataFrame(y_test, columns=mlb.classes)

        for fp in fp_list:
            fp_dict = pd.read_csv(fp, header=None)
            fp_dict.rename(columns={0:'smiles'}, inplace=True)

            #Add corresponding fingerprints for each smiles string
            X_train_merged = pd.merge(X_train, fp_dict, on='smiles')
            X_test_merged = pd.merge(X_test, fp_dict, on='smiles')

            #Write training and test sets to file
            X_train_merged.to_csv(output_dir +'/' + str(i) +
                                  '/X_train.csv', index=False)
            X_test_merged.to_csv(output_dir +'/' + str(i) +
                                 '/X_test.csv', index=False)
            y_train.to_csv(output_dir +'/' + str(i) +
                           '/y_train.csv', index=False)
            y_test.to_csv(output_dir +'/' + str(i) +
                          '/y_test.csv', index=False)
        i += 1


def main():
    parser = argparse.ArgumentParser(description='Convert data pairs to \
                                     multilabel format and generate \
                                     stratified cross-validation sets')
    parser.add_argument('-d', '--data', action='store', nargs=1, dest='data',
                        help='SMILES strings for compounds and corresponding \
                        protein targets (.csv format)')
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
    fp_list = args['fp']

    multilabel_cv(X, y, fp_list, output_dir=args['output'][0], n=10)

if __name__ == "__main__":
    main()
