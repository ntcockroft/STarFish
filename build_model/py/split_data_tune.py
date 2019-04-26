#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Generates a training and testing set for classifier calibration
Intended to be used on datasets that were previously partitioned into
    cross-validation folds (further splits the training data from the fold)
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def explode(df, lst_cols, fill_value=''):
    """
    Allows recovery of compound:label pairs from the multilabel dataframe
    Adapted from: https://stackoverflow.com/questions/12680754/split-explode-pandas-dataframe-string-entry-to-separate-rows

    Args:
        df: DataFrame containg compounds and their labels in a one hot encoded
            format
        lst_cols: List of corresponding labels for each compound - the column
            names for which there was a 1 present in that row
        fill_value: Value which fills in for entries which did not have any
            labels
    Returns:
        A DataFrame of compound:label pairs

    """
    # make sure `lst_cols` is a list
    if lst_cols and not isinstance(lst_cols, list):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)

    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()

    if (lens > 0).all():
        # ALL lists in cells aren't empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .loc[:, df.columns]
    else:
        # at least one list in cells is empty
        return pd.DataFrame({
            col:np.repeat(df[col].values, lens)
            for col in idx_cols
        }).assign(**{col:np.concatenate(df[col].values) for col in lst_cols}) \
          .append(df.loc[lens == 0, idx_cols]).fillna(fill_value) \
          .loc[:, df.columns]


def multilabel_split(X, y, fp_dict, output_dir, test_size=0.1):
    """
    Function to generate stratified random training/test splits for training
    multilabel classifiers

    Args:
        X: Input features
        y: target labels
        fp_dict: A data frame which contains SMILES strings in a column names
        'smiles' and their associated fingerprint bitstring (a column per bit).
        Allows for the look up of pre-computed bitstrings based on the SMILES
        string
        output_dir: output directory to ouput folders containg training and
                    test sets for each fold
        test_size: proportion of dataset that should be used for the test set

    Returns:
        Writes files for the training set input features, training set target
        labels, test set input features, and test set target labels for the
        generated stratified random training/test split
    """
    #Creates a dict of all possible uniprot ids, each with a different key
    mlb = preprocessing.MultiLabelBinarizer(classes=(list(set(y['uniprot']))))

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        stratify=y['uniprot'])
    X_train, X_test = X.iloc[X_train.index], X.iloc[X_test.index]
    y_train, y_test = y.iloc[y_train.index], y.iloc[y_test.index]


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

    #Convert to dataframe and add corresponding column names - for book keeping
    y_train = pd.DataFrame(y_train, columns=mlb.classes)
    y_test = pd.DataFrame(y_test, columns=mlb.classes)


    #Add corresponding fingerprints for each smiles string
    X_train_merged = pd.merge(X_train, fp_dict, on='smiles')
    X_test_merged = pd.merge(X_test, fp_dict, on='smiles')

    #Write training and test sets to file
    X_train_merged.to_csv(output_dir + '/X_train.csv', index=False)
    X_test_merged.to_csv(output_dir + '/X_test.csv', index=False)
    y_train.to_csv(output_dir +  '/y_train.csv', index=False)
    y_test.to_csv(output_dir + '/y_test.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Split the training data \
                                     from a fold into a training and \
                                     calibration set')
    parser.add_argument('-X', '--X_data', action='store', nargs=1, dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs=1, dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    X_multilabel = pd.read_csv(args['input'][0] +
                               args['X'][0], usecols=['smiles'])
    y_multilabel = pd.read_csv(args['input'][0] + args['y'][0])
    fp_dict = pd.read_csv(args['input'][0] + args['X'][0])

    #Recover pairs from multilabel format
    target_list = (y_multilabel > 0)\
    .apply(lambda w: y_multilabel.columns[w.tolist()].tolist(), axis=1)

    smi_target = pd.DataFrame(X_multilabel['smiles'])\
    .join(pd.DataFrame(target_list))

    smi_target.columns = ['smiles', 'uniprot']
    pairs = explode(smi_target, ['uniprot'], fill_value='')

    X = pd.DataFrame(pairs['smiles'], columns=['smiles'])
    y = pd.DataFrame(pairs, columns=['smiles', 'uniprot'])

    multilabel_split(X, y, fp_dict, output_dir=args['output'][0],
                     test_size=0.1)


if __name__ == "__main__":
    main()
