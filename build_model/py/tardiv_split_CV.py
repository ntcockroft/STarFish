#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Starting point for evaluation of how the protein target class diversity impacts
performance
    Identifies unique ChEMBL L2 protein classifications
    Generates a dataset using only unique L2 protein classes as target labels
    Generates a dataset from a specific L2 class with the same number of target
     labels as the unique protein class label dataset (e.g. all from kinase)
    Splits generates subsets for stratified 10 fold cross-validation

Intended to be used on datasets that were previously subset to contain 10 to
100 compound samples per target protein label
"""

import argparse
import pandas as pd
import numpy as np
from split_data_tune import explode
from split_data_CV import multilabel_cv

def find_nearest(df, value):
    """
    Adapted from:
        https://stackoverflow.com/questions/9706041/finding-index-of-an-item-closest-to-the-value-in-a-list-thats-not-entirely-sort

    Function to identify protein target labels with a similar number of
    compound records to a given protein target label

    Args:
        df: A data frame with the protein target labels as the index and a
        column correspoding to the number of compoun records for each
        value: The number of compound records for the protein target label of
        interest

    Returns:
        A string of the protein target label that had the closest number of
        compounds to the given value
    """
    array = np.asarray(df)
    idx = (np.abs(array - value)).argmin()
    idx_label = df.index[idx]
    return idx_label

def main():
    parser = argparse.ArgumentParser(description='Generates a 10-fold \
                                     cross-validation split for various \
                                     datasets that are designed to examine \
                                     how the methods used in building the \
                                     model are impacted by changes to the \
                                     training data')
    parser.add_argument('-X', '--X_data', action='store', nargs=1, dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs=1, dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-p', '--protein_class', action='store', nargs=1,
                        dest='class', help='List of uniprot ids, in a column \
                        named "uniprot" and associated ChEMBL L2 protein \
                        classes in column named "l2" (.csv format) - use full \
                        pathname')
    parser.add_argument('-f', '--fingerprint', action='store', nargs=1,
                        dest='fp', help='File containing corresponding \
                        fingerprints of compounds - use full pathname')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Load data
    X_multilabel = pd.read_csv(args['input'][0] +
                               args['X'][0], usecols=['smiles'])
    y_multilabel = pd.read_csv(args['input'][0] + args['y'][0])
    protein_classes = pd.read_csv(args['class'][0])
    fp_list = args['fp']


    #Recover pairs from multilabel format
    target_list = (y_multilabel > 0) \
    .apply(lambda w: y_multilabel.columns[w.tolist()].tolist(), axis=1)

    smi_target = pd.DataFrame(X_multilabel['smiles']) \
    .join(pd.DataFrame(target_list))

    smi_target.columns = ['smiles', 'uniprot']
    pairs = explode(smi_target, ['uniprot'], fill_value='')

    #Get uniprot id's for unique l2 classes (with largest number of compounds)
    uniprot = pairs['uniprot'].unique()
    protein_class_uniprot = protein_classes[protein_classes['uniprot']
                                            .isin(uniprot)]
    protein_class_uniprot['l2'].fillna('Unclassified', inplace=True)
    uniprot_l2 = protein_class_uniprot[['uniprot', 'l2']]
    uniprot_l2.drop_duplicates('uniprot', inplace=True)
    l2 = uniprot_l2['l2'].unique()

    uniprot_kinase_all = (uniprot_l2[uniprot_l2['l2']
                      .isin(['Kinase'])]['uniprot'])
    uniprot_kinase_size = y_multilabel[uniprot_kinase_all].sum()

    uniprot_kinase = []
    uniprot_tardiv = []
    for class_label in l2:
        uniprot_class_label = uniprot_l2[uniprot_l2['l2'].isin([class_label])]
        uniprot_labels = uniprot_class_label['uniprot']
        uniprot_label =(y_multilabel[uniprot_labels]
                        .sum()
                        .sort_values(ascending=False)
                        .index[0]
                        )
        uniprot_tardiv.append(uniprot_label)

        uniprot_label_size = (y_multilabel[uniprot_labels]
                              .sum()
                              .sort_values(ascending=False)[0]
                              )
        idx_kinase = find_nearest(uniprot_kinase_size,
                                  uniprot_label_size)
        uniprot_kinase.append(idx_kinase)
        uniprot_kinase_size.drop(idx_kinase, axis='index', inplace=True)

    pairs_tardiv = pairs[pairs['uniprot'].isin(uniprot_tardiv)]
    pairs_kinase = pairs[pairs['uniprot'].isin(uniprot_kinase)]

    multilabel_cv(X=pairs_tardiv['smiles'].to_frame(), y=pairs_tardiv,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_tardiv', n=10)
    multilabel_cv(X=pairs_kinase['smiles'].to_frame(), y=pairs_kinase,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_tardiv_kinase', n=10)

if __name__ == "__main__":
    main()
