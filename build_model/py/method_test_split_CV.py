#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Starting point for evaluation of how the number of training samples, number of
protein target labels, and intra-label compound similiairty impact overall
model performance

Generates a 10-fold cross validation split for various datasets that are
designed to examine how the methods used in building the model are impacted
by changes to the training data. The datasets are as follows:
    100 compounds per label (using all targets that have 100 compound examples)
    100 compounds per label, but half the number of targets (randomly selected)
    10 compounds per label (randomly sampled from the
                            all target 100 compound dataset)
    10 compounds per label (randomly sampled from the
                            halved target 100 compound dataset)
    10 compounds per label (selected to be most dissimilar and sampled from the
                            all target 100 compound dataset)
    10 compounds per label (selected to be most dissimilar and sampled from the
                            halved target 100 compound dataset)


Intended to be used on datasets that were previously subset to contain 10 to
100 compound samples per target protein label
"""

import argparse
import pandas as pd
import numpy as np
from rdkit.Chem import rdMolDescriptors, MolFromSmiles
from rdkit import SimDivFilters, DataStructs
from split_data_tune import explode
from split_data_CV import multilabel_cv



def dmat_sim(smiles_target, ntopick=10):
    """
    Function to select most dissimilar compounds from a given set
    Adapted from:
        http://rdkit.blogspot.com/2014/08/optimizing-diversity-picking-in-rdkit.html

    Args:
        smiles_target: DataFrame which contains compound-target activity pairs.
        The compounds should be in the smiles strings format and in a column
        named "smiles"
        ntoppick: The number of dissimiliar compounds to pick from the ranked
        list of dissimilarity

    Returns:
        A DataFrame of compound-target activity pairs that were sampled from
        the input smiles_target DataFrame based on their dissimilarity
    """
    ds = []
    smiles_target.reset_index(drop=True, inplace=True)
    mols = [MolFromSmiles(smi) for smi in smiles_target['smiles']]
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 2) for m in mols]
    for i in range(1, len(fps)):
        ds.extend(DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i],
                                                     returnDistance=True))
    mmp = SimDivFilters.MaxMinPicker()
    ids = mmp.Pick(np.array(ds), len(fps), ntopick)
    smiles_target_dissim = smiles_target.iloc[list(ids)]

    return smiles_target_dissim


def get_dissim_compounds(smi_target_pairs, ncompounds=10):
    """
    Helper function to select most dissimilar compounds from a given set
    Runs dmat_sim() over a DataFrame of compound-target pairs after selecting
    pairs which have the same target in common

    Args:
        smiles_target_pairs: DataFrame which contains compound-target activity
        pairs. The compounds should be in the smiles strings format and in a
        column named "smiles". The target uniprot ids should be in a column
        names "uniprot"
        ntoppick: The number of dissimiliar compounds to pick from the ranked
        list of dissimilarity

    Returns:
        A DataFrame of compound-target activity pairs that were sampled from
        the input smiles_target DataFrame based on their dissimilarity
    """
    target_df = pd.DataFrame(smi_target_pairs['uniprot'].unique())
    target_df.columns = ['uniprot']
    pairs_dissim = pd.DataFrame()
    for target in target_df['uniprot']:
        smi_target = smi_target_pairs.loc[smi_target_pairs['uniprot'] \
                                          == target]
        pairs_dissim = pd.concat([pairs_dissim, dmat_sim(smi_target,
                                                         ntopick=ncompounds)])

    pairs_dissim.reset_index(drop=True, inplace=True)
    return pairs_dissim


def main():
    parser = argparse.ArgumentParser(description='Generates a 10-fold \
                                     cross-validation split for various \
                                     datasets that are designed to examine \
                                     how the methods used in building the \
                                     model are impactedby changes to the \
                                     training data')
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
    fp_list = [args['input'][0] + args['X'][0]]

    #Recover pairs from multilabel format
    target_list = (y_multilabel > 0)\
    .apply(lambda w: y_multilabel.columns[w.tolist()].tolist(), axis=1)

    smi_target = pd.DataFrame(X_multilabel['smiles'])\
    .join(pd.DataFrame(target_list))

    smi_target.columns = ['smiles', 'uniprot']
    pairs = explode(smi_target, ['uniprot'], fill_value='')

    #Only take labels from dataset with at 100 examples of eaech
    subset_100 = pairs.groupby('uniprot').uniprot.transform(len) == 100
    pairs_100 = pairs.loc[subset_100]
    pairs_100.reset_index(inplace=True, drop=True)

    #Subset the 100 example set by halving the number of target labels
    unique_targets = pd.DataFrame(pairs_100['uniprot'].unique())
    targets_half = pd.DataFrame(unique_targets\
                                .sample(round(len(unique_targets)/2)))
    targets_half.columns = ['uniprot']
    pairs_100_half = pd.merge(pairs_100, targets_half, how='inner',
                              on=['uniprot'])

    #Take 10 examples at random from the 100 example set
    pairs_10 = pairs_100.groupby('uniprot').apply(pd.DataFrame.sample, n=10) \
    .reset_index(drop=True)

    #Take 10 examples at random from the halved target label 100 example set
    pairs_10_half = pairs_100_half.groupby('uniprot')\
    .apply(pd.DataFrame.sample, n=10).reset_index(drop=True)

    #Take 10 most dissimilar compounds for each target label from the 100
    #example set
    pairs_10_dissim = get_dissim_compounds(pairs_100)

    #Take 10 most dissimilar compounds for each target label from the halved
    #target label 100 example set
    pairs_10_half_dissim = get_dissim_compounds(pairs_100_half)


    #Split for cross-validation and write train/test files for each set
    multilabel_cv(X=pairs_100['smiles'].to_frame(), y=pairs_100,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_100', n=10)

    multilabel_cv(X=pairs_100_half['smiles'].to_frame(), y=pairs_100_half,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_100_half', n=10)

    multilabel_cv(X=pairs_10['smiles'].to_frame(), y=pairs_10,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_10', n=10)

    multilabel_cv(X=pairs_10_half['smiles'].to_frame(), y=pairs_10_half,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_10_half', n=10)

    multilabel_cv(X=pairs_10_dissim['smiles'].to_frame(), y=pairs_10_dissim,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_10_dissim', n=10)

    multilabel_cv(X=pairs_10_half_dissim['smiles'].to_frame(),
                  y=pairs_10_half_dissim, fp_list=fp_list,
                  output_dir=args['output'][0] +
                  '/cross_validation_10_half_dissim', n=10)

if __name__ == "__main__":
    main()
