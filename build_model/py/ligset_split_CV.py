#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Starting point for systematic evaluation of how the number of training samples
impact model performance
    Identifies top 5 target labels with the most compound records
    Generates subsets of 2500, 2000, 1500, 1000, 500, 100, and 10
     compound-target pairs
    Splits generates subsets for stratified 10 fold cross-validation


"""

import argparse
import pandas as pd
from split_data_CV import multilabel_cv


def sample_pairs(dataset, num_pairs):
    """
    Helper function to sample compound-target pairs from a dataset of compound-
    target pairs

    Args:
        dataset: pandas dataframe with a column named 'uniprot' for the protein
        target labels
        num_pairs: the number of compound-target pairs to sample from the
        dataset

    Returns:
        A DataFrame of compound-target activity pairs that were sampled from
        the input dataset
    """
    sample_pairs_df = (dataset
                       .groupby('uniprot')
                       .apply(pd.DataFrame.sample, n=num_pairs)
                       .reset_index(drop=True)
                       )
    return sample_pairs_df

def main():
    parser = argparse.ArgumentParser(description='Generates compound-pair \
                                     subsets for the top 5 most populated \
                                     protein target labels. Subsets are then \
                                     split for stratified 10-fold \
                                     cross-validation')
    parser.add_argument('-D', '--data', action='store', nargs=1, dest='data',
                        help='File containg standardized smile strings and \
                        associated protein target - protein targets should be \
                        in a column named "uniprot" (.csv format)')
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

    #Load training data
    dataset_info = pd.read_csv(args['input'][0] + args['data'][0])
    fp_list = args['fp']

    #Get top 5 protein targets by number of compound records
    uniprot_top5 = (dataset_info['uniprot']
                    .value_counts()[0:5]
                    .index
                    )
    dataset_info_top5 = dataset_info[dataset_info['uniprot']
                                     .isin(uniprot_top5)]

    #Generate all datasets
    pairs_2500 = sample_pairs(dataset_info_top5, 2500)
    pairs_2000 = sample_pairs(pairs_2500, 2000)
    pairs_1500 = sample_pairs(pairs_2000, 1500)
    pairs_1000 = sample_pairs(pairs_1500, 1000)
    pairs_500 = sample_pairs(pairs_1000, 500)
    pairs_100 = sample_pairs(pairs_500, 100)
    pairs_10 = sample_pairs(pairs_100, 10)



    #Split for cross-validation and write train/test files for each set
    multilabel_cv(X=pairs_2500['smiles'].to_frame(), y=pairs_2500,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_2500', n=10)
    multilabel_cv(X=pairs_2000['smiles'].to_frame(), y=pairs_2000,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_2000', n=10)
    multilabel_cv(X=pairs_1500['smiles'].to_frame(), y=pairs_1500,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_1500', n=10)
    multilabel_cv(X=pairs_1000['smiles'].to_frame(), y=pairs_1000,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_1000', n=10)
    multilabel_cv(X=pairs_500['smiles'].to_frame(), y=pairs_500,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_500', n=10)
    multilabel_cv(X=pairs_100['smiles'].to_frame(), y=pairs_100,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_100', n=10)
    multilabel_cv(X=pairs_10['smiles'].to_frame(), y=pairs_10,
                  fp_list=fp_list, output_dir=args['output'][0] +
                  '/cross_validation_ligset_10', n=10)



if __name__ == "__main__":
    main()
