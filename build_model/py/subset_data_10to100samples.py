#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Subsets data prior to cross validation splitting
Constrains label imbalance to at most 10:1
Requires at least 10 examples (compounds) per label (protein targets)

"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Subsets data so that there \
                                     are at least 10 examples of each protein \
                                     target label, but no more than 100')
    parser.add_argument('-D', '--data', action='store', nargs=1, dest='data',
                        help='File containg standardized smile strings and \
                        associated protein target - protein targets should be \
                        in a column named "uniprot" (.csv format)')
    parser.add_argument('-n', '--name', action='store', nargs=1,
                        dest='name', help='Name of output csv file to write')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Load training data
    dataset_info = pd.read_csv(args['input'][0] + args['data'][0])

    #Only use labels with at least 10 observations
    subset_10 = dataset_info.groupby('uniprot').uniprot.transform(len) >= 10
    dataset_info = dataset_info.loc[subset_10]

    #Find labels with more than 100 examples of each
    subset_gt100 = dataset_info.groupby('uniprot').uniprot.transform(len) > 100
    dataset_info_gt100 = dataset_info.loc[subset_gt100]
    dataset_info_lt100 = dataset_info.loc[~subset_gt100]

    #Take only 100 examples of those
    dataset_info_100 = dataset_info_gt100.groupby('uniprot') \
    .apply(pd.DataFrame.sample, n=100) \
    .reset_index(level='uniprot', drop=True)

    dataset_subset = pd.concat([dataset_info_lt100, dataset_info_100], axis=0)
    dataset_subset.to_csv(args['output'][0] + args['name'][0], index=False)


if __name__ == "__main__":
    main()
