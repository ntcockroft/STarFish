#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Collects results from cross validation for each classifier combination
    Calculates mean values for each metric
    Calculates standard deviation values for each metric

"""

import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Takes a list of files that \
                                     correspond to the results from a cross- \
                                     validation fold. Combines results from \
                                     each file and returns a .csv output file \
                                     that contains mean values and standard \
                                     deviation for each metric present in the \
                                     fold result files')
    parser.add_argument('-i', '--input_files', action='store', nargs='*',
                        dest='input', default=['./'],
                        help='Result files from cross-validation folds - \
                        delimit input file names with a space')
    parser.add_argument('-c', '--column_sort', action='store', nargs='*',
                        dest='column', help='Name of column to sort the \
                        results file by. For example, if a results file \
                        contains the results from multiple different \
                        classifiers that are denoted under a column named \
                        "Model" then "-c Model" should be passed. More than \
                        one column can be passed to sort. For example, \
                        "-c metric n_neighbors" could be passed to sorty by \
                        the unique combinations of those two columns')
    parser.add_argument('-n', '--name', action='store', nargs=1,
                        dest='name', help='Name of output csv file to write')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    cv_results = args['input']
    results_df = pd.DataFrame()
    for filename in cv_results:
        results_df = pd.concat((results_df, pd.read_csv(filename)))

    results_mean = results_df.groupby(args['column']).mean()
    results_std = results_df.groupby(args['column']).std()

    results_mean.to_csv(args['output'][0] + '/' + args['name'][0] + \
                        '_mean.csv', index=True)
    results_std.to_csv(args['output'][0] + '/' + args['name'][0] + \
                       '_std.csv', index=True)

if __name__ == "__main__":
    main()
