#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: cockroft

Trains a LogisticRegression model to be used as a metaclassifier
Uses the probabilities predicted by the base classifiers as input features
Following training, the metaclassifier is used to predict probabilites for
the provided datasets
'''

import argparse
import time
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib


def main():
    parser = argparse.ArgumentParser(description='Train and test \
                                     meta classifier')
    parser.add_argument('-P', '--pred_data', action='store', nargs='*', \
                        dest='P', help='Predicted targets for model \
                        (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs=1, dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Sort P arguements passed - this ensures model predictions are always
    #combined in the same way
    args['P'].sort()

    """
    TRAINING PHASE
    """
    #Collect training data - requires the keyword 'train' be in filename for
    #input features
    train_index_P = [j for j, s in enumerate(args['P']) \
                     if 'train' in s.lower()]
    y = pd.read_csv(args['input'][0] + args['y'][0])

    #Collect and combine predictions to use as input for metaclassiifer
    dataset_stack = pd.DataFrame()
    for i in train_index_P:
        pred = pd.read_csv(args['input'][0] + args['P'][i])
        dataset_stack = pd.concat([dataset_stack, pred], axis=1,
                                  ignore_index=True)

    #Train metaclassifier
    clf = OneVsRestClassifier(LogisticRegression(), n_jobs=-1)

    time_start = time.time()
    clf.fit(dataset_stack, y)
    print('LR training done! Time elapsed: \
          {} seconds'.format(time.time()-time_start))

    joblib.dump(clf, args['input'][0] + 'LRmeta.joblib', compress=True)


    """
    PREDICTION PHASE
    """
    #Identify unqiue datasets - e.g. train, test, NP
    #Assumes file naming convention used in test_model.py output
    stems = []
    for df_name in args['P']:
        name = df_name.split('_')[0]
        stems.append(name)

    #Loop through all data passed to predict on
    for name in set(stems):
        name_index = [j for j, s in enumerate(args['P']) if name in s.lower()]

        #Collect and combine predictions to use as input for metaclassiifer
        dataset_stack = pd.DataFrame()
        for i in name_index:
            pred = pd.read_csv(args['input'][0] + args['P'][i])
            dataset_stack = pd.concat([dataset_stack, pred], axis=1,
                                      ignore_index=True)

        #Make predictions
        stack_proba = clf.predict_proba(dataset_stack)
        stack_proba = pd.DataFrame(stack_proba)
        stack_proba.columns = y.columns
        stack_proba.to_csv(args['output'][0] + '/' + name + "_stack_pred.csv",
                           index=False)

if __name__ == "__main__":
    main()
