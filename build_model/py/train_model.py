#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Trains each specified classifier (RF,KNN, and/or MLP) using chemical
fingerprints as features to predict protein target labels for given datasets
The trained classifiers are saved as .joblib files
"""

import argparse
import os
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

def main():
    parser = argparse.ArgumentParser(description='Train multiple specified \
                                     classifiers')
    parser.add_argument('-X', '--X_data', action='store', nargs=1, dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs=1, dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-c', '--classifier', action='store', nargs='*',
                        dest='classifier', help='Classifier models (write \
                        strings delimited by spaces - options: RF KNN MLP LR)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Dict of classifiers that can be used
    classifiers = {"RF": RandomForestClassifier(n_estimators=1000,
                                                max_features='auto',
                                                n_jobs=-1),
                   "KNN": KNeighborsClassifier(n_neighbors=10,
                                               metric='jaccard', n_jobs=-1),
                   "MLP": MLPClassifier(hidden_layer_sizes=(1000, )),
                   "LR": OneVsRestClassifier(LogisticRegression(), n_jobs=-1)}

    for clf_name in args['classifier']:
        clf = classifiers[clf_name]
        X = pd.read_csv(args['input'][0] +
                        args['X'][0]).drop(columns=['smiles'])
        y = pd.read_csv(args['input'][0] + args['y'][0])

        """
        Special condition for RF - fitting individual models/forests for each
        label. This is needed to avoid memory issues encountered with holding
        all forests in memory at once - the case when using default sklearn
        multilabel implementation of RF.
        """
        time_start = time.time()
        if clf_name == 'RF':
            if not os.path.isdir(args['output'][0] + '/RF_models'):
                os.mkdir(args['output'][0] + '/RF_models')
            for column in y:
                clf.fit(X, y[column])
                joblib.dump(clf, args['output'][0] + '/RF_models/RF_'
                            +  column +'.joblib', compress=False)
        else:
            clf.fit(X, y)
            joblib.dump(clf, args['output'][0] + '/' + clf_name + '.joblib',
                        compress=False)
        print(clf_name + ' training done! Time elapsed: \
              {} seconds'.format(time.time()-time_start))

if __name__ == "__main__":
    main()
