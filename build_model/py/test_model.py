#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Use each specified classifier (RF,KNN, and/or MLP) to predict protein target
labels for given datasets
"""
import argparse
import time
import pandas as pd
from sklearn.externals import joblib


def main():
    parser = argparse.ArgumentParser(description='Test model predictions of \
                                     multiple classifiers')
    parser.add_argument('-X', '--X_data', action='store', nargs='*', dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs='*', dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-c', '--classifier', action='store', nargs='*',
                        dest='classifier',
                        help='Classifier models (.joblib format) - provide \
                        filename without extension (e.g. RF KNN MLP LR)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Sort X/y arguements passed - this makes sure that matching X/y is used
    args['X'].sort()
    args['y'].sort()

    #Loop through all data passed to predict on
    for i in range(len(args['X'])):
        name = args['X'][i].split('_')[1]
        name = name.split('.')[0]

        X = pd.read_csv(args['input'][0] +
                        args['X'][i]).drop(columns=['smiles'])

        #Collect y data that has as filename corresponding to X data
        name_index = [j for j, s in enumerate(args['y']) if name in s.lower()]
        name_index = name_index[0]
        y = pd.read_csv(args['input'][0] + args['y'][name_index])

        #Make predictions using each classifier
        time_start = time.time()
        for clf_name in args['classifier']:
            #Special treatment of RF-see comment in train_model.py for more
            if clf_name == 'RF':
                pred_df = pd.DataFrame()
                for column in y:
                    clf = joblib.load(args['input'][0] + '/RF_models/'+ 'RF_'
                                      + column + '.joblib')
                    pred_df[column] = clf.predict_proba(X)[:, 1]
            else:
                clf = joblib.load(args['input'][0] + '/' +  clf_name +
                                  '.joblib')
                pred_df = clf.predict_proba(X)

                #Standardize format of predictions given from classifiers
                if isinstance(pred_df, list):
                    pred_df = pd.DataFrame([proba_pair[:, 1] for proba_pair
                                            in pred_df]).T
                else:
                    pred_df = pd.DataFrame(pred_df)
            pred_df.columns = y.columns
            pred_df.to_csv(args['output'][0] + name + '_' + clf_name +
                           '_pred.csv', index=False)
            print(clf_name +' prediction done! Time elapsed: \
                  {} seconds'.format(time.time()-time_start))

if __name__ == "__main__":
    main()
