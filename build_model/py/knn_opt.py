#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Check performance of different hyperparameters
for K-Nearest Neighbors Classifier

NOTE: This file must be in the same directory as eval_performance.py
to run and that directory must be in PYTHONPATH
"""
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
import sklearn.metrics as skm
from rdkit.ML.Scoring import Scoring
from eval_performance import get_ranking, true_positive_per_compound, true_positives_recovered, macro_auroc, macro_ap, macro_bedroc

def main():
    parser = argparse.ArgumentParser(description='Tune KNeighborsClassifier')
    parser.add_argument('-X', '--X_data', action='store', nargs=2, dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs=2, dest='y',
                        help='Target outputs for the model (.csv format)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Sort so that training and test data are in a predictable order
    args['X'].sort()
    args['y'].sort()

    X_train = pd.read_csv(args['input'][0] + args['X'][1]) \
    .drop(columns=['smiles'])
    y_train = pd.read_csv(args['input'][0] + args['y'][1])

    X_test = pd.read_csv(args['input'][0] + args['X'][0]) \
    .drop(columns=['smiles'])
    y_test = pd.read_csv(args['input'][0] + args['y'][0])

    # use a full grid over all parameters
    param_grid = {'n_neighbors':[1, 5, 10],
                  'metric': ['minkowski', 'jaccard']}

    results = []
    for params in list(ParameterGrid(param_grid)):
        clf = KNeighborsClassifier(n_jobs=-1,
                                   n_neighbors=params['n_neighbors'],
                                   metric=params['metric'])

        time_start = time.time()
        clf.fit(X_train, y_train)
        pred = clf.predict_proba(X_test)
        pred = pd.DataFrame([proba_pair[:, 1] for proba_pair in pred]).T
        print('Training and prediction done! Time elapsed: \
              {} seconds'.format(time.time()-time_start))

        ranking = get_ranking(y_test, pred)
        tp_cmpd = true_positive_per_compound(ranking)[9]
        tp_all = true_positives_recovered(ranking)[9]

        micro_ap_score = skm.average_precision_score(y_test, pred,
                                                     average='micro')
        macro_ap_score = macro_ap(y_test, pred)

        coverage = skm.coverage_error(y_test, pred)

        micro_auroc_score = skm.roc_auc_score(y_test, pred,
                                              average='micro')
        macro_auroc_score = macro_auroc(y_test, pred)

        scores = pd.DataFrame()
        scores['proba'] = np.array(pred).flatten()
        scores['active'] = np.array(y_test).flatten()
        scores.sort_values(by='proba', ascending=False, inplace=True)

        micro_bedroc_score = Scoring.CalcBEDROC(np.array(scores), col=1,
                                                alpha=20)
        macro_bedroc_score = macro_bedroc(y_test, pred)

        results.append([micro_auroc_score, macro_auroc_score, tp_cmpd, tp_all,
                        micro_ap_score, macro_ap_score, micro_bedroc_score,
                        macro_bedroc_score, coverage] + list(params.values()))

    results = pd.DataFrame(results)
    results.columns = ['micro_AUROC', 'macro_AUROC', 'Frac_1_in_top10',
                       'Frac_all_in_top10', 'micro_AP', 'macro_AP',
                       'micro_BEDROC', 'macro_BEDROC', 'coverage'] \
                       + list(params)
    results.to_csv(args['output'][0] + '/' + 'KNN_opt_results.csv',
                   index=False)


if __name__ == "__main__":
    main()
