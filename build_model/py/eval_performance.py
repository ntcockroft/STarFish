#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cockroft

Evaluates performance of the constructed target fishing models
	Calculates micro and macro averaged AUROC scores
    Calculates micro and macro averaged AP scores
	Calculates micro and macro averaged BEDROC scores
	Assess how many true positive labels (protein targets) are recovered in
        the top of the ranked list
    Calculates coverage error
"""

import argparse
import pandas as pd
import sklearn.metrics as skm
from rdkit.ML.Scoring import Scoring
import numpy as np


def get_ranking(known, predicted):
    """
    Orders the true values present in the "known" data based on ranking in
    the "predicted" data

    Args:
        known: DataFrame of known labels, each row corresponds to a compound
            and each column corresponds to a target label
        predicted: DataFrame of predicted label probabilities, rows and columns
            should match known DataFrame

    Returns:
        A DataFrame of the true labels, but columns are now disjointed and do
        not correspond to individual protein targets. The true labels are now
        ordered by predicted probability rank (high to low)
    """
    predicted.columns = known.columns
    all_ranking = []
    for idx, row in predicted.iterrows():
        ranking = row.sort_values(ascending=False).index
        all_ranking.append(known.iloc[idx].reindex(ranking).values)
    all_ranking = pd.DataFrame(all_ranking)

    return all_ranking


def true_positive_per_compound(ranked_df):
    """
    Takes the output from the "get_ranking" function (known data ranked by
    predicted data ranks)and returns the fraction of compounds for which one
    true positive has been identified

    This information is give for each length of a ranked list:
        e.g. fraction of compounds which had a true positive at rank 1
             fraction of compounds which have a true positive in the top 5,
                 or 10, etc.

    Args:
        ranked_df: The DataFrame of sorted true labels per compound that was
            output by get_ranking()
    Returns:
        An array of values that correspond the cumulative sum of the fraction
        of true positive labels recovered for each rank position. For example,
        the 1st value corresponds to the fraction compounds that have at least
        1 true positive if only looking at the top predicted label. The 5th
        values would correspond to the fraction if looking at the top 5
        predicted labels
    """

    cumsum_df = np.cumsum(ranked_df, axis=1)
    tp_identified = (cumsum_df > 0) * 1
    tp_cmpd_frac = np.sum(tp_identified)/len(tp_identified)

    return tp_cmpd_frac


def true_positives_recovered(ranked_df):
    """
    Takes the output from the "get_ranking" function (known data ranked by
    predicted data ranks)and returns the fraction of true positives that have
    been recovered

    This information is give for each length of a ranked list:
        e.g. fraction of true positives recovered at rank 1
             fraction of true positives recovered in the top 5, or 10, etc.

    Args:
        ranked_df: The DataFrame of sorted true labels per compound that was
            output by get_ranking()
    Returns:
        An array of values that correspond the cumulative sum of the fraction
        of true positive labels recovered for each rank position. For example,
        the 1st value corresponds to the fraction of ALL true positive labels
        recovered if only looking at the top predicted label of each compounds.
        The 5th  values would correspond to the fraction if looking at the top
        5 predicted labels
    """

    sum_df = np.sum(ranked_df)
    num_tp = np.sum(sum_df)
    tp_all_frac = np.cumsum(sum_df/num_tp)

    return tp_all_frac


def macro_auroc(y_true, y_pred):
    """
    Helper function which calculates macro averaged AUROC score using
    metrics.roc_auc_score from sklearn (sklearn "macro" implementation does
    not work for the multilabel format used here)

    Args:
        y_true: DataFrame of known labels, each row corresponds to a compound
            and each column corresponds to a target label
        y_pred: DataFrame of predicted label probabilities, rows and columns
            should match known DataFrame
    Returns:
        The macro averaged auroc score for the predicted labels
    """
    auroc_scores = []
    for column in y_true:
        if np.sum(y_true[column]) != 0:
            auroc_scores.append(skm.roc_auc_score(y_true[column],
                                                  y_pred[column]))
        else:
            continue

    macro_auroc_score = np.mean(auroc_scores)

    return macro_auroc_score


def macro_ap(y_true, y_pred):
    """
    Helper function which calculates macro averaged AUPR score using
    metrics.average_precision_score from sklearn (sklearn "macro"
    implementation does not work for the multilabel format used here)

    Args:
        y_true: DataFrame of known labels, each row corresponds to a compound
            and each column corresponds to a target label
        y_pred: DataFrame of predicted label probabilities, rows and columns
            should match known DataFrame
    Returns:
        The macro averaged average precision score for the predicted labels

    """
    ap_scores = []
    for column in y_true:
        if np.sum(y_true[column]) != 0:
            ap_scores.append(skm.average_precision_score(y_true[column],
                                                         y_pred[column]))
        else:
            continue
    macro_ap_score = np.mean(ap_scores)

    return macro_ap_score


def macro_bedroc(y_true, y_pred, a=20):
    """
    Helper function which calculates macro averaged BEDROC score using
    ML.Scoring.Scoring.CalcBEDROC from rdkit

    Args:
        y_true: DataFrame of known labels, each row corresponds to a compound
            and each column corresponds to a target label
        y_pred: DataFrame of predicted label probabilities, rows and columns
            should match known DataFrame
        a: alpha value for BEDROC calculation. NOTE:only scores computed using
            the same alpha value can be compared
    Returns:
        The macro averaged BEDROC score for the predicted labels

    """
    bedroc_scores = []
    for column in y_true:
        if np.sum(y_true[column]) != 0:
            scores = pd.DataFrame()
            scores['proba'] = np.array(y_pred[column])
            scores['active'] = np.array(y_true[column])
            scores.sort_values(by='proba', ascending=False, inplace=True)
            bedroc_scores.append(Scoring.CalcBEDROC(np.array(scores),
                                                    col=1, alpha=a))
        else:
            continue

    macro_bedroc_score = np.mean(bedroc_scores)

    return macro_bedroc_score


def main():
    parser = argparse.ArgumentParser(description='Evaluate prediction results')
    parser.add_argument('-P', '--pred_data', action='store', nargs='*',
                        dest='P',
                        help='Predicted targets for model (.csv format)')
    parser.add_argument('-y', '--y_data', action='store', nargs='*', dest='y',
                        help='Known target values (.csv format)')
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())

    #Sort P arguements passed to keep result order consistent
    args['P'].sort()

    #Loop through all predictions to evaluate
    for i in range(len(args['y'])):
        name = args['y'][i].split('_')[1]
        name = name.split('.')[0]
        y = pd.read_csv(args['input'][0] + args['y'][i])

        #Collect predictions for corresponding dataset - e.g. train, test
        name_index = [j for j, s in enumerate(args['P']) if name in s.lower()]

        #Generate dictionary to store predictions
        predictions = {}
        for j in name_index:
            pred = pd.read_csv(args['input'][0] + args['P'][j])

            #Get classifier from file name
            clf_name = args['P'][j].split('.')[0]
            clf_name = clf_name.split('_')[1]

            #Check for predictions which don't have the correct dimensions
            #This handles cases in which feature dimenions  were used in
            #stacking that have different dimensions - e.g. MLP hidden layer
            if len(pred.columns) == len(y.columns):
                #Store classifer name and values in dict
                predictions[clf_name] = pred

        #Get values of base classifier predictions and compute mean predictions
        pred_base = [df.values for key, df in predictions.items() if key
                     not in ['stack']]
        if pred_base:
            average_values = sum(pred_base)/len(pred_base)
            predictions['ConsensusAverage'] = pd.DataFrame(average_values)

        results = []
        for clf in predictions:
            pred = predictions[clf]

            ranking = get_ranking(y, pred)
            tp_cmpd = true_positive_per_compound(ranking)[9]
            tp_all = true_positives_recovered(ranking)[9]

            micro_ap_score = skm.average_precision_score(y, pred,
                                                         average='micro')
            macro_ap_score = macro_ap(y, pred)

            coverage = skm.coverage_error(y, pred)

            micro_auroc_score = skm.roc_auc_score(y, pred, average='micro')
            macro_auroc_score = macro_auroc(y, pred)

            scores = pd.DataFrame()
            scores['proba'] = np.array(pred).flatten()
            scores['active'] = np.array(y).flatten()
            scores.sort_values(by='proba', ascending=False, inplace=True)

            micro_bedroc_score = Scoring.CalcBEDROC(np.array(scores),
                                                    col=1, alpha=20)
            macro_bedroc_score = macro_bedroc(y, pred)

            results.append([clf, micro_auroc_score, macro_auroc_score,
                            tp_cmpd, tp_all, micro_ap_score, macro_ap_score,
                            micro_bedroc_score, macro_bedroc_score, coverage])

        results = pd.DataFrame(results)
        results.columns = ['Model', 'micro_AUROC', 'macro_AUROC',
                           'Frac_1_in_top10', 'Frac_all_in_top10', 'micro_AP',
                           'macro_AP', 'micro_BEDROC', 'macro_BEDROC',
                           'coverage']
        print(results)
        results.to_csv(args['output'][0] + '/' + name +  '_results.csv',
                       index=False)

if __name__ == "__main__":
    main()
