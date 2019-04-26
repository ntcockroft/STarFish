#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Get activations for the pentultimate layer of an MLP classifier
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.externals import joblib

def get_activations(clf, X):
    """
    Function which obtains the activations for each layer in a MLP classifer

    Args:
        clf: A trained MLP classifier
        X: Features from the input layer of the MLP classifier

    Returns:
        An array of activations for each layer in the MLP
    """
    hidden_layer_sizes = clf.hidden_layer_sizes
    if not hasattr(hidden_layer_sizes, "__iter__"):
        hidden_layer_sizes = [hidden_layer_sizes]
    hidden_layer_sizes = list(hidden_layer_sizes)
    layer_units = [X.shape[1]] + hidden_layer_sizes + \
        [clf.n_outputs_]
    activations = [X]
    for i in range(clf.n_layers_ - 1):
        activations.append(np.empty((X.shape[0],
                                     layer_units[i + 1])))
    clf._forward_pass(activations)
    return activations



def main():
    parser = argparse.ArgumentParser(description='Takes a trained MLP \
                                     classifier stored as a .joblib file and \
                                     the features used to train it. Outputs \
                                     the activations for the pentultimate \
                                     layer.')
    parser.add_argument('-X', '--X_data', action='store', nargs='*', dest='X',
                        help='Input features for the model (.csv format)')
    parser.add_argument('-c', '--classifier', action='store', nargs=1,
                        dest='classifier', help='Trained MLP classifier file \
                        (.joblib)')
    parser.add_argument("--normalize", dest='normalize', action='store_true',
                        default=False, help="Normalize output activations.")
    parser.add_argument('-i', '--input_directory', action='store', nargs=1,
                        dest='input', default=['./'],
                        help='Directory where input files are stored')
    parser.add_argument('-o', '--output_directory', action='store', nargs=1,
                        dest='output', default=['./'],
                        help='Directory where output files should be written')
    args = vars(parser.parse_args())


    clf_mlp = joblib.load(args['input'][0] + args['classifier'][0])


    for k in range(len(args['X'])):
        name = args['X'][k].split('_')[1]
        name = name.split('.')[0]

        X = pd.read_csv(args['input'][0] +
                        args['X'][k]).drop(columns=['smiles'])


        activations = get_activations(clf_mlp, X)
        act_pent = activations[-2]
        act_pent = pd.DataFrame(act_pent)

        if args['normalize']:
            values = act_pent.stack()
            values_norm = (values - min(values))/(max(values) - min(values))
            act_pent = values_norm.unstack()


        act_pent.to_csv(args['output'][0] + '/' + name + '_MLP-hl_pred.csv',
                        index=False)


if __name__ == "__main__":
    main()
