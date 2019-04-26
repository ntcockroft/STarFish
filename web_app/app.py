#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ntcockroft

Runs STarFish as a web application with Flask
"""
from molvs import Standardizer
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.externals import joblib
import pandas as pd
from flask import Flask, request, render_template


app = Flask(__name__)

@app.route('/')
def main():
    """ Main page of the API """
    return render_template('home.html')

def clean_smiles(smi):
    """
    Helper function which runs the standardization tool on the input smiles
    string

    Args:
        smi: Input smiles string

    Returns:
        The standardized version of the input smiles string
    """
    s = Standardizer(prefer_organic=True)
    try:
        mol = Chem.MolFromSmiles(smi)
        std_mol = s.fragment_parent(mol, skip_standardize=False)
        std_smi = Chem.MolToSmiles(std_mol)
        return std_smi
    except:
        print("Issue with input smiles string. Unable to clean " + str(smi))
    return None


def gen_fp(smi):
    """
    Converts a smiles string into a chemical fingerprint

    Args:
        smi: Input smiles string

    Returns:
        2048 bit string of the chemical fingerprint
    """
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, useFeatures=True)
    fp_bitstr = list(fp.ToBitString())
    fp_bitstr = pd.DataFrame(fp_bitstr).transpose()
    return fp_bitstr


def ensemble_predict(X, clf_names):
    """
    Predicts label probabilities from chemical fingerprint input for each
        trained model

    Args:
        X: 2048 bit string chemical fingerprint
        clf_names: names of the classifiers to be used in prediction

    Returns:
        DataFrame containing the concatenated probability predictions of each
        classifier
    """
    predictions = pd.DataFrame()
    for clf_name in clf_names:
        clf = joblib.load('./models/' +  clf_name + '.joblib')
        pred_df = clf.predict_proba(X)

        #Standardize format of predictions given from classifiers
        if isinstance(pred_df, list):
            pred_df = pd.DataFrame([proba_pair[:, 1] for proba_pair
                                    in pred_df]).T
        else:
            pred_df = pd.DataFrame(pred_df)

        predictions = pd.concat([predictions, pred_df], axis=1,
                                ignore_index=True)

    return predictions


def meta_predict(ensemble_predictions):
    """
    Combines the ensemble of label predictions to produce final label
    probability predictions

    Args:
        ensemble_predictions: DataFrame containing the concatenated probability
        predictions of each classifier

    Returns:
        DataFrame containing probability predictions of each protein target
        label
    """
    meta_clf = joblib.load('./models/LRmeta.joblib')
    meta_prediction = meta_clf.predict_proba(ensemble_predictions)
    meta_prediction = pd.DataFrame(meta_prediction)
    return meta_prediction


@app.route('/predict', methods=['POST'])
def predict():
    """ Predict the targets of a compound """
    target_list = pd.read_csv('./target_list.csv')['targets'].tolist()
    smi = request.form.get("text")
    clean_smi = clean_smiles(smi)
    fp = gen_fp(clean_smi)

    ensemble_pred = ensemble_predict(fp, clf_names=['KNN'])
    meta_pred = meta_predict(ensemble_pred)
    meta_pred = pd.DataFrame(meta_pred**.5)
    meta_pred.columns = target_list
    meta_pred = meta_pred.sort_values(0, axis=1, ascending=False)
    meta_pred = meta_pred.T
    meta_pred['uniprot'] = meta_pred.index
    meta_pred.columns = ['Score', 'Uniprot']

    pd.set_option('display.max_colwidth', -1)
    meta_pred['Uniprot'] = meta_pred['Uniprot'] \
    .apply(lambda x: '<a href="https://www.uniprot.org/uniprot/{0}">{0}</a>' \
           .format(x))
    results = render_template('results.html', table=meta_pred. \
                              to_html(classes='prediction',
                                      index=False, escape=False),
                              structure=clean_smi)

    return results
