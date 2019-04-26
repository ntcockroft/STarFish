#!/bin/sh
#******************************************************************************
# Generates all folders and files needed for running the cross validation for 
# the different method tests
#	-Performs cross validation train/test splitting
#------------------------------------------------------------------------------
# NOTE: The dataset passed to the method_test_split_CV.py script is the all the
# data that was used for cross-valdation. The training and test splits were 
# already re-combined for benchmarking on the NP set so the dataset is pulled
# from there for these subsequent cross-validation tests
#******************************************************************************

source activate STarFish

SCRPT_DIR=`readlink -f  py/`
export PYTHONPATH=$PYTHONPATH:$SCRPT_DIR

mkdir cross_validation_100
for i in {1..10}; do mkdir cross_validation_100/$i; done

mkdir cross_validation_100_half
for i in {1..10}; do mkdir cross_validation_100_half/$i; done

mkdir cross_validation_10
for i in {1..10}; do mkdir cross_validation_10/$i; done

mkdir cross_validation_10_half
for i in {1..10}; do mkdir cross_validation_10_half/$i; done

mkdir cross_validation_10_dissim
for i in {1..10}; do mkdir cross_validation_10_dissim/$i; done

mkdir cross_validation_10_half_dissim
for i in {1..10}; do mkdir cross_validation_10_half_dissim/$i; done

python ./py/method_test_split_CV.py -X X_train.csv -y y_train.csv -i cross_validation/NP/
