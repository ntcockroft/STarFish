#!/bin/sh
#******************************************************************************
# Generates all folders and files needed for running the cross validation for 
# the systematic evaluation of ligand set size on performance
#	-Performs cross validation train/test splitting
#******************************************************************************

source activate STarFish

SCRPT_DIR=`readlink -f  py/`
export PYTHONPATH=$PYTHONPATH:$SCRPT_DIR

python ./py/gen_fp.py -S BindingData_1uM_synthetic.csv \
-f fcfp -n Dict_synthetic_fcfp_all.csv -i ./datasets/ -o ./datasets/


mkdir cross_validation_ligset_2500
for i in {1..10}; do mkdir cross_validation_ligset_2500/$i; done

mkdir cross_validation_ligset_2000
for i in {1..10}; do mkdir cross_validation_ligset_2000/$i; done

mkdir cross_validation_ligset_1500
for i in {1..10}; do mkdir cross_validation_ligset_1500/$i; done

mkdir cross_validation_ligset_1000
for i in {1..10}; do mkdir cross_validation_ligset_1000/$i; done

mkdir cross_validation_ligset_500
for i in {1..10}; do mkdir cross_validation_ligset_500/$i; done

mkdir cross_validation_ligset_100
for i in {1..10}; do mkdir cross_validation_ligset_100/$i; done

mkdir cross_validation_ligset_10
for i in {1..10}; do mkdir cross_validation_ligset_10/$i; done

python ./py/ligset_split_CV.py -D BindingData_1uM_synthetic.csv \
-f ./datasets/Dict_synthetic_fcfp_all.csv -i ./datasets/ 

