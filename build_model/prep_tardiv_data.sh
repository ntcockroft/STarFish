#!/bin/sh
#******************************************************************************
# Generates all folders and files needed for running the cross validation for 
# the systematic evaluation of ligand set size on performance
#	-Performs cross validation train/test splitting
#******************************************************************************

source activate STarFish

SCRPT_DIR=`readlink -f  py/`
export PYTHONPATH=$PYTHONPATH:$SCRPT_DIR

#Uncomment and run if this file was not already generated by prep_ligset_data.sh
#python ./py/gen_fp.py -S BindingData_1uM_synthetic.csv \
#-f fcfp -n Dict_synthetic_fcfp_all.csv -i ./datasets/ -o ./datasets/

mkdir cross_validation_tardiv
for i in {1..10}; do mkdir cross_validation_tardiv/$i; done

mkdir cross_validation_tardiv_kinase
for i in {1..10}; do mkdir cross_validation_tardiv_kinase/$i; done

python ./py/tardiv_split_CV.py -X X_train.csv -y y_train.csv \
-f ./datasets/Dict_synthetic_fcfp_all.csv \
-p ./datasets/chemblID_proteinclass.csv -i cross_validation/NP/

