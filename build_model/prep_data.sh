#!/bin/sh
#******************************************************************************
# Generates all folders and files needed for running the cross validation
#	-Cleans smiles
#	-Subsets synethic dataset to reduce class imbalance
#	-Generates fingerprints
#	-Performs cross validation train/test splitting
#------------------------------------------------------------------------------
# NOTE: Fingerprint files are fairly large - especially  for training 
# data(~300 MB) and it takes a while to generate all of them (~1 hr)
#******************************************************************************
source activate STarFish

python ./py/clean_smiles.py -S NaturalProducts_smiles.csv \
-n NaturalProducts_smiles_clean.csv -i ./datasets/ \
-o ./datasets/

python ./py/clean_smiles.py -S BindingData_1uM.csv \
-n BindingData_1uM_clean.csv -i ./datasets/ \
-o ./datasets/

python ./py/extract_NP.py -s NaturalProducts_smiles_clean.csv \
-k NaturalProducts_inchikey.csv -b BindingData_1uM_clean.csv \
-i ./datasets/ -o ./datasets/

mkdir cross_validation
mkdir cross_validation/NP
for i in {1..10}; do mkdir cross_validation/$i; done

python ./py/subset_data_10to100samples.py \
-D BindingData_1uM_synthetic.csv \
-n BindingData_1uM_synthetic_subset.csv -i ./datasets/ \
-o ./datasets/

python ./py/gen_fp.py -S BindingData_1uM_naturalProduct.csv \
-f fcfp -n Dict_naturalProduct_fcfp.csv -i ./datasets/ -o ./datasets/

python ./py/gen_fp.py -S BindingData_1uM_synthetic_subset.csv \
-f fcfp -n Dict_synthetic_fcfp.csv -i ./datasets/ -o ./datasets/

python ./py/split_data_CV.py -d BindingData_1uM_synthetic_subset.csv \
-f ./datasets/Dict_synthetic_fcfp.csv -i ./datasets/ -o ./cross_validation/

python ./py/split_data_NP.py -d BindingData_1uM_naturalProduct.csv \
-f ./datasets/Dict_naturalProduct_fcfp.csv \
-r ./cross_validation/1/y_test.csv -i ./datasets/ -o ./cross_validation/NP/

#Combine train/test cross-validation sets for final model training
python ./py/recombine_fold_data.py -X X_train.csv X_test.csv \
-y y_train.csv y_test.csv -i ./cross_validation/1/ -o ./cross_validation/NP/

#Prepare data for base classifier tuning
for i in {1..10}; do 
    mkdir cross_validation/$i/tuning; 
    
    python py/split_data_tune.py -X X_train.csv -y y_train.csv \
    -i cross_validation/$i/ -o cross_validation/$i/tuning/
done
