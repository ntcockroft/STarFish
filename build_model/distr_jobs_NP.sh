#!/bin/sh
#******************************************************************************
# NOTE: prep_data.sh should be run first!
# Files should be be copied to HPC cluster
# These scripts are written for a bash shell and a PBS batch scheduling system
#------------------------------------------------------------------------------
# Generates directories for organization, copies .pbs submission files to 
# appropriate locations and submits jobs to HPC cluster for final model
# training and benchmarking on the natural product set
#******************************************************************************

mkdir cross_validation/NP/classifier_combinations/
mkdir cross_validation/NP/classifier_combinations/KNN
mkdir cross_validation/NP/classifier_combinations/MLP
mkdir cross_validation/NP/classifier_combinations/RF
mkdir cross_validation/NP/classifier_combinations/KNN_MLP
mkdir cross_validation/NP/classifier_combinations/KNN_RF
mkdir cross_validation/NP/classifier_combinations/KNN_MLP_RF
mkdir cross_validation/NP/classifier_combinations/MLP_RF
mkdir cross_validation/NP/classifier_combinations/LRbase
mkdir cross_validation/NP/classifier_combinations/MLP-hl
mkdir cross_validation/NP/classifier_combinations/KNN_MLP-hl
mkdir cross_validation/NP/classifier_combinations/KNN_MLP-hl_RF
mkdir cross_validation/NP/classifier_combinations/MLP-hl_RF
cp pbs/run_cv.pbs cross_validation/NP/

cp pbs/run_KNN.pbs \
cross_validation/NP/classifier_combinations/KNN/run_job.pbs

cp pbs/run_MLP.pbs \
cross_validation/NP/classifier_combinations/MLP/run_job.pbs

cp pbs/run_RF.pbs \
cross_validation/NP/classifier_combinations/RF/run_job.pbs

cp pbs/run_KNN_MLP.pbs \
cross_validation/NP/classifier_combinations/KNN_MLP/run_job.pbs

cp pbs/run_KNN_RF.pbs \
cross_validation/NP/classifier_combinations/KNN_RF/run_job.pbs

cp pbs/run_KNN_MLP_RF.pbs \
cross_validation/NP/classifier_combinations/KNN_MLP_RF/run_job.pbs

cp pbs/run_MLP_RF.pbs \
cross_validation/NP/classifier_combinations/MLP_RF/run_job.pbs

cp pbs/run_LRbase.pbs \
cross_validation/NP/classifier_combinations/LRbase/run_job.pbs

cp pbs/run_MLP-hl.pbs \
cross_validation/NP/classifier_combinations/MLP-hl/run_job.pbs

cp pbs/run_KNN_MLP-hl.pbs \
cross_validation/NP/classifier_combinations/KNN_MLP-hl/run_job.pbs

cp pbs/run_KNN_MLP-hl_RF.pbs \
cross_validation/NP/classifier_combinations/KNN_MLP-hl_RF/run_job.pbs

cp pbs/run_MLP-hl_RF.pbs \
cross_validation/NP/classifier_combinations/MLP-hl_RF/run_job.pbs

cd NP; qsub run_cv.pbs;
