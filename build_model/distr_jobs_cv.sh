#!/bin/sh
#******************************************************************************
# NOTE: prep_data.sh should be run first!
# Files should be be copied to HPC cluster
# These scripts are written for a bash shell and a PBS batch scheduling system
#------------------------------------------------------------------------------
# Generates directories for organization, copies .pbs submission files to 
# appropriate locations and submits jobs to HPC cluster for 10 fold cross
# validation
#******************************************************************************

if [ $# -eq 1 ]; then
	top_dir=`pwd`
	for i in {1..10}; do 
	    mkdir $1/$i/classifier_combinations/
	    mkdir $1/$i/classifier_combinations/KNN
	    mkdir $1/$i/classifier_combinations/MLP
	    mkdir $1/$i/classifier_combinations/RF
	    mkdir $1/$i/classifier_combinations/KNN_MLP
	    mkdir $1/$i/classifier_combinations/KNN_RF
	    mkdir $1/$i/classifier_combinations/KNN_MLP_RF
	    mkdir $1/$i/classifier_combinations/MLP_RF
	    mkdir $1/$i/classifier_combinations/LRbase
	    mkdir $1/$i/classifier_combinations/MLP-hl
	    mkdir $1/$i/classifier_combinations/KNN_MLP-hl
	    mkdir $1/$i/classifier_combinations/KNN_MLP-hl_RF
	    mkdir $1/$i/classifier_combinations/MLP-hl_RF
	    cp pbs/run_cv.pbs $1/$i

	    cp pbs/run_KNN.pbs \
	    $1/$i/classifier_combinations/KNN/run_job.pbs

	    cp pbs/run_MLP.pbs \
	    $1/$i/classifier_combinations/MLP/run_job.pbs

	    cp pbs/run_RF.pbs \
	    $1/$i/classifier_combinations/RF/run_job.pbs

	    cp pbs/run_KNN_MLP.pbs \
	    $1/$i/classifier_combinations/KNN_MLP/run_job.pbs

	    cp pbs/run_KNN_RF.pbs \
	    $1/$i/classifier_combinations/KNN_RF/run_job.pbs

	    cp pbs/run_KNN_MLP_RF.pbs \
	    $1/$i/classifier_combinations/KNN_MLP_RF/run_job.pbs

	    cp pbs/run_MLP_RF.pbs \
	    $1/$i/classifier_combinations/MLP_RF/run_job.pbs

	    cp pbs/run_LRbase.pbs \
	    $1/$i/classifier_combinations/LRbase/run_job.pbs

	    cp pbs/run_MLP-hl.pbs \
	    $1/$i/classifier_combinations/MLP-hl/run_job.pbs

	    cp pbs/run_KNN_MLP-hl.pbs \
	    $1/$i/classifier_combinations/KNN_MLP-hl/run_job.pbs

	    cp pbs/run_KNN_MLP-hl_RF.pbs \
	    $1/$i/classifier_combinations/KNN_MLP-hl_RF/run_job.pbs

	    cp pbs/run_MLP-hl_RF.pbs \
	    $1/$i/classifier_combinations/MLP-hl_RF/run_job.pbs


	    cd $1/$i; qsub run_cv.pbs; cd $top_dir;
	done
else
	echo "Either too many arguements were pass or you need to give the 
to the cross-validation 
For example: ./distr_jobs_cv.sh cross_validation/"
fi
