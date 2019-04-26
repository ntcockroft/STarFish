#!/bin/sh
#******************************************************************************
# NOTE: prep_data.sh should be run first!
# Files should be be copied to HPC cluster
# These scripts are written for a bash shell and a PBS batch scheduling system
#------------------------------------------------------------------------------
# Generates directories for organization, copies .pbs submission files to 
# appropriate locations and submits jobs to HPC cluster for base classifier 
# tuning
#******************************************************************************
top_dir=`pwd`

top_dir=`pwd`
for i in {1..10}; do  
    cp pbs/run_KNN_opt.pbs cross_validation/$i/tuning/
    cp pbs/run_MLP_opt.pbs cross_validation/$i/tuning/
    cp pbs/run_RF_opt.pbs cross_validation/$i/tuning/
    
    cd cross_validation/$i/tuning; qsub run_KNN_opt.pbs; qsub run_MLP_opt.pbs; \
    qsub run_RF_opt.pbs; cd $top_dir;
done
