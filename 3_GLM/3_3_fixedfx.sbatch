#!/bin/bash
#3_5_FIXEDFX
#Fixed effects analysis
#Usage: sbatch 3_3_model_run.sbatch [project] [subject] [task] [model]

#################
#set a job name  
#SBATCH --job-name=fixedfx
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=reports/fixedfx.%j.out
#################
# a file for errors from the job
#SBATCH --error=reports/fixedfx.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=05:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
#SBATCH -p normal
#################
#number of nodes you are requesting, the more you ask for the longer you wait
#SBATCH --nodes=1
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem-per-cpu=4G
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
##SBATCH --mail-user=nvelez@stanford.edu

module load biology fsl

PROJECT=$1
SUBJECT=$2
TASK=$3
MODEL=$4

# Find input directory
PROJECT_DIR=$PI_SCRATCH/$PROJECT
MODEL_DIR=$PROJECT_DIR/BIDS_data/derivatives/model
IN_DIR=$MODEL_DIR/task-$TASK/model-$MODEL/$SUBJECT/input_fixedfx
OUT_DIR=${IN_DIR/input_/}

mkdir -p $OUT_DIR

# Number of contrasts to iterate over
NCON=$(ls $IN_DIR/cope* | wc -l)

# Run fixedfx 
for i in `seq 1 $NCON`; do
	COPE=$(printf cope%i_merged.nii.gz $i)
	VARCOPE=$(printf varcope%i_merged.nii.gz $i)
	CON=$(printf con-%02d $i)
	flameo --copefile=$IN_DIR/$COPE --varcopefile=$IN_DIR/$VARCOPE --covsplitfile=$IN_DIR/design.grp --designfile=$IN_DIR/design.mat --ld=$OUT_DIR/$CON --maskfile=$IN_DIR/brain_mask.nii.gz --runmode=fe --tcontrastsfile=$IN_DIR/design.con --dofvarcope=$IN_DIR/dof_file.nii.gz
done


