#!/bin/bash
#SWIST_SUBJ_FMRIPREP
#Runs FMRIPREP on a sngle participant
#Usage: sbatch 2_4_fmriprep.sbatch [SUBJECT_ID]

#################
#set a job name  
#SBATCH --job-name=fmriprep
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=reports/fmriprep.%j.out
#################
# a file for errors from the job
#SBATCH --error=reports/fmriprep.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=01-00:00:00
#################
# Quality of Service (QOS); think of it as sending your job into a special queue; --qos=long for with a max job length of 7 days.
#SBATCH --qos=normal
#################
#number of nodes you are requesting, the more you ask for the longer you wait
#SBATCH --nodes=1
#################
# --mem is memory per node; default is 4000 MB per CPU, remember to ask for enough mem to match your CPU request, since 
# sherlock automatically allocates 4 Gigs of RAM/CPU, if you ask for 8 CPUs you will get 32 Gigs of RAM, so either 
# leave --mem commented out or request >= to the RAM needed for your CPU request.  It will also accept mem. in units, ie "--mem=4G"
#SBATCH --mem-per-cpu=16G
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 4
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
##SBATCH --mail-user=YourSUNetID@stanford.edu

ml load system
ml load singularity

# Get singularity image
FMRIPREP_IMG=$PI_HOME/singularity/fmriprep_1.1.6.img

# Build call to fmriprep
PROJECT_DIR=$PI_SCRATCH/OPUS
BIDS_DIR=$PROJECT_DIR/BIDS_data
OUT_DIR=$BIDS_DIR/derivatives
WORKING_DIR=$PROJECT_DIR/cache/fmriprep
SUBJECT=$1

# Freesurfer license
FS_LICENSE=/home/users/nvelez/freesurfer/license.txt

# Call fmriprep
printf "[DEBUG] Running on subject %s\n" $SUBJECT
printf "[DEBUG] Location of dataset: %s\n" $BIDS_DIR
printf "[DEBUG] Working directory: %s\n" $WORKING_DIR
printf "[DEBUG] Saving output to: %s\n" $OUT_DIR

singularity run --bind $BIDS_DIR:$BIDS_DIR $FMRIPREP_IMG \
	$BIDS_DIR $OUT_DIR participant \
	--participant-label $SUBJECT -w $WORKING_DIR \
	--nthreads 2 --omp-nthreads 4 --mem-mb 64000 \
	--fs-no-reconall
	#--fs-license-file $FS_LICENSE
