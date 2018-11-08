#!/bin/bash
#SWIST_GROUP_MRIQC
#Creates MRIQC summary for the whole group
#NB: ONLY run this after running mriqc on each participant!! 
#Usage: sbatch swist_group_mriqc.sbatch

#################
#set a job name  
#SBATCH --job-name=MRIQC
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=sbatch_reports/MRIQC.%j.out
#################
# a file for errors from the job
#SBATCH --error=sbatch_reports/MRIQC.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=1:30:00
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
#SBATCH --mem=10000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
##SBATCH --mail-user=YourSUNetID@stanford.edu

ml load system
ml load singularity

# Get singularity image
singularity_dir=$PI_HOME/swist_singularity_images
MRIQC_IMG=$singularity_dir/poldracklab_mriqc_0.10.4-2018-03-23-f1e4d5941533.img

# Build call to MRIQC
BIDS_DIR=$SCRATCH/SwiSt/BIDS_data
OUT_DIR=$BIDS_DIR/derivatives/mriqc
WORKING_DIR=$SCRATCH/SwiSt

# Call MRIQC
singularity run $MRIQC_IMG \
	$BIDS_DIR $OUT_DIR group \
	-w $WORKING_DIR

