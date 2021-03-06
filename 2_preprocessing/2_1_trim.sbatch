#!/bin/bash
# 2_1_trim.sbatch
# Trims first three images from all functional files in project
# Usage: sbatch 2_1_trim.sbatch

#################
#set a job name  
#SBATCH --job-name=trim
#################  
#a file for job output, you can check job progress, append the job ID with %j to make it unique
#SBATCH --output=reports/trim.%j.out
#################
# a file for errors from the job
#SBATCH --error=reports/trim.%j.err
#################
#time you think you need; default is 2 hours
#format could be dd-hh:mm:ss, hh:mm:ss, mm:ss, or mm
#SBATCH --time=30
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
#SBATCH --mem=8000
# to request multiple threads/CPUs use the -c option, on Sherlock we use 1 thread/CPU, 16 CPUs on each normal compute node 4Gigs RAM per CPU.  Here we will request just 1.
#SBATCH -c 1
#################
# Have SLURM send you an email when the job ends or fails, careful, the email could end up in your clutter folder
# Also, if you submit hundreds of jobs at once you will get hundreds of emails.
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
# Remember to change this to your email
#SBATCH --mail-user=nvelez@stanford.edu

# Open singularity image with dependencies
module load system singularity
SIMG=$PI_HOME/singularity/sll_fmri_20181028.img
singularity run $SIMG /bin/bash 2_1_trim.sh
