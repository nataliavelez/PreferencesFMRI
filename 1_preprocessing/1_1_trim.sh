# Project directories
PROJECT_DIR=$PI_SCRATCH/OPUS
IN_DIR=$PROJECT_DIR/NIMS_data_anonymized

# Search for anatomical files
FUNC_DIRS=`find $IN_DIR -name "*BOLD_EPI_29mm*"`
printf "%s\n" $FUNC_DIRS

# Number of timepoints to trim
N_TRIM=3

for d in $FUNC_DIRS; do
	# Get file
	IN_F=$(find $d -name "*.nii.gz")
	OUT_F="${IN_F%.nii.gz}_trimmed.nii.gz"

	# Calculate # timepoints to keep
	T_IN=$(fslval $IN_F dim4)
	T_SIZE=`expr $T_IN - $N_TRIM`

	printf "Trimming: %s \n%i --> %i \nOutput: %s\n\n" $IN_F $T_IN $T_SIZE $OUT_F

	# Trim functional image
	echo "Command: fslroi $IN_F $OUT_F $N_TRIM $T_SIZE"
	fslroi $IN_F $OUT_F $N_TRIM $T_SIZE

	unset IN_F
	unset OUT_F
	unset T_IN
	unset T_SIZE

	echo "====="
done
