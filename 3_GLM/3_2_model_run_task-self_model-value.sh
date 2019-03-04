for i in {3..8}; do
	SUBJECT=$(printf sub-%02d $i)
	echo "Submitting job for: " $SUBJECT
	sbatch 3_1_model_run.sbatch OPUS $SUBJECT self value 2
done
