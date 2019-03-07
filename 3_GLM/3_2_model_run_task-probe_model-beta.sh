for i in {4..8}; do
	SUBJECT=$(printf sub-%02d $i)
	echo "Submitting job for: " $SUBJECT
	sbatch 3_1_model_run.sbatch OPUS $SUBJECT probe beta 2
done
