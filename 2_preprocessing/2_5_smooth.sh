for i in {3..8}; do
	SUBJECT=$(printf sub-%02d $i)
	echo "Smoothing data for: " $SUBJECT
	sbatch 2_5_smooth.sbatch OPUS $SUBJECT
done
