for i in {5..23}; do
	SUBJECT=$(printf sub-%02d $i)
	echo "Smoothing data for: " $SUBJECT
	sbatch smoothing_postfmriprep.sbatch SwiSt $SUBJECT
done
