for i in {3..8}; do
	sbatch 1_4_fmriprep.sbatch $(printf sub-%02d $i)
done
