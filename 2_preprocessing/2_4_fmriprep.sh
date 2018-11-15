for i in {3..8}; do
	sbatch 2_4_fmriprep.sbatch $(printf sub-%02d $i)
done
