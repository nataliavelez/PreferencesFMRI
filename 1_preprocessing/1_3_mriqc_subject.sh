for i in {3..8}; do
	sbatch 1_3_mriqc_subject.sbatch $(printf %02d $i)
done
