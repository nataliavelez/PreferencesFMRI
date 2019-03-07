for i in `seq 3 8`; do
	SUBJECT=$(printf sub-%02d $i)
	echo "Submitting job for: " $SUBJECT
	sbatch 3_3_fixedfx.sbatch OPUS $SUBJECT train value
done
