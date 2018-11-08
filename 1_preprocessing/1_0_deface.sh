# Project directories
PROJECT_DIR=$PI_SCRATCH/OPUS
IN_DIR=$PROJECT_DIR/NIMS_data
OUT_DIR=$IN_DIR"_anonymized"

echo "Copying data..."
cp -r $IN_DIR $OUT_DIR

# Search for anatomical files
ANAT_DIRS=`find $OUT_DIR -name "*T1w_9mm_sag*"`
for d in $ANAT_DIRS; do
        ANAT_FILE=`find $d -name "*.nii.gz"`
	echo "Defacing $ANAT_FILE..."
        python -m pydeface $ANAT_FILE

	# Remove unanonymized file
	rm -f $ANAT_FILE
done
