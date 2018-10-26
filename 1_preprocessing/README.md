# 01_preprocessing
Preprocessing steps

In this folder:

* `00_deface.sbatch`: Deface anatomical images before processing
* `01_nims2bids.sbatch`: Organize anonymized data using BIDS convention
* `02_clip_functionals.sbatch`: Removes first three images from each functional file
* `03_mriqc.sbatch`: Call MRIQC (for data quality checks)
* `04_fmriprep.sbatch`: Call fmriprep (preprocessing)
* `05_smooth.sbatch`: Smooth fmriprep outputs 
