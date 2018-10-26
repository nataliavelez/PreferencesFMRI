# 0_preprocessing
Preprocessing steps

In this folder:

* `00_deface`: Deface anatomical images before processing
* `01_nims2bids`: Organize anonymized data using BIDS convention
* `02_clip_functionals`: Removes first three images from each functional file
* `03_mriqc`: Call MRIQC (for data quality checks)
* `04_fmriprep`: Call fmriprep (preprocessing)
* `05_smooth`: Smooth fmriprep outputs 
