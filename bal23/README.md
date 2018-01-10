# Variant Calling label/data->csv->npy

## Introduction
Processing the label/data of reads, to generate csv of each position and compress into npz files

## Structure
- script	contains main scripts in use
  - run_line.sh	the script to be directly used to generate csv file for each chrom pos
  - gen_line.sh	generate csv file for each chrom pos, called by run_line.sh
  - draw.py	generate csv file for each chrom pos, called by gen_line.sh
  - extract_csv.py		generate npz file in batches, called by run）line.sh
- csv-result	store the result csv
  - help_folder_generate.sh	generate folders for each snapshot and parts, and log
  - help_folder_generate.py	called by help_folder_generate.sh, help to gen folders
  - extract_csv.py	copied from /dev/shm/script, each part in each snapshot will take a copy of it 
- ml-result	store the result npz
  - check_npz.py	check info inside npz file
  - help_folder_generate.sh	generate folders for each snapshot and log
- data	store data/label/HG002/ucsc
- abandoned-/	abandoned folder, not deleted yet because too large, please delete them to save space
## Other info
other using directories
bal4:/home/user1/Simon/Lam
bal4:/home/user1/Simon/training
