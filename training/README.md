# Variant Calling training and prediction

## Introduction
***Variant Calling*** This folder contains all the scripts after npz files have been generated, Structure part will explan each script in the logical order

## Struture
scripts in this folder don't take any external arguments
- test_7_train.py	use the provided npz files to train the model
- workflow	record the training process(i.e. what layers are changed each time)
- prediction.py	given the best weight, predict label
- convert_npy_csv.py	use csv to store reference, chrom, pos, label and predict label
- check_prediction.py	summarize the training result

## Other info
other using directories:
bal4:/home/user1/Simon/Lam/
bal23:/dev/shm
