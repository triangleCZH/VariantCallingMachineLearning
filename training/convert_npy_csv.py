from __future__ import print_function, division


import numpy as np
import pandas

import argparse
from os import listdir, path
import sys

FLAGS = None

num_class = 14
num_dimension = 22

label_position_col = 17
data_position_col = 22


def main():

    input_prediction_files = [x for x in listdir(FLAGS.filePredictPath) if x.endswith(FLAGS.predictionFileSuffix)]
    if len(input_prediction_files) == 0:
        sys.exit("Prediction file not found")

    input_label_files = [x for x in listdir(FLAGS.filePath) if x.endswith(FLAGS.labelFileSuffix)]
    if len(input_label_files) == 0:
        sys.exit("File not found")

    if len(input_label_files) != len(input_prediction_files):
        sys.exit("Number of prediction files and label files not match")

    input_prediction_files.sort()
    input_label_files.sort()

    for file_index in range(len(input_prediction_files)):

        filename = input_prediction_files[file_index][0:input_prediction_files[file_index].rfind(FLAGS.predictionFileSuffix)]
        if filename != input_label_files[file_index][0:input_label_files[file_index].rfind(FLAGS.labelFileSuffix)]:
            sys.exit("File name of prediction file and label file not match" + input_predition_files[file_index] + ' ' + input_label_files[file_index])

        input_label_raw = np.load(path.join(FLAGS.filePath, input_label_files[file_index]))
        input_label = input_label_raw['data']
        input_prediction = np.load(path.join(FLAGS.filePredictPath, input_prediction_files[file_index]))
        num_sample = len(input_label)
        num_sample = input_label.shape[0]
        if num_sample != input_prediction.shape[0]:
            sys.exit("Number of samples in prediction file and label file not match" + filename)
        label_file = open(path.join(FLAGS.fileOutputPath, input_prediction_files[file_index]).replace(FLAGS.predictionFileSuffix, FLAGS.outputSuffix), "w")

        for i in range(num_sample):

            #label = np.zeros((9,), dtype=int)

            p = np.argmax(input_prediction[i])
            #FIXME: is this needed, or need to comment out?
            if p == 10:
                p = 0
            if p == 11:
                p = 4
            if p == 12:
                p = 7
            if p == 13:
                p = 9

            #label[8] = p

            for j in range(8):
                label_file.write('{:s}\t'.format(str(input_label[i][100][j].astype(np.float32))))

            #label_file.write('{:d}\t'.format(input_label[i][16]))
            label_file.write('{:d}\n'.format(p))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Extract csv')
    parser.add_argument('--outputSuffix', help='output file pattern', default='.ml.csv')
    parser.add_argument('--filePath', help='File path', default='/home/user1/Simon/Lam/ml-last/')
    parser.add_argument('--fileOutputPath', help='Output path', default='/home/user1/Simon/Lam/ml-last/csv/')
    parser.add_argument('--filePredictPath', help='Predict npy path', default='/home/user1/Simon/Lam/ml-last/predict/')
    parser.add_argument('--predictionFileSuffix', help='Prediction file pattern', default='.predict.npy')
    parser.add_argument('--labelFileSuffix', help='Label file pattern', default='.npz')

    FLAGS, UNPARSED = parser.parse_known_args()

    main()
