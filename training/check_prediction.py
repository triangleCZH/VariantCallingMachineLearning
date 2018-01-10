from __future__ import print_function, division


import numpy as np
import pandas
import argparse
import random
import fnmatch
from os import listdir, path
import sys
FLAGS = None

num_class = 14



def main():
    #FIXME: because later on confusion matrix is indexed by 'reference + 14', so I expand the range by 4, otherwise there could be a indexOutOfBound problem
    total_confusion_matrix = np.zeros((num_class + 4, num_class + 4), dtype=int)
    total_num_sample = 0
    total_num_variant = 0
    total_num_true_positive = 0
    total_num_false_positive = 0
    total_num_false_negative = 0
    total_num_different_class = 0

    #total_num_indel = 0

    input_data_files = [x for x in listdir(FLAGS.filePath) if x.endswith(FLAGS.dataFileSuffix)]
    if len(input_data_files) == 0:
        sys.exit("Data file not found")

    #input_label_files = [x for x in listdir(FLAGS.filePath) if x.endswith(FLAGS.labelFileSuffix)]
    #if len(input_label_files) == 0:
    #    sys.exit("File not found")

    #input_predict_files = [x for x in listdir(FLAGS.filePath) if x.endswith(FLAGS.predictFileSuffix)]
    #if len(input_predict_files) == 0:
    #    sys.exit("File not found")

    num_file = len(input_data_files)
    #if num_file != len(input_label_files) or num_file != len(input_predict_files):
    #    sys.exit("Number of files not match")

    input_data_files.sort()
    #input_label_files.sort()
    #input_predict_files.sort()

    for file_index in range(num_file):

        confusion_matrix = np.zeros((num_class + 4, num_class + 4), dtype=int)
        num_sample = 0
        num_variant = 0
        num_true_positive = 0
        num_false_positive = 0
        num_false_negative = 0
        num_different_class = 0

        num_indel = 0

        #print('Opening file set', file_index)

        file_prefix = input_data_files[file_index].replace(FLAGS.dataFileSuffix, '')
        #if file_prefix != input_label_files[file_index].replace(FLAGS.labelFileSuffix, '') or file_prefix != input_predict_files[file_index].replace(FLAGS.predictFileSuffix, ''):
        #    sys.exit("File name in a file set not match")

        print('File prefix is', file_prefix)

        data = pandas.read_csv(path.join(FLAGS.filePath, input_data_files[file_index]), sep='\t', dtype=float, header=None).values
        data = data.astype(int)
        #label = pandas.read_csv(path.join(FLAGS.filePath, input_label_files[file_index]), sep='\t', dtype=int, header=None).values
        #predict = pandas.read_csv(path.join(FLAGS.filePath, input_predict_files[file_index]), sep='\t', dtype=int, header=None).values
   
        num_sample = data.shape[0]
        #if num_sample != label.shape[0] or num_sample != predict.shape[0]:
        #    sys.exit("Number of samples not match")

        log = open(path.join(FLAGS.filePath, file_prefix + FLAGS.logFileSuffix), 'w')
        diff = open(path.join(FLAGS.filePath, file_prefix + FLAGS.diffFileSuffix), 'w')

        for i in range(num_sample):

            # Skip INDEL
            #if data[i][12] >= 50:
                #num_indel += 1
                #continue

            true_label = int(data[i][7]) #np.argmax(label[i][0:16])
            predict_label = int(data[i][8]) #np.argmax(predict[i][0:16])
            #print("true label%d:     predict label%d:" % (true_label, predict_label))
            reference = -1 #np.argmax(data[i][18:22]) - 18
            for j in range(4):
                if (data[i][j] == 1):
                     reference = j
            if reference == -1:
                print("No reference")
                continue

            is_variant = None
            # 18-21 -> reference, csv 0-3     true_label -> label (AA CC GG TT) TODO   
            if (data[i][0] == 1  and true_label == 0) or (data[i][1] == 1 and true_label == 4) or (data[i][2] == 1  and true_label == 7) or (data[i][3] == 1 and true_label == 9):
                is_variant = False
            else:
                is_variant = True

            predict_variant = None
            if (data[i][0] == 1  and predict_label == 0) or (data[i][1] == 1 and predict_label == 4) or (data[i][2] == 1  and predict_label == 7) or (data[i][3] == 1 and predict_label == 9):
                predict_variant = False
            else:
                predict_variant = True

            if is_variant == True:
                if predict_variant == True:
                    confusion_matrix[predict_label, true_label] += 1
                else:
                    confusion_matrix[reference + 14, true_label] += 1
            else:
                if predict_variant == True:
                    confusion_matrix[predict_label, reference + 14] += 1
                else:
                    confusion_matrix[reference + 14, reference + 14] += 1
            print("predict variant: %s    is variant: %s" % (str(predict_variant), str(is_variant)))
            class_different = False
            if true_label != predict_label:
                class_different = True

            if is_variant == False:
                if class_different == True:
                    num_false_positive += 1
            else:
                num_variant += 1
                if class_different == False:
                    num_true_positive += 1
                else:
                    num_false_negative += 1
                    if predict_variant == True:
                        num_false_positive += 1

            if class_different == True:
                num_different_class += 1
                # Diff file contains: data + True variant(Y/N) + True label + Preidct variant(Y/N) + Predict label + chromosome + location
                for j in range(9):
                    #FIXME: not really sure what should be writen here
                    diff.write('{:d}\t'.format(data[i][j]))
                if is_variant == True:
                    diff.write('Y\t')
                else:
                    diff.write('N\t')
                diff.write('{:d}\t'.format(true_label))
                if predict_variant == True:
                    diff.write('Y\t')
                else:
                    diff.write('N\t')
                diff.write('{:d}\t'.format(predict_label))
                diff.write('{:d}\t'.format(data[i][4])) #label[i][16]))
                diff.write('{:d}\n'.format(int(data[i][5]) * 65536 + int(data[i][6]))) #label[i][17]))

        diff.close()

        log.write('Number of samples:        {:d}\n'.format(num_sample))
        #log.write('Skipped INDEL:            {:d}\n'.format(num_indel))
        log.write('Number of variants:       {:d}\n'.format(num_variant))

        log.write('\n')

        log.write('Number of true positive:  {:d}\n'.format(num_true_positive))
        log.write('Number of false negative: {:d}\n'.format(num_false_negative))
        log.write('Number of false positive: {:d}\n'.format(num_false_positive))

        if num_variant != 0:
            recall = num_true_positive / num_variant
        #FIXME I don't know how to deal with no-variant situation, if it happens, what will be the algorithm?
            precision = num_true_positive / (num_true_positive + num_false_positive)
            F1_measure = 2 * ( (precision * recall ) / (precision + recall) )
        else:
            recall = 0
            precision = 0
            F1_measure = 0
        log.write('\n')

        log.write('Recall:     {:.9f}\n'.format(recall))
        log.write('Precision:  {:.9f}\n'.format(precision))
        log.write('F1 Measure: {:.9f}\n'.format(F1_measure))

        log.write('\n')

        log.write('Diff class:  {:d}\n'.format(num_different_class))

        log.write('\n')

        accuracy = (num_sample - num_indel - num_different_class) / (num_sample - num_indel)
        log.write('Accuracy:   {:.9f}\n'.format(accuracy))

        log.write('\n')
        log.write('Confusion matrix:\n\n')

        for j in range(num_class):
            for k in range(num_class - 1):
                log.write('{:d}\t'.format(confusion_matrix[j][k]))
            log.write('{:d}\n'.format(confusion_matrix[j][num_class - 1]))

        log.write('\n')

        total_confusion_matrix = np.add(total_confusion_matrix, confusion_matrix)
        total_num_sample += num_sample
        total_num_variant += num_variant
        total_num_true_positive += num_true_positive
        total_num_false_positive += num_false_positive
        total_num_false_negative += num_false_negative
        total_num_different_class += num_different_class

        #total_num_indel += num_indel

    log = sys.stdout

    confusion_matrix = total_confusion_matrix
    num_sample = total_num_sample
    num_variant = total_num_variant
    num_true_positive = total_num_true_positive
    num_false_positive = total_num_false_positive
    num_false_negative = total_num_false_negative
    num_different_class = total_num_different_class

    #num_indel = total_num_indel


    log.write('Number of samples:        {:d}\n'.format(num_sample))
    log.write('Skipped INDEL:            {:d}\n'.format(num_indel))
    log.write('Number of variants:       {:d}\n'.format(num_variant))

    log.write('\n')

    log.write('Number of true positive:  {:d}\n'.format(num_true_positive))
    log.write('Number of false negative: {:d}\n'.format(num_false_negative))
    log.write('Number of false positive: {:d}\n'.format(num_false_positive))

    recall = num_true_positive / num_variant
    precision = num_true_positive / (num_true_positive + num_false_positive)
    F1_measure = 2 * ( (precision * recall ) / (precision + recall) )

    log.write('\n')

    log.write('Recall:     {:.9f}\n'.format(recall))
    log.write('Precision:  {:.9f}\n'.format(precision))
    log.write('F1 Measure: {:.9f}\n'.format(F1_measure))

    log.write('\n')

    log.write('Diff class:  {:d}\n'.format(num_different_class))

    log.write('\n')

    accuracy = (num_sample - num_indel - num_different_class) / (num_sample - num_indel)
    log.write('Accuracy:   {:.9f}\n'.format(accuracy))

    log.write('\n')
    log.write('Confusion matrix:\n\n')

    for j in range(num_class):
        for k in range(num_class - 1):
            log.write('{:d}\t'.format(confusion_matrix[j][k]))
        log.write('{:d}\n'.format(confusion_matrix[j][num_class - 1]))

    log.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Check prediction')

    parser.add_argument('--dataFileSuffix', help='Data file suffix', default='.ml.csv')
    #parser.add_argument('--labelFileSuffix', help='Label file suffix', default='.label.csv')
    #parser.add_argument('--predictFileSuffix', help='Prediction file suffix', default='.elsa_ml.csv')

    parser.add_argument('--logFileSuffix', help='Log file suffix', default='.ml.log')
    parser.add_argument('--diffFileSuffix', help='Diff file suffix', default='.ml.diff')

    parser.add_argument('--filePath', help='File path', default='/home/user1/Simon/Lam/ml-last/csv/')

    FLAGS, UNPARSED = parser.parse_known_args()

    main()
