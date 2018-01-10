from __future__ import print_function, division

import subprocess
import shlex
import numpy as np
import pandas

import argparse
from os import listdir, path
import sys
import os as os
import csv
FLAGS = None

num_class = 14
num_dimension = 22

label_position_col = 17
data_position_col = 22

slice_row_num = 101
slice_col_num = 8
slice_thick = 100000

csv_prefix = "HG002_iiiii"
def main():
    print("I start at the main function")
    if FLAGS.randomSeed > 0:
        np.random.seed(FLAGS.randomSeed)

    input_data_files = [x for x in listdir(FLAGS.dataFilePath) if x.endswith(FLAGS.dataFileSuffix)]
    if len(input_data_files) == 0:
        sys.exit("Data file not found")

    input_label_files = [x for x in listdir(FLAGS.dataFilePath) if x.endswith(FLAGS.labelFileSuffix)]
    if len(input_label_files) == 0:
        sys.exit("File not found")

    if len(input_label_files) != len(input_data_files):
        sys.exit("Number of data files and label files not match")

    input_data_files.sort()
    input_label_files.sort()
    print("I'm just before the for loop")
    print(FLAGS.filePath)
    print(FLAGS.dataFilePath)
    for file_index in range(len(input_data_files)):

        filename = input_data_files[file_index][0:input_data_files[file_index].rfind(FLAGS.dataFileSuffix)]
        if filename != input_label_files[file_index][0:input_label_files[file_index].rfind(FLAGS.labelFileSuffix)]:
            sys.exit("File name of data file and label file not match" + input_data_files[file_index] + ' ' + input_label_files[file_index])

        input_label = pandas.read_csv(path.join(FLAGS.dataFilePath, input_label_files[file_index]), dtype=int, sep='\t', header=None).values
        input_data = pandas.read_csv(path.join(FLAGS.dataFilePath, input_data_files[file_index]), dtype=int, sep='\t', header=None).values

        num_sample = input_label.shape[0]
        if num_sample != input_data.shape[0]:
            sys.exit("Number of samples in data file and label file not match" + filename)

        num_validation_sample = int(num_sample * FLAGS.validationPercentage / 100)
        num_test_sample = int(num_sample * FLAGS.testPercentage / 100)
        num_training_sample = num_sample - num_validation_sample - num_test_sample

        training_label = np.zeros((FLAGS.batchSize, num_class), dtype=np.float32)
        training_data = np.zeros((FLAGS.batchSize, slice_row_num, slice_col_num), dtype=np.float32) #np.zeros((FLAGS.batchSize, num_dimension), dtype=np.float32)

        validation_label = np.zeros((FLAGS.batchSize, num_class), dtype=np.float32)
        validation_data = np.zeros((FLAGS.batchSize, slice_row_num, slice_col_num), dtype=np.float32) #np.zeros((FLAGS.batchSize, num_dimension), dtype=np.float32)

        test_label = np.zeros((FLAGS.batchSize, num_class), dtype=np.float32)
        test_data = np.zeros((FLAGS.batchSize, slice_row_num, slice_col_num), dtype=np.float32) #np.zeros((FLAGS.batchSize, num_dimension), dtype=np.float32)

        partition = np.arange(num_sample)
        np.random.shuffle(partition)

        training_index = 0
        validation_index = 0
        test_index = 0

        training_file_index = 0
        validation_file_index = 0
        test_file_index = 0

        max_num_file = (max(num_training_sample, num_validation_sample, num_test_sample) + FLAGS.batchSize - 1) // FLAGS.batchSize
        serialNumberFormat = "{0:0>" + str(int(np.ceil(np.log10(max_num_file)))) + "}"

        for i in range(num_sample):

            label = np.zeros((num_class,), dtype=np.float32)
            data = np.zeros((slice_row_num, slice_col_num), dtype=np.float32) #np.zeros((num_dimension,), dtype=np.float32)
            filename = "/dev/shm/csv-result/result0iiiii/sjjjjj/" + csv_prefix + "_" + input_label[i][16].astype(str) + "_" + input_label[i][17].astype(str) + "_gen.csv"
            #print(filename)
            if os.path.exists(filename)==False: 
                print("%s not exist" % filename)
                #subprocess.call(shlex.split(command)) 
                continue
            """if os.path.getsize(filename) == 0: 
                command = "echo '%s empty' >> /home/user1/Simon/Lam/result00/log " % filename
                subprocess.call(shlex.split(command))
                continue"""
            #command = "sed -i '/there is a break/d' %s" % filename
            #print(command)
            #try:
            #subprocess.call(shlex.split(command))
            #except:
                #continue
            try:
                slice_data = pandas.read_csv(filename, dtype = int, sep='\t', header=None).values
            except:
                continue
            #slice_data = pandas.read_csv("/home/user1/Simon/Lam/result00/" + csv_prefix + "_" + input_label[i][16].astype(str) + "_" + input_label[i][17].astype(str) + ".csv", dtype= int, sep='\t', header=None).values
            #slice_data = pandas.read_csv("/home/user1/Simon/Lam/result00/HG002_1_218267709.csv", dtype= int, sep='\t', header=None).values
            #print("I reach this line")
            """csv_file = open("/home/user1/Simon/Lam/result00/" + csv_prefix + "_" + input_label[i][16].astype(str) + "_" + input_label[i][17].astype(str) + ".csv")
            reader = csv.reader(csv_file, delimiter = '	')
            row_count = 0
            for rowing in reader:
                row = rowing.split("\t")
                if row_count >= slice_row_num: break
                for rows in range(slice_col_num):
                    data[row_count][rows] = float(row[rows])
                row_count += 1"""
            for j in range(min(slice_row_num, len(slice_data))):
                for k in range(slice_col_num):
                    data[j][k] = slice_data[j][k].astype(np.float32) / 255 #np.fromstring(slice_data[j][k], dtype=np.float32) / 255 #all the data has been stored into the 3d array

            #data[slice_row_num - 1][0] = 
            for j in range(slice_col_num):
                if j < 4:
                    data[slice_row_num - 1][j] = input_data[i][18 + j].astype(np.float32) / 255
                elif j == 4 :
                    data[slice_row_num - 1][j] = input_label[i][16].astype(np.float32)
                elif j == 6 :
                    data[slice_row_num - 1][j] = (input_label[i][17] % 65536).astype(np.float32)
                elif j == 5 :
                    data[slice_row_num - 1][j] = ((input_label[i][17] - input_label[i][17] % 65536) / 65536).astype(np.float32) 
                else:
                    label_position = 13
                    label_value = input_label[i][13]
                    for k in range(13): #has set the initial label as the 14-th, so only search through first 13
                        if input_label[i][k] > label_value:
                            label_position = k
                            label_value = input_label[i][k]
                    data[slice_row_num - 1][j] = label_position
            #print("I even reach this line")
            #with open('ml.txt') as f:
                #for line in data:
                    #np.savetxt(f, line, fmt='%.2f')
            #print(slice_data[min(slice_row_num, len(slice_data)) - 1][slice_col_num - 3])
            for j in range(10):
                label[j] = input_label[i][j]
            print("this is lenth of label %d" % (len(input_label[i])))
            for j in range(len(input_label[i])):
                print(input_label[i][j])
            print("end")
            """for j in range(num_dimension):
                data[j] = input_data[i][j] / 255"""
            position = input_label[i][label_position_col]
            
            if input_data[i][data_position_col] != position:
                sys.exit("Position not match")

            # separate variant or not: label[10:14] are no variants
            if input_label[i][0] == 1 and input_data[i][18] == 255:
            	label[0] = 0
            	label[10] = 1
            if input_label[i][4] == 1 and input_data[i][19] == 255:
            	label[4] = 0
            	label[11] = 1
            if input_label[i][7] == 1 and input_data[i][20] == 255:
            	label[7] = 0
            	label[12] = 1
            if input_label[i][9] == 1 and input_data[i][21] == 255:
            	label[9] = 0
            	label[13] = 1

	    #execute the script, acept label[], call chr pos
            if partition[i] < num_training_sample:

                training_label[training_index] = label
                training_data[training_index] = data

                training_index += 1
                if training_index >= FLAGS.batchSize:
                    #np.savez(path.join(FLAGS.filePath, filename + '.' + FLAGS.trainingSetExtension + '.' + serialNumberFormat.format(training_file_index) + '.' + str(FLAGS.batchSize)),
                             #label=training_label,
                             #data=training_data,
                             #)
                    np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.trainingSetExtension + '.' + serialNumberFormat.format(training_file_index) + '.' + str(FLAGS.batchSize), label = training_label, data=training_data,)

                    training_index = 0
                    training_file_index += 1

            elif partition[i] < num_training_sample + num_validation_sample:

                validation_label[validation_index] = label
                validation_data[validation_index] = data

                validation_index += 1
                if validation_index >= FLAGS.batchSize:
                    #np.savez(path.join(FLAGS.filePath, filename + '.' + FLAGS.validationSetExtension + '.' + serialNumberFormat.format(validation_file_index) + '.' + str(FLAGS.batchSize)),
                             #label=validation_label,
                             #data=validation_data,
                             #)
                    np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.validationSetExtension + '.' + serialNumberFormat.format(validation_file_index) + '.' + str(FLAGS.batchSize), label = validation_label, data=validation_data,)

                    validation_index = 0
                    validation_file_index += 1

            else:

                test_label[test_index] = label
                test_data[test_index] = data

                test_index += 1
                if test_index >= FLAGS.batchSize:
                    #np.savez(path.join(FLAGS.filePath, filename + '.' + FLAGS.testSetExtension + '.' + serialNumberFormat.format(test_file_index) + '.' + str(FLAGS.batchSize)),
                             #label=test_label,
                             #data=test_data,
                             #)
                    np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.testSetExtension + '.' + serialNumberFormat.format(test_file_index) + '.' + str(FLAGS.batchSize), label = test_label, data=test_data,)

                    #np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.testSetExtension + '.' + serialNumberFormat.format(test_file_index) + '.' + str(FLAGS.batchSize), label = test_label, data=test_data,)
                    print("I reach 1 %d" % test_index)
                    test_index = 0
                    test_file_index += 1

        if training_index > 0:
            """np.savez(path.join(FLAGS.filePath, filename + '.' + FLAGS.trainingSetExtension + '.' + serialNumberFormat.format(training_file_index) + '.' + str(training_index)),
                     label=training_label[0:training_index],
                     data=training_data[0:training_index],
                     )"""
            np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.trainingSetExtension + '.' + serialNumberFormat.format(training_file_index) + '.' + str(training_index), label = training_label[0:training_index], data=training_data[0:training_index],)
            print(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.trainingSetExtension + '.' + serialNumberFormat.format(training_file_index) + '.' + str(training_index))
            print("I reach 2 %d" % test_index)


        if validation_index > 0:
            """np.savez(path.join(FLAGS.filePath, filename + '.' + FLAGS.validationSetExtension + '.' + serialNumberFormat.format(validation_file_index) + '.' + str(validation_index)),
                     label=training_label[0:validation_index],
                     data=training_data[0:validation_index],
                     )"""
            np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.validationSetExtension + '.' + serialNumberFormat.format(validation_file_index) + '.' + str(validation_index), label = training_label[0:validation_index], data=training_data[0:validation_index],)
            print(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.validationSetExtension + '.' + serialNumberFormat.format(validation_file_index) + '.' + str(validation_index))
            print("I reach 3 %d" % test_index)
        if test_index > 0:
            """np.savez(path.join(FLAGS.filePath, filename + '.' + FLAGS.validationSetExtension + '.' + serialNumberFormat.format(test_file_index) + '.' + str(test_index)),
                     label=training_label[0:test_index],
                     data=training_data[0:test_index],
                     )"""
            np.savez(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.testSetExtension + '.' + serialNumberFormat.format(test_file_index) + '.' + str(test_index), label = training_label[0:test_index], data=training_data[0:test_index],)
            print(filename.replace(FLAGS.dataFilePath, FLAGS.filePath) + '.' + FLAGS.testSetExtension + '.' + serialNumberFormat.format(test_file_index) + '.' + str(test_index))
            #print(training_label[0:validation_index])
            #print(training_data[0:validation_index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Extract csv')

    parser.add_argument('--filePath', help='File path', default='/dev/shm/ml-result/result0iiiii')

    parser.add_argument('--dataFilePath', help='Data file path', default='/dev/shm/csv-result/result0iiiii/sjjjjj')
    parser.add_argument('--dataFileSuffix', help='Data file pattern', default='.data.csv')
    parser.add_argument('--labelFileSuffix', help='Label file pattern', default='.label.csv')

    parser.add_argument('--testPercentage', help='Test set percentage', type=int, default=25)
    parser.add_argument('--validationPercentage', help='Validation set percentage', type=int, default=25)

    parser.add_argument('--trainingSetExtension', help='File extension for training set', default='training')
    parser.add_argument('--testSetExtension', help='File extension for test set', default='test')
    parser.add_argument('--validationSetExtension', help='File extension for validation set', default='validation')

    parser.add_argument('--randomSeed', help='Random seed', type=int, default=0)

    parser.add_argument('--batchSize', help='Batch size', type=int, default=100000)

    FLAGS, UNPARSED = parser.parse_known_args()

    main()
