# Variant Caller using Snapshot data


from __future__ import print_function, division

import keras

from os import listdir, path
import fnmatch
import random
import numpy as np
import argparse
import time
import threading

import sys
sys.path.append('../lib')

from data.data_generator import DataGenerator
from callback.record_epoch import RecordEpoch


num_row = 101
num_col = 8

data_shape = (num_row, num_col)
num_class = 14
#class_weight = { 0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
class_weight = np.ones((num_class,))
class_weight[0] = 25.8
class_weight[1] = 46.3
class_weight[2] = 11.2
class_weight[3] = 55.4
class_weight[4] = 22.9
class_weight[5] = 44.2
class_weight[6] = 11.2
class_weight[7] = 23.0
class_weight[8] = 46.1
class_weight[9]= 25.8
class_weight[10] = 1
class_weight[11] = 1.9
class_weight[12] = 1.9
class_weight[13] = 1



best_epoch_dict = {'epoch_index': 0, 'val_loss': 1, 'val_acc': 0}

FLAGS = None

# this is for preventing overheat on some machines only
sleep_after_epoch = 0

batch_size = 256
"""
class IdentityMatrix(keras.engine.topology.Layer):

    def call(self, x):
        I = keras.backend.eye(num_row)
        I = keras.backend.reshape(I, (1, num_row * num_row))
        I = keras.backend.repeat_elements(I, num_col-kernal_size+1, 0)
        I = keras.backend.repeat_elements(I, batch_size, 0)

        I = keras.backend.reshape(I, (batch_size, num_col-kernal_size+1, num_row, num_row, 1))

        return I

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = 1
        return tuple(shape)
"""

def define_model():
    input = keras.layers.Input(data_shape)
    s = keras.layers.Lambda(lambda x: x[:,0:100, : ])(input)
    print("At the very beginning")
    print(s.get_shape())
    r = keras.layers.Lambda(lambda x: x[:,100:101,0:4])(input)
    print("This is r")
    print(r.get_shape())
    #s = keras.layers.Flatten()(s) #input
    #num_feature_5 = 16 #x=8*100

    r = keras.layers.Flatten()(r) #input
    num_feature_5 = 16 #x=8*100
    print("r after flatten") #flatten make the 2D matrix into 1D
    print(r.get_shape())
    """ x = keras.layers.Dense(num_feature_5,)(x)
    x = keras.layers.core.Activation('selu')(x)

    num_feature_6 = 16

    x = keras.layers.Dense(num_feature_6,)(x)
    x = keras.layers.core.Activation('selu')(x)

    x = keras.layers.Dense(num_class,)(x)
    prediction = keras.layers.core.Activation('softmax')(x)

    model = keras.models.Model(inputs=input, outputs=prediction)"""

    #s = keras.backend.transpose(s)
    print("Initial shape")
    print(s.get_shape())
    s = keras.layers.Conv1D(filters = 8, kernel_size = 1, strides = 1)(s)
    print("After first convolution")
    print(s.get_shape())
    s = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(s)

    s = keras.layers.Conv1D(filters = 8, kernel_size = 1, strides = 1)(s)
    print("After second convolution")
    print(s.get_shape())
    s = keras.layers.AveragePooling1D(pool_size=2, strides=2, padding='valid')(s)
    print("After first avg pooling layer")
    print(s.get_shape())
    s = keras.layers.Conv1D(filters = 8, kernel_size = 1, strides = 1)(s)
    s = keras.layers.AveragePooling1D(pool_size=25, strides=None, padding='valid')(s)
    s = keras.layers.Flatten()(s) #input
    print("After s flatten")
    print(s.get_shape())
    s = keras.layers.Dense(30, #feature
                           name='Dense_1',
                           kernel_initializer=keras.initializers.he_normal(),
                           kernel_regularizer=keras.regularizers.l2(FLAGS.L2),
                           activation='selu',
                          )(s)
    print("After first dense")
    print(s.get_shape())
    s = keras.layers.Dropout(FLAGS.dropout, name='Dropout_1', )(s)
    print("After first dropout")
    print(s.get_shape())
    s = keras.layers.Dense(30,
                           name='Dense_2',
                           kernel_regularizer=keras.regularizers.l2(FLAGS.L2),
                           kernel_initializer=keras.initializers.he_normal(),
                           activation='selu',
                          )(s)
    print("After second dense")
    print(s.get_shape())

    x = keras.layers.concatenate([s, r], axis=1)
    print("After x concat")
    print(x.get_shape())
    x = keras.layers.Dropout(FLAGS.dropout, name='Dropout_2', )(x)
    print("After second  dropout")
    print(x.get_shape())
    x = keras.layers.Dense(30,
                           name='Dense_3',
                           kernel_regularizer=keras.regularizers.l2(FLAGS.L2),
                           kernel_initializer=keras.initializers.he_normal(),
                           activation='selu',
                          )(x)
    print("After third dense")
    print(x.get_shape())
    x = keras.layers.Dropout(FLAGS.dropout, name='Dropout_3', )(x)
    print("After third dropout  dropout")
    print(s.get_shape())
    """x = keras.layers.Dense(30,
                           name='Dense_4',
                           kernel_regularizer=keras.regularizers.l2(FLAGS.L2),
                           kernel_initializer=keras.initializers.he_normal(),
                           activation='selu',
                          )(x)
    print("After fourth dense")
    print(x.get_shape())
    x = keras.layers.Dropout(FLAGS.dropout, name='Dropout_4', )(x)
    print("After fourth dropout")
    print(s.get_shape())"""

    prediction = keras.layers.Dense(num_class,
                                    name='Softmax',
                                    activation='softmax',
                                    kernel_regularizer=keras.regularizers.l2(FLAGS.L2),
                                    kernel_initializer=keras.initializers.he_normal(),
                                    )(x)


    model = keras.models.Model(inputs=input, outputs=prediction)
    return model




def test(model):
    print('Number of test samples:       ', end='')
    test_data_generator = DataGenerator(data_shape, num_class,
                                        FLAGS.filePath, FLAGS.filePattern + FLAGS.testSetExtension + '.*.' + FLAGS.extension,
                                        FLAGS.testBatchSize,
                                        useOddSample=True,
                                        percent=FLAGS.testPercent,
                                        shuffle=True,
                                        )
    
    score = model.evaluate_generator(test_data_generator, 
                                     test_data_generator.num_steps(),
                                     max_queue_size=FLAGS.queueSize,
                                    )

    print('Test loss:    ', score[0])
    print('Test accuracy:', score[1])

def predict(model):
    print("this is filePath %s, this is filePattern %s, this is predictionSetExtension %s, this is extension %s" % (FLAGS.filePath, FLAGS.filePattern, FLAGS.predictionSetExtension, FLAGS.extension))
    #input_files = fnmatch.filter(listdir(FLAGS.filePath), FLAGS.filePattern + FLAGS.predictionSetExtension + '.*.' + FLAGS.extension)
    input_files = fnmatch.filter(listdir(FLAGS.filePath), FLAGS.filePattern + '*.' + FLAGS.extension)
    if len(input_files) == 0:
        sys.exit("File not found: ") # + filename_pattern)

    for file in input_files:
        print('Number of predictions:        ', end='')
        predict_data_generator = DataGenerator(data_shape, num_class,
                                               FLAGS.filePath, file,
                                               FLAGS.testBatchSize,
                                               useOddSample=True,
                                               )
        prediction = model.predict_generator(predict_data_generator,
                                             predict_data_generator.num_steps(),
                                             max_queue_size=FLAGS.queueSize,
                                            )
        #np.save(path.join(FLAGS.filePath, file.replace(FLAGS.predictionSetExtension, FLAGS.predictionExtension).replace('.' + FLAGS.extension, '')), prediction)
        np.save(path.join(FLAGS.predictionFilePath, file.replace(FLAGS.extension, FLAGS.predictionExtension )), prediction) #.replace('.' + FLAGS.extension, '')), prediction)



def main():

    model_serial = define_model()

    # Save model
    model_yaml = model_serial.to_yaml()
    model_yaml_file = open(path.join(FLAGS.modelPath, FLAGS.sessionName + '.model.yaml'), "w")
    model_yaml_file.write(model_yaml)
    model_yaml_file.close()

    #keras.utils.plot_model(model_serial, to_file=path.join(FLAGS.modelPath, FLAGS.sessionName + '.model.png'), show_shapes= True)

    # load weights
    if FLAGS.startEpoch > 0:
        model_serial.load_weights(
                path.join(FLAGS.modelPath, FLAGS.sessionName + '.weights.{:02d}.hdf5'.format(FLAGS.startEpoch)))
        print('Loaded epoch', FLAGS.startEpoch)

    if (FLAGS.testOnly == True or FLAGS.predictOnly) and FLAGS.startEpoch == 0:
        model_serial.load_weights(
                path.join(FLAGS.modelPath, FLAGS.sessionName + '.weights.hdf5'))

    model = model_serial
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.nadam(lr=FLAGS.learningRate),
                  metrics=['accuracy'],
                  )

    if FLAGS.randomSeed > 0:
        random.seed(FLAGS.randomSeed)

    if FLAGS.testOnly == True:
        test(model)
        sys.exit()

    if FLAGS.predictOnly == True:
        predict(model)
        sys.exit()

    trainingParameterText = 'BatchSize : {:02d}; learningRate : {:.8f}; L2 : {:.8f}; Dropout : {:.4f}'.format(FLAGS.batchSize, FLAGS.learningRate, FLAGS.L2, FLAGS.dropout)
    print(trainingParameterText)
    print('_________________________________________________________________')
 

    print('Number of training samples:   ', end='')
    training_data_generator = DataGenerator(data_shape, num_class,
                                            FLAGS.filePath, FLAGS.filePattern + FLAGS.trainingSetExtension + '.*.' + FLAGS.extension,
                                            FLAGS.batchSize,
                                            percent=FLAGS.trainingPercent,
                                            queue_size=FLAGS.queueSize,
                                            shuffle=True,
                                            #class_weight=class_weight,
                                            )
    print('Number of validation samples: ', end='')
    validation_data_generator = DataGenerator(data_shape, num_class,
                                              FLAGS.filePath, FLAGS.filePattern + FLAGS.validationSetExtension + '.*.' + FLAGS.extension,
                                              FLAGS.testBatchSize,
                                              percent=FLAGS.validationPercent,
                                              shuffle=True,
                                              #class_weight=class_weight,
                                              )
    print('Number of test samples:       ', end='')
    test_data_generator = DataGenerator(data_shape, num_class,
                                        FLAGS.filePath, FLAGS.filePattern + FLAGS.testSetExtension + '.*.' + FLAGS.extension,
                                        FLAGS.testBatchSize,
                                        percent=FLAGS.testPercent,
                                        shuffle=True,
                                        )

    print('_________________________________________________________________')

    callback_list = []

    callback_list.append(RecordEpoch(best_epoch_dict,
                                     weigh_filepath=path.join(FLAGS.modelPath,
                                     FLAGS.sessionName + '.weights.{epoch:02d}.hdf5'),
                                     csv_log_filepath=path.join(FLAGS.modelPath,
                                                                FLAGS.sessionName + '.' + FLAGS.logExtension, ),
                                     csv_log_header=trainingParameterText,
                                     patience=FLAGS.earlyStoppingPatience,
                                     sleep_after_epoch=sleep_after_epoch,
                                    ))

    history = model.fit_generator(training_data_generator, training_data_generator.num_steps(),
                                  epochs=FLAGS.maxEpoch,
                                  initial_epoch=FLAGS.startEpoch,
                                  verbose=1,
                                  callbacks=callback_list,
                                  #class_weight=class_weight,
                                  validation_data=validation_data_generator,
                                  validation_steps=validation_data_generator.num_steps(),
                                  max_queue_size=FLAGS.queueSize,
                                  )

    training_data_generator.terminate()

    print('Best epoch:', best_epoch_dict['epoch_index'])
    print('Validation accuracy:', best_epoch_dict['val_acc'])

    model.load_weights(
        path.join(FLAGS.modelPath, FLAGS.sessionName + '.weights.{:02d}.hdf5'.format(best_epoch_dict['epoch_index'])))
    print('Loaded weights of this epoch')

    test(model)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test 7')

    parser.add_argument('-t', dest='testOnly', default=False, action='store_true')
    parser.add_argument('-p', dest='predictOnly', default=True, action='store_true')

    parser.add_argument('--tensorflowLogPath', help='Path for Tensorflow log', default='')
    parser.add_argument('--logExtension', help='File extension of training log', default='log')
    parser.add_argument('--sessionName', help='Name of this training session', default='test_7')
    parser.add_argument('--modelPath', help='Model path', default='/home/user1/Simon/Lam/ml-last/result/')
    parser.add_argument('--filePath', help='File path', default='/home/user1/Simon/Lam/ml-last')


    parser.add_argument('--extension', help='File extension', default='npz')
    parser.add_argument('--filePattern', help='File pattern', default='*')
    parser.add_argument('--trainingSetExtension', help='File extension for training set', default='training')
    parser.add_argument('--testSetExtension', help='File extension for test set', default='test')
    parser.add_argument('--validationSetExtension', help='File extension for validation set', default='validation')
    parser.add_argument('--predictionSetExtension', help='File extension for prediction set', default='predict')

    parser.add_argument('--predictionExtension', help='File extension for prediction result', default='predict')
    parser.add_argument('--predictionFilePath', help='File path for storing prediction files', default='/home/user1/Simon/Lam/ml-last/predict/')

    parser.add_argument('--randomSeed', help='Random seed', type=int, default=0)

    parser.add_argument('--trainingPercent', help='Percentage of training samples to use in each epoch', type=int, default=100)
    parser.add_argument('--validationPercent', help='Percentage of validation samples to use in each epoch', type=int, default=100)
    parser.add_argument('--testPercent', help='Percentage of test samples to use in each epoch', type=int, default=100)

    parser.add_argument('--queueSize', help='Size of data generation queue', type=int, default=1024)

    parser.add_argument('--testBatchSize', help='Batch size for evaluation and prediction', type=int, default=256)
    parser.add_argument('--maxEpoch', help='Maximum no. of epoch', type=int, default=1000)
    parser.add_argument('--earlyStoppingPatience', help='Early stopping patience', type=int, default=50)

    parser.add_argument('--startEpoch', help='Starting epoch', type=int, default=0)

    parser.add_argument('--batchSize', help='Batch size for training', type=int, default=256)
    parser.add_argument('--learningRate', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--L2', help='L2 regualarizer', type=float, default=1e-6)
    parser.add_argument('--dropout', help='Dropout rate', type=float, default=0)
    

    FLAGS, UNPARSED = parser.parse_known_args()

    main()
