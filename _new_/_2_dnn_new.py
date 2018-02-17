from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
from time import sleep as wait
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

########################
###  GLOBAL SETTINGS ###
########################

TRAINING_DATA = "train_data_500.csv"
TEST_DATA = "test_data_500.csv"
number_epoch = 3
mysteps = 10
networks = [[5, 18, 5], [10, 22, 10], [15, 25, 15]]
networks2 = [
            [700, 300], [2100, 900], [3500, 1500], [5600, 2400], [8400, 3600], [12600, 5400],  # 2
            [500, 300, 200, 100], [1500, 800, 400, 300], [2500, 1400, 800, 300], [4000, 2400, 1200, 400], [6000, 3500, 1800, 700], [9000, 5000, 3000, 1000],  # 4
            [240, 210, 170, 150, 130, 100], [720, 630, 510, 450, 390, 300], [1200, 1050, 850, 750, 650, 500], [3000, 2600, 2200, 1800, 1600, 800], [4500, 3900, 3300, 2700, 2400, 1200],  # 6
            [185, 140, 125, 120, 115, 110, 105, 100], [500, 470, 440, 430, 390, 330, 240, 200], [925, 700, 625, 600, 575, 550, 525, 500], [1480, 1120, 1000, 960, 920, 880, 840, 800], [2220, 1680, 1500, 1440, 1380, 1320, 1260, 1200], [3330, 2520, 2250, 2160, 2070, 1980, 1890, 1800]  # 8
            ]


def main():
    # Datamoj = collections.namedtuple('Dataset', ['data', 'target'])
    # d = Datamoj(data=[1,2,3,4,5,6], target=[1,1,1,2,0])
    # print(d.data)
    # print(d.target)

    #  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py
    # Load datasets
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TRAINING_DATA,
                                                                       target_dtype=np.int,  # label
                                                                       features_dtype=np.float32,  # features
                                                                       target_column=-1)  # label je -1 prvok
    # print(training_set)
    print("Počet extrah. vlastnosti: " + str(len(training_set[0][0])))

    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TEST_DATA,
                                                                   target_dtype=np.int,
                                                                   features_dtype=np.float32,
                                                                   target_column=-1)  # label je -1 prvok

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x",
                                                        shape=[len(training_set[0][0])],
                                                        # pocet vlastnosti dokumentu (dlzka FV)
                                                        default_value=None,
                                                        normalizer_fn=None)]


# TRENOVANIE VSETKYCH NN zadefinovanych v hlavicke

    for network in networks:
        # shutil.rmtree('C:\\Users\\Ghost\\PycharmProjects\\DP1_\\_new_\\tmp\\rt_model'+ str(networks.index(network)), ignore_errors=True)

        # Build xzy layer DNN  with x, y, z units respectively.
        classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=network,
                                                n_classes=90,
                                                # na vystupe mame 90 uzlov do kazdeho pride pravdepodobnost danej triedy/categorie, vytazi uzol s najvyssou pravdepodobnostou
                                                model_dir="C:/Users/Ghost/PycharmProjects/DP1_/_new_/tmp/rt_model" + str(networks.index(network)),
                                                optimizer='Adagrad',  # adaptivny gradient
                                                activation_fn=nn.relu,  # sigmoidalna funkcia ak n_classes=2 / tangens hyperbolicky / n_classes>2 softmax_cross_entropy_loss
                                                dropout=None,
                                                config=None) #tf.contrib.learn.RunConfig(save_checkpoints_secs=5)

        # Define the training inputs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(training_set.data)},
                                                            y=np.array(training_set.target),
                                                            batch_size=128,  # 128
                                                            num_epochs=None,  # number of epochs to iterate over data, None = forever
                                                            shuffle=True,
                                                            queue_capacity=1000,
                                                            num_threads=1)

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": np.array(test_set.data)},
                                                           y=np.array(test_set.target),
                                                           batch_size=128,  # 128
                                                           num_epochs=None,  # number of epochs to iterate over data, None = forever
                                                           shuffle=False,
                                                           queue_capacity=1000,
                                                           num_threads=1)

        # Train model
        accuracy_score_test = []
        accuracy_score_train = []
        for e in range(number_epoch):

            classifier.train(input_fn=train_input_fn,
                             hooks=None,
                             steps=mysteps,
                             max_steps=None)

            # Evaluate accuracy on testing data
            accuracy_score_test.append(classifier.evaluate(input_fn=test_input_fn, steps=1)["accuracy"])
            # Evaluate accuracy on training data
            accuracy_score_train.append(classifier.evaluate(input_fn=train_input_fn, steps=1)["accuracy"])

        max_test = max(accuracy_score_test)
        max_train = max(accuracy_score_train)

        print("{0:f}".format(max_test*100) + " ({0:f}), ".format(max_train*100) + f"step: {accuracy_score_test.index(max_test)} " + f"({accuracy_score_train.index(max_train)}), " + str(network))


if __name__ == "__main__":
    main()
