""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import os
import random
import time
import numpy
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from create_FVs import *  # Import FV data

print()
print("------------------- --------- NN experiment pouzitia  ---------- --------------------")
print("Pocet vektorov FV: " + str(len(fvs)))
print()

# Parameters
learning_rate = 0.001
training_epochs = docs
batch_size = 1
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = len(fv_nn)  # dlzka FV, data input
n_classes = 135  # počet vystpunych uzlov, zanrov, resp. labelov/kategorii

# tf Graph input
X = tf.placeholder("float", [1, n_input])
Y = tf.placeholder("float", [1, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training  cycle
    for epoch in range(training_epochs):  # 20 krat
        avg_cost = 0.
        total_batch = int(len(fvs)/batch_size)  # pocet vsetkych FV vektorov deleno pocet FV vektorov na 1 batch
        # print("Total_batch: ")
        # print(total_batch)

        # Loop over all batches
        for i in range(total_batch):
            batch_x = fvs[i]   # 1 FV ako list
            batch_x = numpy.asarray(batch_x)  # z listu na pole
            batch_x = batch_x.reshape(1, n_input)  # prevratime maticu 1 x 1024
            #batch_x = fvs[i:i+2]

            # vygenerujem si batch y, cize Label vektory dlzky 135, kym nebudeme mat orginal z dokumentov 135 labelov
            batch_y = []
            for i in range(135):
                batch_y.append(int(random.choice(['0', '1'])))
            batch_y = numpy.asarray(batch_y)  # z listu na pole
            batch_y = batch_y.reshape(1, 135)  # prevratime maticu 1 x 135

            #print("Batch x a y: ")
            #print(len(batch_x[0]), len(batch_y[0]))

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})

            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))


    print("Optimization Finished!")


    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))

    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #toto je pre databazu obrazkov nie pre nas dataset
    # print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))