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
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from create_FVs import create_fvs  # Import FV function
from create_LVs import create_lvs  # Import LV function


# Parameters for DOCUMENTS
docs = 50


print("------------------- --------- FV ---------- --------------------")
fvs = create_fvs(docs)
print("Pocet vektorov FV: " + str(len(fvs)))
print(fvs[0])
print("Dlzka 1 FV: " + str(len(fvs[0])))

print("------------------- --------- LV  ---------- --------------------")
lvs = create_lvs(docs)
print("Pocet vektorov LV: " + str(len(lvs)))
print(lvs[0 ])
print("Dlzka 1 LV: " + str(len(lvs[0])))

print("------------------- --------- NN priprava pouzitia  ---------- --------------------")
print()

# Training Parameters
learning_rate = 0.001
training_epochs = docs
batch_size = 1
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = len(fvs[0])  # počet vstpnych uzlov, dlzka 1 FV, data input
n_classes = len(lvs[0])   # počet vystpunych uzlov, resp. topics

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

# pripravime si X
batches_x = []
for i in fvs:  # vyberame po 1 FV
    i = numpy.asarray(i)  # transformujeme z listu na pole
    x = i.reshape(1, n_input)  # prevratime maticu 1 x 1024
    batches_x.extend(x)

# print(batches_x[0])
# print(batches_x[0].size)
# print(type(batches_x[0]))
# time.sleep(5)

# pripravime si Y
batches_y = []
for j in lvs:  # vyberame po 1 LV
    j = numpy.asarray(j)  # transformujeme z listu na pole
    y = j.reshape(1, n_classes)  # prevratime maticu 1 x n_classes
    batches_y.extend(y)

# print(batches_y[0])
# print(batches_y[0].size)
# time.sleep(5)

print()
print("------------------- --------- NN trenovanie ---------- --------------------")
print()


with tf.Session() as sess:
    sess.run(init)

    # Training  cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(fvs)/batch_size)  # pocet vsetkych batchov (skupin dat na vstup a vystup)
        print("Total_batch: ")
        print(total_batch)

        # Loop over all batches
        for i in range(total_batch) :
            batch_x = batches_x[i].reshape(1, n_input)  # prevratime maticu na 1 x n_input
            # print(batch_x.size)
            # print(batch_x)
            # print(type(batch_x))

            # batch_x = batches_x[i].reshape(1, n_input)
            # print(batch_x.size)
            # print(batch_x[0])
            # print(type(batch_x))
            # time.sleep(50)
            batch_y = batches_y[i].reshape(1, n_classes)
            # print("Dlzka 1. batchu X a Y: ")
            # print(len(batch_x[0]), len(batch_y[0]))

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