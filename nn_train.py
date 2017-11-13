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
import time
import numpy
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from create_FVs_train import create_fvs_train  # Import train FV function
from create_LVs_train import create_lvs_train  # Import train LV function

from create_FVs_test import create_fvs_test  # Import test LV function
from create_LVs_test import create_lvs_test  # Import test LV function


# Parameters for DOCUMENTS
train_docs = 5
test_docs = 5

# Training Parameters
learning_rate = 0.001
training_epochs = train_docs
batch_size = 20
display_step = 3

print("---------------------------Feature vector train------------------------------")
fvs = create_fvs_train(train_docs)
print("Pocet Feature Vectors: " + str(len(fvs)))
# print(fvs[0])
print("Dlzka 1 FV: " + str(len(fvs[0])))

print("--------------------------Label vector train-------------------------------------")
lvs = create_lvs_train(train_docs)
print("Pocet Label vectors: " + str(len(lvs)))
# print(lvs[0])
print("Dlzka 1 LV: " + str(len(lvs[0])))

print()

print("---------------------------Feature vector TEST------------------------------")
fvs_test = create_fvs_test(test_docs)
print("Pocet Feature Vectors: " + str(len(fvs_test)))
# print(fvs_test[0])
print("Dlzka 1 FV: " + str(len(fvs_test[0])))

print("---------------------------Label vector TEST-------------------------------------")
lvs_test = create_lvs_test(test_docs)
print("Pocet Label vectors: " + str(len(lvs_test)))
# print(lvs_test[0])
print("Dlzka 1 LV: " + str(len(lvs_test[0])))

print()
print("------------------- --------- NN priprava pouzitia  ---------- --------------------")

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = len(fvs[0])  # počet vstpnych uzlov, dlzka 1 FV, data input
n_classes = len(lvs[0])   # počet vystpunych uzlov, resp. topics


# tf Graph input
X = tf.placeholder("float", [None, n_input])  # ak pouzijem float: TypeError: Input 'b' of 'MatMul' Op has type float32 that does not match type int32 of argument 'a'.
Y = tf.placeholder("float", [None, n_classes])


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


# pripravime si maticu X, vsetky FV su spolu
batches_x = (numpy.asarray(fvs[0])).reshape(1, n_input)
for i in fvs[1:]:  # vyberame po jednom FV, ale 0. sme uz pouzili
    i = numpy.asarray(i)  # transformujeme z listu na maticu
    xx = i.reshape(1, n_input)  # prevratime maticu 1 x n_input
    batches_x = numpy.append(batches_x, xx, axis=0)  # lepime matice
print("Priklad FV: ")
print(batches_x[0])
print(batches_x[0].size)


# pripravime si maticu Y, vsetky LV su spolu
batches_y = (numpy.asarray(lvs[0])).reshape(1, n_classes)
for j in lvs[1:]:
    j = numpy.asarray(j)  # transformujeme z listu na maticu
    yy = j.reshape(1, n_classes)  # prevratime maticu 1 x n_classes
    batches_y = numpy.append(batches_y, yy, axis=0)
print("Priklad LV: ")
print(batches_y[0])
print(batches_y[0].size)


print()
print("------------------- --------- NN trenovanie ---------- --------------------")
print()


with tf.Session() as sess:
    sess.run(init)

    # Training  cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(fvs)/batch_size)  # pocet vsetkych batchov (skupin vektorov na vstupe resp. vystupe)
        # print("Total_batches: " + str(total_batch))

        # Loop over all batches
        for i in range(total_batch):
            batch_x = batches_x[i*batch_size:(i*batch_size)+batch_size]
            # print(batch_x.size)
            # print(batch_x)

            batch_y = batches_y[i*batch_size:(i*batch_size)+batch_size]
            # print(batch_y.size)
            # print(batch_y)


            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})

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

    # for i in fvs_test:
    i = numpy.asarray(fvs_test)
    print(i)
    print(i.size)
    print()
    print("*//////")
    print()
    i[:, :len(fvs[0])] = 9
    # i.resize(test_docs, len(fvs[0]), refcheck=False)
    print(i.size)
    print(i)

        # for j in lvs[1:]:
        #     j = numpy.asarray(j)  # transformujeme z listu na maticu
        #     yy = j.reshape(1, n_classes)  # prevratime maticu 1 x n_classes

    print("Accuracy:", accuracy.eval({X: create_fvs_test(train_docs),
                                      Y: create_lvs_test(train_docs)}))