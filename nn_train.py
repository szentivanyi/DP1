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
from time import sleep as wait
import numpy
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from create_FVs_train import create_fvs_train  # Import train FV function
from create_LVs_train import create_lvs_train  # Import train LV function
from create_FVs_test import create_fvs_test  # Import test LV function
from create_LVs_test import create_lvs_test  # Import test LV function
from create_dictionary import create_dictionary
from nltk.corpus import reuters



# Number of documents to process as Feature vectors/Label vectors (from test set and from train set)
count_of_docs = 100

# Number of documents to create dictionary
dict_docs = 500

# Training Parameters
learning_rate = 0.001
training_epochs = count_of_docs
batch_size = 5
display_step = 5



# rozdelim si db dokumentov do trenovacej a testovacej mnoziny (list-u)
train_docs = []
test_docs = []
for doc_id in reuters.fileids():
    if doc_id.startswith("train"):
        train_docs.append(doc_id)
    else:
        test_docs.append(doc_id)

# vytvorim slovnik
dictionary = create_dictionary(train_docs[:dict_docs] + test_docs[:dict_docs], keep_percent=80)


print("-------------------------------------------------------")
print("Training: ")
fvs_train = create_fvs_train(train_docs[:count_of_docs], dictionary)
print("Pocet FVs: " + str(len(fvs_train)))
# print(fvs_train[0])
print("Dlzka 1 FV: " + str(len(fvs_train[0])))
print()
lvs_train = create_lvs_train(train_docs[:count_of_docs])
print("Pocet LVs: " + str(len(lvs_train)))
# print(lvs_train[0])
print("Dlzka 1 LV: " + str(len(lvs_train[0])))
assert len(fvs_train) == len(lvs_train), "PROBLEM s train vektormi, lisi sa dlzka fvs a lvs."

print("-------------------------------------------------------")
print("Testing: ")
fvs_test = create_fvs_test(test_docs[:count_of_docs], dictionary)
print("Pocet FVs: " + str(len(fvs_test)))
# print(fvs_test[0])
print("Dlzka 1 FV: " + str(len(fvs_test[0])))
print()
lvs_test = create_lvs_test(test_docs[:count_of_docs])
print("Pocet LVs: " + str(len(lvs_test)))
# print(lvs_test[0])
print("Dlzka 1 LV: " + str(len(lvs_test[0])))
assert len(fvs_test) == len(lvs_test), "PROBLEM s test vektormi, lisi sa dlzka fvs a lvs."


print("\n----------- NN priprava pouzitia -------------------------")


# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = len(fvs_train[0])  # počet vstpnych uzlov, dlzka 1 FV, data input
n_classes = len(lvs_train[0])   # počet vystpunych uzlov, resp. topics


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


# pripravime si trenovacie data
batches_x = numpy.asarray(fvs_train)
print("Priklad FV: ")
print(batches_x[0])
print(batches_x.size)

batches_y = numpy.asarray(lvs_train)
print("Priklad LV: ")
print(batches_y[0])
print(batches_y.size)


print()
print("------------------- --------- NN trenovanie ---------- --------------------")
print()


with tf.Session() as sess:
    sess.run(init)

    # Training  cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(fvs_train)/batch_size)  # pocet batchov (skupiny vektorov na vstupe, resp. vystupe)
        # print("Total_batches: " + str(total_batch))

        # Loop over all batches
        for i in range(total_batch):
            batch_x = batches_x[i*batch_size:(i*batch_size)+batch_size]
            # print("Batch X: ")
            # print(batch_x.size)
            # print(batch_x)

            batch_y = batches_y[i*batch_size:(i*batch_size)+batch_size]
            # print("Batch Y: ")
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


    # pripravime testovacie data
    m = numpy.asarray(fvs_test)
    n = numpy.asarray(lvs_test)


    print("Accuracy:", accuracy.eval({X: m, Y: n}))