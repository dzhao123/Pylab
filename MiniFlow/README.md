# Miniflow

A computing graph written in python from scratch.

## Getting Started
‘’’Python

import miniflow as mf
import numpy as np

trainfile_X = 'train-images-idx3-ubyte'
trainfile_y = 'train-labels-idx1-ubyte'
testfile_X = 't10k-images-idx3-ubyte'
testfile_y = 't10k-labels-idx1-ubyte'
train_X = mf.DataUtils(filename=trainfile_X).getImage()
train_y = mf.DataUtils(filename=trainfile_y).getLabel()
test_X = mf.DataUtils(testfile_X).getImage()
test_y = mf.DataUtils(testfile_y).getLabel()

with mf.Graph().as_default():

    input_shape = 784
    hidden_layer_1 = 120
    hidden_layer_2 = 84
    hidden_layer_3 = 32
    n_classes = 10

    x = mf.placeholder()
    y_ = mf.placeholder()
    #w1 = mf.Variable(mf.truncated_normal([3,3,1,16]), name = 'w1')
    #conv1 = mf.conv2d(x, w1, [1,2,2,1], padding='same')
    #b1 = mf.Variable(mf.truncated_normal([1,1,1,16]), name = 'b1')
    #y1 = mf.relu(mf.add(conv1,b1))
    w2 = mf.Variable(mf.truncated_normal([5,5,1,6]), name='w2')
    conv2 = mf.conv2d(x, w2, [1,2,2,1], padding='same')
    b2 = mf.Variable(mf.truncated_normal([1,1,1,6]), name = 'b2')
    y2 = mf.relu(mf.add(conv2,b2))
    w3 = mf.Variable(mf.truncated_normal([5,5,6,16]), name='w3')
    conv3 = mf.conv2d(y2, w3, [1,2,2,1], padding='same')
    b3 = mf.Variable(mf.truncated_normal([1,1,1,16]), name = 'b3')
    y3 = mf.relu(mf.add(conv3,b3))
    y3 = mf.merge(y3)
    w4 = mf.Variable(mf.random_normal([input_shape, hidden_layer_1], mu=0.0, sigma=0.1), name = 'w4')
    b4 = mf.Variable(mf.random_normal([hidden_layer_1], mu=0.0, sigma=0.1), name = 'b4')
    y4 = mf.relu(mf.matmul(y3,w4)+b4)
    w5 = mf.Variable(mf.random_normal([hidden_layer_1,hidden_layer_2], mu=0.0, sigma=0.1), name = 'w5')
    b5 = mf.Variable(mf.random_normal([hidden_layer_2], mu=0.0, sigma=0.1), name = 'b5')
    y5 = mf.relu(mf.matmul(y4,w5)+b5)
    w6 = mf.Variable(mf.random_normal([hidden_layer_2,n_classes], mu=0.0, sigma=0.1), name = 'w5')
    b6 = mf.Variable(mf.random_normal([n_classes], mu=0.0, sigma=0.1), name = 'b6')
    y6 = mf.sigmoid(mf.matmul(y5,w6)+b6)
    loss = mf.reduce_sum(mf.square(y_-y6))
    #train_op = mf.GradientDescentOptimizer(learning_rate=0.0005).minimize(loss)
    train_op = mf.ExponentialDecay(learning_rate=0.01, decay_rate=0.01).minimize(loss)
    #train_op = mf.MomentumGDOptimizer(learning_rate=0.005, decay_rate=1e-6, beta=0.9).minimize(loss)
    #train_op = mf.AdamOptimizer(learning_rate=0.001).minimize(loss)
    train_y = mf.onehot_encoding(train_y, 10)#normalization(train_y,10)
    test_y = mf.onehot_encoding(test_y, 10)
    accurate = mf.equal(mf.argmax(y6,1), mf.argmax(y_,1))

with mf.Session() as sess:
    epoches = 1
    batch_size = 3000
    batches = int(len(train_X)/batch_size)

    for step in range(epoches):
        for batch in range(batches):
            loss_value = 0
            accuracy = 0
            mse = 0
            start = batch*batch_size
            end = (batch+1)*batch_size
            X = np.array(train_X[start:end])
            Y = np.array(train_y[start:end])
            feed_dict = {x:np.array(X), y_:np.array(Y)}
            loss_value, _ = sess.run([loss, train_op], feed_dict)
            accuracy = sess.run(accurate, feed_dict)
            mse = loss_value/(end-start)
            print('step:{}, batch:{}, loss:{}, mse:{}, accuracy:{}'.format(step, batch, loss_value, mse, accuracy/(end-start)))
    test_acc = 0
    #for batch in range(len(test_X)):
    test_acc = sess.run(mf.equal(mf.argmax(y6,1),mf.argmax(y_,1)), feed_dict={x:np.array(test_X),y_:np.array(test_y)})
    print('test accuracy:{}'.format(test_acc/len(test_X)))

## Features

- Computational Graph
- Feed Forward and Backward Propagation
- Gradient Descent Optimizer
- Convolutional Layer
- Fully Connected Layer
- MLP and CNN Example on MNIST Dataset
