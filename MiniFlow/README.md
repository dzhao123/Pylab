# Miniflow

A computing graph written in python from scratch.

## Getting Started
``` python

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
    train_op = mf.ExponentialDecay(learning_rate=0.01, decay_rate=0.01).minimize(loss)
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
```
## Result

```
step:0, batch:0, loss:1288.206707465247, mse:0.429402235821749, accuracy:0.908
step:0, batch:1, loss:526.0707037697366, mse:0.17535690125657888, accuracy:0.905
step:0, batch:2, loss:477.68339027107345, mse:0.15922779675702448, accuracy:0.9273333333333333
step:0, batch:3, loss:357.49202307960144, mse:0.11916400769320049, accuracy:0.95
step:0, batch:4, loss:371.9437422040999, mse:0.12398124740136664, accuracy:0.9433333333333334
step:0, batch:5, loss:319.71496349655695, mse:0.10657165449885232, accuracy:0.949
step:0, batch:6, loss:264.84454334452556, mse:0.08828151444817518, accuracy:0.9513333333333334
step:0, batch:7, loss:269.494821454531, mse:0.08983160715151034, accuracy:0.957
step:0, batch:8, loss:263.1669367817608, mse:0.08772231226058694, accuracy:0.961
step:0, batch:9, loss:266.84406589270037, mse:0.08894802196423346, accuracy:0.9516666666666667
step:0, batch:10, loss:254.7459565524755, mse:0.08491531885082516, accuracy:0.9693333333333334
step:0, batch:11, loss:200.9247728171663, mse:0.06697492427238877, accuracy:0.9686666666666667
step:0, batch:12, loss:193.63583051707397, mse:0.06454527683902465, accuracy:0.9566666666666667
step:0, batch:13, loss:206.0093810458327, mse:0.06866979368194423, accuracy:0.9756666666666667
step:0, batch:14, loss:205.71530124131363, mse:0.06857176708043787, accuracy:0.9763333333333334
step:0, batch:15, loss:217.08896871355122, mse:0.07236298957118374, accuracy:0.9606666666666667
step:0, batch:16, loss:193.0210534792457, mse:0.06434035115974857, accuracy:0.9786666666666667
step:0, batch:17, loss:184.72346288978068, mse:0.06157448762992689, accuracy:0.978
step:0, batch:18, loss:141.90972967174852, mse:0.04730324322391617, accuracy:0.9816666666666667
step:0, batch:19, loss:94.39119663702601, mse:0.031463732212342006, accuracy:0.9866666666666667
test accuracy:0.971

```


## Features

- Computational Graph
- Feed Forward and Backward Propagation
- Gradient Descent Optimizer
- Convolutional Layer
- Fully Connected Layer
- MLP and CNN Example on MNIST Dataset
