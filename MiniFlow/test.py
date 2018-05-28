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
train_X = np.array(train_X)[:600]
train_y = np.array(train_y)[:600]
test_X = np.array(test_X)
test_y = np.array(test_y)

with mf.Graph().as_default():

    input_shape = 2592
    hidden_layer_1 = 1024
    hidden_layer_2 = 256
    n_classes = 10

    x = mf.placeholder()
    y_ = mf.placeholder()
    w1 = mf.Variable(mf.truncated_normal(3,3,1,32), name = 'w1')
    conv1 = mf.conv2d(x, w1, [1,3,3,1], padding='valid')
    b1 = mf.Variable(mf.truncated_normal(1,1,1,32), name = 'b1')
    y1 = mf.sigmoid(mf.add(conv1,b1))
    #w2 = mf.Variable(mf.truncated_normal(2,2,32,64))
    #conv2 = mf.conv2d(y1, w2, [1,2,2,1], padding='valid')
    #b2 = mf.Variable(mf.truncated_normal(1,1,1,64), name = 'b2')
    #y2 = mf.add(conv2,b2)
    y2 = mf.merge(y1)
    w3 = mf.Variable(mf.random_normal([input_shape,hidden_layer_1], mu=0.0, sigma=1.0), name = 'w3')
    b4 = mf.Variable(mf.random_normal([hidden_layer_1], mu=0.0, sigma=1.0), name = 'b4')
    y4 = mf.sigmoid(mf.matmul(y2,w3)+b4)
    w5 = mf.Variable(mf.random_normal([hidden_layer_1,n_classes], mu=0.0, sigma=1.0), name = 'w5')
    b6 = mf.Variable(mf.random_normal([n_classes], mu=0.0, sigma=1.0), name = 'b6')
    y6 = mf.sigmoid(mf.matmul(y4,w5)+b6)
    loss = mf.reduce_sum(mf.square(y_-y6))
    #train_op = mf.GradientDescentOptimizer(learning_rate=0.007).minimize(loss)
    train_op = mf.ExponentialDecay(learning_rate=0.01, decay_rate=0.01).minimize(loss)
    train_y = mf.onehot_encoding(train_y, 10)#normalization(train_y,10)
    test_y = mf.onehot_encoding(test_y, 10)
    accurate = mf.equal(mf.argmax(y6,1), mf.argmax(y_,1))

with mf.Session() as sess:
    epoches = 3
    batch_size = 30
    batches = int(len(train_X)/batch_size)

    for step in range(epoches):
        for batch in range(batches):
            loss_value = 0
            accuracy = 0
            mse = 0
            start = batch*batch_size
            end = (batch+1)*batch_size
            for index in range(start,end):
                X = np.array([train_X[index]])#[start:end]
                Y = np.array([train_y[index]])#[start:end]
                #print(X.shape)
                #print(Y.shape)
                feed_dict = {x:X, y_:Y}
                loss_value += sess.run(loss, feed_dict)
                mse += loss_value/len(X)
                sess.run(train_op, feed_dict)
                accuracy += sess.run(accurate, feed_dict)
            print('step:{}, batch:{}, loss:{}, mse:{}, accuracy:{}'.format(step, batch, loss_value, mse, accuracy/(end-start)))
    test_acc = 0
    for batch in range(len(test_X)):
        test_acc += sess.run(mf.equal(mf.argmax(y6,1),mf.argmax(y_,1)), feed_dict={x:[test_X[batch]],y_:[test_y[batch]]})
    print('test accuracy:{}'.format(test_acc/len(test_X)))
