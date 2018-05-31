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
train_X = np.array(train_X)
train_y = np.array(train_y)
test_X = np.array(test_X)
test_y = np.array(test_y)


with mf.Graph().as_default():

    x = mf.placeholder()
    y_ = mf.placeholder()

    input_shape = train_X.shape[1]
    hidden_layer_1 = 256
    hidden_layer_2 = 128
    n_classes = 10

    w1 = mf.Variable(mf.random_normal([input_shape,hidden_layer_1], mu=0.0, sigma=0.1), name = 'w1')
    b1 = mf.Variable(mf.random_normal([hidden_layer_1], mu=0.0, sigma=0.1), name = 'b1')
    y1 = mf.relu(mf.matmul(x,w1)+b1)
    #y1 = batch_average(y1)
    w2 = mf.Variable(mf.random_normal([hidden_layer_1, hidden_layer_2], mu=0.0, sigma=0.1), name = 'w2')
    b2 = mf.Variable(mf.random_normal([hidden_layer_2], mu=0.0, sigma=0.1), name = 'b2')
    y2 = mf.relu(mf.matmul(y1,w2)+b2)
    #y2 = batch_average(y2)
    w3 = mf.Variable(mf.random_normal([hidden_layer_2, n_classes], mu=0.0, sigma=0.1), name = 'w3')
    b3 = mf.Variable(mf.random_normal([n_classes], mu=0.0, sigma=0.1), name = 'b3')
    y3 = mf.relu(mf.matmul(y2,w3)+b3)
    #y3 = batch_average(y3)
    loss = mf.reduce_sum(mf.square(y_-y3))
    train_op = mf.GradientDescentOptimizer(learning_rate=0.0045).minimize(loss)
    #train_op = mf.ExponentialDecay(learning_rate=0.01, decay_rate=0.01).minimize(loss)
    train_y = mf.onehot_encoding(train_y, 10)#normalization(train_y,10)
    test_y = mf.onehot_encoding(test_y, 10)#normalization(test_y,10)
    #feed_dict = {x:train_X, y_:normalization(train_y,10)}
    #eval = equal(argmax(y3,0),argmax(y_,0))
    accurate = mf.equal(mf.argmax(y3,1), mf.argmax(y_,1))

with mf.Session() as sess:
    epoches = 3
    batch_size = 3000
    batches = int(len(train_X)/batch_size)
    #remains = len(train_X) - batches*batch_size

    for step in range(epoches):
        for batch in range(batches):
            loss_value = 0
            accuracy = 0
            mse = 0
            start = batch*batch_size
            end = (batch+1)*batch_size
            for index in range(start,end):
                X = [train_X[index]]#[start:end]
                Y = [train_y[index]]#[start:end]
                feed_dict = {x:X, y_:Y}
                loss_value += sess.run(loss, feed_dict)
                mse += loss_value/len(X)
                sess.run(train_op, feed_dict)
                accuracy += sess.run(accurate, feed_dict)
            print('step:{}, batch:{}, loss:{}, mse:{}, accuracy:{}'.format(step, batch, loss_value, mse, accuracy/(end-start)))
    test_acc = 0
    for batch in range(len(test_X)):
        test_acc += sess.run(mf.equal(mf.argmax(y3,1),mf.argmax(y_,1)), feed_dict={x:[test_X[batch]],y_:[test_y[batch]]})
    print('test accuracy:{}'.format(test_acc/len(test_X)))
