import numpy as np
from data_utils import DataUtils
from sklearn.utils import shuffle


def normalize(x):
    
    encoding = np.zeros((len(x),10))
    encoding[np.arange(len(x)),x] = 1
    return encoding

def sigmoid(x):
    
    return 1/(1 + np.exp(-x))

def prime_sigmoid(x):
    
    return sigmoid(x) * (1 - sigmoid(x))

def prime_sigmoid2(x):
    
    return (1-sigmoid(x))**2 - prime_sigmoid(x)

def layer(weights,x,bias):
    
    return np.add(np.dot(weights,x),bias)
    
def create_weights(mu,sigma,dimension):
    
    np.random.seed(100)
    weights = np.random.normal(mu,sigma,dimension[0]*dimension[1])
    return weights.reshape(dimension[0],dimension[1])
    
def softmax(x):
    
    return np.exp(x)/sum(np.exp(x))

def cross_entropy(p,y):
    return (np.log(p) * y + np.log(1-p) * (1-y))/np.log(np.e)

def forward(x,weights,bias):
    
    weights_1 = weights[0]
    weights_2 = weights[1]
    weights_3 = weights[2]
    bias_1 = bias[0]
    bias_2 = bias[1]
    bias_3 = bias[2]
    
    #forward
    layer_1_i = layer(weights_1,x,bias_1)
    layer_1_o = sigmoid(layer_1_i)
    layer_2_i = layer(weights_2,layer_1_o,bias_2)
    layer_2_o = sigmoid(layer_2_i)
    layer_3_i = layer(weights_3,layer_2_o,bias_3)
    layer_3_o = sigmoid(layer_3_i)
    layer_i = np.array([layer_1_i,layer_2_i,layer_3_i])
    layer_o = np.array([layer_1_o,layer_2_o,layer_3_o])
    weights = np.array([weights_1,weights_2,weights_3])
    bias = np.array([bias_1,bias_2,bias_3])
     
    return layer_i,layer_o#,weights,bias
    

def backward(x,y,layer_i,layer_o,order,weights,bias):
    
    weights_1 = weights[0]
    weights_2 = weights[1]
    weights_3 = weights[2]
    bias_1 = bias[0]
    bias_2 = bias[1]
    bias_3 = bias[2]
    
    layer_1_i = layer_i[0]
    layer_2_i = layer_i[1]
    layer_3_i = layer_i[2]
    layer_1_o = layer_o[0]
    layer_2_o = layer_o[1]
    layer_3_o = layer_o[2]
    
    #p_softmax = softmax(layer_3_o)
    #loss = np.mean(cross_entropy(p_softmax, y),axis=-1,keepdims=True)
    loss = np.subtract(y,layer_3_o)
    #loss = np.mean(np.subtract(y,layer_3_o)**2,axis=-1,keepdims=True)
    
    #backward
    if order == 1:
        layer_b_3 = np.multiply(loss,prime_sigmoid(layer_3_i))
        layer_b_3_w = np.dot(layer_b_3,layer_2_o.T)

        layer_b_2 = np.multiply(np.dot(weights_3.T,layer_b_3),prime_sigmoid(layer_2_i))
        layer_b_2_w = np.dot(layer_b_2,layer_1_o.T)

        layer_b_1 = np.multiply(np.dot(weights_2.T,layer_b_2),prime_sigmoid(layer_1_i))
        layer_b_1_w = np.dot(layer_b_1,x.T)

        layer_b_w = np.array([layer_b_3_w,layer_b_2_w,layer_b_1_w])
        layer_b = np.array([layer_b_3,layer_b_2,layer_b_1])
    
    elif order == 2:
        layer_b_3_1 = np.multiply(loss,prime_sigmoid(layer_3_i))
        layer_b_3_2 = np.multiply(loss,prime_sigmoid2(layer_3_i))
        layer_b_3 = layer_b_3_1 + layer_b_3_2
        layer_b_3_w_1 = np.dot(layer_b_3_1,layer_2_o.T)
        layer_b_3_w_2 = np.dot(layer_b_3_2, (layer_2_o**2).T)
        layer_b_3_w = [layer_b_3_w_1, layer_b_3_w_2]

        layer_b_2_1 = np.multiply(np.dot(weights_3.T,layer_b_3_1),prime_sigmoid(layer_2_i))
        layer_b_2_2 = np.multiply(np.dot((weights_3.T)**2,layer_b_3_2),prime_sigmoid(layer_2_i)**2) + np.multiply(np.dot(weights_3.T,layer_b_3_1),prime_sigmoid2(layer_2_i))
        layer_b_2 = layer_b_2_1 + layer_b_2_2
        layer_b_2_w_1 = np.dot(layer_b_2_1,layer_1_o.T)
        layer_b_2_w_2 = np.dot(layer_b_2_2,(layer_1_o**2).T)
        layer_b_2_w = [layer_b_2_w_1, layer_b_2_w_2]

        layer_b_1_1 = np.multiply(np.dot(weights_2.T,layer_b_2_1),prime_sigmoid(layer_1_i))
        layer_b_1_2 = np.multiply(np.dot((weights_2.T)**2,np.multiply(np.dot((weights_3.T)**2,layer_b_3_2),layer_b_2_1**2)),layer_b_1_1**2) + np.multiply(np.dot(weights_2.T,np.multiply(np.dot(weights_3.T,layer_b_3_1),layer_b_2_2)),layer_b_1_1**2) + np.multiply(np.dot(weights_2.T,np.multiply(np.dot(weights_3.T,layer_b_3_1),layer_b_2_1)),layer_b_1_1)                                    
        layer_b_1 = layer_b_1_1 + layer_b_1_2
        layer_b_1_w_1 = np.dot(layer_b_1_1,x.T)
        layer_b_1_w_2 = np.dot(layer_b_1_2,(x**2).T)
        layer_b_1_w = [layer_b_1_w_1, layer_b_1_w_2]
    
        layer_b_w = np.array([layer_b_3_w,layer_b_2_w,layer_b_1_w])
        layer_b = np.array([layer_b_3,layer_b_2,layer_b_1])
    
    return layer_b_w,layer_b
    


def run(epoches, batch_size, order, learnrate, learnrate2, decay, decay2):
    
    trainfile_X = 'train-images-idx3-ubyte'
    trainfile_y = 'train-labels-idx1-ubyte'
    testfile_X = 't10k-images-idx3-ubyte'
    testfile_y = 't10k-labels-idx1-ubyte'
            
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    train_y = normalize(train_y)
    test_y = normalize(test_y)

    train_X = np.expand_dims(train_X,axis=-1)
    test_X = np.expand_dims(test_X,axis=-1)
    train_y = np.expand_dims(train_y,axis=-1)
    test_y = np.expand_dims(test_y,axis=-1)


    epoches = epoches
    batch_size = batch_size
    batches = int(len(train_X)/batch_size)
    order = order
    
    mu = 0.0
    sigma = 0.1
    image_d = 784
    neuron_d_1 = 256
    neuron_d_2 = 128
    neuron_d_output = 10

    dimension_1 = [neuron_d_1,image_d]
    weights_1 = create_weights(mu,sigma,dimension_1)
    bias_1 = np.zeros((neuron_d_1,1))
    
    dimension_2 = [neuron_d_2,neuron_d_1]
    weights_2 = create_weights(mu,sigma,dimension_2)
    bias_2 = np.zeros((neuron_d_2,1))

    dimension_3 = [neuron_d_output,neuron_d_2]
    weights_3 = create_weights(mu,sigma,dimension_3)
    bias_3 = np.zeros((neuron_d_output,1))

    weights = [weights_1,weights_2,weights_3]
    bias = [bias_1,bias_2,bias_3]

    for epoch in range(epoches):
        learnrate -= learnrate*decay
        learnrate2 -= learnrate2*decay2
        
        #train_X, train_y = shuffle(train_X,train_y,random_state=0)

        for i in range(batches):
            loss = 0
            accuracy = 0
            start = i * batch_size
            end = (i+1) * batch_size
            x = np.concatenate(train_X[start:end],axis=-1)
            y = np.concatenate(train_y[start:end],axis=-1)

            layer_i,layer_o = forward(x,weights,bias)
            layer_b_w,layer_b = backward(x,y,layer_i,layer_o,order,weights,bias)

            if order == 1:
                weights_3 = np.add(weights_3,learnrate * layer_b_w[0])
                weights_2 = np.add(weights_2,learnrate * layer_b_w[1])
                weights_1 = np.add(weights_1,learnrate * layer_b_w[2])
                bias_3 = np.mean(np.add(bias_3,learnrate * layer_b[0]),axis=-1,keepdims=True)
                bias_2 = np.mean(np.add(bias_2,learnrate * layer_b[1]),axis=-1,keepdims=True)
                bias_1 = np.mean(np.add(bias_1,learnrate * layer_b[2]),axis=-1,keepdims=True)

            elif order == 2:
                weights_3 = np.add(np.add(weights_3,learnrate * layer_b_w[0][0]),-learnrate2 * layer_b_w[0][1])
                weights_2 = np.add(np.add(weights_2,learnrate * layer_b_w[1][0]),-learnrate2 * layer_b_w[1][1])
                weights_1 = np.add(np.add(weights_1,learnrate * layer_b_w[2][0]),-learnrate2 * layer_b_w[2][1])
                bias_3 = np.mean(np.add(np.add(bias_3,learnrate * layer_b[0][0]),-learnrate2 * layer_b[0][1]),axis=-1,keepdims=True)
                bias_2 = np.mean(np.add(np.add(bias_2,learnrate * layer_b[1][0]),-learnrate2 * layer_b[1][1]),axis=-1,keepdims=True)
                bias_1 = np.mean(np.add(np.add(bias_1,learnrate * layer_b[2][0]),-learnrate2 * layer_b[2][1]),axis=-1,keepdims=True)
            
            weights = [weights_1,weights_2,weights_3]
            bias = [bias_1,bias_2,bias_3]
            
            loss = sum(sum(abs(np.subtract(y,layer_o[2]))))
            for col in range(batch_size):
                accuracy += int(layer_o[2][:,col].argmax() == y[:,col].argmax())
            accuracy = accuracy/batch_size
            print('epoch:{}, batch:{} , loss:{}, accuracy:{}'.format(epoch, i,loss,accuracy))
            accuracy = 0

    accuracy = 0
    X = np.concatenate(test_X,axis=-1)
    Y = np.concatenate(test_y,axis=-1)
    _,output = forward(X,weights,bias)
    for i in range(len(Y[0])):
        accuracy += int(output[2][:,i].argmax() == Y[:,i].argmax())
    print('test accuracy:{}'.format(float(accuracy/len(Y[0]))))

    del weights
    del bias

    return accuracy


if __name__ == '__main__':

    memo = []
    accuracy = run(epoches=70, batch_size=300, order=2, learnrate=0.01, learnrate2=0.0001, decay=0.0, decay2=1e-7)
    memo.append(accuracy)