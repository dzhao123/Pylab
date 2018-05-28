import random
from data_utils import DataUtils
from C45 import C45
from collections import Counter
import heapq

class randomForest(object):

    def __init__(self, data, label):
        self.data = data
        self.label = label
        #train_X, train_y = self.randomSample()
        self.forest = []
        self.buildForest()


    def randomSample(self, size = 200):
        #train_X = [random.sample(row, size) for row in self.data]
        #train_y = self.label
        #print(len(self.data[0]))
        row_inds = [num for num in range(len(self.data[0]))]
        col_inds = [num for num in range(len(self.data))]

        sampled_row_inds = random.sample(row_inds, size)
        sampled_col_inds = random.sample(col_inds, size)

        train_X = [self.data[row_index] for row_index in sampled_col_inds]
        train_y = [self.label[row_index] for row_index in sampled_col_inds]

        return train_X, train_y, row_inds

    def buildForest(self, number_of_trees = 10):

        for num in range(number_of_trees):
            train_X, train_y, attribute = self.randomSample()
            tree = C45()
            tree.fit(train_X, train_y, attribute)
            self.forest.append(tree.tree)


    def iterativelyTest(self, data, node):

        if node.isLeaf:
            return node.attribute
        elif data[node.attribute] < node.threshold:
            return self.iterativelyTest(data, node.children[0])
        else:
            return self.iterativelyTest(data, node.children[1])


    def test(self, curData, curLabel):
        counter = 0
        res = []
        for index, data in enumerate(curData):
            temp = []
            for tree in self.forest:
                temp.append(self.iterativelyTest(data, tree))
            top = max(Counter(temp).items(), key = lambda x: x[1])[0]
            if top == curLabel[index]:
                counter += 1
        print(counter/len(curLabel))


if __name__ == '__main__':

    #with open('data') as file:
    #    data = file.readlines()

    #X = [[float(value) for value in item.split(',')[:-1]] for item in data]
    #y = [item.split(',')[-1][:-1] for item in data]
    #import random


    trainfile_X = 'train-images-idx3-ubyte'
    trainfile_y = 'train-labels-idx1-ubyte'
    testfile_X = 't10k-images-idx3-ubyte'
    testfile_y = 't10k-labels-idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    X = train_X[:500]
    y = train_y[:500]

    X_test = test_X[:80]
    y_test = test_y[:80]

    #print(X.shape)
    #print(y.shape)
    forest = randomForest(X,y)
    forest.buildForest()
    forest.test(X, y)
