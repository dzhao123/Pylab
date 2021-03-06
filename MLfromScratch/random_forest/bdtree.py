##############################
#Attribute  0,1,2,3,4,5,6

#Data       1,2,3,4,5,6,7   1  Label
#           8,9,1,2,3,4,5   2
#           3,4,6,7,8,7,5   3
##############################
import numpy as np
from collections import Counter
import math
from node import Node

#class Node(object):
#    def __init__(self, isLeaf, attribute, threshold):
#        self.isLeaf = isLeaf
#        self.attribute = attribute
#        self.threshold = threshold
#        self.children = []



class C45(object):
    def __init__(self):
        self.data = []
        self.attributes = []
        self.classes = []
        self.tree = []


    def allSameClass(self, curData):
        for row in curData[1:]:
            if curData[0][-1] != row[-1]:
                return False
        return curData[0][-1]



    def majorClass(self, curData):

        classes = []
        for row in curData:
            classes.append(row[-1])
        classes,value = max(Counter(classes).items(),key = lambda x: x[1])
        return classes



    def entropy(self, classes):

        totalSample = 0
        prob = []

        totalSample = len(classes)
        for key,value in Counter(classes).items():
            prob.append(value/totalSample)
        return -np.dot(prob, np.log(prob)/np.log(2))



    def gain(self, curData, splitData):

        purityBeforeSplit = float('inf')
        purityAfterSplit = float('inf')

        classBeforeSplit = []
        classAfterSplit = []
        weights = []


        for row in curData:
            classBeforeSplit.append(row[-1])

        for data in splitData:
            temp = []
            for row in data:
                temp.append(row[-1])
            classAfterSplit.append(temp)

        weights = [len(classAfterSplit[0])/len(classBeforeSplit), len(classAfterSplit[1])/len(classBeforeSplit)]

        purityBeforeSplit = self.entropy(classBeforeSplit)
        purityAfterSplit = weights[0]*self.entropy(classAfterSplit[0])+weights[1]*self.entropy(classAfterSplit[1])
        return purityBeforeSplit - purityAfterSplit



    def splitBestAttributes(self, curData, curAttributes):

        threshold = -1*float('inf')
        max_infoGain = -1*float('inf')
        less = []
        greater = []
        best_attribute = -1
        best_threshold = -1*float('inf')
        best_less = []
        best_greater = []

        for attribute in curAttributes:
            curData.sort(key = lambda x: x[attribute])
            for index in range(1, len(curData)):
                if curData[index-1][attribute] != curData[index][attribute]:
                    threshold = (curData[index-1][attribute] + curData[index][attribute])/2
                    less = curData[:index]
                    greater = curData[index:]
                    infoGain = self.gain(curData, [less,greater])
                    if infoGain > max_infoGain:
                        max_infoGain = infoGain
                        best_threshold = threshold
                        best_attribute = attribute
                        best_less = less
                        best_greater = greater

        return best_attribute, best_threshold, [best_less,best_greater], max_infoGain




    def sameElementInData(self, curData, curAttributes):

        for attribute in curAttributes:
            for ele in [item[attribute] for item in curData]:
                if ele == curData[0][attribute]:
                    continue
                else:
                    return False
        return True


    def classPortion(self, data):

        temp = []
        for item in data:
            temp.append(item[-1])

        portion = dict(Counter(temp))

        for key,value in portion.items():
            portion[key] = value/len(temp)

        return portion




    def recursivelyGenerateTree(self, curData, curAttributes, max_infoGain):

        allSameClass = self.allSameClass(curData)

        if len(curData) == 0:
            return Node(True, 'empty', None)

        elif allSameClass is not False:
            return Node(True, allSameClass, None)

        elif not curAttributes:
            return Node(True, self.majorClass(curData), None)

        else:
            best_attribute, best_threshold, splited_data, max_infoGain = self.splitBestAttributes(curData, curAttributes)
            remainingAttributes = curAttributes[:]
            remainingAttributes.remove(best_attribute)
            #print('remainingAttributes:', remainingAttributes)
            #print('best_attribute:', best_attribute)
            #print('best_threshold:', best_threshold)
            #print('splited_data:', splited_data)
            #print('\n\n')
            node = Node(False, best_attribute, best_threshold)
            node.children = [self.recursivelyGenerateTree(data, remainingAttributes, max_infoGain) for data in splited_data]

            return node



    def generateTree(self, data):
        attributes = self.attributes
        max_infoGain = 0
        self.tree = self.recursivelyGenerateTree(data, attributes, max_infoGain)



    def dataProcess(self, data, label):
        dataProcessed = []
        for index, row in enumerate(data):
            temp = []
            for num in row:
                temp.append(num)
            temp.append(label[index])
            dataProcessed.append(temp)
        return dataProcessed



    def fit(self, x_train, y_train):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.data = x_train
        self.classes = y_train
        self.attributes = [num for num in range(len(x_train[1]))]
        x_processed = self.dataProcess(x_train, y_train)
        self.generateTree(x_processed)



    def visualize(self, node):

        left_child = node.children[0]
        right_child = node.children[1]

        if left_child.isLeaf:
            print(node.attribute, "<", node.threshold, ":", left_child.attribute)
        else:
            print(node.attribute, "<", node.threshold, ":", )
            self.visualize(left_child)

        if right_child.isLeaf:
            print(node.attribute, ">=", node.threshold, ":", right_child.attribute)
        else:
            print(node.attribute, ">=", node.threshold, ":")
            self.visualize(right_child)



    def visualize2(self, node):

        if node.isLeaf:
            print(node.attribute)

        else:
            print(node.attribute, "<", node.threshold, ":")
            self.visualize2(node.children[0])
            print(node.attribute, ">=", node.threshold, ":")
            self.visualize2(node.children[1])



    def iterativelyTest(self, data, node):
        if node.isLeaf:
            return node.attribute
        elif data[node.attribute] < node.threshold:
            print('<', node.attribute)
            return self.iterativelyTest(data, node.children[0])
        else:
            print('>=', node.attribute)
            return self.iterativelyTest(data, node.children[1])


    def test(self, curData, curLabel):

        counter = 0
        for index, data in enumerate(curData):
            res = self.iterativelyTest(data, self.tree)
            #if type(res) is dict:
            #    if curLabel[index] in res:
            #        counter += 1
            #else:
            if curLabel[index] == res:
                counter += 1
        #    if curLabel[index] in self.iterativelyTest(data, self.tree):
        #        counter += 1

        return counter/len(curLabel)



class Node(object):
    def __init__(self, isLeaf, attribute, threshold):
        self.isLeaf = isLeaf
        self.attribute = attribute
        self.threshold = threshold
        self.children = []




if __name__ == '__main__':
    from data_utils import DataUtils

    trainfile_X = 'train-images-idx3-ubyte'
    trainfile_y = 'train-labels-idx1-ubyte'
    testfile_X = 't10k-images-idx3-ubyte'
    testfile_y = 't10k-labels-idx1-ubyte'
    train_X = DataUtils(filename=trainfile_X).getImage()
    train_y = DataUtils(filename=trainfile_y).getLabel()
    test_X = DataUtils(testfile_X).getImage()
    test_y = DataUtils(testfile_y).getLabel()

    a = C45()
    a.fit(train_X, train_y)
    #a.visualize2(a.tree)
    print(a.test(test_X, test_y))
