import numpy as np
from collections import Counter
import math
from node import Node


class btree(object):
    def __init__(self, min_split = 2, method = 'infoGain'):
        self.inputs = []
        self.outputs = []
        self.shape = []
        self.features = []
        self.label = []
        self.info = 0
        self.tree = []
        self.attribute = []


    def entropy(self, partition,label):
        left_part = label[:partition]
        right_part = label[partition:]
        left = Counter(left_part)
        right = Counter(right_part)
        entrol = 0
        entror = 0

        for keyl,vall in left.items():
            pl = vall/len(left_part)
            entrol += pl*math.log(pl,2)

        for keyr,valr in right.items():
            pr = valr/len(right_part)
            entror += pr*math.log(pr,2)

        return entrol, entror


    def infoGain(self, partition, label):
        left_portion = partition/len(label)
        right_portion = 1 - left_portion
        left,right = self.entropy(partition,label)
        return self.info+left_portion*left+right_portion*right


    def maxInfoGain_AttributeValue(self, attribute, label):
        temp = []
        max_infoGain = -100
        argmax = 0
        attr = 0

        #if len(attribute) < 3:
        #    print(attribute)
        #    max_infoGain = self.info
        #    return attribute[0], max_infoGain

        for partition in range(1,len(attribute)):
            infoGain = self.infoGain(partition, label)
            if infoGain > max_infoGain:
                max_infoGain = infoGain
                argmax = partition

        attr = (attribute[argmax-1] + attribute[argmax])/2
        return attr, max_infoGain #argmax, attr


    def data_parsing(self, inputs, outputs, attribute_index, attribute_value):
        left_index = []
        right_index = []
        left_data = []
        right_data = []
        left_label = []
        right_label = []
        output_left = []
        output_right = []
        output_label_left = []
        output_label_right = []
        goal_attribute = []

        #inputs = [[6,1],[7,2]]
        #outputs = [[3],[3]]
        #attribute_index = 1
        #attribute_value = 6

        for attribute in inputs:
            if attribute[-1] == attribute_index:
                goal_attribute = attribute

        #goal_attributes = [6,1]


        for index, value in enumerate(goal_attribute[:-1]):
            if value < attribute_value:
                left_index.append(index)
            else:
                right_index.append(index)
        # left_index = []
        # right_index = [0]
        for index, attribute in enumerate(inputs):
            if attribute[-1] == attribute_index:
                #print(attribute)
                continue
            else:
                left_data = [attribute[i] for i in left_index]
                right_data = [attribute[i] for i in right_index]
                #if left_index:
                left_data.append(attribute[-1])
                #else:
                #    left_data.append(outputs[index])
                #if right_index:
                right_data.append(attribute[-1])
                #else:
                #    right_data.append(outputs[])
                #if left_index:
                left_label = [outputs[index][i] for i in left_index]
                #else:
                #    left_label = outputs[index]

                #if right_index:
                right_label = [outputs[index][i] for i in right_index]
                #else:
                #    right_label = outputs[index]
                output_left.append(left_data)
                output_right.append(right_data)
                output_label_left.append(left_label)
                output_label_right.append(right_label)


        return output_left, output_right, output_label_left, output_label_right


    def build(self, inputs, outputs):
        #print('inputs:', inputs)
        #print('outputs:', outputs)

        if inputs is None:
            return

        if len(outputs) == 1 and len(outputs[0]) == 1:
            return


        max_infoGain = -100
        attribute_index = 0
        attribute_value = 0
        for attr_index in range(len(inputs)):
            #print(inputs[attr_index])
            #print(attr_index)
            attribute = inputs[attr_index][-1]
            features_label = sorted(list(zip(inputs[attr_index][:-1],outputs[attr_index])))
            features = [x for x,y in features_label]
            self.features = features
            label = [y for x,y in features_label]
            self.label = label
            attrValue, infoGain = self.maxInfoGain_AttributeValue(features,label)
            if infoGain > max_infoGain:
                max_infoGain = infoGain
                attribute_index = attribute
                attribute_value = attrValue
        #print([attribute_index, attribute_value])
        self.tree.append([attribute_index, attribute_value])
        data_left, data_right, label_left, label_right = self.data_parsing(inputs, outputs, attribute_index, attribute_value)

        self.shape = np.array(data_left).shape
        left = self.build(data_left, label_left)
        self.shape = np.array(data_right).shape
        right = self.build(data_right, label_right)


    def dataProcess(self, inputs_shape):
        return np.array([self.outputs]*inputs_shape[1])

    def inputsProcess(self, inputs):
        attributes = [i for i in range(len(inputs[0]))]
        #for index, attribute in enumerate(inputs):
        #    np.append(attribute,index)
        #np.insert(inputs,inputs.shape[1],attributes,axis=1)
        inputs = np.vstack((inputs, attributes))
        return inputs.T



    def train(self, inputs, outputs):
        #self.inputs = inputs
        #self.outputs = outputs
        inputs_shape = inputs.shape
        self.shape = inputs.shape
        #inputs = inputs.T
        self.inputs = inputs
        self.outputs = outputs
        outputs = self.dataProcess(inputs_shape)
        #print(outputs)
        label_counter = Counter(self.outputs)
        for key,val in label_counter.items():
            p = val/len(self.outputs)
            self.info += -p*math.log(p,2)
        new_inputs = self.inputsProcess(inputs)
        self.attribute = [i for i in range(inputs_shape[1])]
        attribute = self.attribute
        #print(new_inputs)
        #print(outputs)
        self.build(new_inputs, outputs)

if __name__ == '__main__':
    inputs = np.array([[1,2,1],[3,2,4],[5,6,7]])
    labels = np.array([1,2,3])
    a = btree()
    a.train(inputs,labels)
    print(a.tree)
