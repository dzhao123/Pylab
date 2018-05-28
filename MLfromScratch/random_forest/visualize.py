#Decision tree for MNIST dataset by arthur503.
#Data format: 'class	label1:pixel	label2:pixel ...'
#Warning: without fix overfitting!
#
#Test change pixel data into more categories than 0/1:
#int(pixel)/50: 37%
#int(pixel)/64: 45.9%
#int(pixel)/96: 52.3%
#int(pixel)/128: 62.48%
#int(pixel)/152: 59.1%
#int(pixel)/176: 57.6%
#int(pixel)/192: 54.0%
#
#Result:
#Train with 10k, test with 60k: 77.79%
#Train with 60k, test with 10k: 87.3%
#Time cost: 3 hours.

from numpy import *
import operator

def calcShannonEntropy(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featureVec in dataSet:
		currentLabel = featureVec[0]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 1
		else:
			labelCounts[currentLabel] += 1
	shannonEntropy = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEntropy -= prob  * log2(prob)
	return shannonEntropy

#get all rows whose axis item equals value.
def splitDataSet(dataSet, axis, value):
	subDataSet = []
	for featureVec in dataSet:
		if featureVec[axis] == value:
			reducedFeatureVec = featureVec[:axis]
			reducedFeatureVec.extend(featureVec[axis+1:])	#if axis == -1, this will cause error!
			subDataSet.append(reducedFeatureVec)
	return subDataSet

def chooseBestFeatureToSplit(dataSet):
	#Notice: Actucally, index 0 of numFeatures is not feature(it is class label).
	numFeatures = len(dataSet[0])
	baseEntropy = calcShannonEntropy(dataSet)
	bestInfoGain = 0.0
	bestFeature = numFeatures - 1 	#DO NOT use -1! or splitDataSet(dataSet, -1, value) will cause error!
	#feature index start with 1(not 0)!
	for i in range(numFeatures)[1:]:
		featureList = [example[i] for example in dataSet]
		featureSet = set(featureList)
		newEntropy = 0.0
		for value in featureSet:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEntropy(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

#classify on leaf of decision tree.
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount:
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

#Create Decision Tree.
def createDecisionTree(dataSet, features):
	print ('create decision tree... length of features is:'+str(len(features)))
	classList = [example[0] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeatureIndex = chooseBestFeatureToSplit(dataSet)
	bestFeatureLabel = features[bestFeatureIndex]
	myTree = {bestFeatureLabel:{}}
	del(features[bestFeatureIndex])
	featureValues = [example[bestFeatureIndex] for example in dataSet]
	featureSet = set(featureValues)
	for value in featureSet:
		subFeatures = features[:]
		myTree[bestFeatureLabel][value] = createDecisionTree(splitDataSet(dataSet, bestFeatureIndex, value), subFeatures)
	return myTree

def line2Mat(line):
	mat = line.strip().split(' ')
	for i in range(len(mat)-1):
		pixel = mat[i+1].split(':')[1]
		#change MNIST pixel data into 0/1 format.
		mat[i+1] = int(pixel)/128
	return mat

#return matrix as a list(instead of a matrix).
#features is the 28*28 pixels in MNIST dataset.
def file2Mat(fileName):
	f = open(fileName)
	lines = f.readlines()
	matrix = []
	for line in lines:
		mat = line2Mat(line)
		matrix.append(mat)
	f.close()
	print ('Read file '+str(fileName) + ' to array done! Matrix shape:'+str(shape(matrix)))
	return matrix

#Classify test file.
def classify(inputTree, featureLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featureIndex = featureLabels.index(firstStr)
	predictClass = '-1'
	for key in secondDict.keys():
		if testVec[featureIndex] == key:
			if type(secondDict[key]) == type({}):
				predictClass = classify(secondDict[key], featureLabels, testVec)
			else:
				predictClass = secondDict[key]
	return predictClass

def classifyTestFile(inputTree, featureLabels, testDataSet):
	rightCnt = 0
	for i in range(len(testDataSet)):
		classLabel = testDataSet[i][0]
		predictClassLabel = classify(inputTree, featureLabels, testDataSet[i])
		if classLabel == predictClassLabel:
			rightCnt += 1
		if i % 200 == 0:
			print ('num '+str(i)+'. ratio: ' + str(float(rightCnt)/(i+1)))
	return float(rightCnt)/len(testDataSet)

def getFeatureLabels(length):
	strs = []
	for i in range(length):
		strs.append('#'+str(i))
	return strs

#Normal file
#trainFile = 'train_60k.txt'
#testFile = 'test_10k.txt'
#Scaled file
#trainFile = 'train_60k_scale.txt'
#testFile = 'test_10k_scale.txt'
#Test file
#trainFile = 'test_only_1.txt'
#testFile = 'test_only_2.txt'

#train decision tree.
#dataSet = file2Mat(trainFile)
#Actually, the 0 item is class, not feature labels.
#featureLabels = getFeatureLabels(len(dataSet[0]))
from data_utils import DataUtils

trainfile_X = 'train-images-idx3-ubyte'
trainfile_y = 'train-labels-idx1-ubyte'
testfile_X = 't10k-images-idx3-ubyte'
testfile_y = 't10k-labels-idx1-ubyte'
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()



dataSet = train_X[:500]
featureLabels = train_y[:500]

print ('begin to create decision tree...')
myTree = createDecisionTree(dataSet, featureLabels)
print ('create decision tree done.')

#predict with decision tree.
#testDataSet = file2Mat(testFile)
#featureLabels = getFeatureLabels(len(testDataSet[0]))

testDataSet = test_X[:100]
featureLabels = test_y[:100]

rightRatio = classifyTestFile(myTree, featureLabels, testDataSet)
print ('total right ratio: ' + str(rightRatio))
