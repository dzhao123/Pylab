from data_utils import DataUtils


trainfile_X = 'train-images-idx3-ubyte'
trainfile_y = 'train-labels-idx1-ubyte'
testfile_X = 't10k-images-idx3-ubyte'
testfile_y = 't10k-labels-idx1-ubyte'
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()


#print(train_X[0])
from sklearn.tree import DecisionTreeClassifier

a = DecisionTreeClassifier()
a.fit(train_X[:200],train_y[:200])
print(a.score(train_X[:200],train_y[:200]))
print(a.decision_path(train_X[:200]))
