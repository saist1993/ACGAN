'''
	The file reads the matrtices created by the linear_svm_eval and shuffels the data, then runs svm classifier over it.

'''
import pickle
import argparse
from sklearn import svm
from sklearn.utils import shuffle


parser = argparse.ArgumentParser()
parser.add_argument('--manualSeed', default=0, type=int, help='manual seed')
parser.add_argument('--data',default='matrices.dat', help='feature vector matrix')
parser.add_argument('--percentage', default=1, type=int, help='the size of dataset to train on')
parser.add_argument('--fit',default='fit.svm', help='the result of fit on the dataset by svm classifier')
parser.add_argument('--predict',default='predict.dat', help='the final predection matrix')


opt = parser.parse_args()
data = pickle.load(open(opt.data))


# Now select samples from it like a ba-mofo
X, Y = shuffle(data['trainX'], data['trainY'], random_state=opt.manualseed)
x_train = X[:len(X) / opt.percentage]
y_train = Y[:len(Y) / opt.percentage]


#After splitting the data run fit and predict on them.
svmier = svm.LinearSVC()
svmier.fit(x_train, y_train)
pickle.dump(svmier, open(opt.fit, 'w+'))
predection = svmier.predict(data['testX'])
pickle.dump(predection, open(opt.predict, 'w+'))


print "done with classifying, check %s for predection" % (opt.predict)