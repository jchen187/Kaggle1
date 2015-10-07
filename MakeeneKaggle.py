xtrain = open('X_train.txt','r')
ytrain = open('Y_train.txt','r')
xtest = open('X_test.txt','r')

import numpy as np
features_train = np.loadtxt(xtrain)
labels_train = np.loadtxt(ytrain)
features_test = np.loadtxt(xtest)

from sklearn import svm
classifier = svm.SVC()
classifier.fit(features_train,labels_train)
prediction = classifier.predict(features_test)

ytest = open('Y_test.csv','w')
ytest.write("Id,Category\n")
for i in xrange(len(features_test)):
    #using int because if not you will get a float
    ytest.write("%d,%d\n" %(i+1,prediction[i]))
    #ytest.write("{},{}\n".format(i+1,int(prediction[i])))
ytest.close
