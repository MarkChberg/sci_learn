# Author Mark
from sklearn import svm
from sklearn import datasets


# to learn more about how to calculate the svm if you have time
iris_data = datasets.load_iris()

clf = svm.SVC()
clf.fit(iris_data.get('data'), iris_data.get('target'))

print(clf)

cl = clf.predict(iris_data.get('data')[1:2])

print(cl)