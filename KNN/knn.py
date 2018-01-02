# Author Mark

from sklearn import datasets
from sklearn import neighbors # to import package neighbor

iris_data = datasets.load_iris()
print(iris_data)

clf = neighbors.KNeighborsClassifier(n_neighbors=10)
clf.fit(iris_data.get('data'), iris_data.get('target'))

print(clf)

cl = clf.predict([[5.9,3,5.1,1.8]])
print(str(cl))

print(iris_data.get('target_names')[cl[0]])