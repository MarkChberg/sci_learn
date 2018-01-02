# Author Mark

import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree


all_data_file = open('/Users/mark/PycharmProjects/sci_learn/DecisionTree/classone.csv', 'r')
print(all_data_file)
reader = csv.reader(all_data_file) # input a csv iter item and return a reader
headers = reader.__next__()
print(headers)

label_list = []
feature_list = []

for row in reader:
    label_list.append(row[len(row) - 1]) # make result a list
    row_dict = {}
    for i in range(1, len(row) - 1):
        row_dict[headers[i]] = row[i]  # make every feature a dictionary eg: {'age':'youth','income':'high'....}
    feature_list.append(row_dict) # in order to match the rule
print(label_list)
for row in feature_list:
    print(row)

vec = DictVectorizer()
vec_data = vec.fit_transform(feature_list).toarray()
print(vec_data)
# print(vec.get_feature_names())
lb = preprocessing.LabelBinarizer()
lb_data = lb.fit_transform(label_list)
print(lb_data)

clf_tree = DecisionTreeClassifier(criterion='entropy')
clf_data = clf_tree.fit(vec_data, lb_data)
print(str(clf_data))
with open('classone_result.dot','w') as out_file:
    tree.export_graphviz(clf_data,feature_names=vec.get_feature_names(), out_file = out_file)

# predict start
pre_row = vec_data[0:1] # 需要一个二维array
print(pre_row)
predictions = clf_tree.predict(X=pre_row)
print(str(predictions))
all_data_file.close()