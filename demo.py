from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)



class ScrappyKNN():
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        pass

    def predict(self,X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self,row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0

        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])

            if dist < best_dist:
                best_dist = dist
                best_index = i
            return self.y_train[best_index]

from sklearn.datasets import load_iris

import numpy as np
iris = load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .5)


clf = ScrappyKNN()
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
# #viz
# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")
#
#
#
# #predict
#
# print(test_data[0],test_target[0])
# print(iris.feature_names,iris.target_names)
