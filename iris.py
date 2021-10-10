import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#the following statement initializes and loads the dataset
iris_dataset = load_iris()

print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: {}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))

print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

################################################################

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

# Same for the test samples
print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))

#create knn classifier
knn = KNeighborsClassifier(n_neighbors=1)

#fit the classifier to the data
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:{}".format(X_new.shape)) 

#here the prediction part begins, we make use of k next neighbor algorithm here
prediction = knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted target name:{}".format(iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score (np.mean):{:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score (knn.score):{:.2f}".format(knn.score(X_test, y_test)))
