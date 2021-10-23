# Random Forest classifier for Iris Dataset

import matplotlib.pyplot as plt
# Importing the libraries
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Loading the iris dataset
iris_data = load_iris()
print("dir of iris_data:", dir(iris_data))
print("target names:", iris_data.target_names[:5])

# Creating a dataframe to view the data
data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
print("Viewing head of dataframe:\n", data.head())

# Creating a correlation matrix between all the features
corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='cubehelix')
plt.title("correlation matrix")
plt.show()

# Splitting the dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=1)
print("length of X_train:", len(X_train))
print("length of X_test:", len(X_test))

# Creating the Random forest classifier object
model = RandomForestClassifier(n_estimators=3)

# Training the model with our training dataset
model.fit(X_train, y_train)

# Checking the accuracy of our model
print("Model score:", model.score(X_test, y_test))

# Predicting the outcome from our model using the testing dataset
predicted = model.predict(X_test)

# Let's build a confusion matrix
cm = confusion_matrix(y_test, predicted)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='GnBu')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title("confusion matrix")
plt.show()

# Some metrics about the mode
print("CLASSIFICATION REPORT CHART FOR RANDOM FOREST CLASSIFIER: ", "\n\n", classification_report(y_test, predicted),
      "\n")
