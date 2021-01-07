import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


wine = datasets.load_wine()

x=wine.data
y=wine.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

#Name of the classes
classes=['class_0','class_1','class_2']

#classifier to fit training data
# Kernel changes dimensions of planes making it easier to classify data points
classifier = svm.SVC(kernel="poly")
classifier.fit(x_train,y_train)

#Prediction on the x-test
y_prediction = classifier.predict(x_test)

#Accuracy of the model
accurate = metrics.accuracy_score(y_test, y_prediction)

print(accurate)

