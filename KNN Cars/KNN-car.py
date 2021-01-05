import pandas as pd
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

#Changes String column values to Integers
fix=preprocessing.LabelEncoder()
buying=fix.fit_transform(list(data["buying"]))
maint=fix.fit_transform(list(data["maint"]))
door=fix.fit_transform(list(data["door"]))
persons=fix.fit_transform(list(data["persons"]))
lug_boot=fix.fit_transform(list(data["lug_boot"]))
persons=fix.fit_transform(list(data["persons"]))
safety=fix.fit_transform(list(data["safety"]))
cls=fix.fit_transform(list(data["class"]))

predict="class"

X=list(zip(buying,maint,door,persons,lug_boot,safety))
Y=list(cls)


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.2)

#Find the closest k data points
model = KNeighborsClassifier(n_neighbors=7)

model.fit(x_train,y_train)
acc=model.score(x_test,y_test)
print(acc)

predicted=model.predict(x_test)

# The Classes of Cars
names =["unacc","acc","good","vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]],"Data: ", x_test[x], "Actual: ", names[y_test[x]])
    Neighbours = model.kneighbors([x_test[x]],7,True)
    print("Neighbours: ", Neighbours)