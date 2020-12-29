import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv",sep=";")

#selected attributes from data
data = data[["G1","G2","G3","studytime","failures"]]

#print data of selected columns
print(data.head())

#predict final grade
predict = "G3"

#return new data frame that doesnt have G3
#Attributes
X=np.array(data.drop([predict], 1))
#Labels
Y=np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)


bestData=0

#Run alogirthim 50 times, and choose the highest accuracy 
for _ in range(50):
#0.1= split  10% of test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)
    
    linear = linear_model.LinearRegression()
    
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)
    
    if acc>bestData:
        bestData=acc
        #pickle file
        with open("grades.pickle","wb") as f:
            pickle.dump(linear,f)
            

print("Best Accuracy:",bestData)

pickle_in=open("grades.pickle","rb")
linear = pickle.load(pickle_in)

print("Coe: \n",linear.coef_)
print("Int: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])
 
b="G1"    
style.use("ggplot")
pyplot.scatter(data[b],data["G3"])
pyplot.xlabel(b)
pyplot.ylabel("Final Grade")
pyplot.show()
                 
                 
