import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

#read data and each data value is seperated by a comma(",")
data = pd.read_csv("heart.csv",sep=",")

"""
Binary
0=No,Woman,Alive,
1=Yes,Men,Dead

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)
"""

#selected attributes from data
data = data[["Death","high_blood_pressure","creatinine_phosphokinase","serum_sodium","time","ejection_fraction"]]

#print data of selected columns first 5 rows
print(data.head())

#predict if they wiill die or not before the follow up period
prediction = "Death"

#return new data frame that doesnt have Death
#Attributes
X=np.array(data.drop([prediction], 1))

#Labels
Y=np.array(data[prediction])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)

bestData=0

#Run alogirthim 50 times, and choose the highest accuracy 
for _ in range(50):
#0.1= split  10% of test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size = 0.1)
    
    linear = linear_model.LinearRegression()
    
    linear.fit(x_train,y_train)
    accuracy=linear.score(x_test,y_test)
    
    if accuracy>bestData:
        bestData=accuracy
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

#Xlabel 
b="time"    
style.use("ggplot")
pyplot.scatter(data[b],data["Death"])
pyplot.xlabel(b)
pyplot.ylabel("Death")
pyplot.show()



