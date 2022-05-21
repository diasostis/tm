#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#load data from xlsx
TitanicData = pd.read_excel("Project38Titanic.xlsx",)
TitanicData = pd.DataFrame(TitanicData, columns=["Class","Age","Sex","Survived"])

#print 10 rows of df
print(TitanicData.head(10))

#check num of rows and cols
print(TitanicData.shape)

#info about the data
print(TitanicData.info())

#num of missing values in cols
print(TitanicData.isnull().sum())

#drop 2 & 3 row
TitanicData = TitanicData.drop(TitanicData.index[0:2])
print(TitanicData.head(10))
print("\n")

#reset df index
TitanicData = TitanicData.reset_index(drop=True)
print(TitanicData.head(10))
print("\n")

#num of survivors // Sum
print(TitanicData["Survived"].value_counts())
print("\n")

#convert categorical cols to numeric
TitanicData.replace({"Survived":{"Yes":1,"No":0}, "Sex":{"Male":0, "Female":1}, "Class":{"First":1, "Second":2, "Third":3, "Crew":4}, "Age":{"Adult":0, "Child":1}}, inplace=True)
print(TitanicData.head(50))

#num of survivors visualized
sns.set()
sns.countplot("Survived", data=TitanicData)
plt.show()

#num of Male / Female
print(TitanicData["Sex"].value_counts())
print("\n")

#num of survivors gender wise visualized
sns.countplot("Sex", hue="Survived", data=TitanicData)
plt.show()

#num of Class
print(TitanicData["Class"].value_counts())
print("\n")

#num of survivors class wise visualized
sns.countplot("Class", hue="Survived", data=TitanicData)
plt.show()

#keep df of values except "Survived" and create a "Survived" one
X = TitanicData[["Class","Age","Sex"]].copy()
Y = TitanicData["Survived"]

print(X)
print("\n")
print(Y)

#split data to training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)
print(X.shape, X_train.shape, X_test.shape)

#Logistic Regression
model = LogisticRegression()

#training the model with training data
model.fit(X_train, Y_train)

#model evaluation // accuracy on train data
X_train_pred = model.predict(X_train)
train_data_acc = accuracy_score(Y_train, X_train_pred)
print("\nAccuracy score of train data: ", train_data_acc)

#model evaluation // accuracy on test data
X_test_pred = model.predict(X_test)
test_data_acc = accuracy_score(Y_test, X_test_pred)
print("\nAccuracy score of test data: ", test_data_acc)