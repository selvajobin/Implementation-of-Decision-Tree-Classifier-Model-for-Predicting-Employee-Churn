# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SELVA JOBIN S
RegisterNumber:  212223220102
*/
```
```
import pandas as pd
df = pd.read_csv("/content/Employee.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['left'].value_counts())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['salary'] = le.fit_transform(df['salary'])

x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]

print(x)
y = df['left']
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Y Predicted : \n\n",y_pred)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
```
df.head()
```
![Screenshot 2025-04-07 105243](https://github.com/user-attachments/assets/ec794435-680e-4181-82d0-1483eb08d819)
```
df.info()
```
![Screenshot 2025-04-07 105342](https://github.com/user-attachments/assets/2c50d04a-756e-41c4-9f07-e22e9243243f)
```
df.isnull().sum()
```
![Screenshot 2025-04-07 105442](https://github.com/user-attachments/assets/28debc9b-8bbf-4e79-a253-3bd05fd32e87)
```
df['left'].value_counts()
```
![Screenshot 2025-04-07 105528](https://github.com/user-attachments/assets/93c26a1f-4b9d-4236-994c-0120794da0ce)
```
print(x)
y = df['left']
print(y)
print("Y Predicted : \n\n",y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![Screenshot 2025-04-07 105930](https://github.com/user-attachments/assets/325977c0-9975-4316-9ea6-43292ebebbf6)![Screenshot 2025-04-07 105956](https://github.com/user-attachments/assets/014b5f31-73d8-48fc-adbd-64aacd430426)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
