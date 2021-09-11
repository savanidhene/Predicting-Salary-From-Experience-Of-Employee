#Step1: Importing the required packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#Step2: Loading the dataset
data_set = pd.read_csv('Salary_Data.csv')
#print(data_set.head())

#Step3: Extracting the target and predictor variables
predictors = ['YearsExperience']
target = ['Salary']
x = data_set[predictors].values
y = data_set[target].values
#print(x)
#print(y)

#Step4: Splitting the Dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Step5: Fitting the simple Linear Regression Model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)       #training the model
print("Training Accuracy = ",regressor.score(x_train, y_train))
print("Testing Accuracy = ", regressor.score(x_test, y_test))

#Step6: Prediction
exp = float(input("Enter the Year's of Experiance: "))
sal = regressor.predict([[exp]])

print("Predicted Salary for given Experience = ",sal)

#Step7: Visualization of Results for Test Dataset
y_pred = regressor.predict(x_test)

plt.scatter(x_test, y_test, color='green')      #Actual Values
plt.plot(x_test, y_pred, color='red')           #Predicted Values
plt.title('Experience vs Salary')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()