# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Keerthan D
RegisterNumber: 212224240075

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load data
data = pd.read_csv("Salary.csv")

# Print head of the data
print("Data Head:")
print(data.head())

# Print info of the data
print("\nData Info:")
print(data.info())

# Print count of missing values
print("\nMissing Values Count:")
print(data.isnull().sum())

# Print head of salary column
print("\nSalary Column Head:")
print(data["Salary"].head())

# Encode categorical variable 'Position'
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

# Prepare features and target
x = data[["Position", "Level"]]
y = data[["Salary"]]

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Initialize and train Decision Tree Regressor model
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Predict on test set
y_pred = dt.predict(x_test)

# Calculate and print Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:")
print(mse)

# Calculate and print R2 score
r2 = metrics.r2_score(y_test, y_pred)
print("\nR2 Score:")
print(r2)

# Predict salary for a new input, making sure to pass proper feature names in dataframe
input_data = pd.DataFrame([[5, 6]], columns=["Position", "Level"])
predicted_salary = dt.predict(input_data)
print("\nPredicted Salary for Position=5 and Level=6:")
print(predicted_salary[0])


```

## Output:

![image](https://github.com/user-attachments/assets/e4f307bf-9669-4bdc-b570-52bcff4f9c55)

![image](https://github.com/user-attachments/assets/c47f70b4-183a-466d-b291-d98aceffcc9c)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
