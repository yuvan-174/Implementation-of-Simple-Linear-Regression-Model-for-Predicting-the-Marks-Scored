# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Prepare the Data
2. Train the Model
3. Make Predictions
4. Evaluate and Visualize

## Program:
```py
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: YUVAN SUNDAR S
RegisterNumber: 212223040250 
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Hours': eval(input()),
    'Marks': eval(input())
    }

df = pd.DataFrame(data)

X = df[['Hours']]  
y = df['Marks']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Hours')
plt.ylabel('Marks')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()


```

## Output:
![Screenshot 2024-08-28 090619](https://github.com/user-attachments/assets/2883d23e-e7e5-473e-8d2f-36253ef51dc4)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
