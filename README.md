# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

# Startup Profit Prediction Algorithm (4-Step)

## Step 1: Data Preparation
- Import necessary libraries: `numpy`, `pandas`, `StandardScaler`, `LinearRegression`.
- Load the dataset `50_Startups.csv`.
- Extract features (`R&D Spend`, `Administration`, `Marketing Spend`) and target (`Profit`).

## Step 2: Feature Scaling
- Create two `StandardScaler` instances: one for features, one for the target.
- Fit and transform the features and target using their respective scalers.

## Step 3: Model Training
- Initialize a `LinearRegression` model.
- Train the model using the scaled features and target.

## Step 4: Prediction
- Scale new input data using the feature scaler.
- Predict the scaled target using the trained model.
- Inverse transform the prediction to original scale.
- Print the predicted profit value.

## Program:
```python
#Program to implement the linear regression using gradient descent.
#Developed by: B Surya Prakash
#RegisterNumber: 212224230281

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):

    X= np.c_[np.ones(len(X1)), X1]

    theta = np.zeros(x.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (x).dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1,1)
        theta -= learning_rate * (1/ len(X1)) * X.T.dot(errors)
    return theta

```
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv("50_Startups.csv")
print(data.head())
X = data.iloc[:, :-2].values.astype(float)  
y = data.iloc[:, -1].values.astype(float).reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X1_Scaled = scaler_X.fit_transform(X)
Y1_Scaled = scaler_y.fit_transform(y)

print('Name:B Surya Prakash')
print('Register No:212224230281')
print(X1_Scaled)
print(Y1_Scaled)

model = LinearRegression()
model.fit(X1_Scaled, Y1_Scaled)

new_data = np.array([[165349.2,136897.8,471784.1]])
new_Scaled = scaler_X.transform(new_data)
prediction = model.predict(new_Scaled)
pre = scaler_y.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:

<img width="990" height="125" alt="image" src="https://github.com/user-attachments/assets/8136084c-b37a-4d27-b001-5391311d11a9" />

<img width="997" height="327" alt="image" src="https://github.com/user-attachments/assets/8417534a-5db2-4277-beb4-54c6956861e4" />

<img width="1004" height="163" alt="image" src="https://github.com/user-attachments/assets/b01dd161-99a8-4559-a731-f235216fbc46" />

<img width="1016" height="38" alt="image" src="https://github.com/user-attachments/assets/3fb7d1cb-2209-4457-8a86-3d29414d58b6" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
