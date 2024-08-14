import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

header = ['preg', 'plas', 'pres', 'skin', 'est', 'mass', 'pedi', 'age', 'class']
data= pd.read_csv("./data/pima-indians-diabetes.data.csv")

array = data.values
X=array[:, 0:8]
Y=array[:,8]

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X=scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_pred_binary = (y_pred>0.5).astype(int)

acc = accuracy_score(y_pred_binary, Y_test)
print(acc)

df_Y_test - pd.DataFrame(Y_test)
df_Y_pred_binary = pd.DataFrame(y_pred_binary)

plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, clolr='blue', label='Actual Values', marker='o')
plt.scatter(rangelen(Y_pred_binary), color='red', label='Predictted Values', marker='x')

plt.clf()
plt.title ('Comparison of Actual and Predicted values')
plt.xlabel('index')
plt.ylabel('Class(0 or 1)')
plt.legend()
plt.show()
plt.savefig('./result/scatter.png')