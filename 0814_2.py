import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score


data = pd.read_csv('./data/2.iris.csv')
header=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

array = data.values

X=data.iloc[:, :-1].values
Y=data.iloc[:, -1].values

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X=scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

model = LogisticRegression(max_iter=200)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

fold = KFold(n_splits=10, shuffle=True, random_state=0)
acc=cross_val_score(model, rescaled_X, Y, cv=fold, scoring='accuracy')

conf_matrix = confusion_matrix(Y_pred, Y_test)
print(conf_matrix)
print(classification_report(Y_pred, Y_test))

# KFold 교차검증 정확도 분포 시각화
plt.figure(figsize=(8, 6))
plt.hist(acc, bins=10, edgecolor='black', color='skyblue')
plt.title('Distribution of Cross-validation Accuracy Scores')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

