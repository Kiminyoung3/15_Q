import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#파일 불러오기
data = pd.read_csv('./data/1.salary.csv')
# header = ['Experience Years', 'Salary']->헤더 이미 있으므로 따로 설정할 필요 없음

#데이터 전처리:Min-Max 스케일링
array = data.values

#데이터 독립변수 슬라이싱_변수의 개수 확인하도록
X=array[:, 0] #독립변수
Y=array[:, 1] #종속변수

fig, ax=plt.subplots()
plt.clf()
plt.scatter(X, Y, label="Actual Data Points", color="green", marker="x", s=30, alpha=0.5)
plt.title("Actual Data Points")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

#소수점으로 통일. 지금은 필요없는 작업
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_scaled = scaler.fit_transform(X)

X1=X.reshape(-1, 1)
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
mae = mean_absolute_error(Y_test, y_pred)
mse = mean_squared_error(Y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R^2): {r2:.2f}")

plt.figure(figsize=(10, 6))

plt.scatter(range(len(Y_test)), Y_test, color='green', label='Actual Values', marker='o')

plt.plot(range(len(y_pred)), y_pred, color='red', label='predictted Values', marker='*')

plt.title("Scatter Plot of Salary vs. Experience Years")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

# 결과를 파일로 저장
plt.savefig("./result/scatter2.png")

# 그래프 보여주기
plt.show()