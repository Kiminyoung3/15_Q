import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 파일 불러오기
data = pd.read_csv('./data/1.salary.csv')

# 데이터 전처리
X = data.iloc[:, 0].values.reshape(-1, 1)  # 독립변수
Y = data.iloc[:, 1].values  # 종속변수

# 훈련 세트와 테스트 세트로 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 모델 학습
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

# 전체 데이터에 대한 회귀선 추가
plt.figure(figsize=(10, 6))

# 실제 데이터 산점도
plt.scatter(X, Y, color='pink', label='Actual Data Points', marker='x', s=30, alpha=0.5)

# 회귀선 추가
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Fitted Line')

# 테스트 데이터와 예측 결과를 비교하기 위해 산점도 추가
plt.scatter(X_test, Y_test, color='blue', label='Test Data Points', marker='o')
plt.plot(X_test, y_pred, color='purple', linestyle='--', label='Predicted Values')

# 그래프에 제목과 레이블 추가
plt.title("Scatter Plot of Salary vs. Experience Years")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

# 결과를 파일로 저장
plt.savefig("./result/scatter2.png")

# 그래프 보여주기
plt.show()
