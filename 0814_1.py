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
#헤더 이미 있으므로 따로 설정할 필요 없음
#header = ['Experience Years', 'Salary']

#데이터 프레임 제거 및 독립/종속변수 설정
array = data.values
#print(array)

#데이터 독립변수 슬라이싱_변수의 개수 확인하도록
X=array[:, 0] #독립변수. [행, 열] 형식이다. [:, 0]은 모든행과 첫 번째 열만 불러오겠다는 뜻
Y=array[:, 1] #종속변수

#모델 예측할 때 2열 이상을 가져오도록 되어있는데 이번 데이터에는 열이 1개이므로 임의로 행렬을 변경해줌.
X=X.reshape(-1, 1)

#소수점으로 통일. 지금은 필요없는 작업
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_scaled = scaler.fit_transform(X)

#데이터 분할(Train/Test) ->컴퓨터가 학습할 할당량을 제공하는 과정.
#train: 학습시킴
#test: 학습 확인(test_size=0.2는 전체의 20%만 test로 확인하겠다는 뜻. 40개의 데이터 중 8개만 뽑게 된다.)
#train: 20%를 제외한 나머지 80%를 학습에 사용하게 됨.
# ㄴ그래서 코드를 재생할때마다 랜덤으로 8개를 뽑아 테스트에 사용하기때문에 매번 그래프 모양이 달라진다.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)

#그래프로 그려지게 되는 방정식을 찾아내게 된 것. y=ax+b에서 a값과 b값을 찾아낸 것이다.
model.coef_
model.intercept_

# 예측. 컴퓨터가 학습한 결과 확인. 찾아낸 방정식에 따라 근속연수에 따른 연봉 예측하도록 X_test값을 넣어줌
y_pred = model.predict(X_test)
#예측값과 실제값의 괴리가 얼마인지 절댓값 평균을 통해 예측하는 것. 값이 작을수록 오차가 작으므로 좋다.
error = mean_absolute_error(y_pred, Y_test)
print(error)

fig, ax=plt.subplots()
plt.clf()
plt.scatter(X, Y, label="Actual Data Points", color="green", marker="x", s=30, alpha=0.5)
plt.title("Actual Data Points")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

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

#여기에 초기화 넣으면 안돼~~~그래프 안나온다.
plt.title("Scatter Plot of Salary vs. Experience Years")
plt.xlabel("Experience Years")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)

# 결과를 파일로 저장
plt.savefig("./result/scatter2.png")

# 그래프 보여주기
plt.show()