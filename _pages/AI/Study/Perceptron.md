---
title: "Perceptron"
date: "2025-06-25"
thumbnail: "/assets/img/thumbnail/opencv.png"
---

# Code
---
**Perceptron Model**
```
# 퍼셉트론 모델
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 모델 구조 (Model Architecture) ---
# Perceptron 클래스 정의: 단층 퍼셉트론의 구조와 학습 매개변수를 초기화합니다.
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10000):
        # 가중치 초기화: 입력 크기만큼의 0으로 구성된 배열 (각 입력 특성에 대응)
        self.weights = np.zeros(input_size)
        # 편향(bias) 초기화: 단일 값
        self.bias = 0
        # 학습률(learning rate)과 에폭(epochs) 수 초기화
        self.lr = lr
        self.epochs = epochs
        # 에폭별 총 오차 (잘못 분류된 샘플 수)를 저장할 리스트
        self.errors = []

    # --- 2. 활성화 함수 (Activation Function) ---
    # 계단 함수 (Step Function): 이진 분류에 사용되는 선형 활성화 함수
    # 입력 x가 0보다 크면 1, 아니면 0을 반환합니다.
    def activation(self, x):
        return np.where(x > 0, 1, 0)

    # --- 3. 예측 (순전파 - Forward Propagation) ---
    # 주어진 입력에 대해 퍼셉트론의 최종 출력을 계산합니다.
    def predict(self, x):
        # 가중치와 입력의 내적 (선형 결합) + 편향
        linear_output = np.dot(x, self.weights) + self.bias
        # 활성화 함수를 적용하여 최종 예측값 반환
        return self.activation(linear_output)

    # --- 4. 학습 알고리즘 (Training Algorithm) ---
    # 퍼셉트론 학습 규칙 (Perceptron Learning Rule)을 사용하여 가중치와 편향을 업데이트합니다.
    def train(self, X, y):
        # 지정된 에폭(epoch) 수만큼 반복 학습
        for epoch in range(self.epochs):
            total_error = 0 # 현재 에폭에서 잘못 분류된 샘플 수를 세기 위한 변수

            # 모든 학습 데이터 샘플에 대해 예측 및 가중치 업데이트 수행
            for xi, target in zip(X, y):
                prediction = self.predict(xi) # 현재 입력에 대한 예측값 계산
                
                # 가중치 및 편향 업데이트 양 계산
                # (목표값 - 예측값) * 학습률. 예측이 틀렸을 경우에만 0이 아닌 값을 가집니다.
                update = self.lr * (target - prediction)
                
                # 가중치 업데이트: 오차와 입력값에 비례하여 조정
                self.weights += update * xi
                # 편향 업데이트: 오차에 비례하여 조정
                self.bias += update
                
                # 업데이트가 발생했는지 (즉, 예측이 틀렸는지) 확인하여 total_error에 추가
                total_error += int(update != 0.0) # 0.0이 아니면 (업데이트가 있었다면) 1을 더함
            
            # 현재 에폭의 총 오차 (잘못 분류된 샘플 수)를 저장
            self.errors.append(total_error)
            # 현재 에폭과 총 오차를 출력 (학습 진행 상황 모니터링)
            print(f"Epoch {epoch + 1}/{self.epochs} - Total Error: {total_error}")  

# 결정 경계 시각화
from matplotlib.colors import ListedColormap

def plot_decision_boundary(X, y, model):
  cmap_light = ListedColormap(['#FFAAAA','#AAAAFF'])
  cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

  h = .02 # mesh grid 간격
  x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
  y_min, y_max = X[:,0].min() - 1, X[:,0].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  plt.figure(figsize=(8,6))
  plt.contourf(xx, yy, Z, cmap=cmap_light)

  # 실제 데이터 포인트 표시
  plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap_bold, edgecolor='k', s=100, marker='o')
  plt.xlabel('Input 1')
  plt.ylabel('Input 2')
  plt.title('Perceptron Decision Boundary')
  plt.show()

# 오류 시각화
plt.figure(figsize=(8,5))
plt.plot(range(1, len(ppn_and.errors) + 1), ppn_and.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of Errors')
plt.title('Perceptron Learning Error Over Epochs (AND Gate)')
plt.grid(True)
plt.show()
```
**AND Gate**
```
# AND 게이트 데이터
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0,0,0,1])

# 퍼셉트론 모델 훈련
ppn_and = Perceptron(input_size=2)
ppn_and.train(X_and, y_and)

#예측 결과 확인
print("\nAND Gate Test:")
for x in X_and:
  print(f"Input: {x}, Predicted Output: {ppn_and.predict(x)}")

# AND 게이트 결정 경계 시각화
plot_decision_boundary(X_and, y_and, ppn_and)
```
**OR Gate**
```
# OR 게이트 데이터
X_or = np.array([[0,0],[0,1],[1,0],[1,1]])
y_or = np.array([0,1,1,1])

# OR 퍼셉트론 모델 훈련
ppn_or = Perceptron(input_size=2)
ppn_or.train(X_or, y_or)

#예측 결과 확인
print("\nOR Gate Test:")
for x in X_or:
    print(f"Input: {x}, Predicted Output: {ppn_or.predict(x)}")

# OR 게이트 결정 경계 시각화
plot_decision_boundary(X_and, y_and, ppn_and)
```
**NAND Gate**
```
# NAND 게이트 데이터
X_nand = np.array([[0,0],[0,1],[1,0],[1,1]])
y_nand = np.array([1,1,1,0])

# NAND 퍼셉트론 모델 훈련
ppn_nand = Perceptron(input_size=2)
ppn_nand.train(X_nand, y_nand)

# 예측 결과 확인
print("\nNAND Gate Test:")
for x in X_nand:
    print(f"Input: {x}, Predicted Output: {ppn_nand.predict(x)}")

# NAND 게이트 결정 경계 시각화
plot_decision_boundary(X_and, y_and, ppn_and)
```
**XOR Gate**
```
# XOR 게이트 데이터
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

# XOR 퍼셉트론 모델 훈련
ppn_xor = Perceptron(input_size=2, epochs=20)
ppn_xor.train(X_xor, y_xor)

# 예측 결과 확인
print("\nXOR Gate Test:")
for x in X_xor:
    print(f"Input: {x}, Predicted Output: {ppn_xor.predict(x)}")

# XOR 게이트 결정 경계 시각화
plot_decision_boundary(X_and, y_and, ppn_and)
```
# 분석
---
**AND Gate**
>![alt text](../../assets/img/opencv/perceptron/and_result.png)예상결과<br/>
![alt text](../../assets/img/opencv/perceptron/and_d_b.png)게이트 결정 경계 시각화<br/>
![alt text](../../assets/img/opencv/perceptron/and_err.png)오류 시각화

<br/>

**OR Gate**
>![alt text](../../assets/img/opencv/perceptron/or_result.png)예상결과<br/>
![alt text](../../assets/img/opencv/perceptron/or_d_b.png)게이트 결정 경계 시각화<br/>
![alt text](../../assets/img/opencv/perceptron/or_err.png)오류 시각화

<br/>

**NAND Gate**
>![alt text](../../assets/img/opencv/perceptron/nand_result.png)예상결과<br/>
![alt text](../../assets/img/opencv/perceptron/nand_d_b.png)게이트 결정 경계 시각화<br/>
![alt text](../../assets/img/opencv/perceptron/nand_err.png)오류 시각화

<br/>

**XOR Gate**
>![alt text](../../assets/img/opencv/perceptron/xor_result.png)예상결과<br/>
![alt text](../../assets/img/opencv/perceptron/xor_d_b.png)게이트 결정 경계 시각화<br/>
![alt text](../../assets/img/opencv/perceptron/xor_err.png)오류 시각화

# 고찰
---
**XOR Gate**
퍼셉트론은 가장 기본적인 형태의 인공 신경망으로, 입력을 받아 가중치를 곱한 후 활성화 함수를 적용하여 출력을 계산.
하지만 이 단일 계층 구조로는 XOR 게이트 문제를 해결할 수 없음.

|입력A|입력B|출력|
|:---:|:---:|:---:|
|0|0|1|
|0|1|1|
|1|0|1|
|1|1|0|

이 문제는 선형적으로 분리할 수 없는 데이터이기 때문에, 단일 퍼셉트론으로는 어떠한 선형 결정 경계로도 올바른 출력을 분리해낼 수 없음.

**MLP(Multi Layer Perceptron)의 필요성**
비선형적으로 분리 가능한 데이털르 해결하기 위해서는 하나 이상의 은닉층을 갖는 다층 퍼셉트론이 필요함. 은닉층을 통해 비선형성을 학습할 수 있고, 이로 인해 더 복잡한 함수도 근사할 수 있게 됨.

**이진 분류와 손실 함수**
출력이 0 또는 1인 이진 분류 문제에서는, 손실 함수로 Binary Cross Entropy를 사용함.
이 손실 함수는 예측값과 실제값 사이의 차이를 측정하며, 모델의 출력이 정답에 가까워질수록 손실 값은 0에 수렴.

**Gradient Descent를 통한 학습**
손실 함수 값을 줄이기 위해 경사하강법(gradient descent)을 사용하여 가중치를 업데이트함. 이때 중요한 요소는 학습률.
>학습률이 너무 작으면, 학습 속도가 매우 느려지고 극소에 갇힐 수 있음.

>학습률이 너무 크면, 최적점을 지나치거나 발산하여 학습이 실패할 수 있음.

따라서 적절한 학습률 설정이 모델 성능에 큰 영향을 미침.

**활성화 함수: ReLU와 Sigmoid**
>ReLU<br/>
- 계산이 빠르고
- 비선형성을 유지할 수 있어 딥러닝에 널리 쓰임.

>Sigmoid<br/>
- 출력이 0과 1 사이의 확률로 해석 될 수 있음.

**Epoch와 Overfitting 문제**
모델을 학습할 때 전체 데이터셋을 여러 번 반복해서 학습하는 과정을 Epoch이라고 함.
하지만 너무 많은 Epoch을 사용하면 모델이 학습 데이터에 과도하게 적응하면서 과적합이 발생할 수 있음.
이를 방지하기 위해 다음과 같은 기법을 사용할 수 있음.
>Dropout<br/>
- 학습 중 일부 뉴런을 임의로 비활성화시켜 과적합 방지.

>Early Stopping<br/>
- 검증 손실이 더 이상 개선되지 않으면 학습을 조기 종료.

**XOR Gate by MLP**
```
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 모델 구조 (Model Architecture) ---
# MLP 클래스 정의: 신경망의 전체 구조와 학습 매개변수를 초기화합니다.
class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.05, epochs=50000):
        # 학습률(learning rate)과 에폭(epochs) 수 초기화
        self.lr = lr
        self.epochs = epochs
        # 에폭별 총 오차를 저장할 리스트
        self.errors = []

        # 가중치 초기화: 입력층 -> 은닉층
        # np.random.randn()으로 작은 난수로 초기화하여 학습 초기에 기울기 소실/폭주 방지
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        # 은닉층 편향 초기화: 0으로 초기화
        self.bias_hidden = np.zeros((1, hidden_size))

        # 가중치 초기화: 은닉층 -> 출력층
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        # 출력층 편향 초기화: 0으로 초기화
        self.bias_output = np.zeros((1, output_size))

    # --- 2. 활성화 함수 (Activation Function) ---
    # 시그모이드(Sigmoid) 함수: 비선형성을 도입하여 복잡한 패턴 학습 가능하게 함
    def sigmoid(self, x):
        # np.clip으로 입력값을 제한하여 np.exp() 계산 시 발생할 수 있는 오버플로우(overflow) 방지
        x = np.clip(x, -500, 500)
        # 시그모이드 함수 정의: 1 / (1 + e^-x). 출력값이 0과 1 사이
        return 1 / (1 + np.exp(-x))

    # 시그모이드 함수의 미분 (기울기 계산): 역전파(Backpropagation)에 필수
    # f'(x) = f(x) * (1 - f(x)) 임을 활용 (여기서 x는 이미 시그모이드 함수를 통과한 값)
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # --- 3. 예측 (순전파 - Forward Propagation) ---
    # 주어진 입력에 대해 신경망의 최종 출력을 계산
    def predict(self, x):
        # 입력 x를 (1, input_size) 형태의 2차원 배열로 변환하여 행렬 연산 준비
        x = x.reshape(1, -1)

        # 1. 입력층 -> 은닉층 계산
        # 입력 x와 input_hidden 가중치 행렬의 내적(dot product) 후 은닉층 편향을 더함
        hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        # 은닉층 활성화: 시그모이드 함수를 통과시켜 비선형성을 부여
        hidden_output = self.sigmoid(hidden_input)

        # 2. 은닉층 -> 출력층 계산
        # 은닉층 출력과 hidden_output 가중치 행렬의 내적 후 출력층 편향을 더함
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        # 출력층 활성화: 시그모이드 함수를 통과시켜 최종 출력 생성
        output = self.sigmoid(output_input)
        
        # 예측 결과를 스칼라 값 (단일 숫자)으로 반환
        return output[0, 0]

    # --- 4. 학습 알고리즘 (Training Algorithm) ---
    # 순전파(Forward Propagation) 및 역전파(Backward Propagation)를 포함
    def train(self, X, y):
        # 목표 레이블 y를 (데이터 개수, 1) 형태로 변환하여 행렬 연산 준비
        y = y.reshape(-1, 1)

        # 지정된 에폭(epoch) 수만큼 반복 학습
        for epoch in range(self.epochs):
            total_error = 0 # 현재 에폭의 총 오차를 계산하기 위한 변수

            # 모든 학습 데이터 샘플에 대해 순전파 및 역전파 수행
            for xi, target in zip(X, y):
                # 단일 입력 xi와 목표값 target을 2차원 배열로 변환 (행렬 연산 준비)
                xi = xi.reshape(1, -1)     # 예: (1, 2)
                target = target.reshape(1, -1) # 예: (1, 1)

                # --- 4-1. 순방향 전파 (Forward Pass): 각 층의 입력 및 출력 값 저장 ---
                # 은닉층 계산
                # 입력 xi와 입력-은닉 가중치 행렬의 내적 후 은닉층 편향을 더함 (가중 합)
                hidden_input = np.dot(xi, self.weights_input_hidden) + self.bias_hidden
                # 활성화 함수인 시그모이드 함수를 통과하여 은닉층의 출력 생성
                hidden_output = self.sigmoid(hidden_input)

                # 출력층 계산
                # 은닉층 출력과 은닉-출력 가중치 행렬의 내적 후 출력층 편향을 더함 (가중 합)
                output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
                # 활성화 함수를 통과하여 최종 출력 생성
                output = self.sigmoid(output_input)

                # 오류 계산: MSE(Mean Squared Error)를 사용하여 예측과 정답 사이의 오차 계산
                error = target - output
                # 현재 샘플의 오차 제곱을 총 오차에 더함 (손실 함수)
                total_error += np.sum(error ** 2)

                # --- 4-2. 역방향 전파 (Backward Pass): 오류를 역전파하여 기울기 계산 ---
                # 1. 출력층의 오류 기울기(델타) 계산: (오차 * 출력층 활성화 함수의 미분값)
                delta_output = error * self.sigmoid_derivative(output)
                
                # 2. 은닉층의 오류 기울기(델타) 계산 (★★핵심★★)
                # 출력층의 델타를 은닉층 -> 출력층 가중치(self.weights_hidden_output.T)를 통해 역방향으로 전파
                hidden_error = np.dot(delta_output, self.weights_hidden_output.T)
                # 전파된 오차에 은닉층 활성화 함수의 미분값을 곱하여 최종 은닉층 델타 계산
                delta_hidden = hidden_error * self.sigmoid_derivative(hidden_output)

                # --- 4-3. 가중치 & 편향 업데이트: 계산된 기울기를 사용하여 업데이트 ---
                # 은닉층 -> 출력층 가중치 업데이트
                # hidden_output.T (은닉층 출력의 전치)와 delta_output의 행렬 곱을 학습률과 곱하여 가중치 조정
                self.weights_hidden_output += self.lr * np.dot(hidden_output.T, delta_output)
                # 출력층 편향 업데이트
                self.bias_output += self.lr * delta_output

                # 입력층 -> 은닉층 가중치 업데이트
                # xi.T (입력의 전치)와 delta_hidden의 행렬 곱을 학습률과 곱하여 가중치 조정
                self.weights_input_hidden += self.lr * np.dot(xi.T, delta_hidden)
                # 은닉층 편향 업데이트
                self.bias_hidden += self.lr * delta_hidden

            # 에폭별 총 오차를 저장
            self.errors.append(total_error)
            # 1000 에폭마다 현재 오차 출력 (학습 진행 상황 모니터링)
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - Total Error: {total_error:.6f}")

# XOR 학습
mlp_xor = SimpleMLP()
mlp_xor.train(X_xor, y_xor)

# 예측 결과
print("\nXOR Gate Test:")
for x in X_xor:
    pred = mlp_xor.predict(np.array([x]))[0][0]
    print(f"Input: {x}, Predicted Output: {pred}")

# 결정 경계 시각화 함수
def plot_decision_boundary(X, y, model):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    h = .02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=cmap_bold, edgecolor='k', s=100)
    plt.title("MLP Decision Boundary (XOR Gate)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

# 결정 경계 시각화
plot_decision_boundary(X_xor, y_xor, mlp_xor)

# 손실 시각화
plt.figure(figsize=(8, 5))
plt.plot(mlp_xor.losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Time (XOR Gate)")
plt.grid(True)
plt.show()
```
>![alt text](../../assets/img/opencv/perceptron/xor_mlp_result.png)예상결과<br/>
![alt text](../../assets/img/opencv/perceptron/xor_mlp_d_b.png)게이트 결정 결계 시각화<br/>
![alt text](../../assets/img/opencv/perceptron/xor_loss.png)손실 시각화
