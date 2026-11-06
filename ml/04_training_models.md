# 4장. 모델 훈련

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `04_training_linear_models.ipynb`

## 이 장을 시작하기 전에

2장과 3장에서는 `model.fit(X, y)` 한 줄로 모델을 훈련했습니다. 그런데 `fit()` 안에서는 실제로 무슨 일이 벌어질까요? 4장은 바로 그 **상자 안을 열어 보는 장**입니다.

내부를 알면 좋은 점이 세 가지 있습니다.

1. **알맞은 모델과 하이퍼파라미터를 빨리 고를 수 있습니다.** 학습률을 왜 줄여야 하는지, 왜 스케일링이 필요한지 이유를 알게 됩니다.
2. **문제가 생겼을 때 원인을 찾을 수 있습니다.** 손실이 발산하거나, 과대적합되거나, 학습이 안 끝날 때 무엇을 바꿔야 할지 보입니다.
3. **신경망의 기초가 됩니다.** 여기서 배우는 **경사 하강법**은 10장 이후 신경망 학습에서 그대로 쓰입니다.

이 장의 흐름을 한 문장으로 말하면 이렇습니다.

> 가장 단순한 **선형 회귀**를 두 가지 방식(공식으로 한 번에 / 조금씩 내려가며)으로 풀어 보고 → 곡선 데이터에 쓰는 **다항 회귀**로 넓히고 → 모델이 너무 단순한지 복잡한지 **진단**하는 법을 배우고 → 과대적합을 막는 **규제**를 걸고 → 마지막으로 같은 아이디어를 **분류**에 적용합니다.

### 실습 데이터

이 장은 대부분 **직접 만든 가짜 데이터**를 씁니다. 정답(데이터를 만든 식)을 우리가 알고 있으니, 모델이 그 정답을 잘 찾아내는지 바로 확인할 수 있다는 장점이 있습니다.

| 데이터 | 만드는 식 | 쓰이는 곳 |
|---|---|---|
| 선형 데이터 (200개) | `y = 4 + 3x + 잡음`, x는 0~2 | 4.1, 4.2 |
| 2차 데이터 (200개) | `y = 0.5x² + x + 2 + 잡음`, x는 −3~3 | 4.3, 4.4, 조기 종료 |
| 작은 선형 데이터 (20개) | `y = 1 + 0.5x + 잡음/1.5`, x는 0~3 | 4.5 릿지·라쏘·엘라스틱넷 |
| 붓꽃(iris) | 사이킷런 내장, 150송이, 3품종 | 4.6 로지스틱 회귀 |

## 목차

0. [미리 알아 둘 기호와 용어](#0-미리-알아-둘-기호와-용어)
1. [선형 회귀 (4.1)](#1-선형-회귀-41)
2. [경사 하강법 (4.2)](#2-경사-하강법-42)
3. [다항 회귀 (4.3)](#3-다항-회귀-43)
4. [학습 곡선 (4.4)](#4-학습-곡선-44)
5. [규제가 있는 선형 모델 (4.5)](#5-규제가-있는-선형-모델-45)
6. [로지스틱 회귀 (4.6)](#6-로지스틱-회귀-46)
7. [연습문제](#7-연습문제)
8. [전체 흐름 요약](#8-전체-흐름-요약)

---

## 0. 미리 알아 둘 기호와 용어

수식이 나오면 겁먹기 쉬운데, 이 장에서 쓰는 기호는 몇 개 안 됩니다. 먼저 정리해 두겠습니다.

| 기호 | 읽는 법 | 뜻 | 이 장의 예 |
|---|---|---|---|
| `m` | 엠 | 샘플(데이터 행) 개수 | 200 |
| `n` | 엔 | 특성(데이터 열) 개수 | 1 (x 하나) |
| `x` | 엑스 | 입력 특성 | 꽃잎 너비 |
| `y` | 와이 | 정답(타깃) | 집값, 품종 |
| `ŷ` | 와이 햇 | 모델의 **예측값** (모자 = 추정했다는 뜻) | |
| `θ` | 세타 | 모델 **파라미터**. 훈련으로 찾아야 하는 값 | 절편 θ₀, 기울기 θ₁ |
| `η` | 에타 | **학습률**. 한 번에 얼마나 움직일지 | 0.1 |
| `α` | 알파 | **규제 강도** | 0.1 |
| `Σ` | 시그마 | 모두 더하기 | |
| `Xᵀ` | 엑스 전치 | 행렬의 행과 열을 뒤바꾼 것 (코드로 `X.T`) | |
| `@` | (코드) | 행렬 곱셈 | `X @ theta` |

**자주 나오는 용어**

| 용어 | 쉬운 설명 |
|---|---|
| **비용 함수 (손실 함수)** | 모델이 얼마나 **틀렸는지**를 숫자 하나로 나타낸 것. 작을수록 좋음. 훈련 = 이 숫자를 줄이는 θ 찾기 |
| **그레이디언트 (기울기)** | 파라미터를 조금 바꿨을 때 비용이 얼마나, 어느 방향으로 바뀌는지 |
| **에포크 (epoch)** | 훈련 데이터 전체를 한 바퀴 다 보는 것 |
| **과소적합** | 모델이 너무 단순해서 훈련 데이터조차 잘 못 맞힘 |
| **과대적합** | 모델이 훈련 데이터를 **외워 버려서** 새 데이터에서 성능이 떨어짐 |

---

## 1. 선형 회귀 (4.1)

> **이 절에서 하는 일**: 점들 사이를 가장 잘 지나가는 **직선 하나**를 공식으로 한 번에 구합니다.

### 1-1. 선형 회귀는 어떤 모델인가

중학교 때 배운 일차 함수 `y = ax + b`를 떠올리면 됩니다. 선형 회귀는 이 a(기울기)와 b(절편)를 데이터로부터 찾아내는 모델입니다.

머신러닝에서는 기호만 바꿔서 이렇게 씁니다.

```
ŷ = θ₀ + θ₁x₁
```

- `θ₀` = 절편 (b) — 편향(bias)이라고도 부릅니다
- `θ₁` = 기울기 (a) — 가중치(weight)라고 부릅니다

특성이 여러 개라면 그냥 계속 더하면 됩니다.

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ
```

예를 들어 집값을 예측한다면 `집값 = θ₀ + θ₁×방 개수 + θ₂×면적 + …` 같은 식입니다. 각 특성에 **중요도(가중치)를 곱해서 더한다**고 생각하면 됩니다.

> 💡 **편리한 트릭**: 모든 샘플에 값이 항상 1인 특성 `x₀ = 1`을 하나 추가하면, `θ₀`도 `θ₀ × 1`로 쓸 수 있습니다. 그러면 식 전체를 `ŷ = θ₀x₀ + θ₁x₁ + …` 처럼 **모두 같은 모양**으로 쓸 수 있고, 행렬 곱 한 번(`X @ θ`)으로 계산이 끝납니다. 아래 코드의 `add_dummy_feature`가 바로 이 일을 합니다.

### 1-2. "가장 잘 맞는 직선"의 기준: MSE

직선은 무수히 많이 그릴 수 있습니다. 어떤 직선이 "가장 좋은" 직선일까요?

각 점에서 **직선까지의 세로 거리(오차)**를 재고, 그 오차를 **제곱해서 평균**낸 값이 가장 작은 직선을 고릅니다. 이것이 **MSE(평균 제곱 오차)**입니다.

```
MSE = (1/m) × Σ (예측값 − 정답)²
```

왜 그냥 더하지 않고 제곱할까요?
- 그냥 더하면 +3과 −3이 서로 상쇄되어 0이 되어 버립니다. 제곱하면 모두 양수가 됩니다.
- 제곱하면 **큰 오차에 더 큰 벌점**이 붙습니다. 오차 2는 4, 오차 10은 100이 됩니다.
- 수학적으로 **미분하기 쉽습니다.** 이게 뒤에서 아주 중요해집니다.

> 💡 2장에서는 성능을 **RMSE**(MSE에 루트를 씌운 것)로 평가했는데 왜 여기선 MSE를 쓸까요? 루트는 값이 커지면 같이 커지는 함수라서, **MSE가 가장 작은 θ와 RMSE가 가장 작은 θ는 똑같습니다.** 그렇다면 계산이 더 간단한 MSE를 쓰는 게 이득입니다.

### 1-3. 데이터 만들기

```python
import numpy as np

rng = np.random.default_rng(seed=42)
m = 200  # 샘플 수
X = 2 * rng.random((m, 1))                     # 0~2 사이 값 200개
y = 4 + 3 * X + rng.standard_normal((m, 1))    # 정답 = 4 + 3x + 잡음
```

**한 줄씩 읽기**

1. `rng = np.random.default_rng(seed=42)`
   난수(무작위 수) 생성기를 만듭니다. `seed=42`로 고정하면 **실행할 때마다 똑같은 난수**가 나와서 결과를 재현할 수 있습니다. 42는 관례일 뿐 특별한 의미는 없습니다.

2. `X = 2 * rng.random((m, 1))`
   `rng.random((200, 1))`은 0 이상 1 미만의 수를 200행 1열로 만듭니다. 여기에 2를 곱하면 0~2 사이가 됩니다.
   왜 `(200,)`이 아니라 `(200, 1)`일까요? 사이킷런은 X를 **"행 = 샘플, 열 = 특성"인 2차원 표**로 받기 때문입니다. 특성이 1개뿐이어도 열이 1개인 표로 만들어야 합니다.

3. `y = 4 + 3 * X + rng.standard_normal((m, 1))`
   진짜 관계는 **절편 4, 기울기 3**입니다. 여기에 평균 0, 표준편차 1인 정규분포 잡음을 더해 현실 데이터처럼 흩어지게 만듭니다.
   우리의 목표는 모델이 **4와 3에 최대한 가까운 값을 찾아내는 것**입니다.

### 1-4. 방법 ①: 정규 방정식 — 공식으로 한 번에 풀기

MSE를 가장 작게 만드는 θ를 **딱 한 번의 계산으로** 구하는 공식이 이미 알려져 있습니다. 이걸 **정규 방정식**이라고 합니다.

```
θ = (XᵀX)⁻¹ Xᵀ y
```

공식을 외울 필요는 없습니다. "**MSE를 미분해서 0이 되는 지점을 풀면 이 식이 나온다**" 정도만 알아 두세요. 고등학교 때 이차 함수의 최솟값을 미분 = 0으로 찾았던 것과 같은 원리입니다.

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # 모든 샘플 앞에 1을 붙임 (x0 = 1)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
theta_best
# array([[3.69084138],
#        [3.32960458]])
```

**한 줄씩 읽기**

1. `add_dummy_feature(X)`
   `X`의 모양은 `(200, 1)`입니다. 여기에 1로 채운 열을 맨 앞에 붙여서 `(200, 2)`로 만듭니다.
   ```
   X          X_b
   [[1.55]    [[1.  1.55]
    [0.88]  →  [1.  0.88]
    ...]       ...]
   ```
2. `X_b.T` — 전치(행과 열 바꾸기)
3. `np.linalg.inv(...)` — 역행렬 (숫자로 치면 "나누기"에 해당)
4. `@` — 행렬 곱셈

**결과 해석**

θ₀ = 3.69, θ₁ = 3.33이 나왔습니다. 정답 (4, 3)과 조금 다른 이유는 **잡음** 때문입니다. 데이터 200개가 무작위로 흩어져 있어서, 그 데이터만 보고는 정확히 4와 3을 알아낼 수 없습니다. 데이터가 더 많을수록 정답에 가까워집니다.

**예측해 보기**

```python
X_new = np.array([[0], [2]])          # x = 0, x = 2 두 점
X_new_b = add_dummy_feature(X_new)    # 앞에 1 붙이기
y_predict = X_new_b @ theta_best
y_predict
# array([[ 3.69084138],     ← x=0일 때: 3.69 + 3.33×0
#        [10.35005055]])    ← x=2일 때: 3.69 + 3.33×2
```

직선은 점 두 개만 있으면 그릴 수 있으니, 양 끝 x=0과 x=2만 예측했습니다.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")  # 빨간 실선: 예측 직선
plt.plot(X, y, "b.")                                    # 파란 점: 실제 데이터
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])   # 보이는 범위 [x최소, x최대, y최소, y최대]
plt.grid()
plt.legend(loc="upper left")
plt.show()
```

`"r-"`는 red + 실선, `"b."`는 blue + 점이라는 뜻의 짧은 서식 문자열입니다.

### 1-5. 사이킷런으로 하면?

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
# (array([3.69084138]), array([[3.32960458]]))

lin_reg.predict(X_new)
# array([[ 3.69084138],
#        [10.35005055]])
```

직접 계산한 값과 **소수점까지 똑같습니다.** 차이점은 두 가지입니다.
- 사이킷런은 절편(`intercept_`)과 기울기(`coef_`)를 **따로 저장**하므로 `add_dummy_feature`를 할 필요가 없습니다.
- 속성 이름 끝의 `_`는 사이킷런의 약속으로, "**훈련해서 알아낸 값**"이라는 표시입니다.

### 1-6. 사이킷런은 사실 다른 방법을 쓴다: 유사역행렬

```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
# array([[3.69084138],
#        [3.32960458]])

np.linalg.pinv(X_b) @ y     # 같은 결과
```

`LinearRegression`은 정규 방정식 대신 **유사역행렬(pseudoinverse)**을 씁니다. `lstsq`는 "least squares(최소 제곱)"의 줄임말이고, `pinv`는 유사역행렬을 직접 구하는 함수입니다.

왜 굳이 다른 방법을 쓸까요?

| 상황 | 정규 방정식 | 유사역행렬 |
|---|---|---|
| 특성이 샘플보다 많거나, 똑같은 특성이 중복될 때 | 역행렬이 존재하지 않아 **계산이 깨짐** | **항상 계산됨** |
| 특성이 많을 때 속도 | 더 느림 | 더 빠름 |

즉, 유사역행렬이 **더 튼튼하고 빠릅니다.** 결과는 같습니다.

### 1-7. 이 방법의 한계

정규 방정식이든 유사역행렬이든, **특성 수가 많아지면 급격히 느려집니다.**

- 특성이 2배가 되면 계산 시간은 약 4~8배로 늘어납니다.
- 특성이 수십만 개면 사실상 쓸 수 없습니다.
- 데이터 전체를 **한꺼번에 메모리에 올려야** 합니다.

반면 샘플 수가 늘어나는 것에는 비교적 강합니다(샘플 2배 → 시간 약 2배).

> 📌 **기억할 것**: 공식으로 한 번에 푸는 방법은 정확하지만, 특성이 많거나 데이터가 너무 크면 못 씁니다. 그래서 다음 절의 **경사 하강법**이 필요합니다.

---

## 2. 경사 하강법 (4.2)

> **이 절에서 하는 일**: 공식 없이, **조금씩 내려가면서** 최솟값을 찾아가는 방법을 배웁니다. 머신러닝·딥러닝에서 가장 중요한 알고리즘입니다.

### 2-1. 아이디어: 안개 낀 산에서 내려오기

짙은 안개가 낀 산 중턱에 서 있다고 상상해 보세요. 주변이 하나도 보이지 않습니다. 가장 낮은 골짜기로 내려가려면 어떻게 해야 할까요?

**발밑의 경사를 느끼고, 가장 가파르게 내려가는 방향으로 한 걸음 내딛습니다.** 그리고 다시 경사를 느끼고, 또 한 걸음. 이걸 반복하면 결국 골짜기 바닥에 도착합니다.

경사 하강법이 정확히 이 방식입니다.

| 산 비유 | 경사 하강법 |
|---|---|
| 내가 서 있는 위치 | 현재 파라미터 θ |
| 그 위치의 높이 | 비용(MSE) |
| 발밑의 경사 | **그레이디언트** |
| 한 걸음의 보폭 | **학습률 η** |
| 골짜기 바닥 | 비용이 최소인 θ (= 정답) |

**순서**
1. θ를 **아무 값으로나** 시작합니다 (무작위 초기화).
2. 지금 위치에서 그레이디언트(경사)를 계산합니다.
3. 경사의 **반대 방향**으로 η만큼 이동합니다. (경사가 오르막이면 반대로 가야 내려가니까요.)
4. 경사가 거의 0이 될 때(= 바닥)까지 2~3을 반복합니다.

### 2-2. 학습률(보폭)이 중요한 이유

- **보폭이 너무 작으면**: 바닥까지 가는 데 한참 걸립니다.
- **보폭이 너무 크면**: 골짜기를 건너뛰어 반대편 벽으로 올라가 버리고, 점점 더 높이 튀어 올라 **발산**합니다.
- **적당하면**: 빠르게 바닥에 도착합니다.

### 2-3. 경사 하강법에서 조심할 두 가지

**① 지역 최솟값**
산에는 작은 웅덩이가 여러 개 있을 수 있습니다. 웅덩이에 빠지면 "여기가 바닥이다"라고 착각하고 멈춰 버립니다.

하지만 다행히 **선형 회귀의 MSE는 그릇 모양(볼록 함수)**이라서 웅덩이가 없습니다. 바닥이 딱 하나뿐이라, 충분히 오래 내려가면 반드시 진짜 바닥에 도착합니다.

**② 특성의 스케일**
특성 하나는 0~1 범위이고 다른 하나는 0~10,000 범위라면, 그릇이 **한쪽으로 아주 길쭉한 모양**이 됩니다. 이러면 바닥까지 가는 길이 매우 길어집니다.

> ⚠️ **경사 하강법을 쓸 때는 반드시 `StandardScaler` 등으로 특성의 스케일을 맞추세요.** (1절의 정규 방정식은 스케일과 상관없습니다.)

### 2-4. 배치 경사 하강법 — 데이터 전체를 보고 한 걸음

"배치(batch)"는 **매 걸음마다 훈련 데이터 전체를 다 보고** 경사를 계산한다는 뜻입니다.

MSE의 그레이디언트는 이렇게 계산됩니다. (미분 과정은 생략하고 결과만 쓰면 됩니다.)

```
그레이디언트 = (2/m) × Xᵀ × (예측값 − 정답)
```

말로 풀면: **"모든 샘플의 오차에 각 특성값을 곱해서 평균 낸 것"**입니다. 오차가 크면 경사도 크고, 오차가 0이면 경사도 0입니다.

```python
eta = 0.1        # 학습률 (보폭)
n_epochs = 1000  # 1000걸음 내려가기
m = len(X_b)     # 샘플 수 200

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))   # θ를 무작위로 시작

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)   # ① 경사 계산
    theta = theta - eta * gradients                 # ② 반대 방향으로 한 걸음

theta
# array([[3.69084138],
#        [3.32960458]])
```

**한 줄씩 읽기**

1. `theta = rng.standard_normal((2, 1))` — θ₀, θ₁을 아무 값으로 시작합니다. 실제로는 (0.305, −1.040)에서 출발합니다.
2. `X_b @ theta - y` — 현재 θ로 200개를 모두 예측한 뒤 정답을 뺍니다. **샘플별 오차** 200개가 나옵니다.
3. `X_b.T @ (오차)` — 오차와 각 특성을 곱해 합칩니다. θ₀, θ₁ 각각에 대한 경사 2개가 나옵니다.
4. `2 / m * ...` — 평균을 내고 MSE 미분에서 나온 2를 곱합니다.
5. `theta - eta * gradients` — 경사의 **반대 방향**으로 η만큼 이동합니다. 빼기(−)가 바로 "반대 방향"의 의미입니다.

**처음 3걸음을 직접 따라가 보면** (η = 0.1)

| 걸음 | 그레이디언트 | θ₀, θ₁ | MSE |
|---|---|---|---|
| 시작 | — | 0.305, −1.040 | 66.38 |
| 1 | −15.40, −17.99 | 1.845, 0.759 | 22.32 |
| 2 | −8.77, −10.29 | 2.722, 1.788 | 7.96 |
| 3 | −4.98, −5.90 | 3.220, 2.378 | 3.27 |

- 그레이디언트가 **음수** → θ를 키우면 비용이 줄어든다는 뜻 → θ가 커집니다.
- 바닥에 가까워질수록 **그레이디언트가 작아져서** 걸음도 자연스럽게 작아집니다.
- 1000걸음 후에는 정규 방정식과 **소수점까지 똑같은 값**(3.69, 3.33)에 도착합니다.

**학습률을 바꿔 보면 (그림 4-8)**

```python
import matplotlib as mpl

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_epochs = 1000
    n_shown = 20
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < n_shown:   # 처음 20걸음의 직선만 그림
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle="solid", color=color)
        gradients = 2 / m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
        theta_path.append(theta)   # 뒤에서 경로를 그리려고 저장
    plt.xlabel("$x_1$")
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")
    return theta_path

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

plt.figure(figsize=(10, 4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.subplot(132); theta_path_bgd = plot_gradient_descent(theta, eta=0.1)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
plt.show()
```

- `plt.cm.OrRd(값)` — 0~1 사이 값을 연한 주황 → 진한 빨강 색으로 바꿔 줍니다. 걸음이 진행될수록 선이 진해집니다.
- `plt.subplot(131)` — "1행 3열로 나눈 칸 중 1번째"라는 뜻입니다.

| η | 20걸음 후 θ | 1000걸음 후 θ | 해석 |
|---|---|---|---|
| 0.02 | 3.29, 2.47 | 3.69, 3.33 | 너무 느림. 아직 도착 전 |
| **0.1** | 3.81, 3.23 | 3.69, 3.33 | 적당. 빠르게 근처 도착 |
| 0.5 | −50, −59 | −2×10⁵⁹ … | 너무 큼. **발산** — 값이 폭발 |

> 💡 그럼 학습률은 어떻게 정할까요? 2장에서 배운 **그리드 서치**로 여러 값을 시험해 보면 됩니다. 반복 횟수는 넉넉히 크게 잡고, 경사가 아주 작아지면(= 거의 바닥이면) 멈추게 하는 방식을 씁니다.

### 2-5. 확률적 경사 하강법(SGD) — 샘플 하나만 보고 한 걸음

배치 경사 하강법은 매 걸음마다 **데이터 전체**를 봐야 합니다. 데이터가 수백만 개라면 한 걸음 내딛는 데에도 한참 걸리겠죠.

**SGD(Stochastic Gradient Descent)**는 반대로, 매 걸음마다 **샘플을 딱 하나 무작위로 골라서** 그것만 보고 경사를 계산합니다.

비유하자면:
- **배치 GD** = 반 학생 200명 모두에게 의견을 묻고 나서 한 걸음 → 정확하지만 느림
- **SGD** = 아무 학생 한 명에게만 묻고 바로 한 걸음 → 빠르지만 그 한 명의 의견이 치우칠 수 있어 **갈팡질팡**

| | 배치 GD | SGD |
|---|---|---|
| 한 걸음에 보는 데이터 | 전체 | 1개 |
| 속도 | 느림 | 매우 빠름 |
| 경로 | 매끄럽게 직진 | 지그재그로 흔들림 |
| 바닥에 도착하면 | 멈춤 | 계속 주변을 맴돔 |
| 웅덩이(지역 최솟값) | 빠질 수 있음 | 흔들림 덕분에 **빠져나올 수 있음** |
| 메모리에 안 들어가는 큰 데이터 | 어려움 | 가능 |

**맴도는 문제는 어떻게 해결할까? → 학습 스케줄**

처음엔 보폭을 크게 해서 빨리 내려가고, **점점 보폭을 줄여서** 바닥 근처에서는 조심조심 움직이게 합니다. 이렇게 학습률을 바꾸는 규칙을 **학습 스케줄**이라고 합니다.

```python
theta_path_sgd = []   # 그림용으로 경로 저장

n_epochs = 50
t0, t1 = 5, 50        # 학습 스케줄용 숫자

def learning_schedule(t):
    return t0 / (t + t1)   # t가 커질수록 작아짐

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

for epoch in range(n_epochs):           # 50바퀴
    for iteration in range(m):          # 한 바퀴에 200걸음
        random_index = rng.integers(m)              # 0~199 중 아무 번호 하나
        xi = X_b[random_index : random_index + 1]   # 그 샘플 하나
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)    # 샘플 1개로 경사 계산
        eta = learning_schedule(epoch * m + iteration)  # 지금 보폭
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

theta
# array([[3.69826475],
#        [3.30748311]])
```

**한 줄씩 읽기**

1. `learning_schedule(t)` — t는 지금까지 걸은 총 걸음 수입니다. 보폭이 이렇게 줄어듭니다.

   | 총 걸음 수 t | 0 | 50 | 500 | 5,000 | 9,999 (마지막) |
   |---|---|---|---|---|---|
   | 보폭 η | 0.1 | 0.05 | 0.009 | 0.001 | 0.0005 |

2. `rng.integers(m)` — 0~199 중 무작위 번호 하나. 매번 새로 뽑기 때문에 한 바퀴 동안 어떤 샘플은 여러 번, 어떤 샘플은 한 번도 안 뽑힐 수 있습니다.
3. `X_b[random_index : random_index + 1]` — 왜 그냥 `X_b[random_index]`라고 안 쓸까요? 그렇게 쓰면 1차원 `(2,)`가 되어 행렬 곱 모양이 안 맞습니다. 슬라이싱(`:`)을 쓰면 **`(1, 2)` 2차원 모양이 유지**됩니다.
4. `2 * xi.T @ ...` — 배치 GD와 같은 식인데, 샘플이 1개라 `m`으로 나눌 필요가 없습니다.
5. `epoch * m + iteration` — 예: 3바퀴째(epoch=2)의 10번째 걸음이면 2×200+10 = 410. 지금까지 걸은 총 걸음 수입니다.

**결과 해석**

배치 GD는 1000바퀴를 돌았는데, SGD는 **50바퀴만에** (3.698, 3.307)로 거의 같은 곳에 도착했습니다. 대신 정확히 같지는 않습니다(조금 흔들리니까요).

> ⚠️ SGD는 데이터가 **섞여 있어야** 합니다. 만약 데이터가 정답 순으로 정렬되어 있으면, 처음엔 한쪽 클래스만, 나중엔 다른 쪽만 보게 되어 엉뚱하게 학습됩니다.

**사이킷런 `SGDRegressor`**

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())

sgd_reg.intercept_, sgd_reg.coef_
# (array([3.68899733]), array([3.33054574]))
```

| 매개변수 | 뜻 |
|---|---|
| `max_iter=1000` | 최대 1000바퀴 |
| `tol=1e-5`, `n_iter_no_change=100` | "100바퀴 동안 손실이 0.00001보다 덜 줄면 멈춰라" |
| `penalty=None` | 규제를 쓰지 않음 (5절에서 다룸) |
| `eta0=0.01` | 처음 보폭. 사이킷런의 기본 스케줄로 점점 줄어듦 |
| `random_state=42` | 결과 재현용 |

`y.ravel()`은 `(200, 1)` 모양을 `(200,)`으로 납작하게 폅니다. `SGDRegressor`는 정답을 1차원으로 받기 때문입니다.

### 2-6. 미니배치 경사 하강법 — 몇십 개씩 보고 한 걸음

배치(전체)와 SGD(1개)의 **중간**입니다. 매 걸음마다 **20개 정도의 작은 묶음(미니배치)**을 보고 경사를 계산합니다.

- 1개보다는 덜 흔들리고, 전체보다는 훨씬 빠릅니다.
- 컴퓨터(특히 **GPU**)는 여러 개를 **한꺼번에 계산**하는 데 강해서, 20개를 보는 시간이 1개를 보는 시간과 거의 비슷합니다.
- 그래서 **딥러닝은 거의 모두 미니배치**를 씁니다.

```python
from math import ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)   # 200 ÷ 20 = 10묶음

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

theta_path_mgd = []
for epoch in range(n_epochs):
    shuffled_indices = rng.permutation(m)       # ① 매 바퀴 순서 섞기
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size        # 0, 20, 40, ...
        xi = X_b_shuffled[idx : idx + minibatch_size]   # ② 20개씩 잘라 오기
        yi = y_shuffled[idx : idx + minibatch_size]
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
```

**SGD 코드와 다른 점**

- `rng.permutation(m)` — 0~199를 무작위 순서로 섞은 목록. SGD처럼 매번 아무거나 뽑는 게 아니라, **섞은 뒤 순서대로 20개씩** 잘라 쓰므로 한 바퀴에 **모든 샘플을 정확히 한 번씩** 봅니다. 카드를 섞은 뒤 위에서부터 20장씩 나눠 주는 것과 같습니다.
- `ceil(...)` — 올림. 200이 20으로 나누어떨어지지 않는 경우 마지막 묶음까지 챙기기 위해서입니다.
- `2 / minibatch_size` — 묶음 크기로 평균을 냅니다.

**세 방법이 걸어간 길 비교 (그림 4-11)**

```python
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(7, 4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left")
plt.xlabel(r"$\theta_0$")
plt.ylabel(r"$\theta_1$   ", rotation=0)
plt.axis([2.7, 4.5, 2.6, 3.7])
plt.show()
```

x축을 θ₀, y축을 θ₁으로 놓고, 각 방법이 걸어간 발자국을 이은 그림입니다. `[:, 0]`은 "모든 걸음의 θ₀"이라는 뜻입니다.

- 🔵 **배치**: 곧장 바닥으로 가서 멈춤
- 🔴 **SGD**: 크게 지그재그
- 🟢 **미니배치**: 그 중간

### 2-7. 지금까지 배운 방법 한눈에 비교

| 방법 | 샘플이 많을 때 | 메모리에 다 안 들어갈 때 | 특성이 많을 때 | 스케일링 필요? | 사이킷런 |
|---|---|---|---|---|---|
| 정규 방정식 | 빠름 | ❌ | 느림 | 불필요 | — |
| 유사역행렬(SVD) | 빠름 | ❌ | 느림 | 불필요 | `LinearRegression` |
| 배치 GD | 느림 | ❌ | 빠름 | **필요** | — |
| SGD | 빠름 | ✅ | 빠름 | **필요** | `SGDRegressor` |
| 미니배치 GD | 빠름 | ✅ | 빠름 | **필요** | — |

> 📌 **기억할 것**: 방법이 달라도 **결과로 나오는 모델은 같은 직선**입니다. 다른 건 "거기까지 가는 속도와 메모리 사용량"뿐입니다.

---

## 3. 다항 회귀 (4.3)

> **이 절에서 하는 일**: 데이터가 직선이 아니라 **곡선**일 때도 선형 회귀를 쓰는 요령을 배웁니다.

### 3-1. 아이디어: 특성을 제곱해서 하나 더 만들기

데이터가 포물선 모양이라면 직선으로는 잘 맞출 수 없습니다. 그런데 이렇게 생각해 보세요.

```
ŷ = θ₀ + θ₁x + θ₂x²
```

여기서 `x²`를 **새로운 특성 하나**라고 보면, 이 식은 특성이 2개(x, x²)인 **평범한 선형 회귀**입니다. 즉, **특성만 미리 만들어 주면 선형 회귀 그대로 곡선을 학습**할 수 있습니다. 이게 다항 회귀입니다.

> 💡 "선형"이라는 말은 그래프가 직선이라는 뜻이 아니라, **θ들을 단순히 곱하고 더하기만 한다**는 뜻입니다. 그래서 곡선을 그려도 여전히 "선형 모델"이라고 부릅니다.

### 3-2. 2차 데이터 만들기

```python
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3                          # −3 ~ 3
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))  # 정답: 0.5x² + 1x + 2
```

- `6 * (0~1) − 3` → −3~3 범위가 됩니다.
- 정답 식은 **0.5x² + 1x + 2**. 모델이 이 숫자들을 찾아낼 수 있는지 보겠습니다.

### 3-3. `PolynomialFeatures`로 x² 만들기

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

X[0]        # array([1.64373629])
X_poly[0]   # array([1.64373629, 2.701869  ])
```

첫 번째 샘플을 보면 원래 x = 1.644였는데, 변환 후 `[1.644, 2.702]`가 되었습니다. 2.702 = 1.644² 이죠. **x 옆에 x²가 새 열로 붙었습니다.**

| 매개변수 | 뜻 |
|---|---|
| `degree=2` | 2차(제곱)까지 만든다 |
| `include_bias=False` | 1로 된 열은 만들지 않는다. `LinearRegression`이 절편을 알아서 처리하므로 필요 없음 |

```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# (array([2.00540719]), array([[1.11022126, 0.50526985]]))
```

**결과 해석**

| | 절편 | x의 계수 | x²의 계수 |
|---|---|---|---|
| 정답 | 2 | 1 | 0.5 |
| 모델 | 2.005 | 1.110 | 0.505 |

거의 정확하게 찾아냈습니다! `coef_`의 순서는 `X_poly`의 열 순서(x, x²)와 같습니다.

**곡선 그리기 (그림 4-13)**

```python
X_new = np.linspace(-3, 3, 100).reshape(100, 1)   # −3~3을 100칸으로 촘촘히
X_new_poly = poly_features.transform(X_new)       # 새 데이터도 x² 추가
y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
```

- `np.linspace(-3, 3, 100)` — −3부터 3까지 **같은 간격으로** 100개의 수. 점을 촘촘히 찍어야 곡선이 매끄럽게 그려집니다.
- `.reshape(100, 1)` — 사이킷런에 넣으려고 2차원으로 바꿉니다.
- 새 데이터에는 `fit_transform`이 아니라 **`transform`**만 씁니다. 이미 "x²를 만든다"는 규칙을 배웠으니 그대로 적용만 하면 됩니다.

### 3-4. 특성이 여러 개일 때 주의

> ⚠️ 특성이 a, b 두 개이고 `degree=3`이면, `PolynomialFeatures`는 a², a³, b², b³뿐 아니라 **ab, a²b, ab²** 같은 **곱하기 조합(교차항)**까지 모두 만듭니다.
>
> 그래서 특성 수가 **폭발적으로** 늘어납니다. 예를 들어 특성 10개에 degree=3이면 285개가 됩니다. 차수를 함부로 높이지 마세요.

---

## 4. 학습 곡선 (4.4)

> **이 절에서 하는 일**: 내 모델이 **너무 단순한지(과소적합) 너무 복잡한지(과대적합)** 진단하는 방법을 배웁니다.

### 4-1. 차수를 너무 높이면 무슨 일이? (그림 4-14)

같은 2차 데이터에 1차, 2차, 300차 모델을 맞춰 보겠습니다.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

for style, width, degree in (("r-+", 2, 1), ("b--", 2, 2), ("g-", 1, 300)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = make_pipeline(polybig_features, std_scaler, lin_reg)
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    label = f"{degree} degree{'s' if degree > 1 else ''}"
    plt.plot(X_new, y_newbig, style, label=label, linewidth=width)
```

**코드 포인트**
- `for style, width, degree in (...)` — 튜플 3개를 하나씩 꺼내며 반복. 한 번 돌 때마다 (선 모양, 굵기, 차수)가 바뀝니다.
- `make_pipeline(다항 특성 → 스케일링 → 선형 회귀)` — 세 단계를 하나로 묶습니다. `fit` 한 번에 순서대로 처리됩니다.
- `StandardScaler`를 넣는 이유: 3의 300제곱 같은 수는 상상할 수 없이 커서, 스케일을 맞추지 않으면 계산이 엉망이 됩니다.
- `'s' if degree > 1 else ''` — "1 degree", "2 degrees"처럼 영어 복수형을 맞추는 작은 장치입니다.

**결과**

| 차수 | 그래프 모양 | 진단 |
|---|---|---|
| 1 | 포물선을 무시한 직선 | **과소적합** — 너무 단순 |
| 2 | 데이터와 잘 맞는 포물선 | ✅ 적당 |
| 300 | 점 하나하나를 지나가려고 미친 듯이 구불거림 | **과대적합** — 훈련 데이터를 외움 |

시험 공부에 비유하면:
- **과소적합** = 공부를 너무 대충 해서 연습 문제도 못 풂
- **과대적합** = 연습 문제의 **답을 통째로 외워서**, 연습 문제는 100점인데 처음 보는 시험 문제는 못 풂

### 4-2. 그런데 실제로는 정답 차수를 모른다

이번엔 데이터를 우리가 만들었으니 2차가 정답인 걸 압니다. 하지만 현실 데이터는 모릅니다. 어떻게 판단할까요?

1. **교차 검증** (2장): 훈련 점수는 좋은데 검증 점수가 나쁘면 과대적합, 둘 다 나쁘면 과소적합
2. **학습 곡선** (지금 배울 것): 훈련 데이터 양을 조금씩 늘려 가며 **훈련 오차와 검증 오차의 변화**를 그래프로 그림

### 4-3. 선형 회귀의 학습 곡선 (그림 4-15)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.axis([0, 160, 0, 2.5])
```

**`learning_curve`가 하는 일**

"훈련 데이터를 1개, 5개, 9개, …, 160개 쓰면 각각 훈련 오차와 검증 오차가 얼마인가?"를 자동으로 계산해 줍니다.

| 인자 | 뜻 |
|---|---|
| `train_sizes=np.linspace(0.01, 1.0, 40)` | 훈련 데이터를 1%부터 100%까지 40단계로 늘려 봄 |
| `cv=5` | 5-폴드 교차 검증. 200개 중 160개가 훈련용 → 그래서 x축이 160까지 |
| `scoring="neg_root_mean_squared_error"` | RMSE에 마이너스를 붙인 값 (사이킷런은 "클수록 좋은" 점수를 쓰기 때문) |

- `train_scores`는 40단계 × 5폴드 = `(40, 5)` 모양의 표입니다.
- `.mean(axis=1)` — 5폴드의 평균. 앞에 `-`를 붙여 다시 양수 RMSE로 바꿉니다.

**그래프 읽기**

🔴 **훈련 오차(빨강)**
- 데이터가 1~2개면 직선이 그 점을 완벽하게 지나가므로 오차 0.
- 데이터가 늘수록 직선 하나로는 다 못 맞추니 오차가 커지다가, 약 **1.62**에서 평평해집니다.

🔵 **검증 오차(파랑)**
- 데이터가 적을 때는 제대로 배운 게 없으니 처음 보는 데이터에서 오차가 큼(약 3.6).
- 데이터가 늘수록 점점 좋아지다가, 약 **1.64**에서 평평해집니다.

**진단: 과소적합**
두 곡선이 **높은 곳에서 거의 붙어서 평평**합니다. 모델이 할 수 있는 만큼 다 했는데도 오차가 크다는 뜻입니다.

> 💡 **과소적합이면 데이터를 더 모아도 소용없습니다.** 곡선이 이미 평평하니까요. 더 복잡한 모델(예: 다항 특성 추가)이나 더 좋은 특성이 필요합니다.

### 4-4. 10차 다항 회귀의 학습 곡선 (그림 4-16)

```python
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
```

**선형 회귀와 비교**

| | 선형 회귀 | 10차 다항 |
|---|---|---|
| 최종 훈련 오차 | 약 1.62 | 약 **0.97** (훨씬 낮음) |
| 최종 검증 오차 | 약 1.64 | 약 1.03 |
| 두 곡선 사이 | 거의 붙음 | **간격이 있음** |

**진단: 과대적합 경향**
훈련 데이터에서는 훨씬 잘하는데 검증 데이터에서는 그만큼 못합니다. 이 **간격**이 과대적합의 신호입니다. 다만 데이터가 늘수록 간격이 줄어듭니다.

> 💡 **과대적합이면 데이터를 더 모으는 게 효과가 있습니다.** 두 곡선이 가까워질 때까지 데이터를 추가하세요.

### 4-5. 편향과 분산 — 오차는 어디서 오는가

모델의 오차는 세 가지로 나눌 수 있습니다.

| 오차 종류 | 원인 | 결과 | 해결 방법 |
|---|---|---|---|
| **편향** (bias) | 잘못된 가정. 예: 포물선 데이터를 직선이라고 가정 | 과소적합 | 더 복잡한 모델 |
| **분산** (variance) | 데이터의 작은 차이에 너무 민감. 예: 300차 모델 | 과대적합 | 더 단순한 모델, 데이터 추가, **규제** |
| **줄일 수 없는 오차** | 데이터 자체의 잡음 | — | 데이터를 깨끗하게 (센서 고치기, 이상치 제거) |

모델을 복잡하게 하면 편향은 줄지만 분산이 늘고, 단순하게 하면 그 반대입니다. 그래서 둘 사이에서 **균형점**을 찾아야 하고, 이걸 **편향/분산 트레이드오프**라고 합니다.

---

## 5. 규제가 있는 선형 모델 (4.5)

> **이 절에서 하는 일**: 과대적합을 막기 위해 모델에 **"너무 튀지 마"라는 제약(규제)**을 거는 방법 네 가지를 배웁니다.

### 5-0. 규제란?

과대적합된 모델의 특징은 **가중치(θ)가 아주 큰 값**이 되는 것입니다. 300차 모델이 미친 듯이 구불거린 것도 고차항의 가중치가 엄청 컸기 때문입니다.

그래서 아이디어는 간단합니다.

> **비용 함수에 "가중치가 크면 벌점"을 추가한다.**

그러면 모델은 "오차를 줄이는 것"과 "가중치를 작게 유지하는 것" 사이에서 타협하게 되고, 덜 구불거리는 모델이 됩니다.

벌점의 세기는 **α(알파)**로 조절합니다.
- α = 0 → 벌점 없음 = 일반 선형 회귀
- α가 크면 → 가중치를 거의 0으로 눌러서 **평평한 선**

**실습 데이터: 일부러 과대적합하기 쉬운 상황**

```python
rng = np.random.default_rng(seed=42)
m = 20                     # 샘플이 겨우 20개
X = 3 * rng.random((m, 1))
y = 1 + 0.5 * X + rng.standard_normal((m, 1)) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)
```

데이터가 적을수록 과대적합이 쉽게 일어나므로, 규제의 효과를 보기 좋습니다.

### 5-1. 릿지 회귀 (Ridge) — 가중치를 "제곱"으로 벌점

```
비용 = MSE + (α/m) × (θ₁² + θ₂² + … + θₙ²)
```

**가중치를 제곱해서 더한 값**을 벌점으로 붙입니다. 이걸 **ℓ₂ 규제**라고도 부릅니다.

두 가지 주의점:
- **절편 θ₀은 벌점에서 뺍니다.** 절편까지 줄이면 예측 전체가 아래로 내려가 버리기 때문입니다.
- **반드시 스케일링을 먼저 합니다.** 단위가 큰 특성(예: 원 단위 가격)은 가중치가 원래 작고, 단위가 작은 특성은 가중치가 원래 큽니다. 스케일이 다르면 벌점이 불공평하게 걸립니다.

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84414523])
```

- `alpha=0.1` — 벌점 세기
- `solver="cholesky"` — 정규 방정식처럼 **공식으로 한 번에** 푸는 방법(숄레스키 분해)
- `predict([[1.5]])` — x = 1.5일 때 예측. 대괄호 2개는 "샘플 1개, 특성 1개"인 2차원 입력이라는 뜻입니다.

**α를 바꿔 가며 비교 (그림 4-17)**

```python
def plot_model(model_class, polynomial, alphas, **model_kwargs):
    plt.plot(X, y, "b.", linewidth=3)
    for alpha, style in zip(alphas, ("b:", "g--", "r-")):
        if alpha > 0:
            model = model_class(alpha, **model_kwargs)
        else:
            model = LinearRegression()           # α=0이면 그냥 선형 회귀
        if polynomial:
            model = make_pipeline(
                PolynomialFeatures(degree=10, include_bias=False),
                StandardScaler(),
                model)
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        plt.plot(X_new, y_new_regul, style, linewidth=2,
                 label=fr"$\alpha = {alpha}$")
    plt.legend(loc="upper left")
    plt.xlabel("$x_1$")
    plt.axis([0, 3, 0, 3.5])
    plt.grid()

plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
plt.show()
```

이 함수는 뒤에서 라쏘에도 재사용하는 **그림 그리기 도우미**입니다.

**코드 포인트**
- `model_class` — `Ridge`나 `Lasso` 같은 **클래스 자체**를 인자로 받습니다. 그래서 같은 함수로 여러 모델을 그릴 수 있습니다.
- `**model_kwargs` — 추가로 넘긴 키워드 인자(`random_state=42`)를 모아 두었다가 모델에 그대로 전달합니다.
- `zip(alphas, 스타일들)` — α 3개와 선 모양 3개를 짝지어 하나씩 꺼냅니다.
- `polynomial=True`면 10차 다항 특성 + 스케일링을 앞에 붙입니다.
- `fr"$\alpha = {alpha}$"` — `f`는 변수를 넣는 f-string, `r`은 `\a`를 특수 문자로 해석하지 말라는 raw string입니다.

**그림 읽기**

| | 왼쪽 (단순 직선 + 릿지) | 오른쪽 (10차 다항 + 릿지) |
|---|---|---|
| α = 0 | 일반 직선 | 점을 따라 **심하게 구불거림** (과대적합) |
| α 중간 | 조금 완만해짐 | 덜 구불거림 |
| α 큼 | 거의 **수평선** | 매끄러운 곡선 |

> 💡 **직접 확인해 본 숫자**: 10차 다항 모델에서 가장 큰 가중치가 α = 0.00001일 때 약 **23**, α = 1일 때 약 **0.48**로 줄어듭니다. 그런데 10개 가중치 중 **0이 된 것은 하나도 없습니다.** 릿지는 가중치를 **작게** 만들 뿐 **없애지는** 않습니다. 이게 다음에 배울 라쏘와의 가장 큰 차이입니다.

**SGD로 릿지 하기**

```python
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
# array([1.83659707])   ← Ridge(1.844)와 거의 같음
```

- `penalty="l2"` — 릿지와 같은 벌점을 쓰겠다는 뜻
- `alpha=0.1 / m` — `SGDRegressor`는 α를 정의하는 방식이 `Ridge`와 조금 달라서, 같은 결과를 내려면 **샘플 수 m으로 나눠 줘야** 합니다.
- `tol=None` — 일찍 멈추지 말고 1000바퀴를 끝까지 돌라는 뜻

**다른 풀이 방법으로도 확인**

```python
ridge_reg = Ridge(alpha=0.1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84415109])   ← cholesky와 거의 같음
```

`sag`는 경사 하강법의 한 변형입니다. 풀이 방법이 달라도 같은 답에 도착한다는 걸 보여 줍니다.

**릿지 공식을 직접 계산해서 확인**

릿지에도 정규 방정식 같은 공식이 있습니다.

```
θ = (XᵀX + αA)⁻¹ Xᵀ y
```

정규 방정식에 `αA`가 더해진 모양입니다. `A`는 대각선이 1인 행렬인데, 절편은 규제하지 않으니 **맨 왼쪽 위만 0**입니다.

```python
alpha = 0.1
A = np.array([[0., 0.],
              [0., 1.]])        # 절편 자리는 0 → 규제 안 함
X_b = np.c_[np.ones(m), X]      # 앞에 1 열 붙이기 (add_dummy_feature와 같음)
np.linalg.inv(X_b.T @ X_b + alpha * A) @ X_b.T @ y
# array([[1.23749481],
#        [0.40443361]])

ridge_reg.intercept_, ridge_reg.coef_
# (array([1.23754037]), array([0.40440715]))   ← 일치!
```

`np.c_[a, b]`는 a와 b를 **열 방향으로 나란히** 붙이는 NumPy 단축 문법입니다.

### 5-2. 라쏘 회귀 (Lasso) — 가중치를 "절댓값"으로 벌점

```
비용 = MSE + 2α × (|θ₁| + |θ₂| + … + |θₙ|)
```

제곱 대신 **절댓값**을 씁니다. **ℓ₁ 규제**라고 부릅니다. 겉보기엔 작은 차이 같지만 결과는 아주 다릅니다.

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
# array([1.87550211])
```

```python
# 그림 4-18: 앞에서 만든 plot_model을 그대로 재사용
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 1e-2, 1), random_state=42)
```

**라쏘의 가장 중요한 특징: 쓸모없는 특성의 가중치를 정확히 0으로 만든다**

오른쪽 그림에서 α = 0.01인 선을 보면, 10차 다항 모델인데도 거의 **2차 곡선**처럼 보입니다. 고차항들의 가중치가 0이 되어 사라졌기 때문입니다.

> 💡 **직접 확인해 본 숫자**: α = 0.01일 때 10개 가중치 중 0이 아닌 것은 **2개뿐**이고, α = 1이면 **0개** — 모두 0이 되어 그냥 수평선이 됩니다.

이게 왜 좋을까요?
- 중요한 특성만 남기고 나머지는 자동으로 버려 줍니다 → **자동 특성 선택**
- 대부분의 가중치가 0인 모델을 **희소 모델**이라고 합니다. 해석하기 쉽습니다.

**왜 라쏘는 0을 만들고, 릿지는 못 만들까? (그림 4-19)**

벌점이 θ를 0 쪽으로 **당기는 힘**을 생각해 보세요.

| | 라쏘 (절댓값) | 릿지 (제곱) |
|---|---|---|
| 당기는 힘 | θ가 크든 작든 **항상 같은 힘** | θ가 0에 가까워질수록 **힘이 약해짐** |
| 비유 | 일정한 힘으로 끌고 가는 줄 | 가까워질수록 느슨해지는 고무줄 |
| 결과 | 0까지 끌려가 **딱 0에 도착** | 0 근처까지는 오지만 **0에 닿지 못함** |

그림 4-19의 핵심 코드는 이 "당기는 힘"을 그대로 표현합니다.

```python
def bgd_path(theta, X, y, l1, l2, core=1, eta=0.05, n_iterations=200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = (core * 2 / len(X) * X.T @ (X @ theta - y)   # MSE가 당기는 힘
                     + l1 * np.sign(theta)                        # 라쏘: 부호만 봄 → 항상 ±1
                     + l2 * theta)                                # 릿지: θ에 비례 → 작아지면 약해짐
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)
```

- `np.sign(theta)` — 양수면 +1, 음수면 −1, 0이면 0. 크기와 상관없이 **방향만** 알려 줍니다. 그래서 힘이 일정합니다.
- `l2 * theta` — θ 자체에 비례하므로 θ가 작아지면 힘도 작아집니다.
- 나머지 그림 코드(`np.meshgrid`, `contourf`)는 θ₁·θ₂ 평면 전체에 비용을 등고선으로 칠하는 부분이라 개념 이해에는 꼭 필요하지 않습니다.

> ⚠️ 라쏘를 SGD로 훈련할 때는 바닥 근처에서 계속 튕기는 경향이 있으니 **학습률을 점점 줄여야** 합니다.

### 5-3. 엘라스틱넷 (Elastic Net) — 둘을 섞기

릿지와 라쏘의 벌점을 **비율 r로 섞은 것**입니다.

```
비용 = MSE + r × (라쏘 벌점) + (1 − r) × (릿지 벌점)
```

- r = 0 → 완전히 릿지
- r = 1 → 완전히 라쏘

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)   # l1_ratio가 r
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
# array([1.8645014])
```

### 5-4. 그래서 무엇을 쓸까?

| 상황 | 추천 | 이유 |
|---|---|---|
| 잘 모르겠다 | **릿지** | 규제가 조금이라도 있는 게 거의 항상 낫다 |
| 쓸모 있는 특성이 몇 개뿐일 것 같다 | **라쏘** 또는 엘라스틱넷 | 나머지를 0으로 정리해 줌 |
| 특성이 샘플보다 많다 / 서로 비슷한 특성이 많다 | **엘라스틱넷** | 이런 상황에서 라쏘는 불안정하게 행동함 |

### 5-5. 조기 종료 (Early Stopping) — 가장 좋을 때 멈추기

규제의 또 다른 방법은 아주 단순합니다. **검증 오차가 가장 낮을 때 훈련을 멈추는 것**입니다.

경사 하강법으로 오래 훈련하면 처음에는 훈련 오차와 검증 오차가 같이 줄어듭니다. 그런데 어느 순간부터 모델이 훈련 데이터를 **외우기 시작하면**, 훈련 오차는 계속 줄어도 검증 오차는 다시 올라갑니다. 그 **꺾이는 지점**에서 멈추면 됩니다.

딥러닝의 아버지로 불리는 제프리 힌턴은 이 방법을 "**아름다운 공짜 점심**"이라고 불렀습니다. 간단한데 효과가 좋기 때문입니다.

```python
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# 2차 데이터를 다시 만들어서 앞 절반은 훈련, 뒤 절반은 검증
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))
X_train, y_train = X[: m // 2], y[: m // 2, 0]
X_valid, y_valid = X[m // 2 :], y[m // 2 :, 0]

# 90차 다항 특성 + 스케일링 → 일부러 과대적합하기 쉬운 모델
preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')    # 지금까지 최고 기록 (처음엔 무한대)
train_errors, val_errors = [], []

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)          # ① 한 바퀴만 훈련
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)   # ② 검증 오차 측정
    if val_error < best_valid_rmse:                     # ③ 최고 기록이면
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)                  #    그 모델을 복사해 저장

    y_train_predict = sgd_reg.predict(X_train_prep)     # (그림용 기록)
    train_error = root_mean_squared_error(y_train, y_train_predict)
    val_errors.append(val_error)
    train_errors.append(train_error)
```

**한 줄씩 읽기**

1. `m // 2` — 정수 나눗셈. 200 // 2 = 100. 앞 100개는 훈련, 뒤 100개는 검증.
2. `y[: m // 2, 0]` — `(100, 1)` 모양에서 0번 열만 꺼내 `(100,)` 1차원으로 만듭니다.
3. `preprocessing.fit_transform(X_train)` / `transform(X_valid)` — 스케일러는 **훈련 데이터로만** 평균과 표준편차를 배웁니다. 검증 데이터에는 그 값을 적용만 합니다. 검증 데이터 정보가 훈련에 섞이는 것(데이터 누설)을 막기 위해서입니다.
4. `float('inf')` — 무한대. 첫 번째 비교에서 무조건 "최고 기록"이 되도록 하는 장치입니다.
5. `partial_fit()` — **딱 한 바퀴만** 훈련하고, 다음에 부르면 **이어서** 훈련합니다. `fit()`은 부를 때마다 처음부터 다시 시작하므로 여기서 쓰면 안 됩니다.
6. `deepcopy(sgd_reg)` — 모델을 **통째로 복사**합니다. 그냥 `best_model = sgd_reg`라고 쓰면 같은 모델을 가리키는 이름표만 하나 더 생기는 것이라서, 다음 바퀴에 훈련하면 `best_model`도 같이 바뀌어 버립니다. 사진을 찍어 두는 것(복사)과 같은 사람을 다른 이름으로 부르는 것(대입)의 차이입니다.

**그림으로 확인 (그림 4-20)**

```python
best_epoch = np.argmin(val_errors)   # 검증 오차가 가장 작은 바퀴 번호
plt.annotate('Best model',
             xy=(best_epoch, best_valid_rmse),            # 화살표 끝
             xytext=(best_epoch, best_valid_rmse + 0.5),  # 글자 위치
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot([0, n_epochs], [best_valid_rmse, best_valid_rmse], "k:", linewidth=2)
plt.plot(val_errors, "b-", linewidth=3, label="Validation set")
plt.plot(best_epoch, best_valid_rmse, "bo")
plt.plot(train_errors, "r--", linewidth=2, label="Training set")
```

**결과 해석**

- 직접 돌려 보면 **184번째 바퀴**에서 검증 RMSE가 약 **1.105**로 가장 낮습니다.
- 그 뒤로 훈련 오차(빨강)는 계속 내려가지만, 검증 오차(파랑)는 다시 올라가서 500바퀴째에는 약 1.318이 됩니다. **외우기가 시작된 것**입니다.
- 우리는 184번째 모델을 `best_model`로 저장해 두었으니 그걸 쓰면 됩니다.

> 💡 SGD는 곡선이 울퉁불퉁해서 "지금이 최저점인지" 바로 알기 어렵습니다. 그래서 바로 멈추지 않고, **최고 기록 모델을 저장해 두면서 계속 돌리다가** 한동안 기록이 안 깨지면 멈추고 저장한 모델로 돌아가는 방식을 씁니다.

---

## 6. 로지스틱 회귀 (4.6)

> **이 절에서 하는 일**: 지금까지 배운 회귀 모델을 살짝 바꿔서 **분류** 문제를 풉니다.

### 6-1. 회귀로 분류를 한다고?

이메일이 스팸인지 아닌지 같은 문제를 생각해 봅시다. 답은 "예/아니오" 둘 중 하나입니다. 숫자를 예측하는 선형 회귀로 어떻게 풀 수 있을까요?

아이디어는 이렇습니다.
1. 선형 회귀처럼 `θ₀ + θ₁x₁ + …`를 계산합니다. 이 값은 −∞부터 +∞까지 아무 값이나 될 수 있습니다.
2. 이 값을 **0~1 사이로 눌러 주는 함수**에 넣습니다. 그러면 **확률**처럼 쓸 수 있습니다.
3. 확률이 0.5 이상이면 "예(1)", 아니면 "아니오(0)"로 판단합니다.

이게 **로지스틱 회귀**입니다. 이름은 회귀지만 실제로는 **분류 모델**입니다.

### 6-2. 0~1로 눌러 주는 함수: 시그모이드

```
σ(t) = 1 / (1 + e⁻ᵗ)
```

이 함수에 숫자를 넣으면 이렇게 나옵니다.

| 입력 t | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| 출력 σ(t) | 0.018 | 0.119 | **0.5** | 0.881 | 0.982 |

- 아주 큰 음수를 넣으면 0에 가까워지고, 아주 큰 양수를 넣으면 1에 가까워집니다.
- **t = 0일 때 정확히 0.5**입니다. 그래서 "t가 0보다 크면 1, 작으면 0"으로 판단하는 것과 같습니다.
- 그래프는 S자 모양입니다.

```python
# 그림 4-21: 시그모이드 그래프
lim = 6
t = np.linspace(-lim, lim, 100)
sig = 1 / (1 + np.exp(-t))

plt.figure(figsize=(8, 3))
plt.plot([-lim, lim], [0, 0], "k-")       # y=0 가로선
plt.plot([-lim, lim], [0.5, 0.5], "k:")   # y=0.5 점선
plt.plot([-lim, lim], [1, 1], "k:")       # y=1 점선
plt.plot([0, 0], [-1.1, 1.1], "k-")       # x=0 세로선
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \dfrac{1}{1 + e^{-t}}$")
```

> 💡 **로짓(logit)**: 시그모이드에 넣기 전의 값 t를 로짓이라고 부릅니다. 신경망에서도 계속 나오는 용어이니 알아 두세요.

### 6-3. 로지스틱 회귀는 어떻게 훈련할까?

**원하는 것**: 정답이 1인 샘플에는 높은 확률을, 정답이 0인 샘플에는 낮은 확률을 내는 모델.

**비용 함수 (로그 손실)**
```
정답이 1일 때 벌점:  −log(p̂)        ← p̂이 1이면 0, 0에 가까우면 무한대
정답이 0일 때 벌점:  −log(1 − p̂)    ← p̂이 0이면 0, 1에 가까우면 무한대
```

말로 하면: **"확신을 갖고 틀리면 엄청 큰 벌점"**입니다. 정답이 1인데 "1일 확률 1%"라고 말하면 벌점이 아주 큽니다.

**좋은 소식 두 가지**
- 정규 방정식 같은 **한 번에 푸는 공식은 없지만**,
- 이 비용 함수도 **그릇 모양(볼록)**이라 경사 하강법으로 **반드시 바닥을 찾습니다.**
- 그레이디언트 식도 선형 회귀와 거의 똑같습니다: "(예측 확률 − 정답) × 특성값의 평균"

### 6-4. 붓꽃 데이터 살펴보기

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
list(iris)
# ['data', 'target', 'frame', 'target_names', 'DESCR',
#  'feature_names', 'filename', 'data_module']

iris.data.head(3)
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2

iris.target.head(3)    # 0, 0, 0
iris.target_names      # array(['setosa', 'versicolor', 'virginica'])
```

| 항목 | 내용 |
|---|---|
| 샘플 | 붓꽃 150송이 (품종마다 50송이) |
| 특성 4개 | 꽃받침(sepal) 길이·너비, 꽃잎(petal) 길이·너비 (단위 cm) |
| 정답 | 0 = setosa, 1 = versicolor, 2 = virginica |
| `as_frame=True` | 판다스 표(DataFrame)로 받기 |

> ⚠️ `iris.target.head(3)`이 0, 0, 0인 것처럼, 데이터가 **품종 순서대로 정렬**되어 있습니다. 앞에서부터 자르면 한 품종만 들어가므로 반드시 섞어서 나눠야 합니다. `train_test_split`은 기본으로 섞어 줍니다.

### 6-5. 꽃잎 너비 하나로 virginica 구별하기

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

**한 줄씩 읽기**

1. `iris.data[["petal width (cm)"]]` — **대괄호 2개**에 주의하세요. 대괄호 1개면 1차원 Series가 되고, 2개면 열이 1개인 **2차원 표**가 됩니다. 사이킷런은 2차원을 원합니다.
2. `.values` — 판다스 표를 NumPy 배열로 바꿉니다. 모양은 `(150, 1)`.
3. `iris.target_names[iris.target]` — 0/1/2 숫자를 품종 이름으로 바꿉니다. `['setosa', 'versicolor', 'virginica']`에서 각 번호의 이름을 꺼내는 것입니다.
4. `== 'virginica'` — virginica면 True, 아니면 False. **3개 클래스 문제를 "virginica인가 아닌가"의 이진 문제로** 바꿨습니다.
5. `train_test_split(...)` — 기본으로 75%는 훈련, 25%는 테스트로 섞어서 나눕니다.

**꽃잎 너비에 따른 확률 그려 보기 (그림 4-23)**

```python
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)   # 꽃잎 너비 0~3cm를 1000칸으로
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
plt.plot(X_train[y_train == 0], y_train[y_train == 0], "bs")   # 아닌 꽃: 아래쪽(0)
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")   # virginica: 위쪽(1)

decision_boundary
# np.float64(1.6516516516516517)
```

**한 줄씩 읽기**

1. `.reshape(-1, 1)` — `-1`은 "개수는 알아서 계산해 줘"라는 뜻. 결과는 `(1000, 1)`.
2. `predict_proba(X_new)` — 각 너비마다 `[아닐 확률, virginica일 확률]` 두 개를 돌려줍니다. 모양 `(1000, 2)`. 두 값의 합은 항상 1입니다.
3. `y_proba[:, 1] >= 0.5` — virginica 확률이 50% 이상인 곳은 True.
4. `X_new[...][0, 0]` — True인 너비들 중 **첫 번째** 값 = 확률이 처음으로 50%를 넘는 지점 = **결정 경계**.

**결과 해석**

꽃잎 너비에 따라 virginica일 확률이 이렇게 변합니다.

| 꽃잎 너비 | 1.0cm | 1.5cm | **1.65cm** | 1.7cm | 2.0cm |
|---|---|---|---|---|---|
| virginica 확률 | 7.6% | 36% | **약 50%** | 55% | 79% |

- **약 1.65cm가 경계**입니다. 이보다 넓으면 virginica, 좁으면 아니라고 판단합니다.
- 2cm 이상은 꽤 확실하고, 1cm 이하는 확실히 아닙니다. 그 사이는 두 품종이 섞여 있는 애매한 구간입니다.

```python
log_reg.predict([[1.7], [1.5]])
# array([ True, False])   ← 1.7은 경계보다 넓으니 True, 1.5는 좁으니 False
```

### 6-6. 특성 2개로 늘려 보기 (그림 4-24)

이번엔 꽃잎 **길이와 너비** 두 개를 함께 씁니다.

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)
```

> 💡 **`C`는 무엇?** 로지스틱 회귀도 기본으로 **릿지(ℓ₂) 규제**가 걸려 있습니다. 그런데 세기를 α가 아니라 **C = 1/α**로 조절합니다. **C가 클수록 규제가 약해집니다.** α와 반대 방향이니 헷갈리지 마세요.

**그림 그리는 코드 — 격자(meshgrid) 기법**

특성이 2개면 그래프가 평면이 됩니다. 평면 위 **모든 지점**에서 확률을 계산해 등고선을 그리려면, 평면을 바둑판처럼 촘촘한 격자로 나눠야 합니다.

```python
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]      # 격자점 10만 개를 샘플 목록으로
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)       # 다시 바둑판 모양으로

contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)              # 등고선에 확률 숫자 표시
```

이 3단계는 앞으로 결정 경계를 그릴 때마다 나오니 익혀 두면 좋습니다.

1. **`np.meshgrid`** — 가로 500칸 × 세로 200칸 바둑판의 모든 교차점 좌표를 만듭니다. `x0`에는 각 점의 가로 좌표, `x1`에는 세로 좌표가 들어갑니다. 둘 다 `(200, 500)` 모양.
2. **`.ravel()` + `np.c_`** — 바둑판을 한 줄로 펴서 `[가로, 세로]` 두 열짜리 표 `(100000, 2)`로 만듭니다. 이제 모델에 넣을 수 있습니다.
3. **`.reshape(x0.shape)`** — 예측 결과(10만 개)를 다시 `(200, 500)` 바둑판 모양으로 접어서 등고선을 그립니다.

**결정 경계 직선**

```python
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])
plt.plot(left_right, boundary, "k--", linewidth=3)
```

결정 경계는 확률이 정확히 50%인 곳, 즉 **로짓이 0인 곳**입니다.

```
θ₀ + θ₁ × 길이 + θ₂ × 너비 = 0
→ 너비 = −(θ₁ × 길이 + θ₀) / θ₂
```

이 식에 길이 2.9와 7을 넣어 양 끝점을 구하고 직선으로 이었습니다.

**결과**: 결정 경계가 **직선**입니다. 로지스틱 회귀는 선형 모델이라서 경계도 항상 직선(특성이 더 많으면 평면)입니다. 다른 평행선들은 15%, 30%, …, 90% 확률 등고선입니다.

### 6-7. 소프트맥스 회귀 — 클래스가 3개 이상일 때

로지스틱 회귀는 "예/아니오" 두 가지만 구별합니다. 붓꽃 세 품종을 한 번에 구별하려면 **소프트맥스 회귀**를 씁니다.

**작동 방식**

1. 클래스마다 **따로 점수(로짓)**를 계산합니다. 클래스마다 자기만의 θ가 있습니다.
2. 점수들을 **소프트맥스 함수**에 넣어 확률로 바꿉니다.

```
클래스 k의 확률 = e^(k의 점수) / (모든 클래스의 e^(점수) 합)
```

3. 확률이 가장 높은 클래스를 답으로 고릅니다.

**숫자로 따라가 보기** — 꽃잎 길이 5cm, 너비 2cm인 꽃

| | setosa | versicolor | virginica |
|---|---|---|---|
| ① 점수(로짓) | −10.39 | 3.61 | 6.77 |
| ② e^점수 | ≈ 0.00003 | ≈ 37 | ≈ 871 |
| ③ 합계로 나누기 (합 ≈ 908) | **0.00** | **0.04** | **0.96** |

- e^x를 하면 모든 값이 **양수**가 되고, 큰 점수는 **훨씬 더 크게** 벌어집니다.
- 합계로 나누면 세 확률의 합이 정확히 1이 됩니다.
- 가장 높은 virginica(96%)가 답입니다.

**비용 함수: 크로스 엔트로피**

```
비용 = −(1/m) × Σ log(정답 클래스의 예측 확률)
```

"**정답 클래스에 준 확률이 낮을수록 큰 벌점**"입니다. 클래스가 2개일 때는 로지스틱 회귀의 로그 손실과 같습니다.

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]                      # 0, 1, 2 그대로 → 클래스 3개
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5, 2]])
# array([2])            ← 2번 = virginica

softmax_reg.predict_proba([[5, 2]]).round(2)
# array([[0.  , 0.04, 0.96]])
```

- 클래스가 3개 이상이면 `LogisticRegression`이 **알아서 소프트맥스 회귀로** 동작합니다. 따로 설정할 필요가 없습니다.
- `C=30` — 규제를 약하게 걸었습니다(C가 클수록 약함).
- `.round(2)` — 소수점 둘째 자리까지 반올림.

**세 품종의 영역 그리기 (그림 4-25)**

```python
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])   # 품종별 배경색

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)    # versicolor 확률 → 등고선
zz = y_predict.reshape(x0.shape)         # 예측 품종 → 배경 색칠

plt.contourf(x0, x1, zz, cmap=custom_cmap)   # 영역을 색으로 채움
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
```

- 앞에서 배운 **바둑판 3단계**를 그대로 씁니다.
- `contourf`의 f는 fill(채우기). 영역을 색으로 칠합니다. `contour`는 선만 그립니다.
- 세 영역의 경계가 모두 **직선**입니다(선형 모델이니까요).

---

## 7. 연습문제

### 7-1. 개념 문제 (1~11번) 정리

| 번호 | 질문 | 답 |
|---|---|---|
| 1 | 특성이 수백만 개면 어떤 방법? | **SGD나 미니배치 GD**. 메모리에 들어간다면 배치 GD도 가능. 정규 방정식·SVD는 특성이 많으면 너무 느림 |
| 2 | 특성마다 스케일이 많이 다르면? | 비용 함수가 길쭉한 그릇이 되어 경사 하강법이 아주 느려짐 → **스케일링**하세요. 규제 모델도 작은 값의 특성을 무시하게 됨. 정규 방정식·SVD는 괜찮음 |
| 3 | 로지스틱 회귀를 경사 하강법으로 훈련하면 웅덩이(지역 최솟값)에 빠질까? | **아니오.** 비용 함수가 그릇 모양(볼록)이라 웅덩이가 없음 |
| 4 | 경사 하강법 종류가 달라도 같은 모델이 나올까? | 비용 함수가 볼록하고 학습률이 적당하면 **거의 비슷**. 단, 학습률을 줄이지 않으면 SGD·미니배치는 바닥 주변을 계속 맴돌아 조금씩 다름 |
| 5 | 배치 GD에서 검증 오차가 매 바퀴 계속 오르면? | 훈련 오차도 같이 오르면 학습률이 너무 커서 **발산** → 학습률을 낮추기. 훈련 오차는 그대로라면 **과대적합** → 훈련을 멈추기 |
| 6 | 미니배치 GD에서 검증 오차가 오르자마자 멈추는 게 좋을까? | **아니오.** 무작위성 때문에 잠깐 오를 수 있음. 최고 모델을 저장해 두다가, 오래 개선이 없을 때 멈추고 저장한 모델로 돌아가기 |
| 7 | 가장 먼저 바닥 근처에 도착하는 방법은? 실제로 바닥에 멈추는 방법은? | 가장 빠름: **SGD**. 실제로 멈춤: **배치 GD**뿐 (나머지는 학습률을 점점 줄여야 멈춤) |
| 8 | 다항 회귀에서 검증 오차가 훈련 오차보다 훨씬 크면? | **과대적합**. ① 차수 낮추기 ② 규제(릿지·라쏘) ③ 데이터 늘리기 |
| 9 | 릿지에서 훈련·검증 오차가 비슷하게 높으면? | **과소적합**(편향이 큼) → **α를 낮춰** 규제를 약하게 |
| 10 | 릿지 vs 선형 회귀 / 라쏘 vs 릿지 / 엘라스틱넷 vs 라쏘 | 규제가 조금 있는 게 대개 나음 → 릿지. 중요한 특성이 몇 개뿐이면 라쏘. 라쏘가 불안정한 경우(특성 > 샘플, 비슷한 특성 많음)엔 엘라스틱넷 |
| 11 | 사진을 실외/실내, 낮/밤으로 분류하려면? | 둘은 **동시에 성립 가능**(실외+낮, 실내+밤 …)하므로 소프트맥스 하나가 아니라 **로지스틱 회귀 2개** |

### 7-2. 12번: 사이킷런 없이 소프트맥스 회귀 직접 만들기

> **문제**: NumPy만 써서 소프트맥스 회귀를 배치 경사 하강법 + 조기 종료로 구현하고, 붓꽃 데이터에 적용하세요.

사이킷런이 내부에서 하는 일을 **직접 손으로** 해 보는 문제입니다. 6절에서 배운 수식을 그대로 코드로 옮기면 됩니다. 단계별로 따라가 보겠습니다.

#### ① 데이터 불러오기 + 편향 열 붙이기

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"].values

X_with_bias = np.c_[np.ones(len(X)), X]     # 맨 앞에 1로 된 열
```

`add_dummy_feature`를 쓰지 않고 직접 1 열을 붙였습니다. 결과 모양은 `(150, 3)` — [1, 꽃잎 길이, 꽃잎 너비].

#### ② 훈련 60% / 검증 20% / 테스트 20%로 직접 나누기

```python
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)                         # 150

test_size = int(total_size * test_ratio)              # 30
validation_size = int(total_size * validation_ratio)  # 30
train_size = total_size - test_size - validation_size # 90

rng = np.random.default_rng(seed=42)
rnd_indices = rng.permutation(total_size)             # 0~149를 섞은 순서

X_train = X_with_bias[rnd_indices[:train_size]]          # 앞 90개
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]] # 가운데 30개
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]            # 뒤 30개
y_test = y[rnd_indices[-test_size:]]
```

카드를 섞은 뒤 위에서 90장, 그다음 30장, 마지막 30장으로 나누는 것과 같습니다. `[-test_size:]`는 "뒤에서 30개"라는 뜻입니다.

#### ③ 정답을 원-핫 벡터로 바꾸기

소프트맥스의 비용 함수(크로스 엔트로피)는 정답을 **확률 형태**로 받아야 합니다. 정답이 2(virginica)라면 `[0, 0, 1]`처럼, 정답 자리만 1이고 나머지는 0인 벡터로 바꿉니다. 이걸 **원-핫 벡터**라고 합니다.

```python
def to_one_hot(y):
    return np.diag(np.ones(y.max() + 1))[y]

y_train[:10]
# array([1, 1, 2, 0, 2, 2, 1, 2, 2, 0])

to_one_hot(y_train[:10])
# array([[0., 1., 0.],    ← 1
#        [0., 1., 0.],    ← 1
#        [0., 0., 1.],    ← 2
#        [1., 0., 0.],    ← 0
#        ...])

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)
```

**`to_one_hot`이 동작하는 원리**

1. `y.max() + 1` = 3 (클래스 개수)
2. `np.ones(3)` = `[1, 1, 1]`
3. `np.diag(...)` = 대각선만 1인 3×3 표
   ```
   [[1, 0, 0],    ← 0번 행
    [0, 1, 0],    ← 1번 행
    [0, 0, 1]]    ← 2번 행
   ```
4. `[y]` — 이 표에서 정답 번호에 해당하는 **행을 꺼내 옵니다.** 정답이 2면 2번 행 `[0, 0, 1]`.

#### ④ 스케일링 (편향 열은 빼고)

```python
mean = X_train[:, 1:].mean(axis=0)
std = X_train[:, 1:].std(axis=0)
X_train[:, 1:] = (X_train[:, 1:] - mean) / std
X_valid[:, 1:] = (X_valid[:, 1:] - mean) / std
X_test[:, 1:] = (X_test[:, 1:] - mean) / std
```

- `[:, 1:]` — 모든 행의 1번 열부터 끝까지. 0번 열(편향 = 1)은 건드리지 않습니다.
- 평균과 표준편차는 **훈련 데이터에서만** 구하고, 세 세트 모두에 같은 값을 적용합니다. `StandardScaler`를 직접 만든 것입니다.

#### ⑤ 소프트맥스 함수 만들기

```python
def softmax(logits):
    exps = np.exp(logits)                          # 모든 점수에 e^x
    exp_sums = exps.sum(axis=1, keepdims=True)     # 샘플(행)별 합
    return exps / exp_sums                         # 합으로 나누기

n_inputs = X_train.shape[1]          # 3 (특성 2개 + 편향)
n_outputs = len(np.unique(y_train))  # 3 (클래스 3개)
```

6-7의 "숫자로 따라가 보기" 표를 그대로 코드로 옮긴 것입니다.

- `axis=1` — 가로 방향(한 샘플의 세 클래스)으로 더합니다.
- `keepdims=True` — 왜 필요할까요? `exps`는 `(90, 3)`인데 합은 `(90,)`이 됩니다. 그러면 나눗셈 모양이 안 맞습니다. `keepdims=True`를 쓰면 합이 `(90, 1)`이 되어, 각 행이 **자기 행의 합으로** 나눠집니다(브로드캐스팅).

#### ⑥ 훈련하기 (배치 경사 하강법)

소프트맥스의 그레이디언트도 선형 회귀와 모양이 똑같습니다.

```
그레이디언트 = (1/m) × Xᵀ × (예측 확률 − 정답 원-핫)
```

```python
eta = 0.5
n_epochs = 5001
m = len(X_train)       # 90
epsilon = 1e-5

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))   # (3, 3) 무작위 시작

for epoch in range(n_epochs):
    logits = X_train @ Theta                         # 점수 (90, 3)
    Y_proba = softmax(logits)                        # 확률 (90, 3)
    if epoch % 1000 == 0:                            # 1000바퀴마다 검증 손실 출력
        Y_proba_valid = softmax(X_valid @ Theta)
        xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
        print(epoch, xentropy_losses.sum(axis=1).mean())
    error = Y_proba - Y_train_one_hot                # 확률 오차
    gradients = 1 / m * X_train.T @ error            # 경사
    Theta = Theta - eta * gradients                  # 한 걸음
```

```
0 2.973977344302766
1000 0.09313918303944392
2000 0.08904698946125779
3000 0.08847558719791135
4000 0.08861927669821175
5000 0.08894346006981158
```

**코드 포인트**

- `Theta` `(3, 3)` — 행은 특성(편향, 길이, 너비), 열은 클래스(setosa, versicolor, virginica). **열 하나가 클래스 하나의 θ**입니다.
- `+ epsilon` — 확률이 정확히 0이면 log(0) = −∞가 되어 계산이 깨집니다. 아주 작은 수를 더해 이를 막습니다.
- `Y_one_hot * np.log(...)` — 원-핫은 정답 자리만 1이므로, 곱하면 **정답 클래스의 log 확률만 남습니다.** 크로스 엔트로피 공식 그대로입니다.
- `epoch % 1000 == 0` — 1000으로 나눈 나머지가 0일 때, 즉 0, 1000, 2000…번째에만 출력합니다.

**결과 해석**

손실이 2.97 → 0.09로 크게 줄었습니다. 그런데 **3000바퀴 이후로 다시 조금씩 오릅니다** (0.0885 → 0.0889). 과대적합이 시작되는 신호입니다.

```python
logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)              # 확률이 가장 큰 클래스 번호
accuracy_score = (y_predict == y_valid).mean()  # 맞힌 비율
accuracy_score
# np.float64(0.9333333333333333)
```

- `argmax(axis=1)` — 각 행에서 가장 큰 값의 **위치(번호)**. 즉 예측 클래스.
- `(y_predict == y_valid).mean()` — True는 1, False는 0으로 계산되므로 평균이 곧 **정확도**입니다.
- 검증 정확도 **93.3%**.

#### ⑦ 릿지(ℓ₂) 규제 추가하기

```python
alpha = 0.01   # 규제 세기

# ... (앞부분 같음)
for epoch in range(n_epochs):
    ...
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()                  # 가중치 제곱 합 (편향 행 제외)
    total_loss = xentropy_losses.sum(axis=1).mean() + alpha * l2_loss
    ...
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]   # 규제의 경사 추가
    Theta = Theta - eta * gradients
```

```
0 3.0065
1000 0.2711
...
5000 0.2711
```

- `Theta[1:]` — 0번 행(편향)은 빼고 규제합니다. 5-1에서 배운 대로입니다.
- `½θ²`를 미분하면 `θ`가 되므로, 규제의 경사는 `α × θ`입니다.
- `np.r_[a, b]` — a와 b를 **세로(행) 방향으로** 이어 붙입니다. 편향 자리엔 0, 나머지 자리엔 `α × θ`가 들어간 `(3, 3)` 행렬이 됩니다.

**결과**: 손실 숫자는 0.27로 더 커졌습니다(벌점이 더해졌으니까요). 하지만 **검증 정확도는 93.3% → 96.7%로 올랐습니다.** 규제가 과대적합을 막아 준 것입니다.

#### ⑧ 조기 종료 추가하기

```python
eta = 0.5
n_epochs = 50_001
C = 100        # 규제 (α = 1/C = 0.01, 위와 같은 세기)
best_loss = np.inf

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    Y_proba_valid = softmax(X_valid @ Theta)                   # 매 바퀴 검증 손실 계산
    xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()
    total_loss = xentropy_losses.sum(axis=1).mean() + 1 / C * l2_loss
    if epoch % 1000 == 0:
        print(epoch, total_loss.round(4))
    if total_loss < best_loss:          # 좋아졌으면 기록 갱신
        best_loss = total_loss
    else:                               # 나빠졌으면 멈춤!
        print(epoch - 1, best_loss.round(4))
        print(epoch, total_loss.round(4), "early stopping!")
        break
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), 1 / C * Theta[1:]]
    Theta = Theta - eta * gradients
```

```
0 3.0065
402 0.2711
403 0.2711 early stopping!
```

- `50_001` — 숫자 사이 `_`는 읽기 편하라고 넣는 구분자일 뿐, 50001과 같습니다.
- 조기 종료를 하려면 **매 바퀴** 검증 손실을 계산해야 합니다.
- `break` — 반복문을 즉시 빠져나갑니다.

**결과**: 5000바퀴 대신 **403바퀴**에서 멈췄습니다. 정확도는 그대로 96.7%. **같은 성능을 10배 이상 빨리** 얻었습니다.

> 💡 여기서는 손실이 한 번만 올라도 바로 멈췄습니다. 배치 GD는 곡선이 매끄러워서 이렇게 해도 괜찮습니다. SGD였다면 연습문제 6번 답처럼 좀 더 기다려 봐야 합니다.

#### ⑨ 최종 평가

```python
# 새 데이터도 훈련 때와 똑같이 처리해야 함: 스케일링 → 편향 열 추가
X_new = (X_new - mean) / std
X_new_with_bias = np.c_[np.ones(len(X_new)), X_new]

# 테스트 세트 정확도
logits = X_test @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)
accuracy_score = (y_predict == y_test).mean()
accuracy_score
# np.float64(0.9666666666666667)
```

테스트 정확도도 **96.7%**입니다. 보통은 테스트가 검증보다 조금 낮게 나오는데, 이번엔 같았습니다. 데이터가 작아서 무작위 시드를 바꾸면 결과가 조금씩 달라질 수 있습니다.

---

## 8. 전체 흐름 요약

### 이 장의 이야기를 한 장으로

```
y = 4 + 3x + 잡음 (직선 데이터)
   │
   ├─ 공식으로 한 번에: 정규 방정식 / 유사역행렬 / LinearRegression     ← 4.1
   │     → 정확하지만 특성이 많으면 느림
   │
   └─ 조금씩 내려가기: 배치 GD → SGD(+학습 스케줄) → 미니배치 GD         ← 4.2
         → 큰 데이터에도 쓸 수 있음. 스케일링 필수
   ↓ 데이터가 곡선이라면?
PolynomialFeatures로 x² 추가 + LinearRegression                       ← 4.3
   ↓ 몇 차가 적당한지 어떻게 알지?
learning_curve로 진단                                                  ← 4.4
   · 두 곡선이 높고 붙어 있다 → 과소적합 → 모델을 복잡하게
   · 두 곡선 사이에 간격이 있다 → 과대적합 → 데이터 추가 / 규제
   ↓ 과대적합을 막으려면?
Ridge(제곱 벌점) / Lasso(절댓값 벌점, 0으로 만듦) / ElasticNet / 조기 종료  ← 4.5
   ↓ 분류에도 쓰려면?
로지스틱 회귀 (시그모이드 → 확률 → 0.5 기준)                            ← 4.6
소프트맥스 회귀 (클래스별 점수 → 확률 → 가장 큰 것)
```

### 모델과 비용 함수 정리

| 모델 | 예측하는 법 | 비용 함수 | 한 번에 푸는 공식 | 사이킷런 |
|---|---|---|---|---|
| 선형 회귀 | 가중치 합 | MSE | 있음 | `LinearRegression`, `SGDRegressor` |
| 다항 회귀 | 거듭제곱 특성의 가중치 합 | MSE | 있음 | `PolynomialFeatures` + `LinearRegression` |
| 릿지 | 가중치 합 | MSE + 가중치 제곱 벌점 | 있음 | `Ridge` |
| 라쏘 | 가중치 합 | MSE + 가중치 절댓값 벌점 | 없음 | `Lasso` |
| 엘라스틱넷 | 가중치 합 | MSE + 둘을 섞은 벌점 | 없음 | `ElasticNet` |
| 로지스틱 회귀 | 시그모이드(가중치 합) | 로그 손실 | 없음 (그래도 그릇 모양) | `LogisticRegression` |
| 소프트맥스 회귀 | 소프트맥스(클래스별 가중치 합) | 크로스 엔트로피 | 없음 (그래도 그릇 모양) | `LogisticRegression` (클래스 3개 이상) |

### 실험 결과 모음

| 실험 | 결과 |
|---|---|
| 정규 방정식 / `LinearRegression` / 배치 GD | θ₀ = 3.691, θ₁ = 3.330 (정답 4, 3) — 세 방법 모두 같음 |
| SGD 직접 구현 (50바퀴) | θ₀ = 3.698, θ₁ = 3.307 |
| `SGDRegressor` | θ₀ = 3.689, θ₁ = 3.331 |
| 학습률 0.5 | 발산 (값이 10⁵⁹까지 폭발) |
| 2차 다항 회귀 | 0.505x² + 1.110x + 2.005 (정답 0.5x² + x + 2) |
| 학습 곡선 RMSE (훈련 / 검증) | 선형 ≈ 1.62 / 1.64 → 과소적합, 10차 ≈ 0.97 / 1.03 → 과대적합 경향 |
| 10차 다항 + 릿지 | α를 키우면 가중치가 작아지지만 0은 안 됨 |
| 10차 다항 + 라쏘 (α = 0.01) | 가중치 10개 중 2개만 남음 |
| 조기 종료 (90차 다항 + SGD) | 184바퀴째 최고 (검증 RMSE ≈ 1.105) |
| 로지스틱 회귀 결정 경계 | 꽃잎 너비 ≈ 1.65cm |
| 소프트맥스 예측 (길이 5, 너비 2) | virginica 96% |
| NumPy 소프트맥스 직접 구현 | 규제 없음 93.3% → 규제 + 조기 종료 96.7% (403바퀴에 종료) |

### 하이퍼파라미터, 키우면 어떻게 될까?

| 하이퍼파라미터 | 키우면 | 헷갈리기 쉬운 점 |
|---|---|---|
| 학습률 η (`eta0`) | 빨라지지만 너무 크면 발산 | — |
| 다항 차수 `degree` | 모델이 복잡해짐 → 과대적합 위험 | 특성 수가 폭발함 |
| `alpha` (릿지·라쏘·엘라스틱넷) | 규제가 **강해짐** | `SGDRegressor`는 α를 m으로 나눠야 같아짐 |
| `C` (로지스틱 회귀) | 규제가 **약해짐** | α와 **반대 방향** (C = 1/α) |
| `l1_ratio` (엘라스틱넷) | 라쏘에 가까워짐 | 0이면 릿지, 1이면 라쏘 |

### 자주 쓴 코드 패턴

| 패턴 | 하는 일 | 처음 나온 곳 |
|---|---|---|
| `add_dummy_feature(X)` 또는 `np.c_[np.ones(m), X]` | 맨 앞에 1 열 붙이기 (편향용) | 1-4 |
| `A @ B`, `A.T` | 행렬 곱, 전치 | 1-4 |
| `theta = theta - eta * gradients` | 경사 하강 한 걸음 | 2-4 |
| `X[i : i+1]` | 샘플 하나를 2차원 모양 그대로 꺼내기 | 2-5 |
| `rng.permutation(m)` | 순서 섞기 (매 바퀴 섞기, 직접 분할) | 2-6 |
| `y.ravel()` | `(m, 1)` → `(m,)` 납작하게 | 2-5 |
| `np.linspace(a, b, n).reshape(-1, 1)` | 그래프용 촘촘한 입력 만들기 | 3-3 |
| `learning_curve(..., scoring="neg_root_mean_squared_error")` | 학습 곡선 | 4-3 |
| `partial_fit()` + `deepcopy()` | 한 바퀴씩 훈련 + 최고 모델 사진 찍어 두기 | 5-5 |
| `meshgrid` → `ravel` + `np.c_` → `reshape` | 바둑판 위에 결정 경계 그리기 | 6-6 |
| `np.diag(np.ones(K))[y]` | 원-핫 인코딩 | 7-2 |
| `sum(axis=1, keepdims=True)` | 행별 합을 나눗셈에 쓸 수 있는 모양으로 | 7-2 |
| `(pred == y).mean()` | 정확도 계산 | 7-2 |
