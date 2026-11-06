# 4장. 모델 훈련

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `04_training_linear_models.ipynb`
>
> 2·3장에서 블랙박스로 사용한 `fit()`의 내부를 다룬다. 선형 회귀를 닫힌 형태 해와 경사 하강법으로 풀고, 다항 회귀·학습 곡선·규제를 거쳐 로지스틱·소프트맥스 회귀로 분류까지 확장한다. 경사 하강법은 이후 신경망 학습의 기초가 된다.

## 이 노트의 구성

각 소절은 같은 틀로 정리한다.

| 항목 | 내용 |
|---|---|
| **목적** | 이 단계에서 무엇을 해결하려는가 |
| **개념·원리** | 필요한 정의, 수식, 동작 원리 |
| **코드** | 노트북 코드 (주석 포함) |
| **결과 해석** | 출력값이 의미하는 것 |
| **정리** | 핵심 요약 |

## 장 전체 구조

| 단계 | 절 | 핵심 질문 | 결과물 |
|---|---|---|---|
| 1. 선형 회귀 | 4.1 | 최적 파라미터를 공식으로 구할 수 있는가 | 정규 방정식, 유사역행렬 |
| 2. 경사 하강법 | 4.2 | 공식을 쓸 수 없을 때 어떻게 최적화하는가 | 배치·확률적·미니배치 GD |
| 3. 다항 회귀 | 4.3 | 비선형 데이터에 선형 모델을 쓸 수 있는가 | `PolynomialFeatures` |
| 4. 학습 곡선 | 4.4 | 모델이 과소적합인가 과대적합인가 | 학습 곡선, 편향/분산 |
| 5. 규제 | 4.5 | 과대적합을 어떻게 줄이는가 | 릿지·라쏘·엘라스틱넷·조기 종료 |
| 6. 로지스틱 회귀 | 4.6 | 회귀 모델로 분류할 수 있는가 | 로지스틱·소프트맥스 회귀 |

## 데이터

이 장은 대부분 **생성식을 알고 있는 인공 데이터**를 사용한다. 정답 파라미터를 알고 있으므로 모델이 이를 복원하는지 바로 검증할 수 있다.

| 데이터 | 샘플 수 | 생성식 | 사용 절 |
|---|---|---|---|
| 선형 데이터 | 200 | $y = 4 + 3x + \epsilon$, $x \in [0, 2)$ | 4.1, 4.2 |
| 2차 데이터 | 200 | $y = 0.5x^2 + x + 2 + \epsilon$, $x \in [-3, 3)$ | 4.3, 4.4, 4.5 조기 종료 |
| 작은 선형 데이터 | 20 | $y = 1 + 0.5x + \epsilon / 1.5$, $x \in [0, 3)$ | 4.5 릿지·라쏘·엘라스틱넷 |
| 붓꽃(iris) | 150 | `load_iris()` — 3품종, 특성 4개 | 4.6 |

($\epsilon$: 표준정규분포 잡음)

## 목차

1. [선형 회귀 (4.1)](#1-선형-회귀-41)
   - [1-1. 모델과 비용 함수](#1-1-모델과-비용-함수)
   - [1-2. 실습 데이터 생성](#1-2-실습-데이터-생성)
   - [1-3. 정규 방정식](#1-3-정규-방정식)
   - [1-4. LinearRegression과 유사역행렬](#1-4-linearregression과-유사역행렬)
   - [1-5. 계산 복잡도](#1-5-계산-복잡도)
2. [경사 하강법 (4.2)](#2-경사-하강법-42)
   - [2-1. 경사 하강법의 원리](#2-1-경사-하강법의-원리)
   - [2-2. 배치 경사 하강법](#2-2-배치-경사-하강법)
   - [2-3. 학습률의 영향](#2-3-학습률의-영향)
   - [2-4. 확률적 경사 하강법](#2-4-확률적-경사-하강법)
   - [2-5. SGDRegressor](#2-5-sgdregressor)
   - [2-6. 미니배치 경사 하강법](#2-6-미니배치-경사-하강법)
   - [2-7. 알고리즘 비교](#2-7-알고리즘-비교)
3. [다항 회귀 (4.3)](#3-다항-회귀-43)
   - [3-1. 다항 특성 변환](#3-1-다항-특성-변환)
   - [3-2. 다항 회귀 훈련과 예측](#3-2-다항-회귀-훈련과-예측)
4. [학습 곡선 (4.4)](#4-학습-곡선-44)
   - [4-1. 차수에 따른 적합도](#4-1-차수에-따른-적합도)
   - [4-2. 학습 곡선: 과소적합 모델](#4-2-학습-곡선-과소적합-모델)
   - [4-3. 학습 곡선: 과대적합 모델](#4-3-학습-곡선-과대적합-모델)
   - [4-4. 편향/분산 트레이드오프](#4-4-편향분산-트레이드오프)
5. [규제가 있는 선형 모델 (4.5)](#5-규제가-있는-선형-모델-45)
   - [5-1. 규제의 원리](#5-1-규제의-원리)
   - [5-2. 릿지 회귀](#5-2-릿지-회귀)
   - [5-3. 라쏘 회귀](#5-3-라쏘-회귀)
   - [5-4. ℓ1과 ℓ2 규제의 차이](#5-4-ℓ1과-ℓ2-규제의-차이)
   - [5-5. 엘라스틱넷](#5-5-엘라스틱넷)
   - [5-6. 조기 종료](#5-6-조기-종료)
6. [로지스틱 회귀 (4.6)](#6-로지스틱-회귀-46)
   - [6-1. 확률 추정](#6-1-확률-추정)
   - [6-2. 훈련과 비용 함수](#6-2-훈련과-비용-함수)
   - [6-3. 붓꽃 데이터](#6-3-붓꽃-데이터)
   - [6-4. 결정 경계: 특성 1개](#6-4-결정-경계-특성-1개)
   - [6-5. 결정 경계: 특성 2개](#6-5-결정-경계-특성-2개)
   - [6-6. 소프트맥스 회귀](#6-6-소프트맥스-회귀)
7. [연습문제 (4.7)](#7-연습문제-47)
8. [정리](#8-정리)

---

## 1. 선형 회귀 (4.1)

### 1-1. 모델과 비용 함수

**목적**
선형 회귀 모델의 형태와 "좋은 모델"의 기준(비용 함수)을 정의한다.

**개념·원리**

**선형 회귀 모델**: 입력 특성의 가중치 합에 편향(절편)을 더해 예측한다.

```math
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
```

벡터 형태 ($x_0 = 1$을 추가):

```math
\hat{y} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x}
```

| 기호 | 의미 |
|---|---|
| $\hat{y}$ | 예측값 |
| $n$ | 특성 수 |
| $x_i$ | $i$번째 특성값 ($x_0 = 1$: 편향용 더미 특성) |
| $\theta_j$ | $j$번째 파라미터. $\theta_0$ = 편향, $\theta_1 \sim \theta_n$ = 가중치 |
| $h_{\boldsymbol{\theta}}$ | 파라미터 $\boldsymbol{\theta}$를 사용하는 가설 함수 |

**비용 함수 (MSE)**: 훈련 = 비용 함수를 최소화하는 $\boldsymbol{\theta}$ 탐색.

```math
\text{MSE}(\boldsymbol{\theta}) = \frac{1}{m}\sum_{i=1}^{m}\left(\boldsymbol{\theta}^\top \mathbf{x}^{(i)} - y^{(i)}\right)^2
```

| 항목 | 이유 |
|---|---|
| 평가는 RMSE, 훈련은 MSE | $\sqrt{\cdot}$는 단조 증가 → 최솟값 위치가 같고, MSE가 미분이 간단 |
| $x_0 = 1$ 추가 | 편향과 가중치를 하나의 벡터로 다룸 → 행렬 곱 한 번으로 예측 |

**정리**
- 선형 회귀 = 특성의 가중치 합 + 편향
- 훈련 = MSE를 최소화하는 $\boldsymbol{\theta}$ 찾기
- 풀이 방법 2가지: 닫힌 형태 해(1-3) / 경사 하강법(2장)

---

### 1-2. 실습 데이터 생성

**목적**
생성식을 알고 있는 선형 데이터를 만들어 모델이 정답 파라미터를 복원하는지 검증한다.

**코드**

```python
import numpy as np

rng = np.random.default_rng(seed=42)          # 난수 생성기, 시드 고정 → 재현 가능
m = 200                                        # 샘플 수
X = 2 * rng.random((m, 1))                     # [0, 2) 균등분포, 열 벡터 (200, 1)
y = 4 + 3 * X + rng.standard_normal((m, 1))    # 진짜 관계 y = 4 + 3x + 잡음
```

```python
# 그림 4-1: 데이터 산점도
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.show()
```

**원리**
- `rng.random((m, 1))`: [0, 1) 균등분포 → ×2 하여 [0, 2)
- `rng.standard_normal((m, 1))`: 평균 0, 표준편차 1 정규분포 잡음
- `(m, 1)` 모양: 사이킷런은 X를 2차원(샘플 × 특성)으로 기대 → 특성이 1개여도 열 벡터

**정리**
- 정답 파라미터: $\theta_0 = 4$, $\theta_1 = 3$

---

### 1-3. 정규 방정식

**목적**
MSE를 최소화하는 $\boldsymbol{\theta}$를 반복 없이 공식 한 번으로 계산한다.

**개념·원리**

**정규 방정식** (Normal Equation): MSE의 그레이디언트를 0으로 놓고 풀어 얻은 닫힌 형태 해.

```math
\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^\top \mathbf{X}\right)^{-1} \mathbf{X}^\top \mathbf{y}
```

- $\mathbf{X}$: 각 행이 샘플, 첫 열이 1인 $m \times (n+1)$ 행렬
- $\mathbf{y}$: 타깃 벡터

**코드**

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)                                # 각 샘플 앞에 x0 = 1 추가 → (200, 2)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y       # (XᵀX)⁻¹ Xᵀ y
theta_best
# array([[3.69084138],
#        [3.32960458]])
```

| 코드 | 수식 |
|---|---|
| `X_b.T` | $\mathbf{X}^\top$ (전치) |
| `@` | 행렬 곱 |
| `np.linalg.inv()` | 역행렬 $(\cdot)^{-1}$ |

**코드: 예측**

```python
X_new = np.array([[0], [2]])                 # 직선 양 끝 두 점
X_new_b = add_dummy_feature(X_new)           # x0 = 1 추가
y_predict = X_new_b @ theta_best             # ŷ = Xθ
y_predict
# array([[ 3.69084138],
#        [10.35005055]])

# 그림 4-2: 예측 직선
plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")
plt.show()
```

**결과 해석**

| 파라미터 | 정답 | 정규 방정식 |
|---|---|---|
| $\theta_0$ | 4 | 3.691 |
| $\theta_1$ | 3 | 3.330 |

- 정답과의 차이는 **잡음** 때문. 샘플 200개로는 정확한 복원이 불가능하며, 샘플이 많을수록 정답에 가까워진다
- x = 0 → 3.69, x = 2 → 10.35 → 두 점을 이어 예측 직선을 그린다

**정리**
- 정규 방정식 = 반복 없이 한 번에 최적해
- 특성 스케일링이 필요 없다

---

### 1-4. LinearRegression과 유사역행렬

**목적**
사이킷런의 선형 회귀 구현과 내부 계산 방식을 확인한다.

**개념·원리**

`LinearRegression`은 정규 방정식 대신 **유사역행렬**(pseudoinverse, 무어-펜로즈 역행렬)을 사용한다.

```math
\hat{\boldsymbol{\theta}} = \mathbf{X}^{+} \mathbf{y}
```

- $\mathbf{X}^{+}$는 **특잇값 분해(SVD)** $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$로 계산: $\mathbf{X}^{+} = \mathbf{V}\boldsymbol{\Sigma}^{+}\mathbf{U}^\top$
- $\boldsymbol{\Sigma}^{+}$: 아주 작은 특잇값은 0으로, 나머지는 역수로 바꾼 뒤 전치

| 비교 | 정규 방정식 | 유사역행렬 (SVD) |
|---|---|---|
| $\mathbf{X}^\top\mathbf{X}$가 역행렬이 없을 때 ($m < n$, 중복 특성) | 계산 불가 | 항상 계산 가능 |
| 특성 수에 대한 복잡도 | $O(n^{2.4}) \sim O(n^3)$ | $O(n^2)$ |

**코드**

```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)                     # 편향 열을 직접 추가할 필요 없음
lin_reg.intercept_, lin_reg.coef_     # 편향과 가중치를 따로 저장
# (array([3.69084138]), array([[3.32960458]]))

lin_reg.predict(X_new)
# array([[ 3.69084138],
#        [10.35005055]])
```

```python
# LinearRegression의 기반: scipy.linalg.lstsq (least squares)
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
# array([[3.69084138],
#        [3.32960458]])

np.linalg.pinv(X_b) @ y               # 유사역행렬을 직접 계산 → 동일 결과
```

**결과 해석**

| 방법 | $\theta_0$ | $\theta_1$ |
|---|---|---|
| 정규 방정식 | 3.691 | 3.330 |
| `LinearRegression` | 3.691 | 3.330 |
| `np.linalg.lstsq` | 3.691 | 3.330 |
| `np.linalg.pinv` | 3.691 | 3.330 |

- 네 방법 결과 **완전히 동일**

**정리**
- `intercept_` = $\theta_0$, `coef_` = $\theta_1 \sim \theta_n$ (끝의 `_` = 학습으로 얻은 값)
- 유사역행렬이 정규 방정식보다 안정적이고 빠르다

---

### 1-5. 계산 복잡도

**목적**
닫힌 형태 해의 적용 한계를 파악한다.

**개념·원리**

| 방법 | 특성 수 $n$ | $n$이 2배일 때 | 샘플 수 $m$ |
|---|---|---|---|
| 정규 방정식 | $O(n^{2.4}) \sim O(n^3)$ | 약 5.3 ~ 8배 | $O(m)$ |
| SVD | $O(n^2)$ | 약 4배 | $O(m)$ |
| 예측 | $O(n)$ | 2배 | $O(m)$ |

**결과 해석**
- 특성이 수십만 개 → 사실상 사용 불가
- 샘플 수에는 선형 → 메모리에 들어가기만 하면 대용량도 처리 가능
- 전체 훈련 세트를 한 번에 메모리에 올려야 한다

**정리**
- 특성이 매우 많거나 데이터가 메모리에 들어가지 않으면 → **경사 하강법**(2장)

---

## 2. 경사 하강법 (4.2)

### 2-1. 경사 하강법의 원리

**목적**
닫힌 형태 해 없이 비용 함수를 최소화하는 일반적인 방법을 이해한다.

**개념·원리**

**경사 하강법** (Gradient Descent): 비용 함수의 **그레이디언트 반대 방향**으로 파라미터를 반복적으로 조정해 최솟값을 찾는다.

비유: 안개 낀 산에서 발밑의 가장 가파른 내리막 방향으로 한 걸음씩 이동.

```math
\boldsymbol{\theta}^{(\text{next})} = \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})
```

| 단계 | 내용 |
|---|---|
| 1 | $\boldsymbol{\theta}$를 무작위로 초기화 (random initialization) |
| 2 | 현재 $\boldsymbol{\theta}$에서 그레이디언트 계산 |
| 3 | 그레이디언트 반대 방향으로 $\eta$만큼 이동 |
| 4 | 그레이디언트 ≈ 0 (수렴)까지 2~3 반복 |

| 요소 | 내용 |
|---|---|
| 학습률 $\eta$ | 스텝 크기. 너무 작으면 느림, 너무 크면 발산 |
| 지역 최솟값 | 전역 최솟값이 아닌 곳에 갇힐 위험 |
| **볼록 함수** | 선형 회귀의 MSE는 볼록 → 지역 최솟값 없음, 전역 최솟값 하나. 립시츠 연속 → 그레이디언트가 급변하지 않음 |
| 특성 스케일 | 스케일이 다르면 비용 함수가 길쭉한 그릇 모양 → 수렴 매우 느림 |

**정리**
- 경사 하강법 사용 시 **반드시 특성 스케일링** (`StandardScaler`)
- 그레이디언트를 계산할 때 사용하는 샘플 수에 따라 3종류: 배치(전체) / 확률적(1개) / 미니배치(일부)

---

### 2-2. 배치 경사 하강법

**목적**
매 스텝 전체 훈련 세트로 그레이디언트를 계산하는 기본 경사 하강법을 구현한다.

**개념·원리**

MSE의 편도함수:

```math
\frac{\partial}{\partial \theta_j}\text{MSE}(\boldsymbol{\theta}) = \frac{2}{m}\sum_{i=1}^{m}\left(\boldsymbol{\theta}^\top\mathbf{x}^{(i)} - y^{(i)}\right)x_j^{(i)}
```

그레이디언트 벡터 (모든 편도함수를 한 번에):

```math
\nabla_{\boldsymbol{\theta}}\text{MSE}(\boldsymbol{\theta}) = \frac{2}{m}\mathbf{X}^\top\left(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\right)
```

- 의미: 샘플별 오차 × 특성값의 평균. 오차가 0이면 그레이디언트도 0
- 매 스텝 전체 $\mathbf{X}$ 사용 → **배치** 경사 하강법

**코드**

```python
eta = 0.1                                   # 학습률
n_epochs = 1000                             # 에포크 수
m = len(X_b)                                # 샘플 수

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))         # 파라미터 무작위 초기화

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)   # ∇MSE = (2/m) Xᵀ(Xθ − y)
    theta = theta - eta * gradients                 # θ ← θ − η∇MSE

theta
# array([[3.69084138],
#        [3.32960458]])
```

| 코드 | 의미 |
|---|---|
| `X_b @ theta - y` | 샘플별 오차 `(200, 1)` |
| `X_b.T @ (오차)` | 파라미터별 오차 기여도 합 `(2, 1)` |
| `theta - eta * gradients` | 그레이디언트 반대 방향으로 이동 |
| 에포크 (epoch) | 훈련 세트 전체를 한 번 사용하는 단위 |

**결과 해석**

처음 3스텝 추적 ($\eta = 0.1$, 직접 계산):

| 스텝 | 그레이디언트 | $\theta_0$, $\theta_1$ | MSE |
|---|---|---|---|
| 0 | — | 0.305, −1.040 | 66.38 |
| 1 | −15.40, −17.99 | 1.845, 0.759 | 22.32 |
| 2 | −8.77, −10.29 | 2.722, 1.788 | 7.96 |
| 3 | −4.98, −5.90 | 3.220, 2.378 | 3.27 |

- 그레이디언트가 음수 → $\theta$가 증가하는 방향으로 이동
- 최솟값에 가까울수록 그레이디언트가 작아져 스텝도 자동으로 작아짐
- 1000 에포크 후 정규 방정식과 **소수점까지 동일** (3.691, 3.330)

**정리**
- 배치 GD = 매 스텝 전체 데이터 → 정확하고 매끄럽게 수렴
- 단점: 훈련 세트가 크면 매우 느림

---

### 2-3. 학습률의 영향

**목적**
학습률에 따라 수렴 양상이 어떻게 달라지는지 비교한다.

**코드**

```python
# 그림 4-8
import matplotlib as mpl

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_epochs = 1000
    n_shown = 20
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < n_shown:                                     # 처음 20스텝만 직선 표시
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))  # 진행할수록 진한 색
            plt.plot(X_new, y_predict, linestyle="solid", color=color)
        gradients = 2 / m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
        theta_path.append(theta)                                # 2-6 경로 비교용
    plt.xlabel("$x_1$")
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.title(fr"$\eta = {eta}$")
    return theta_path

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

plt.figure(figsize=(10, 4))
plt.subplot(131)                                                # 1행 3열 중 1번째
plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0)
plt.subplot(132)
theta_path_bgd = plot_gradient_descent(theta, eta=0.1)
plt.gca().axes.yaxis.set_ticklabels([])
plt.subplot(133)
plt.gca().axes.yaxis.set_ticklabels([])
plot_gradient_descent(theta, eta=0.5)
plt.show()
```

**결과 해석** (직접 계산)

| $\eta$ | 20스텝 후 $\boldsymbol{\theta}$ | 1000스텝 후 $\boldsymbol{\theta}$ | 판정 |
|---|---|---|---|
| 0.02 | 3.29, 2.47 | 3.69, 3.33 | 너무 작음 → 느림 |
| 0.1 | 3.81, 3.23 | 3.69, 3.33 | 적절 |
| 0.5 | −50.3, −59.3 | $\approx -2 \times 10^{59}$ | 너무 큼 → **발산** |

**정리**
- 적절한 학습률은 **그리드 서치**로 탐색
- 반복 횟수는 크게 잡고, 그레이디언트 벡터의 노름이 허용오차 $\epsilon$보다 작아지면 중지
- 볼록 함수 + 고정 학습률이면 허용오차 $\epsilon$ 도달까지 $O(1/\epsilon)$ 반복

---

### 2-4. 확률적 경사 하강법

**목적**
대용량 데이터에서도 빠르게 훈련하기 위해 샘플 1개씩 사용하는 경사 하강법을 구현한다.

**개념·원리**

**확률적 경사 하강법** (Stochastic GD, SGD): 매 스텝 **샘플 1개를 무작위로** 선택해 그레이디언트를 계산한다.

| 비교 | 배치 GD | SGD |
|---|---|---|
| 스텝당 샘플 | $m$개 | 1개 |
| 속도 | 느림 | 매우 빠름 |
| 경로 | 매끄러움 | 불규칙 (지그재그) |
| 최솟값 근처 | 정지 | 계속 요동 |
| 지역 최솟값 | 갇힐 수 있음 | 요동 덕분에 탈출 가능 |
| 외부 메모리 학습 | 불가 | 가능 |

**학습 스케줄** (learning schedule): 요동 문제를 해결하기 위해 학습률을 점진적으로 감소 (담금질 기법과 유사).

```math
\eta(t) = \frac{t_0}{t + t_1}
```

- $t$: 누적 스텝 수
- 너무 빨리 줄이면 지역 최솟값에 갇히거나 중간에 멈춤, 너무 천천히 줄이면 계속 요동

**코드**

```python
theta_path_sgd = []                       # 2-6 경로 비교용

n_epochs = 50
t0, t1 = 5, 50                            # 학습 스케줄 하이퍼파라미터

def learning_schedule(t):
    return t0 / (t + t1)

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))       # 무작위 초기화

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = rng.integers(m)                     # 무작위 샘플 1개 (복원 추출)
        xi = X_b[random_index : random_index + 1]          # 슬라이싱 → (1, 2) 2차원 유지
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)           # 샘플 1개 → m으로 나누지 않음
        eta = learning_schedule(epoch * m + iteration)     # 누적 스텝 수로 학습률 계산
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

theta
# array([[3.69826475],
#        [3.30748311]])
```

| 코드 | 의미 |
|---|---|
| `rng.integers(m)` | 복원 추출 → 한 에포크에 어떤 샘플은 여러 번, 어떤 샘플은 0번 |
| `X_b[i : i+1]` | `X_b[i]`는 1차원 `(2,)` → 행렬 곱 불가. 슬라이싱으로 `(1, 2)` 유지 |
| `epoch * m + iteration` | 누적 스텝 수 $t$ |

**결과 해석**

학습 스케줄에 따른 학습률:

| $t$ | 0 | 50 | 500 | 5,000 | 9,999 (마지막) |
|---|---|---|---|---|---|
| $\eta$ | 0.1 | 0.05 | 0.009 | 0.001 | 0.0005 |

| 방법 | 에포크 | $\theta_0$ | $\theta_1$ |
|---|---|---|---|
| 배치 GD | 1000 | 3.691 | 3.330 |
| SGD | **50** | 3.698 | 3.307 |

- 1/20 에포크로 거의 같은 위치 도달. 다만 요동 때문에 정확히 같지는 않음
- 그림 4-10: 첫 에포크 20스텝의 직선이 불규칙하게 움직임

**정리**
- SGD는 샘플이 **IID**(독립 동일 분포)여야 한다 → 레이블 순으로 정렬된 데이터는 반드시 섞기
- 학습 스케줄로 요동을 줄여 수렴시킨다

---

### 2-5. SGDRegressor

**목적**
사이킷런의 SGD 기반 선형 회귀를 사용한다.

**코드**

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())          # fit()은 1차원 타깃을 기대 → (200, 1)을 (200,)로

sgd_reg.intercept_, sgd_reg.coef_
# (array([3.68899733]), array([3.33054574]))
```

| 매개변수 | 의미 |
|---|---|
| `max_iter=1000` | 최대 에포크 수 |
| `tol=1e-5`, `n_iter_no_change=100` | 100 에포크 동안 손실 개선이 $10^{-5}$ 미만이면 중지 |
| `penalty=None` | 규제 없음 (5장에서 사용) |
| `eta0=0.01` | 초기 학습률 (기본 학습 스케줄로 감소) |

**결과 해석**
- $\theta_0 = 3.689$, $\theta_1 = 3.331$ → 정규 방정식 해와 매우 가까움

**정리**
- `SGDRegressor`는 기본으로 매 에포크 훈련 세트를 섞는다

---

### 2-6. 미니배치 경사 하강법

**목적**
배치 GD와 SGD의 장점을 절충한 방법을 구현하고, 세 방법의 이동 경로를 비교한다.

**개념·원리**

**미니배치 경사 하강법**: 매 스텝 **작은 무작위 샘플 묶음**(미니배치)으로 그레이디언트를 계산한다.

```math
\nabla_{\boldsymbol{\theta}} = \frac{2}{b}\mathbf{X}_B^\top\left(\mathbf{X}_B\boldsymbol{\theta} - \mathbf{y}_B\right) \qquad (B: \text{크기 } b\text{인 미니배치})
```

- 행렬 연산 최적화(특히 **GPU**)로 SGD보다 빠르고 덜 요동
- 신경망 학습의 표준 방식

**코드**

```python
from math import ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)    # 200 / 20 = 10 (나누어떨어지지 않으면 올림)

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

theta_path_mgd = []
for epoch in range(n_epochs):
    shuffled_indices = rng.permutation(m)         # 매 에포크 섞기 (비복원)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_b_shuffled[idx : idx + minibatch_size]          # 20개씩
        yi = y_shuffled[idx : idx + minibatch_size]
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
```

| 코드 | 의미 |
|---|---|
| `rng.permutation(m)` | 무작위 순열 → **비복원**. 한 에포크에 모든 샘플을 정확히 1번 사용 |
| `2 / minibatch_size` | 배치 크기로 평균 |

**코드: 경로 비교 (그림 4-11)**

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
plt.grid()
plt.show()
```

- `[:, 0]`, `[:, 1]`: 경로에서 $\theta_0$, $\theta_1$ 추출 → 파라미터 공간 위 이동 궤적

**결과 해석**

| 방법 | 경로 |
|---|---|
| 배치 (파랑) | 최솟값으로 직진 후 정지 |
| SGD (빨강) | 크게 요동 |
| 미니배치 (초록) | 중간 정도 요동 |

**정리**
- 미니배치 = 배치의 안정성 + SGD의 속도
- 배치 GD만 실제로 최솟값에 멈춘다. SGD·미니배치는 학습률을 줄여야 수렴

---

### 2-7. 알고리즘 비교

**목적**
선형 회귀 훈련 알고리즘의 특성을 한눈에 비교한다.

**결과 해석** (교재 표 4-1)

| 알고리즘 | $m$이 클 때 | 외부 메모리 학습 | $n$이 클 때 | 하이퍼파라미터 수 | 스케일 조정 | 사이킷런 |
|---|---|---|---|---|---|---|
| 정규 방정식 | 빠름 | X | 느림 | 0 | 불필요 | — |
| SVD | 빠름 | X | 느림 | 0 | 불필요 | `LinearRegression` |
| 배치 GD | 느림 | X | 빠름 | 2 | 필요 | — |
| SGD | 빠름 | O | 빠름 | ≥2 | 필요 | `SGDRegressor` |
| 미니배치 GD | 빠름 | O | 빠름 | ≥2 | 필요 | — |

**정리**
- 훈련 방법만 다를 뿐 **최종 모델은 동일** (예측 방식이 같음)
- 특성이 많거나 데이터가 크면 경사 하강법 계열

---

## 3. 다항 회귀 (4.3)

### 3-1. 다항 특성 변환

**목적**
비선형 데이터에 선형 모델을 적용하기 위해 거듭제곱 특성을 추가한다.

**개념·원리**

**다항 회귀**: 각 특성의 거듭제곱을 새 특성으로 추가한 뒤 선형 모델을 훈련한다.

```math
\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2
```

- $x^2$를 새로운 특성으로 보면 일반 선형 회귀와 같음
- $\boldsymbol{\theta}$에 대해 선형 → 여전히 "선형 모델"

특성 $n$개, 차수 $d$일 때 생성되는 특성 수 (편향 포함):

```math
\frac{(n+d)!}{d!\,n!}
```

- 교차항까지 생성: 특성 $a, b$와 $d = 3$ → $a^2, a^3, b^2, b^3$ + $ab, a^2b, ab^2$

**코드**

```python
# 2차 데이터 생성
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3                          # [-3, 3)
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))  # 정답: 0.5x² + x + 2

# 그림 4-12: 데이터 산점도
plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([-3, 3, 0, 10])
plt.grid()
plt.show()
```

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]           # array([1.64373629])
X_poly[0]      # array([1.64373629, 2.701869  ])   ← [x, x²]
```

| 매개변수 | 의미 |
|---|---|
| `degree=2` | 2차항까지 생성 |
| `include_bias=False` | $x^0 = 1$ 열 생성 안 함 (`LinearRegression`이 절편을 별도 학습) |

**결과 해석**
- $x = 1.644$ → $[1.644,\ 2.702]$ ($2.702 = 1.644^2$)

**정리**
- `PolynomialFeatures`로 특성 확장 → 선형 모델로 곡선 학습
- 특성 수가 조합적으로 폭발 (예: $n = 10$, $d = 3$ → 편향 제외 285개)

---

### 3-2. 다항 회귀 훈련과 예측

**목적**
확장된 특성으로 선형 회귀를 훈련하고 생성식을 복원하는지 확인한다.

**코드**

```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# (array([2.00540719]), array([[1.11022126, 0.50526985]]))
```

```python
# 그림 4-13: 예측 곡선
X_new = np.linspace(-3, 3, 100).reshape(100, 1)   # 균등 간격 100점 → 매끄러운 곡선
X_new_poly = poly_features.transform(X_new)       # 새 데이터는 transform만
y_new = lin_reg.predict(X_new_poly)

plt.figure(figsize=(6, 4))
plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.legend(loc="upper left")
plt.axis([-3, 3, 0, 10])
plt.grid()
plt.show()
```

**결과 해석**

| 항 | 정답 | 모델 |
|---|---|---|
| 절편 | 2 | 2.005 |
| $x$ | 1 | 1.110 |
| $x^2$ | 0.5 | 0.505 |

- $\hat{y} = 0.505x^2 + 1.110x + 2.005$ → 생성식을 거의 정확히 복원
- `coef_` 순서 = `X_poly` 열 순서 ($x$, $x^2$)

**정리**
- 다항 회귀 = `PolynomialFeatures` + `LinearRegression`
- 차수가 높을수록 과대적합 위험 → 4장에서 진단

---

## 4. 학습 곡선 (4.4)

### 4-1. 차수에 따른 적합도

**목적**
다항 차수가 적합도에 미치는 영향을 관찰한다.

**코드**

```python
# 그림 4-14
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

plt.figure(figsize=(6, 4))

for style, width, degree in (("r-+", 2, 1), ("b--", 2, 2), ("g-", 1, 300)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()                    # 고차항 값의 크기 차이 → 수치 안정화
    lin_reg = LinearRegression()
    polynomial_regression = make_pipeline(polybig_features, std_scaler, lin_reg)
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    label = f"{degree} degree{'s' if degree > 1 else ''}"
    plt.plot(X_new, y_newbig, style, label=label, linewidth=width)

plt.plot(X, y, "b.", linewidth=3)
plt.legend(loc="upper left")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([-3, 3, 0, 10])
plt.grid()
plt.show()
```

**결과 해석**

| 차수 | 결과 | 진단 |
|---|---|---|
| 1 | 포물선을 무시한 직선 | **과소적합** |
| 2 | 데이터와 잘 맞는 포물선 | 적절 (생성식과 같은 차수) |
| 300 | 훈련 샘플을 지나가려 심하게 구불거림 | **과대적합** |

**정리**
- 실제 문제에서는 정답 차수를 모른다 → 진단 방법 필요
  1. **교차 검증** (2장): 훈련 점수만 좋으면 과대적합, 둘 다 나쁘면 과소적합
  2. **학습 곡선** (4-2, 4-3)

---

### 4-2. 학습 곡선: 과소적합 모델

**목적**
훈련 세트 크기에 따른 훈련·검증 오차 변화로 모델 상태를 진단한다.

**개념·원리**

**학습 곡선**: 훈련 세트 크기(또는 훈련 반복)에 따른 훈련 오차와 검증 오차를 그린 그래프.

`learning_curve()` 동작:
1. 훈련 세트 크기를 단계적으로 늘림
2. 각 크기에서 교차 검증으로 모델 훈련
3. 훈련 점수와 검증 점수 반환

**코드**

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, valid_scores = learning_curve(
    LinearRegression(), X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
train_errors = -train_scores.mean(axis=1)     # 폴드 평균 후 부호 반전 → RMSE
valid_errors = -valid_scores.mean(axis=1)

# 그림 4-15
plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.grid()
plt.legend(loc="upper right")
plt.axis([0, 160, 0, 2.5])
plt.show()
```

| 인자·반환값 | 의미 |
|---|---|
| `train_sizes=np.linspace(0.01, 1.0, 40)` | 훈련 폴드의 1% ~ 100%, 40단계 |
| `cv=5` | 5-폴드 → 훈련 폴드 최대 $200 \times 4/5 = 160$개 |
| `scoring="neg_root_mean_squared_error"` | "클수록 좋음" 규약 → 음수 RMSE |
| `train_scores` | `(40, 5)` — 크기별 × 폴드별 |

**결과 해석** (직접 계산)

| 곡선 | 시작 | 훈련 160개 | 변화 |
|---|---|---|---|
| 훈련 오차 | 0 | ≈ 1.62 | 샘플 1~2개는 완벽히 맞춤 → 증가 후 평평 |
| 검증 오차 | ≈ 3.62 | ≈ 1.64 | 일반화 불가 상태에서 감소 후 평평 |

- 두 곡선이 **높은 값에서 붙어 평평** → 전형적인 **과소적합**

**정리**
- 과소적합이면 **훈련 데이터를 추가해도 효과 없음**
- 대응: 더 복잡한 모델, 더 나은 특성

---

### 4-3. 학습 곡선: 과대적합 모델

**목적**
복잡한 모델(10차 다항)의 학습 곡선 형태를 확인한다.

**코드**

```python
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")

# 그림 4-16
train_errors = -train_scores.mean(axis=1)
valid_errors = -valid_scores.mean(axis=1)

plt.figure(figsize=(6, 4))
plt.plot(train_sizes, train_errors, "r-+", linewidth=2, label="train")
plt.plot(train_sizes, valid_errors, "b-", linewidth=3, label="valid")
plt.legend(loc="upper right")
plt.xlabel("Training set size")
plt.ylabel("RMSE")
plt.grid()
plt.axis([0, 160, 0, 2.5])
plt.show()
```

**결과 해석** (훈련 160개, 직접 계산)

| 모델 | 훈련 RMSE | 검증 RMSE | 곡선 형태 | 진단 |
|---|---|---|---|---|
| 선형 회귀 | 1.62 | 1.64 | 높은 값에서 붙음 | 과소적합 |
| 10차 다항 | **0.97** | 1.03 | 훈련 오차 낮음, **두 곡선 간격** | 과대적합 경향 |

- 훈련 세트가 커질수록 두 곡선의 간격이 줄어든다

**정리**

| 학습 곡선 형태 | 진단 | 대응 |
|---|---|---|
| 두 곡선이 높은 값에서 붙어 평평 | 과소적합 | 복잡한 모델, 더 나은 특성 (데이터 추가 무효) |
| 훈련 오차 낮음 + 두 곡선 간격 | 과대적합 | 데이터 추가, 모델 단순화, 규제 |

---

### 4-4. 편향/분산 트레이드오프

**목적**
일반화 오차의 구성 요소를 이해하고 모델 복잡도와의 관계를 정리한다.

**개념·원리**

일반화 오차 = **편향** + **분산** + **줄일 수 없는 오차**

| 오차 | 원인 | 결과 | 대응 |
|---|---|---|---|
| 편향 (bias) | 잘못된 가정 (예: 2차 데이터를 선형으로 가정) | 과소적합 | 모델 복잡도 ↑ |
| 분산 (variance) | 훈련 데이터의 작은 변동에 과도하게 민감 (자유도 높은 모델) | 과대적합 | 모델 복잡도 ↓, 데이터 ↑, 규제 |
| 줄일 수 없는 오차 | 데이터 자체의 잡음 | — | 데이터 정제 (센서 수리, 이상치 제거) |

**정리**
- 복잡도 ↑ → 편향 ↓, 분산 ↑ / 복잡도 ↓ → 편향 ↑, 분산 ↓
- 둘 사이의 균형점을 찾는 것이 모델 선택의 핵심

---

## 5. 규제가 있는 선형 모델 (4.5)

### 5-1. 규제의 원리

**목적**
과대적합을 줄이기 위해 모델의 자유도를 제한하는 방법을 이해한다.

**개념·원리**

**규제** (regularization): 모델을 제한해 과대적합을 감소시킨다.
- 다항 모델 → 차수 감소
- 선형 모델 → **가중치 크기 제한**: 비용 함수에 가중치 벌점 항 추가

```math
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \text{(규제 항)}
```

| 규칙 | 이유 |
|---|---|
| 편향 $\theta_0$은 규제하지 않음 | 절편을 줄이면 예측 전체가 아래로 이동 |
| 규제 항은 훈련 시에만 사용 | 평가는 규제 없는 성능 지표로 |
| **스케일링 필수** | 규제는 가중치 크기에 민감 → 특성 단위가 다르면 불공평한 벌점 |
| 하이퍼파라미터 $\alpha$ | $\alpha = 0$ → 규제 없음, $\alpha$ ↑ → 가중치 ↓ |

**코드: 실습 데이터**

```python
rng = np.random.default_rng(seed=42)
m = 20                                                # 샘플 20개 → 과대적합하기 쉬운 상황
X = 3 * rng.random((m, 1))
y = 1 + 0.5 * X + rng.standard_normal((m, 1)) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)
```

**코드: 공통 시각화 함수** (5-2, 5-3에서 사용)

```python
def plot_model(model_class, polynomial, alphas, **model_kwargs):
    plt.plot(X, y, "b.", linewidth=3)
    for alpha, style in zip(alphas, ("b:", "g--", "r-")):    # α 3개와 선 스타일을 짝지음
        if alpha > 0:
            model = model_class(alpha, **model_kwargs)        # Ridge, Lasso 등 클래스를 인자로 받음
        else:
            model = LinearRegression()                        # α = 0 = 규제 없음
        if polynomial:
            model = make_pipeline(                            # 10차 다항 + 스케일링
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
```

**정리**
- 규제 = 비용 함수에 가중치 벌점을 더해 "오차 최소화"와 "가중치 작게 유지" 사이 균형
- 규제 방식: 릿지(ℓ2), 라쏘(ℓ1), 엘라스틱넷(혼합), 조기 종료

---

### 5-2. 릿지 회귀

**목적**
가중치 제곱합으로 규제하는 릿지 회귀를 적용한다.

**개념·원리**

**릿지 회귀** (Ridge, 티호노프 규제): 비용 함수에 **ℓ2 노름의 제곱**을 추가.

```math
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + \frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2
```

닫힌 형태 해:

```math
\hat{\boldsymbol{\theta}} = \left(\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{A}\right)^{-1}\mathbf{X}^\top\mathbf{y}
```

- $\mathbf{A}$: 왼쪽 위(편향 자리)만 0인 $(n+1) \times (n+1)$ 단위 행렬

**코드**

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")   # 숄레스키 분해로 닫힌 형태 해 계산
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84414523])
```

```python
# 그림 4-17
plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.ylabel("$y$  ", rotation=0)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
plt.gca().axes.yaxis.set_ticklabels([])
plt.show()
```

**코드: 다른 풀이 방법**

```python
# SGD로 릿지
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,   # SGDRegressor의 α 정의 차이 → m으로 나눔
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
# array([1.83659707])

# 확률적 평균 경사 하강법
ridge_reg = Ridge(alpha=0.1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84415109])
```

**코드: 닫힌 형태 해 직접 계산**

```python
alpha = 0.1
A = np.array([[0., 0.], [0., 1.]])        # 편향 자리 0
X_b = np.c_[np.ones(m), X]                # 편향 열 추가 (열 방향 결합)
np.linalg.inv(X_b.T @ X_b + alpha * A) @ X_b.T @ y
# array([[1.23749481],
#        [0.40443361]])

ridge_reg.intercept_, ridge_reg.coef_
# (array([1.23754037]), array([0.40440715]))
```

**결과 해석**

| 풀이 방법 | `predict([[1.5]])` |
|---|---|
| `Ridge(solver="cholesky")` | 1.8441 |
| `Ridge(solver="sag")` | 1.8442 |
| `SGDRegressor(penalty="l2")` | 1.8366 |

- 닫힌 형태 해 직접 계산 $(1.2375, 0.4044)$ = `sag` 결과 → 일치

그림 4-17:

| $\alpha$ | 왼쪽: 선형 + 릿지 | 오른쪽: 10차 다항 + 릿지 |
|---|---|---|
| 0 | 일반 직선 | 심하게 구불거림 (과대적합) |
| 증가 | 기울기 감소 → 수평선에 가까워짐 | 점점 매끄러운 곡선 |

10차 다항 모델의 가중치 (직접 계산):

| $\alpha$ | 최대 가중치 크기 | 0이 아닌 가중치 |
|---|---|---|
| $10^{-5}$ | ≈ 23.25 | 10 / 10 |
| 1 | ≈ 0.48 | 10 / 10 |

**정리**
- 릿지 = ℓ2 규제, 가중치를 **줄이지만 0으로 만들지 않음**
- 닫힌 형태 해(`cholesky`)와 경사 하강법(`sag`, `SGDRegressor`) 모두 가능
- 규제 없는 선형 회귀보다 대부분 나음 → 기본 선택

---

### 5-3. 라쏘 회귀

**목적**
가중치 절댓값 합으로 규제하는 라쏘 회귀를 적용하고, 특성 선택 효과를 확인한다.

**개념·원리**

**라쏘 회귀** (Lasso, Least Absolute Shrinkage and Selection Operator): 비용 함수에 **ℓ1 노름**을 추가.

```math
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + 2\alpha\sum_{i=1}^{n}\left|\theta_i\right|
```

- 덜 중요한 특성의 가중치를 **정확히 0**으로 만든다 → **자동 특성 선택**, **희소 모델** 생성

**코드**

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
# array([1.87550211])
```

```python
# 그림 4-18
plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.ylabel("$y$  ", rotation=0)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 1e-2, 1), random_state=42)
plt.gca().axes.yaxis.set_ticklabels([])
plt.show()
```

**결과 해석**
- 그림 4-18 오른쪽 $\alpha = 0.01$: 10차 다항인데 거의 2차 곡선 형태 → 고차항 가중치가 0

10차 다항 모델의 가중치 (직접 계산):

| $\alpha$ | 0이 아닌 가중치 |
|---|---|
| 0.01 | **2** / 10 |
| 1 | **0** / 10 (수평선) |

**정리**
- 라쏘 = ℓ1 규제, 가중치를 **정확히 0**으로 → 특성 선택
- 최솟값 근처에서 요동 → SGD로 훈련할 때 학습률을 점진적으로 감소
- 특성 수 > 샘플 수, 특성 간 강한 상관이 있으면 불안정

---

### 5-4. ℓ1과 ℓ2 규제의 차이

**목적**
라쏘는 가중치를 0으로 만들고 릿지는 그렇지 않은 이유를 그레이디언트로 설명한다.

**개념·원리**

ℓ1 항 $\lvert\theta_i\rvert$는 $\theta_i = 0$에서 미분 불가 → **서브그레이디언트 벡터** 사용:

```math
g(\boldsymbol{\theta}, J) = \nabla_{\boldsymbol{\theta}}\text{MSE}(\boldsymbol{\theta}) + 2\alpha\,\text{sign}(\boldsymbol{\theta}),
\qquad
\text{sign}(\theta_i) = \begin{cases} -1 & \theta_i < 0 \\ 0 & \theta_i = 0 \\ +1 & \theta_i > 0 \end{cases}
```

| 비교 | ℓ1 (라쏘) | ℓ2 (릿지) |
|---|---|---|
| 규제 항의 그레이디언트 | $\text{sign}(\theta)$ — 크기와 무관하게 **일정** | $\theta$ — 0에 가까울수록 **약해짐** |
| 이동 경로 | 한 축이 먼저 0에 도달 → 축을 따라 이동 | 원점을 향해 직진하며 감속 |
| 최솟값 위치 | 축 위 (일부 $\theta_i = 0$) | 0 근처, 0은 아님 |

**코드** (그림 4-19 핵심 함수)

```python
def bgd_path(theta, X, y, l1, l2, core=1, eta=0.05, n_iterations=200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = (core * 2 / len(X) * X.T @ (X @ theta - y)   # MSE 그레이디언트
                     + l1 * np.sign(theta)                        # ℓ1 서브그레이디언트 (크기 일정)
                     + l2 * theta)                                # ℓ2 그레이디언트 (θ에 비례)
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)
```

- `core=0`: MSE 항 제거 → 규제 항만으로 움직이는 경로 (그림의 노란 점선)
- 나머지 그림 코드: `np.meshgrid`로 $\theta_1$–$\theta_2$ 격자 생성 → `contourf`로 비용 등고선 표시

```python
# 그림 4-19 전체 코드
t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5

t1s = np.linspace(t1a, t1b, 500)
t2s = np.linspace(t2a, t2b, 500)
t1, t2 = np.meshgrid(t1s, t2s)
T = np.c_[t1.ravel(), t2.ravel()]
Xr = np.array([[1, 1], [1, -1], [1, 0.5]])
yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]

J = (1 / len(Xr) * ((T @ Xr.T - yr.T) ** 2).sum(axis=1)).reshape(t1.shape)   # MSE 등고선

N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)    # ℓ1 노름
N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)    # ℓ2 노름

t_min_idx = np.unravel_index(J.argmin(), J.shape)
t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]

t_init = np.array([[0.25], [-1]])

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10.1, 8))

for i, N, l1, l2, title in ((0, N1, 2.0, 0, "Lasso"), (1, N2, 0, 2.0, "Ridge")):
    JR = J + l1 * N1 + l2 * 0.5 * N2 ** 2           # 규제 포함 비용

    tr_min_idx = np.unravel_index(JR.argmin(), JR.shape)
    t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]

    levels = np.exp(np.linspace(0, 1, 20)) - 1
    levelsJ = levels * (J.max() - J.min()) + J.min()
    levelsJR = levels * (JR.max() - JR.min()) + JR.min()
    levelsN = np.linspace(0, N.max(), 10)

    path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
    path_JR = bgd_path(t_init, Xr, yr, l1, l2)
    path_N = bgd_path(theta=np.array([[2.0], [0.5]]), X=Xr, y=yr,
                      l1=np.sign(l1) / 3, l2=np.sign(l2), core=0)
    ax = axes[i, 0]                                    # 왼쪽: 규제 항만
    ax.grid()
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.contourf(t1, t2, N / 2.0, levels=levelsN)
    ax.plot(path_N[:, 0], path_N[:, 1], "y--")
    ax.plot(0, 0, "ys")
    ax.plot(t1_min, t2_min, "ys")
    ax.set_title(fr"$\ell_{i + 1}$ penalty")
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$", rotation=0)

    ax = axes[i, 1]                                    # 오른쪽: MSE + 규제
    ax.grid()
    ax.axhline(y=0, color="k")
    ax.axvline(x=0, color="k")
    ax.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
    ax.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
    ax.plot(path_N[:, 0], path_N[:, 1], "y--")
    ax.plot(0, 0, "ys")
    ax.plot(t1_min, t2_min, "ys")
    ax.plot(t1r_min, t2r_min, "rs")
    ax.set_title(title)
    ax.axis([t1a, t1b, t2a, t2b])
    if i == 1:
        ax.set_xlabel(r"$\theta_1$")

plt.show()
```

**결과 해석**

| 그림 위치 | 내용 |
|---|---|
| 왼쪽 위 (ℓ1 벌점만) | 경로가 $\theta_2 = 0$ 축에 먼저 도달 → 축을 따라 원점으로 |
| 왼쪽 아래 (ℓ2 벌점만) | 원점을 향해 직선으로 이동 |
| 오른쪽 위 (라쏘) | 최솟값(빨간 사각형)이 $\theta_2 = 0$ 축 위 → 가중치 하나가 정확히 0 |
| 오른쪽 아래 (릿지) | 최솟값이 원점 쪽으로 당겨졌지만 축 위는 아님 |

**정리**
- ℓ1: 일정한 힘으로 당김 → 0에 도달 / ℓ2: 0에 가까울수록 힘이 약해짐 → 0에 도달하지 않음
- 미분 불가능한 지점은 서브그레이디언트로 처리

---

### 5-5. 엘라스틱넷

**목적**
릿지와 라쏘를 결합한 규제를 적용하고, 규제 모델 선택 기준을 정리한다.

**개념·원리**

**엘라스틱넷** (Elastic Net): ℓ1과 ℓ2 규제를 혼합 비율 $r$로 결합.

```math
J(\boldsymbol{\theta}) = \text{MSE}(\boldsymbol{\theta}) + r\left(2\alpha\sum_{i=1}^{n}\left|\theta_i\right|\right) + (1-r)\left(\frac{\alpha}{m}\sum_{i=1}^{n}\theta_i^2\right)
```

- $r = 0$ → 릿지, $r = 1$ → 라쏘

**코드**

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)   # l1_ratio = r
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
# array([1.8645014])
```

**결과 해석**

| 모델 ($\alpha = 0.1$) | `predict([[1.5]])` |
|---|---|
| 릿지 | 1.844 |
| 엘라스틱넷 ($r = 0.5$) | 1.865 |
| 라쏘 | 1.876 |

- 엘라스틱넷 결과가 릿지와 라쏘 사이

**정리: 규제 모델 선택 기준**

| 상황 | 선택 |
|---|---|
| 기본 | 규제 없는 선형 회귀 대신 **릿지** |
| 유용한 특성이 일부뿐일 것으로 예상 | **라쏘** 또는 **엘라스틱넷** |
| 특성 수 > 샘플 수, 특성 간 강한 상관 | **엘라스틱넷** (라쏘가 불안정) |
| 라쏘처럼 쓰되 안정적으로 | 엘라스틱넷 + `l1_ratio` ≈ 1 |

---

### 5-6. 조기 종료

**목적**
반복 학습에서 검증 오차가 최소인 시점에 훈련을 멈춰 과대적합을 막는다.

**개념·원리**

**조기 종료** (early stopping): 검증 오차가 최솟값에 도달하면 훈련 중지.
- 훈련이 진행되면 훈련 오차는 계속 감소, 검증 오차는 감소하다가 다시 증가 (과대적합 시작)
- 그 꺾이는 지점의 모델 사용
- 제프리 힌턴: "아름다운 공짜 점심 (beautiful free lunch)"

**코드**

```python
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# 2차 데이터 재생성 후 앞/뒤 절반으로 분할
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))
X_train, y_train = X[: m // 2], y[: m // 2, 0]     # 앞 100개 훈련, (100, 1) → (100,)
X_valid, y_valid = X[m // 2 :], y[m // 2 :, 0]     # 뒤 100개 검증

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),  # 과대적합하기 쉬운 90차
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)    # 훈련 세트로만 fit (데이터 누설 방지)
X_valid_prep = preprocessing.transform(X_valid)
sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')                         # 초기 최고 기록 = 무한대
train_errors, val_errors = [], []                      # 그림용

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)         # 1 에포크만 훈련, 이전 상태에서 이어감
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:                    # 최고 기록 갱신 시
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)                 # 모델 전체 복사 저장

    y_train_predict = sgd_reg.predict(X_train_prep)
    train_error = root_mean_squared_error(y_train, y_train_predict)
    val_errors.append(val_error)
    train_errors.append(train_error)
```

| 코드 | 의미 |
|---|---|
| `partial_fit()` | 1 에포크만 훈련하고 상태 유지. `fit()`은 매번 초기화 |
| `deepcopy()` | 객체 전체 복사. 단순 대입은 같은 객체를 참조 → 이후 훈련에 함께 변경됨 |

```python
# 그림 4-20
best_epoch = np.argmin(val_errors)                   # 검증 오차 최소 에포크
plt.figure(figsize=(6, 4))
plt.annotate('Best model',
             xy=(best_epoch, best_valid_rmse),            # 화살표 끝
             xytext=(best_epoch, best_valid_rmse + 0.5),  # 텍스트 위치
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot([0, n_epochs], [best_valid_rmse, best_valid_rmse], "k:", linewidth=2)
plt.plot(val_errors, "b-", linewidth=3, label="Validation set")
plt.plot(best_epoch, best_valid_rmse, "bo")
plt.plot(train_errors, "r--", linewidth=2, label="Training set")
plt.legend(loc="upper right")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.axis([0, n_epochs, 0, 3.5])
plt.grid()
plt.show()
```

**결과 해석** (직접 계산)

| 시점 | 검증 RMSE |
|---|---|
| **184 에포크 (최적)** | **≈ 1.105** |
| 500 에포크 | ≈ 1.318 |

- 184 에포크 이후 훈련 오차는 계속 감소, 검증 오차는 증가 → 과대적합 시작
- `best_model`에 184 에포크 시점 모델이 저장됨

**정리**
- 조기 종료 = 검증 오차 최소 시점의 모델 사용
- SGD·미니배치는 곡선이 불규칙 → 즉시 중지하지 말고 **최고 모델 저장 → 일정 기간 개선 없으면 중지 → 저장 모델로 복귀**

---

## 6. 로지스틱 회귀 (4.6)

### 6-1. 확률 추정

**목적**
회귀 모델의 출력을 확률로 변환해 이진 분류에 사용한다.

**개념·원리**

**로지스틱 회귀**: 선형 회귀처럼 가중치 합을 계산한 뒤 **시그모이드 함수**로 0~1 확률로 변환.

```math
\hat{p} = h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma\left(\boldsymbol{\theta}^\top\mathbf{x}\right),
\qquad
\sigma(t) = \frac{1}{1 + e^{-t}}
```

예측:

```math
\hat{y} = \begin{cases} 0 & \hat{p} < 0.5 \\ 1 & \hat{p} \geq 0.5 \end{cases}
\quad\Longleftrightarrow\quad
\hat{y} = \begin{cases} 0 & \boldsymbol{\theta}^\top\mathbf{x} < 0 \\ 1 & \boldsymbol{\theta}^\top\mathbf{x} \geq 0 \end{cases}
```

**로짓** (logit): 시그모이드의 입력 $t = \boldsymbol{\theta}^\top\mathbf{x}$. 시그모이드의 역함수로, 로그 오즈와 같다.

```math
\text{logit}(p) = \log\left(\frac{p}{1-p}\right)
```

**코드**

```python
# 그림 4-21
lim = 6
t = np.linspace(-lim, lim, 100)
sig = 1 / (1 + np.exp(-t))

plt.figure(figsize=(8, 3))
plt.plot([-lim, lim], [0, 0], "k-")
plt.plot([-lim, lim], [0.5, 0.5], "k:")
plt.plot([-lim, lim], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \dfrac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left")
plt.axis([-lim, lim, -0.1, 1.1])
plt.gca().set_yticks([0, 0.25, 0.5, 0.75, 1])
plt.grid()
plt.show()
```

**결과 해석**

| $t$ | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| $\sigma(t)$ | 0.018 | 0.119 | **0.5** | 0.881 | 0.982 |

- S자 곡선, 출력 범위 (0, 1), $t = 0$에서 정확히 0.5

**정리**
- 로지스틱 회귀 = 선형 결합 → 시그모이드 → 확률 → 0.5 기준 분류
- $\boldsymbol{\theta}^\top\mathbf{x}$의 부호만으로 클래스가 결정된다

---

### 6-2. 훈련과 비용 함수

**목적**
로지스틱 회귀의 비용 함수와 최적화 방법을 이해한다.

**개념·원리**

목표: 양성 샘플($y = 1$)에는 높은 확률, 음성 샘플($y = 0$)에는 낮은 확률.

샘플 하나의 비용:

```math
c(\boldsymbol{\theta}) = \begin{cases} -\log(\hat{p}) & y = 1 \\ -\log(1 - \hat{p}) & y = 0 \end{cases}
```

- $y = 1$인데 $\hat{p} \to 0$ → $-\log(\hat{p}) \to \infty$ (확신을 갖고 틀리면 큰 벌점)

전체 비용 (**로그 손실**):

```math
J(\boldsymbol{\theta}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log\left(\hat{p}^{(i)}\right) + \left(1 - y^{(i)}\right)\log\left(1 - \hat{p}^{(i)}\right)\right]
```

편도함수:

```math
\frac{\partial}{\partial\theta_j}J(\boldsymbol{\theta}) = \frac{1}{m}\sum_{i=1}^{m}\left(\sigma\left(\boldsymbol{\theta}^\top\mathbf{x}^{(i)}\right) - y^{(i)}\right)x_j^{(i)}
```

| 성질 | 내용 |
|---|---|
| 닫힌 형태 해 | **없음** |
| 볼록성 | **볼록** → 경사 하강법으로 전역 최솟값 보장 |
| 그레이디언트 구조 | 선형 회귀와 동일: (예측 − 정답) × 특성값의 평균 |
| 사이킷런 기본 규제 | ℓ2, 강도는 **$C = 1/\alpha$** ($C$ ↑ → 규제 ↓) |

**정리**
- 로그 손실 = 볼록 → 경사 하강법으로 최적화
- `LogisticRegression`의 규제 하이퍼파라미터는 $\alpha$가 아니라 $C$ (방향 반대)

---

### 6-3. 붓꽃 데이터

**목적**
로지스틱 회귀 실습에 사용할 데이터 구조를 확인한다.

**코드**

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)         # DataFrame 형태로 로드
list(iris)
# ['data', 'target', 'frame', 'target_names', 'DESCR',
#  'feature_names', 'filename', 'data_module']

print(iris.DESCR)                       # 데이터셋 설명

iris.data.head(3)
#    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# 0                5.1               3.5                1.4               0.2
# 1                4.9               3.0                1.4               0.2
# 2                4.7               3.2                1.3               0.2

iris.target.head(3)                     # 샘플이 섞여 있지 않음
# 0    0
# 1    0
# 2    0

iris.target_names
# array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
```

**결과 해석**

| 항목 | 내용 |
|---|---|
| 샘플 | 150 (품종별 50) |
| 특성 | 꽃받침(sepal) 길이·너비, 꽃잎(petal) 길이·너비 (cm) |
| 타깃 | 0 setosa / 1 versicolor / 2 virginica |
| 순서 | **클래스 순으로 정렬** |

**정리**
- 정렬된 데이터 → 반드시 섞어서 분할 (`train_test_split` 기본 동작)

---

### 6-4. 결정 경계: 특성 1개

**목적**
꽃잎 너비 하나로 Iris virginica를 감지하는 이진 분류기를 만들고 결정 경계를 찾는다.

**개념·원리**

**결정 경계** (decision boundary): 두 클래스의 추정 확률이 50%로 같은 지점.

```math
\theta_0 + \theta_1 x_1 = 0 \quad\Longrightarrow\quad x_1 = -\frac{\theta_0}{\theta_1}
```

**코드**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values                 # [[ ]] → 2차원 유지, .values → NumPy (150, 1)
y = iris.target_names[iris.target] == 'virginica'          # 0/1/2 → 품종 이름 → virginica 여부 (bool)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)   # 75/25, 섞어서 분할

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

```python
# 그림 4-23
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)             # 꽃잎 너비 0~3cm, (1000, 1)
y_proba = log_reg.predict_proba(X_new)                     # (1000, 2): [음성 확률, 양성 확률]
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]      # 양성 확률이 처음 0.5 이상인 x

plt.figure(figsize=(8, 3))
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2,
         label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
plt.arrow(x=decision_boundary, y=0.08, dx=-0.3, dy=0,
          head_width=0.05, head_length=0.1, fc="b", ec="b")
plt.arrow(x=decision_boundary, y=0.92, dx=0.3, dy=0,
          head_width=0.05, head_length=0.1, fc="g", ec="g")
plt.plot(X_train[y_train == 0], y_train[y_train == 0], "bs")   # 음성 샘플 (y=0 위치)
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")   # 양성 샘플 (y=1 위치)
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.legend(loc="center left")
plt.axis([0, 3, -0.02, 1.02])
plt.grid()
plt.show()

decision_boundary
# np.float64(1.6516516516516517)

log_reg.predict([[1.7], [1.5]])
# array([ True, False])
```

**결과 해석**

꽃잎 너비별 virginica 확률 (직접 계산):

| 꽃잎 너비 (cm) | 1.0 | 1.5 | **1.65** | 1.7 | 2.0 |
|---|---|---|---|---|---|
| 확률 | 0.076 | 0.360 | **≈ 0.50** | 0.547 | 0.792 |

- 결정 경계 ≈ **1.65cm** → 1.7은 True, 1.5는 False
- 2cm 이상: 확실한 virginica / 1cm 이하: 확실히 아님 / 그 사이: 두 클래스가 겹치는 구간

**정리**
- `predict_proba()`: 클래스별 확률, `predict()`: 0.5 기준 클래스
- 결정 경계 = 확률 50% = $\boldsymbol{\theta}^\top\mathbf{x} = 0$

---

### 6-5. 결정 경계: 특성 2개

**목적**
특성 2개일 때 결정 경계와 확률 등고선을 2차원 평면에 시각화한다.

**개념·원리**

```math
\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0 \quad\Longrightarrow\quad x_2 = -\frac{\theta_1 x_1 + \theta_0}{\theta_2}
```

- 결정 경계 = **직선** (특성이 더 많으면 초평면) → 로지스틱 회귀는 **선형 분류기**

**코드**

```python
# 그림 4-24
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(C=2, random_state=42)       # C = 1/α
log_reg.fit(X_train, y_train)

# 등고선용 격자
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]                    # 격자점 하나당 샘플 하나
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)                     # 다시 격자 모양으로

# 결정 경계 직선: x2 = −(θ1·x1 + θ0) / θ2
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])

plt.figure(figsize=(10, 4))
plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], "bs")
plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], "g^")
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)                            # 등고선에 확률값 표시
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.27, "Not Iris virginica", color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", color="g", ha="center")
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.axis([2.9, 7, 0.8, 2.7])
plt.grid()
plt.show()
```

격자 시각화 3단계 (이후 장에서도 반복 사용):

| 단계 | 코드 | 결과 |
|---|---|---|
| 1. 격자 생성 | `np.meshgrid(a, b)` | 좌표 행렬 2개 `(200, 500)` |
| 2. 샘플 목록화 | `np.c_[x0.ravel(), x1.ravel()]` | `(100000, 2)` → 모델 입력 |
| 3. 격자로 복원 | `.reshape(x0.shape)` | `(200, 500)` → 등고선 |

**결과 해석**
- 점선 = 결정 경계 (확률 50%), **직선**
- 평행한 실선 = 15% ~ 90% 확률 등고선. 경계에서 멀어질수록 확신이 커짐

**정리**
- 로지스틱 회귀의 결정 경계는 항상 선형
- $C$가 클수록 규제가 약해 훈련 데이터에 더 맞춘다

---

### 6-6. 소프트맥스 회귀

**목적**
로지스틱 회귀를 다중 클래스로 일반화한다.

**개념·원리**

**소프트맥스 회귀** (다항 로지스틱 회귀): 이진 분류기 여러 개를 결합하지 않고 하나의 모델로 다중 클래스를 직접 처리.

1. 클래스 $k$별 점수 (클래스마다 별도 파라미터 벡터 $\boldsymbol{\theta}^{(k)}$):

```math
s_k(\mathbf{x}) = \left(\boldsymbol{\theta}^{(k)}\right)^\top\mathbf{x}
```

2. **소프트맥스 함수**로 확률 변환:

```math
\hat{p}_k = \sigma\left(\mathbf{s}(\mathbf{x})\right)_k = \frac{\exp\left(s_k(\mathbf{x})\right)}{\sum_{j=1}^{K}\exp\left(s_j(\mathbf{x})\right)}
```

3. 가장 높은 확률의 클래스 선택:

```math
\hat{y} = \underset{k}{\operatorname{argmax}}\ \hat{p}_k
```

비용 함수 (**크로스 엔트로피**):

```math
J(\boldsymbol{\Theta}) = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}y_k^{(i)}\log\left(\hat{p}_k^{(i)}\right)
```

그레이디언트:

```math
\nabla_{\boldsymbol{\theta}^{(k)}}J(\boldsymbol{\Theta}) = \frac{1}{m}\sum_{i=1}^{m}\left(\hat{p}_k^{(i)} - y_k^{(i)}\right)\mathbf{x}^{(i)}
```

| 항목 | 내용 |
|---|---|
| $\exp$ 사용 | 모든 점수를 양수로 + 큰 점수를 더 크게 벌림 |
| 합으로 나눔 | 확률 합 = 1 |
| $K = 2$일 때 | 크로스 엔트로피 = 로지스틱 회귀의 로그 손실 |
| 적용 범위 | 상호 배타적 클래스만 (다중 출력 X) |

**코드**

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]                                   # 0, 1, 2 → 3 클래스
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)   # 클래스 ≥ 3 → 자동으로 소프트맥스
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5, 2]])
# array([2])

softmax_reg.predict_proba([[5, 2]]).round(2)
# array([[0.  , 0.04, 0.96]])
```

```python
# 그림 4-25
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])   # 클래스별 영역 색

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)       # versicolor 확률 → 등고선
zz = y_predict.reshape(x0.shape)            # 예측 클래스 → 영역 채우기

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

plt.contourf(x0, x1, zz, cmap=custom_cmap)  # 영역 채우기
contour = plt.contour(x0, x1, zz1, cmap="hot")   # 등고선
plt.clabel(contour, inline=1)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="center left")
plt.axis([0.5, 7, 0, 3.5])
plt.grid()
plt.show()
```

**결과 해석**

꽃잎 길이 5cm, 너비 2cm의 계산 과정 (직접 계산):

| 단계 | setosa | versicolor | virginica |
|---|---|---|---|
| 점수 $s_k$ | −10.39 | 3.61 | 6.77 |
| $\exp(s_k)$ | ≈ 0.00003 | ≈ 37 | ≈ 871 |
| $\div$ 합(≈ 908) = $\hat{p}_k$ | **0.00** | **0.04** | **0.96** |

- 예측: 클래스 2 (virginica)
- 그림 4-25: 클래스 간 결정 경계가 모두 **직선**. 세 경계가 만나는 점에서 세 확률이 모두 33%

**정리**
- 소프트맥스 회귀 = 클래스별 선형 점수 → 소프트맥스 → argmax
- 비용 = 크로스 엔트로피, 그레이디언트는 로지스틱 회귀와 같은 구조
- 사이킷런: `LogisticRegression`에 3개 이상 클래스 타깃을 넣으면 됨

---

## 7. 연습문제 (4.7)

### 7-1. 개념 문제 (1 ~ 11)

| # | 질문 | 답 |
|---|---|---|
| 1 | 특성이 수백만 개일 때 사용할 알고리즘 | SGD, 미니배치 GD (메모리에 들어가면 배치 GD). 정규 방정식·SVD는 $n$에 대해 너무 느림 |
| 2 | 특성 스케일이 크게 다를 때 문제와 해결 | 비용 함수가 길쭉해져 GD 수렴 지연 → **스케일링**. 규제 모델은 작은 값 특성을 무시하게 됨. 정규 방정식·SVD는 영향 없음 |
| 3 | 로지스틱 회귀에서 GD가 지역 최솟값에 갇히는가 | 아니오. 비용 함수가 **볼록** |
| 4 | GD 알고리즘들이 같은 모델에 도달하는가 | 볼록 + 적절한 학습률이면 비슷. 학습률을 줄이지 않으면 SGD·미니배치는 최적점 주변 요동으로 약간씩 다름 |
| 5 | 배치 GD에서 검증 오차가 매 에포크 증가 | 훈련 오차도 증가 → 학습률 과대로 **발산** → 학습률 감소. 훈련 오차 유지 → **과대적합** → 훈련 중지 |
| 6 | 미니배치 GD에서 검증 오차가 오르면 즉시 중지? | 아니오. 무작위성으로 일시 증가 가능. 주기적 저장 → 오래 개선 없으면 최고 모델로 복귀 |
| 7 | 최적점 근처 도달이 가장 빠른 것 / 실제 수렴하는 것 | 가장 빠름: SGD. 실제 수렴: 배치 GD (나머지는 학습률 감소 필요) |
| 8 | 다항 회귀에서 검증 오차 ≫ 훈련 오차 | 과대적합 → ① 차수 감소 ② 규제(ℓ2, ℓ1) ③ 훈련 데이터 증가 |
| 9 | 릿지에서 훈련·검증 오차가 비슷하게 높음 | 과소적합(높은 편향) → **$\alpha$ 감소** |
| 10 | 릿지 vs 선형 / 라쏘 vs 릿지 / 엘라스틱넷 vs 라쏘 | 약간의 규제가 대체로 나음 → 릿지. 소수 특성만 중요하면 라쏘. 라쏘가 불안정한 경우($n > m$, 강한 상관) 엘라스틱넷 (`l1_ratio` ≈ 1) |
| 11 | 실외/실내 + 낮/밤 분류 | 상호 배타적이지 않음 → 소프트맥스 1개가 아니라 **로지스틱 회귀 2개** |

---

### 7-2. 문제 12 — 데이터 준비

**목적**
사이킷런 없이 NumPy만으로 소프트맥스 회귀 + 조기 종료를 구현하기 위해 데이터를 준비한다.

**코드: 편향 열 추가와 분할 (60 / 20 / 20)**

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"].values

X_with_bias = np.c_[np.ones(len(X)), X]               # 편향 열 추가 (x0 = 1)

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)                          # 150

test_size = int(total_size * test_ratio)               # 30
validation_size = int(total_size * validation_ratio)   # 30
train_size = total_size - test_size - validation_size  # 90

rng = np.random.default_rng(seed=42)
rnd_indices = rng.permutation(total_size)              # 섞은 인덱스

X_train = X_with_bias[rnd_indices[:train_size]]            # 앞 90개
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]  # 가운데 30개
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]             # 뒤 30개
y_test = y[rnd_indices[-test_size:]]
```

**코드: 원-핫 인코딩**

크로스 엔트로피는 타깃을 **클래스 확률 벡터**로 받는다 → 클래스 인덱스를 원-핫 벡터로 변환.

```python
def to_one_hot(y):
    return np.diag(np.ones(y.max() + 1))[y]    # 단위 행렬에서 레이블 번호의 행을 선택

y_train[:10]
# array([1, 1, 2, 0, 2, 2, 1, 2, 2, 0])

to_one_hot(y_train[:10])
# array([[0., 1., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.],
#        [1., 0., 0.],
#        [0., 0., 1.],
#        [0., 0., 1.],
#        [0., 1., 0.],
#        [0., 0., 1.],
#        [0., 0., 1.],
#        [1., 0., 0.]])

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)
```

| 코드 | 의미 |
|---|---|
| `np.diag(np.ones(3))` | 3×3 단위 행렬 |
| `단위행렬[y]` | 팬시 인덱싱 → 각 레이블 번호의 행 = 원-핫 벡터 |

**코드: 스케일링**

```python
mean = X_train[:, 1:].mean(axis=0)              # 편향 열(0번) 제외
std = X_train[:, 1:].std(axis=0)
X_train[:, 1:] = (X_train[:, 1:] - mean) / std  # 훈련 세트 통계량으로
X_valid[:, 1:] = (X_valid[:, 1:] - mean) / std  # 세 세트 모두 변환
X_test[:, 1:] = (X_test[:, 1:] - mean) / std
```

**정리**
- `train_test_split`, `OneHotEncoder`, `StandardScaler`의 역할을 직접 구현
- 스케일링 통계량은 훈련 세트에서만 계산

---

### 7-3. 문제 12 — 소프트맥스 회귀 훈련

**목적**
6-6의 수식을 코드로 옮겨 배치 경사 하강법으로 훈련한다.

**개념·원리**

행렬 형태의 그레이디언트:

```math
\nabla_{\boldsymbol{\Theta}}J = \frac{1}{m}\mathbf{X}^\top\left(\hat{\mathbf{P}} - \mathbf{Y}\right)
```

- $\hat{\mathbf{P}}$: 예측 확률 행렬 $(m \times K)$, $\mathbf{Y}$: 원-핫 타깃 행렬 $(m \times K)$
- $\log(0)$ 방지를 위해 $\log(\hat{p} + \epsilon)$ 사용

**코드**

```python
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = exps.sum(axis=1, keepdims=True)   # (m, 1) 유지 → 행별 나눗셈 브로드캐스팅
    return exps / exp_sums

n_inputs = X_train.shape[1]            # 3 (특성 2 + 편향)
n_outputs = len(np.unique(y_train))    # 3 (클래스 3)
```

```python
eta = 0.5
n_epochs = 5001
m = len(X_train)
epsilon = 1e-5

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))     # (3, 3): 행 = 특성, 열 = 클래스

for epoch in range(n_epochs):
    logits = X_train @ Theta                           # 클래스별 점수 (90, 3)
    Y_proba = softmax(logits)                          # 클래스별 확률 (90, 3)
    if epoch % 1000 == 0:
        Y_proba_valid = softmax(X_valid @ Theta)
        xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))  # 정답 클래스 항만 남음
        print(epoch, xentropy_losses.sum(axis=1).mean())                        # 크로스 엔트로피
    error = Y_proba - Y_train_one_hot                  # P̂ − Y
    gradients = 1 / m * X_train.T @ error              # (1/m) Xᵀ(P̂ − Y)
    Theta = Theta - eta * gradients
```

```
0 2.973977344302766
1000 0.09313918303944392
2000 0.08904698946125779
3000 0.08847558719791135
4000 0.08861927669821175
5000 0.08894346006981158
```

```python
Theta
# array([[-1.23843975,  5.9045082 , -4.65088428],
#        [-6.53446858, -2.85065468,  7.07247328],
#        [-6.919744  , -0.36831947,  7.08286012]])

logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)                  # 확률 최대 클래스
accuracy_score = (y_predict == y_valid).mean()      # True=1, False=0 → 평균 = 정확도
accuracy_score
# np.float64(0.9333333333333333)
```

**결과 해석**
- 검증 손실 2.974 → 0.088까지 감소
- 3000 에포크 이후 소폭 상승 (0.0885 → 0.0889) → 약한 과대적합
- 검증 정확도 **0.933**

**정리**
- 소프트맥스 회귀의 핵심 계산 = `softmax(X @ Θ)` → `Xᵀ(P̂ − Y) / m` → `Θ −= η∇`

---

### 7-4. 문제 12 — ℓ2 규제 추가

**목적**
ℓ2 규제로 과대적합을 줄인다.

**개념·원리**

```math
J(\boldsymbol{\Theta}) = \text{크로스 엔트로피} + \alpha \cdot \frac{1}{2}\sum \boldsymbol{\Theta}_{[1:]}^2,
\qquad
\nabla \mathrel{+}= \alpha\,\boldsymbol{\Theta}_{[1:]}
```

- $\boldsymbol{\Theta}_{[1:]}$: 편향 행(0번) 제외. $\frac{1}{2}\theta^2$의 미분 = $\theta$

**코드**

```python
eta = 0.5
n_epochs = 5001
m = len(X_train)
epsilon = 1e-5
alpha = 0.01                                           # 규제 하이퍼파라미터

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    if epoch % 1000 == 0:
        Y_proba_valid = softmax(X_valid @ Theta)
        xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
        l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()                     # 편향 행 제외
        total_loss = xentropy_losses.sum(axis=1).mean() + alpha * l2_loss
        print(epoch, total_loss.round(4))
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]  # 편향 행 0 + 나머지 αθ (행 방향 결합)
    Theta = Theta - eta * gradients
```

```
0 3.0065
1000 0.2711
2000 0.2711
3000 0.2711
4000 0.2711
5000 0.2711
```

```python
logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)
accuracy_score = (y_predict == y_valid).mean()
accuracy_score
# np.float64(0.9666666666666667)
```

**결과 해석**
- 손실값은 규제 항이 더해져 커졌지만 (0.088 → 0.271)
- 검증 정확도 **0.933 → 0.967** 향상

**정리**
- 규제는 편향 행을 제외하고 적용
- 손실값 크기가 아니라 검증 성능으로 판단

---

### 7-5. 문제 12 — 조기 종료 추가

**목적**
매 에포크 검증 손실을 확인해 증가하는 순간 훈련을 멈춘다.

**코드**

```python
eta = 0.5
n_epochs = 50_001                                      # _ 는 자릿수 구분자
m = len(X_train)
epsilon = 1e-5
C = 100                                                # 규제 (α = 1/C = 0.01)
best_loss = np.inf

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    Y_proba_valid = softmax(X_valid @ Theta)           # 매 에포크 검증 손실 계산
    xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()
    total_loss = xentropy_losses.sum(axis=1).mean() + 1 / C * l2_loss
    if epoch % 1000 == 0:
        print(epoch, total_loss.round(4))
    if total_loss < best_loss:                         # 개선 → 기록 갱신
        best_loss = total_loss
    else:                                              # 악화 → 중지
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

```python
logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)
accuracy_score = (y_predict == y_valid).mean()
accuracy_score
# np.float64(0.9666666666666667)
```

**결과 해석**
- 5000 에포크 대신 **403 에포크**에서 중지 (약 1/12)
- 검증 정확도는 그대로 0.967 → 같은 성능을 더 짧은 훈련으로

**정리**
- 배치 GD는 손실 곡선이 매끄러워 1회 증가 즉시 중지해도 됨
- SGD·미니배치는 인내 기간(patience)을 두어야 함 (연습문제 6번)

---

### 7-6. 문제 12 — 최종 평가

**목적**
전체 평면에서의 예측을 시각화하고 테스트 세트로 최종 성능을 측정한다.

**코드**

```python
custom_cmap = mpl.colors.ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new = (X_new - mean) / std                           # 훈련 때와 같은 스케일링
X_new_with_bias = np.c_[np.ones(len(X_new)), X_new]    # 편향 열 추가

logits = X_new_with_bias @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)

zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
plt.xlabel("Petal length")
plt.ylabel("Petal width")
plt.legend(loc="upper left")
plt.axis([0, 7, 0, 3.5])
plt.grid()
plt.show()
```

```python
logits = X_test @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)

accuracy_score = (y_predict == y_test).mean()
accuracy_score
# np.float64(0.9666666666666667)
```

**결과 해석**

| 단계 | 종료 에포크 | 검증 정확도 | 테스트 정확도 |
|---|---|---|---|
| 기본 | 5000 | 0.933 | — |
| + ℓ2 규제 ($\alpha = 0.01$) | 5000 | 0.967 | — |
| + 조기 종료 ($C = 100$) | **403** | 0.967 | **0.967** |

- 테스트 정확도 = 검증 정확도. 데이터가 작아 시드에 따라 결과가 달라질 수 있음

**정리**
- 새 데이터도 훈련 때와 동일한 전처리(스케일링 → 편향 열)를 거쳐야 한다

---

## 8. 정리

### 8-1. 흐름과 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| 닫힌 형태 해 | `np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y`, `LinearRegression` | 특성이 적을 때 정확하고 빠름 |
| 배치 GD | `theta -= eta * 2/m * X_b.T @ (X_b @ theta - y)` | 스케일링 필수, 학습률 탐색 |
| SGD | `learning_schedule(t)`, `SGDRegressor` | 데이터 섞기, 학습률 감소 |
| 미니배치 GD | `rng.permutation(m)` + 배치 슬라이싱 | 신경망 학습의 표준 |
| 다항 회귀 | `PolynomialFeatures(degree=d)` | 특성 수 폭발 주의 |
| 진단 | `learning_curve(..., scoring="neg_root_mean_squared_error")` | 곡선 모양으로 과소/과대적합 판단 |
| 규제 | `Ridge`, `Lasso`, `ElasticNet` | 스케일링 필수, 편향 제외 |
| 조기 종료 | `partial_fit()` + `deepcopy()` | 최고 모델 보존 |
| 분류 | `LogisticRegression` | 이진 = 시그모이드, 다중 = 소프트맥스 |

### 8-2. 결과 변화

| 절 | 실험 | 결과 |
|---|---|---|
| 1-3, 1-4 | 정규 방정식 / `LinearRegression` / `lstsq` / `pinv` | $\boldsymbol{\theta} = (3.691, 3.330)$, 정답 (4, 3) |
| 2-2 | 배치 GD ($\eta = 0.1$, 1000 에포크) | (3.691, 3.330) — 정규 방정식과 동일 |
| 2-3 | 배치 GD ($\eta = 0.5$) | 발산 |
| 2-4 | SGD (50 에포크) | (3.698, 3.307) |
| 2-5 | `SGDRegressor` | (3.689, 3.331) |
| 3-2 | 2차 다항 회귀 | $0.505x^2 + 1.110x + 2.005$ (정답 $0.5x^2 + x + 2$) |
| 4-2, 4-3 | 학습 곡선 RMSE (훈련 / 검증) | 선형 1.62 / 1.64, 10차 0.97 / 1.03 |
| 5-2 ~ 5-5 | 릿지 / 엘라스틱넷 / 라쏘 `predict([[1.5]])` | 1.844 / 1.865 / 1.876 |
| 5-3 | 10차 다항 + 라쏘 ($\alpha = 0.01$) | 0이 아닌 가중치 2 / 10 |
| 5-6 | 조기 종료 (90차 + SGD) | 184 에포크, 검증 RMSE 1.105 |
| 6-4 | 로지스틱 회귀 결정 경계 | 꽃잎 너비 1.65cm |
| 6-6 | 소프트맥스 `predict_proba([[5, 2]])` | [0.00, 0.04, 0.96] |
| 7-5, 7-6 | NumPy 소프트맥스 (규제 + 조기 종료) | 403 에포크, 검증·테스트 정확도 0.967 |

### 8-3. 모델 비교

| 모델 | 출력 | 비용 함수 | 닫힌 형태 해 | 사이킷런 |
|---|---|---|---|---|
| 선형 회귀 | $\boldsymbol{\theta}^\top\mathbf{x}$ | MSE | O | `LinearRegression`, `SGDRegressor` |
| 다항 회귀 | $\boldsymbol{\theta}^\top \text{poly}(\mathbf{x})$ | MSE | O | `PolynomialFeatures` + `LinearRegression` |
| 릿지 | $\boldsymbol{\theta}^\top\mathbf{x}$ | MSE + ℓ2 | O | `Ridge` |
| 라쏘 | $\boldsymbol{\theta}^\top\mathbf{x}$ | MSE + ℓ1 | X | `Lasso` |
| 엘라스틱넷 | $\boldsymbol{\theta}^\top\mathbf{x}$ | MSE + ℓ1·ℓ2 혼합 | X | `ElasticNet` |
| 로지스틱 회귀 | $\sigma(\boldsymbol{\theta}^\top\mathbf{x})$ | 로그 손실 | X (볼록) | `LogisticRegression` |
| 소프트맥스 회귀 | $\text{softmax}(\boldsymbol{\Theta}^\top\mathbf{x})$ | 크로스 엔트로피 | X (볼록) | `LogisticRegression` (클래스 ≥ 3) |

하이퍼파라미터 방향:

| 하이퍼파라미터 | 값 ↑ | 주의 |
|---|---|---|
| 학습률 $\eta$ (`eta0`) | 빠름, 과하면 발산 | |
| 다항 차수 `degree` | 복잡도 ↑, 과대적합 위험 | 특성 수 폭발 |
| `alpha` (Ridge·Lasso·ElasticNet) | 규제 **강화** | `SGDRegressor`는 $\alpha/m$으로 맞춤 |
| `C` (LogisticRegression) | 규제 **약화** | $C = 1/\alpha$ |
| `l1_ratio` | 라쏘에 가까워짐 | 0 = 릿지, 1 = 라쏘 |

### 8-4. 핵심 원칙

1. 특성이 적으면 닫힌 형태 해, 특성이 많거나 데이터가 크면 경사 하강법을 쓴다.
2. 경사 하강법과 규제 모델은 반드시 특성 스케일을 맞춘다.
3. 학습률은 너무 작으면 느리고 너무 크면 발산한다. SGD·미니배치는 학습률을 점점 줄인다.
4. 학습 곡선으로 진단한다: 두 곡선이 높고 붙으면 과소적합(데이터 추가 무효), 간격이 있으면 과대적합(데이터 추가 유효).
5. 규제 모델은 릿지를 기본으로, 특성 선택이 필요하면 라쏘·엘라스틱넷을 쓴다. 편향은 규제하지 않는다.
6. 반복 학습에서는 조기 종료로 최고 모델을 보존한다.
7. 로지스틱 회귀는 확률을 출력하는 선형 분류기이며, 다중 클래스는 소프트맥스로 확장한다. `C`는 규제의 역수다.
