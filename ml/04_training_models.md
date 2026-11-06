# 4장. 모델 훈련

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `04_training_linear_models.ipynb`
>
> 2·3장에서는 모델을 **블랙박스**처럼 `fit()`만 호출했다. 4장은 그 상자를 열어 **모델이 실제로 어떻게 학습하는지** 본다. 가장 단순한 선형 회귀를 두 방식(정규 방정식 / 경사 하강법)으로 직접 구현하고, 다항 회귀 → 학습 곡선 → 규제 → 로지스틱·소프트맥스 회귀로 넓혀 간다. 여기서 배우는 **경사 하강법**이 이후 신경망 학습의 기초가 된다.
>
> 실습 데이터
>
> | 데이터 | 생성식 / 출처 | 쓰이는 절 |
> |---|---|---|
> | 선형 데이터 (200개) | `y = 4 + 3x + 잡음`, `x ∈ [0, 2)` | 4.1, 4.2 |
> | 2차 데이터 (200개) | `y = 0.5x² + x + 2 + 잡음`, `x ∈ [-3, 3)` | 4.3, 4.4, 조기 종료 |
> | 작은 선형 데이터 (20개) | `y = 1 + 0.5x + 잡음/1.5`, `x ∈ [0, 3)` | 4.5 릿지·라쏘·엘라스틱넷 |
> | 붓꽃(iris) | `load_iris()` — 150개, 3 품종, 특성 4개 | 4.6 |

## 목차

1. [선형 회귀 (4.1)](#1-선형-회귀-41)
2. [경사 하강법 (4.2)](#2-경사-하강법-42)
3. [다항 회귀 (4.3)](#3-다항-회귀-43)
4. [학습 곡선 (4.4)](#4-학습-곡선-44)
5. [규제가 있는 선형 모델 (4.5)](#5-규제가-있는-선형-모델-45)
6. [로지스틱 회귀 (4.6)](#6-로지스틱-회귀-46)
7. [연습문제](#7-연습문제)
8. [전체 흐름 요약](#8-전체-흐름-요약)

---

## 0. 환경 설정

```python
import sys
assert sys.version_info >= (3, 10)

from packaging.version import Version
import sklearn
assert Version(sklearn.__version__) >= Version("1.6.1")

import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

2·3장과 같음. 이 장에서는 `root_mean_squared_error`(1.4+) 때문에 버전을 확인한다.

---

## 1. 선형 회귀 (4.1)

### 1-1. 모델과 비용 함수

선형 회귀는 입력 특성의 **가중치 합 + 편향(절편)**으로 예측한다.

```
ŷ = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ      → 벡터로 쓰면  ŷ = θᵀx
```

| 기호 | 의미 |
|---|---|
| `ŷ` | 예측값 |
| `n` | 특성 수 |
| `xᵢ` | i번째 특성값 (`x₀ = 1`로 두면 편향을 가중치처럼 다룰 수 있음) |
| `θⱼ` | j번째 모델 파라미터. `θ₀`은 편향, `θ₁…θₙ`은 가중치 |

**훈련 = 비용 함수를 최소화하는 θ 찾기.** 회귀에서는 보통 **MSE**(평균 제곱 오차)를 최소화한다.

```
MSE(θ) = (1/m) Σ (θᵀx⁽ⁱ⁾ − y⁽ⁱ⁾)²
```

> 💡 평가 지표는 RMSE를 썼는데 훈련은 MSE로 한다. **같은 θ에서 최솟값**이 나오고(제곱근은 단조 증가), MSE가 미분하기 더 쉽기 때문.

### 1-2. 데이터 만들기

```python
import numpy as np

rng = np.random.default_rng(seed=42)
m = 200  # 샘플 수
X = 2 * rng.random((m, 1))                     # 열 벡터
y = 4 + 3 * X + rng.standard_normal((m, 1))    # 열 벡터
```

| 줄 | 의미 |
|---|---|
| `np.random.default_rng(seed=42)` | 새 방식의 난수 생성기. 시드 고정 → 매번 같은 데이터 |
| `rng.random((m, 1))` | [0, 1) 균등분포, 모양 `(200, 1)` → ×2 하면 [0, 2) |
| `rng.standard_normal((m, 1))` | 평균 0, 표준편차 1 정규분포 = **잡음** |
| `4 + 3 * X` | 진짜 관계: 절편 4, 기울기 3. 모델이 이 값을 찾아내는지 확인할 것 |

> ⚠️ `X`를 `(200,)`가 아니라 `(200, 1)`로 만드는 이유: 사이킷런은 **2차원 X**(샘플 × 특성)를 기대한다.

### 1-3. 정규 방정식 (Normal Equation)

MSE를 최소화하는 θ를 **한 번에 계산하는 공식**이 있다.

```
θ̂ = (XᵀX)⁻¹ Xᵀ y
```

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # 각 샘플에 x0 = 1 추가
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
theta_best
# array([[3.69084138],
#        [3.32960458]])
```

| 줄 | 의미 |
|---|---|
| `add_dummy_feature(X)` | 맨 앞에 1로 채운 열 추가 → `(200, 2)`. 편향 θ₀에 곱해질 `x₀ = 1` |
| `X_b.T` | 전치 행렬 |
| `@` | 행렬 곱 (`np.dot`과 같음) |
| `np.linalg.inv()` | 역행렬 |

- 진짜 값 (4, 3)에 대해 (3.69, 3.33). **잡음** 때문에 정확히 같을 수는 없다.

**예측하기**

```python
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)  # x0 = 1 추가
y_predict = X_new_b @ theta_best
y_predict
# array([[ 3.69084138],
#        [10.35005055]])
```

- x=0과 x=2 두 점만 예측하면 직선을 그릴 수 있다 (그림 4-2).

```python
plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")  # 빨간 실선 = 예측 직선
plt.plot(X, y, "b.")                                    # 파란 점 = 데이터
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])   # [x최소, x최대, y최소, y최대]
plt.grid()
plt.legend(loc="upper left")
plt.show()
```

### 1-4. 사이킷런 `LinearRegression`

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

- 정규 방정식과 **완전히 같은 결과**
- 사이킷런은 편향(`intercept_`)과 가중치(`coef_`)를 **따로** 저장 → `add_dummy_feature` 필요 없음
- 끝에 `_`가 붙은 속성 = 학습으로 얻은 값

### 1-5. 내부 구현: 최소 제곱법과 유사역행렬

```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
# array([[3.69084138],
#        [3.32960458]])

np.linalg.pinv(X_b) @ y     # 유사역행렬로 직접 계산 → 같은 결과
```

- `LinearRegression`은 내부적으로 `scipy.linalg.lstsq()`(least squares)를 쓴다
- `lstsq`는 **θ̂ = X⁺y**를 계산. `X⁺`는 **유사역행렬**(무어-펜로즈 역행렬)
- 유사역행렬은 **SVD(특잇값 분해)**로 구한다

**왜 정규 방정식 대신 유사역행렬?**

| | 정규 방정식 `(XᵀX)⁻¹` | 유사역행렬 `X⁺` (SVD) |
|---|---|---|
| `XᵀX`가 역행렬이 없을 때 (m < n, 중복 특성) | **계산 불가** | 항상 계산 가능 |
| 계산 복잡도 (특성 수 n) | O(n²·⁴) ~ O(n³) | O(n²) |

### 1-6. 계산 복잡도 정리

- 특성 수 n이 2배 → 정규 방정식 약 5.3~8배, SVD 약 4배 느려짐 → **특성이 많으면(수십만) 매우 느림**
- 샘플 수 m에 대해서는 **선형** O(m) → 메모리에만 들어가면 대용량도 OK
- 예측은 샘플·특성 수에 모두 **선형** → 빠름

> 💡 그래서 특성이 아주 많거나 데이터가 메모리에 다 안 들어갈 때는 다른 방법이 필요하다 → **경사 하강법**

---

## 2. 경사 하강법 (4.2)

### 2-1. 아이디어

> 짙은 안개 속 산에서 계곡으로 내려가려면? **발밑 경사가 가장 가파른 방향**으로 한 걸음씩 내려간다.

1. θ를 **무작위로 초기화**
2. 현재 θ에서 비용 함수의 **그레이디언트(기울기)**를 계산
3. 그레이디언트의 **반대 방향**으로 조금 이동
4. 그레이디언트가 0에 가까워질 때(최솟값)까지 반복

| 개념 | 설명 |
|---|---|
| **학습률 η (eta)** | 한 걸음의 크기. 너무 작으면 오래 걸리고, 너무 크면 발산 |
| **지역 최솟값** | 전역 최솟값이 아닌 웅덩이에 갇힐 위험 |
| **볼록 함수** | 선형 회귀의 MSE는 볼록(그릇 모양) → **지역 최솟값이 없음**. 전역 최솟값 하나뿐 |
| **특성 스케일** | 스케일이 다르면 그릇이 길쭉해져 수렴이 매우 느려짐 → `StandardScaler` 필수 |

> ⚠️ 경사 하강법을 쓸 때는 **반드시 특성 스케일을 맞춘다.** (정규 방정식·SVD는 스케일 상관없음)

### 2-2. 배치 경사 하강법 (Batch GD)

MSE의 그레이디언트 벡터:

```
∇θ MSE(θ) = (2/m) Xᵀ(Xθ − y)
```

매 스텝마다 **전체 훈련 세트**로 그레이디언트를 계산한다 → "배치".

```python
eta = 0.1        # 학습률
n_epochs = 1000
m = len(X_b)     # 샘플 수

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))   # 파라미터 무작위 초기화

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

theta
# array([[3.69084138],
#        [3.32960458]])
```

| 줄 | 의미 |
|---|---|
| `theta = rng.standard_normal((2, 1))` | θ₀, θ₁을 정규분포 난수로 시작 |
| `X_b @ theta - y` | 모든 샘플의 **예측 오차** 벡터 `(200, 1)` |
| `X_b.T @ (오차)` | 각 파라미터에 대한 오차 기여도 합 `(2, 1)` |
| `2 / m * ...` | MSE 미분 계수 |
| `theta - eta * gradients` | 기울기 반대 방향으로 η만큼 이동 = **경사 하강 스텝** |
| **에포크(epoch)** | 훈련 세트를 한 바퀴 다 도는 것 |

- 결과가 정규 방정식과 **소수점까지 동일**. 경사 하강법이 제대로 수렴했다.

**학습률 비교 (그림 4-8)**

```python
import matplotlib as mpl

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_epochs = 1000
    n_shown = 20
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < n_shown:   # 처음 20 스텝만 직선을 그림
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle="solid", color=color)
        gradients = 2 / m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
        theta_path.append(theta)   # 나중에 경로 비교 그림(4-11)용
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

- `plt.cm.OrRd(값)`: 0~1 값을 주황→빨강 색으로 변환 → 스텝이 진행될수록 진해짐
- `plt.subplot(131)`: 1행 3열 중 1번째 칸

| η | 결과 |
|---|---|
| 0.02 | 너무 작음 → 20 스텝으로는 아직 한참 멀었음 |
| 0.1 | 적당 → 몇 스텝 만에 정답 근처 |
| 0.5 | 너무 큼 → 직선이 위아래로 튀며 **발산** |

> 💡 적절한 학습률은 **그리드 서치**로 찾고, 반복 횟수는 크게 잡되 그레이디언트 벡터가 아주 작아지면(허용오차 ε 이하) 멈춘다.

### 2-3. 확률적 경사 하강법 (SGD)

배치 GD는 매 스텝 **전체 데이터**를 쓰므로 데이터가 크면 느리다. SGD는 매 스텝 **샘플 하나를 무작위로** 골라 그레이디언트를 계산한다.

| | 배치 GD | SGD |
|---|---|---|
| 스텝당 계산 | 전체 m개 | 1개 → 매우 빠름 |
| 경로 | 매끄럽게 수렴 | 지그재그로 요동 |
| 최솟값 근처 | 멈춤 | 계속 맴돔 (정확히 수렴 X) |
| 지역 최솟값 | 갇힐 수 있음 | 요동 덕분에 **탈출 가능** |
| 대용량 / 외부 메모리 | 어려움 | 가능 |

**학습 스케줄**: 요동 문제를 해결하려고 학습률을 **처음엔 크게, 점점 작게** 줄인다 (담금질 기법과 비슷).

```python
theta_path_sgd = []   # 그림용 경로 저장

n_epochs = 50
t0, t1 = 5, 50        # 학습 스케줄 하이퍼파라미터

def learning_schedule(t):
    return t0 / (t + t1)   # t=0 → 0.1, t가 커질수록 0에 가까워짐

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = rng.integers(m)              # 0~m-1 중 하나
        xi = X_b[random_index : random_index + 1]   # 샘플 1개 (2차원 유지)
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)    # SGD는 m으로 나누지 않음
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients
        theta_path_sgd.append(theta)

theta
# array([[3.69826475],
#        [3.30748311]])
```

| 줄 | 의미 |
|---|---|
| `rng.integers(m)` | 무작위 인덱스 1개. **복원 추출**이라 한 에포크에 어떤 샘플은 여러 번, 어떤 샘플은 0번 |
| `X_b[i : i+1]` | `X_b[i]`는 1차원이 되므로 슬라이싱으로 **(1, 2) 모양 유지** |
| `2 * xi.T @ ...` | 샘플 1개라 m=1 → `/ m` 없음 |
| `epoch * m + iteration` | 지금까지 진행한 **전체 스텝 수** → 학습률이 계속 감소 |

- 배치 GD는 1000 에포크, SGD는 **50 에포크**만에 비슷한 결과 (3.698, 3.307)
- 그림 4-10: 첫 에포크 20 스텝 동안 직선이 불규칙하게 움직이는 모습

> ⚠️ SGD는 샘플이 **IID**(독립·동일 분포)여야 한다. 레이블 순으로 정렬된 데이터라면 섞고 시작할 것. (`SGDRegressor`는 기본으로 매 에포크 섞는다)

**사이킷런 `SGDRegressor`**

```python
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                       n_iter_no_change=100, random_state=42)
sgd_reg.fit(X, y.ravel())   # fit()은 1차원 타깃을 기대

sgd_reg.intercept_, sgd_reg.coef_
# (array([3.68899733]), array([3.33054574]))
```

| 매개변수 | 의미 |
|---|---|
| `max_iter=1000` | 최대 에포크 수 |
| `tol=1e-5`, `n_iter_no_change=100` | 100 에포크 동안 손실 개선이 1e-5보다 작으면 중지 |
| `penalty=None` | 규제 없음 (4.5에서 `"l2"` 등 사용) |
| `eta0=0.01` | 초기 학습률 (기본 스케줄로 점점 감소) |
| `y.ravel()` | `(200, 1)` → `(200,)` 1차원으로 펼침 |

### 2-4. 미니배치 경사 하강법

배치(전체)와 SGD(1개)의 중간 — **작은 무작위 묶음(미니배치)**으로 그레이디언트를 계산한다.

- 행렬 연산 최적화, 특히 **GPU** 덕분에 SGD보다 빠르고 덜 요동함
- 신경망 학습의 표준 방식

```python
from math import ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)   # 200/20 = 10

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

theta_path_mgd = []
for epoch in range(n_epochs):
    shuffled_indices = rng.permutation(m)          # 매 에포크 섞기
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_b_shuffled[idx : idx + minibatch_size]   # 20개 묶음
        yi = y_shuffled[idx : idx + minibatch_size]
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
```

| 줄 | 의미 |
|---|---|
| `rng.permutation(m)` | 0~m-1을 무작위로 섞은 배열 → **비복원**. 한 에포크에 모든 샘플을 정확히 한 번씩 사용 |
| `ceil(m / size)` | 올림. 나누어떨어지지 않으면 마지막 배치는 작아짐 |
| `2 / minibatch_size` | 배치 크기로 평균 |

**세 방식의 파라미터 경로 비교 (그림 4-11)**

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

- `[:, 0]`, `[:, 1]`: 경로 배열에서 θ₀, θ₁만 뽑아 x축·y축으로
- 배치(파랑)는 곧장 최솟값에 도착해 멈춤 / SGD(빨강)는 크게 요동 / 미니배치(초록)는 그 중간

### 2-5. 선형 회귀 알고리즘 비교 (교재 표 4-1)

| 알고리즘 | 큰 m | 외부 메모리 학습 | 큰 n | 하이퍼파라미터 | 스케일 조정 | 사이킷런 |
|---|---|---|---|---|---|---|
| 정규 방정식 | 빠름 | X | 느림 | 0 | 불필요 | — |
| SVD | 빠름 | X | 느림 | 0 | 불필요 | `LinearRegression` |
| 배치 GD | 느림 | X | 빠름 | 2 | 필요 | — |
| SGD | 빠름 | O | 빠름 | ≥2 | 필요 | `SGDRegressor` |
| 미니배치 GD | 빠름 | O | 빠름 | ≥2 | 필요 | — |

> 💡 훈련 방법이 달라도 **결과 모델은 같다** (예측 방식 동일). 다른 것은 훈련 속도와 메모리 사용.

---

## 3. 다항 회귀 (4.3)

데이터가 직선이 아니라면? → **각 특성의 거듭제곱을 새 특성으로 추가**한 뒤 선형 모델을 훈련한다. 모델 자체는 여전히 "선형"(θ에 대해 선형).

### 3-1. 2차 데이터 만들기

```python
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3                          # [-3, 3)
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))  # 진짜 식: 0.5x² + 1x + 2
```

- 그림 4-12: 포물선 모양. 직선으로는 잘 맞출 수 없다.

### 3-2. `PolynomialFeatures`로 특성 추가

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

X[0]        # array([1.64373629])
X_poly[0]   # array([1.64373629, 2.701869  ])   ← [x, x²]
```

| 매개변수 | 의미 |
|---|---|
| `degree=2` | 2차항까지 생성 |
| `include_bias=False` | 1로 된 열(x⁰)을 만들지 않음. `LinearRegression`이 절편을 따로 학습하므로 불필요 |

```python
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# (array([2.00540719]), array([[1.11022126, 0.50526985]]))
```

- 학습 결과: **ŷ = 0.505x² + 1.110x + 2.005**
- 진짜 식 **0.5x² + 1x + 2**와 거의 같음 → 다항 회귀 성공

**예측 곡선 그리기 (그림 4-13)**

```python
X_new = np.linspace(-3, 3, 100).reshape(100, 1)   # -3~3을 100등분
X_new_poly = poly_features.transform(X_new)       # 새 데이터는 transform만
y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
```

- `np.linspace(시작, 끝, 개수)`: 균등 간격 배열 → 매끄러운 곡선용
- `.reshape(100, 1)`: 1차원 → 열 벡터 (사이킷런 입력 형식)

### 3-3. 특성 조합 폭발

> ⚠️ 특성이 여러 개면 `PolynomialFeatures`는 **교차항**도 만든다. 특성 a, b에 degree=3이면 a², a³, b², b³뿐 아니라 **ab, a²b, ab²**까지 생성.
>
> 특성 n개, 차수 d → 특성 수 = **(n+d)! / (d!·n!)** → 아주 빠르게 폭발하므로 주의.

---

## 4. 학습 곡선 (4.4)

### 4-1. 차수가 너무 높으면? (그림 4-14)

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

| 차수 | 결과 |
|---|---|
| 1 (직선) | **과소적합** — 포물선을 못 따라감 |
| 2 | 데이터 생성식과 같은 차수 → 가장 적절 |
| 300 | **과대적합** — 훈련 샘플을 지나가려고 심하게 구불거림 |

- `StandardScaler`를 넣는 이유: x³⁰⁰ 같은 값은 엄청나게 크거나 작아짐 → 스케일을 맞춰야 수치가 안정
- `f"...{'s' if degree > 1 else ''}"`: 복수형 처리 (1 degree / 2 degrees)

**실제 문제에서는 진짜 차수를 모른다.** 과대·과소적합을 어떻게 판단할까?

1. **교차 검증** (2장): 훈련 점수는 좋은데 검증 점수가 나쁘면 과대적합, 둘 다 나쁘면 과소적합
2. **학습 곡선**: 훈련 세트 크기를 늘려 가며 훈련 오차와 검증 오차를 그래프로 비교

### 4-2. 선형 회귀의 학습 곡선 (그림 4-15)

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

| 인자 / 반환값 | 의미 |
|---|---|
| `train_sizes=np.linspace(0.01, 1.0, 40)` | 훈련 폴드의 1%~100%를 40단계로 |
| `cv=5` | 5-폴드 교차 검증 → 훈련 폴드 최대 200×4/5 = **160개** (그래서 x축이 160까지) |
| `scoring="neg_root_mean_squared_error"` | 점수는 "클수록 좋음" → 음수 RMSE |
| `train_scores` | 모양 `(40, 5)` — 크기별 × 폴드별 |
| `-...mean(axis=1)` | 폴드 평균 후 부호를 뒤집어 RMSE로 |

곡선 읽기:
- **훈련 오차**: 샘플 1~2개면 완벽히 맞춰 0 → 샘플이 늘수록 직선이 다 못 맞춰서 증가 → 약 1.6에서 평평
- **검증 오차**: 샘플이 적으면 일반화를 못해 큼 → 점점 감소 → 약 1.6에서 평평
- 두 곡선이 **높은 곳에서 가깝게 붙어 평평** → 전형적인 **과소적합**

> 💡 과소적합이면 **샘플을 더 모아도 소용없다.** 더 복잡한 모델을 쓰거나 더 나은 특성을 찾아야 한다.

### 4-3. 10차 다항 회귀의 학습 곡선 (그림 4-16)

```python
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
```

선형 회귀와의 차이:
- 훈련 오차가 **훨씬 낮음** (전체 크기에서 약 0.97 vs 1.6)
- 두 곡선 사이에 **간격**이 있음 → 훈련 데이터에서 더 잘함 = **과대적합**의 신호
- 훈련 세트가 커지면 두 곡선이 점점 가까워짐

> 💡 과대적합이면 **검증 오차가 훈련 오차에 가까워질 때까지 훈련 데이터를 더 추가**하는 것이 한 방법.

### 4-4. 편향 / 분산 트레이드오프

일반화 오차 = **편향** + **분산** + **줄일 수 없는 오차**

| 오차 | 원인 | 결과 | 해결 |
|---|---|---|---|
| 편향 | 잘못된 가정 (예: 2차를 직선으로) | 과소적합 | 모델 복잡도 ↑ |
| 분산 | 작은 변동에 과민 (자유도 높은 모델) | 과대적합 | 모델 복잡도 ↓, 데이터 ↑, 규제 |
| 줄일 수 없는 오차 | 데이터 자체의 잡음 | — | 데이터 정제 (고장 센서 수리, 이상치 제거) |

- 복잡도 ↑ → 분산 ↑, 편향 ↓ / 복잡도 ↓ → 분산 ↓, 편향 ↑ → 그래서 **트레이드오프**

---

## 5. 규제가 있는 선형 모델 (4.5)

과대적합을 줄이는 좋은 방법은 모델을 **규제**(regularize, 제약)하는 것. 다항 모델이면 차수를 줄이면 되고, 선형 모델이면 **가중치를 제한**한다.

### 5-0. 실습 데이터 (작고 잡음 많은 선형 데이터)

```python
rng = np.random.default_rng(seed=42)
m = 20
X = 3 * rng.random((m, 1))
y = 1 + 0.5 * X + rng.standard_normal((m, 1)) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)
```

- 샘플이 **20개**뿐 → 과대적합하기 쉬운 상황을 일부러 만듦

### 5-1. 릿지 회귀 (Ridge, ℓ₂ 규제)

비용 함수에 **가중치 제곱합**을 더한다.

```
J(θ) = MSE(θ) + (α/m) Σ θᵢ²      (i = 1…n, 편향 θ₀은 규제하지 않음)
```

| 항목 | 의미 |
|---|---|
| `α` (alpha) | 규제 강도. α=0이면 일반 선형 회귀, α가 크면 가중치가 0에 가까워져 **평평한 직선** |
| 편향 제외 | 절편까지 줄이면 예측이 전체적으로 내려가 버림 |
| 스케일 필수 | 규제는 가중치 크기에 민감 → **반드시 `StandardScaler`** |

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84414523])
```

- `solver="cholesky"`: 숄레스키 분해로 **닫힌 형태 해**를 바로 계산 (정규 방정식의 릿지 버전)

**α에 따른 변화 (그림 4-17)**

```python
def plot_model(model_class, polynomial, alphas, **model_kwargs):
    plt.plot(X, y, "b.", linewidth=3)
    for alpha, style in zip(alphas, ("b:", "g--", "r-")):
        if alpha > 0:
            model = model_class(alpha, **model_kwargs)
        else:
            model = LinearRegression()           # α=0은 규제 없음과 같음
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

| 줄 | 의미 |
|---|---|
| `**model_kwargs` | 추가 키워드 인자를 딕셔너리로 받아 모델에 그대로 전달 (`random_state=42`) |
| `zip(alphas, styles)` | α 3개와 선 스타일 3개를 짝지어 반복 |
| `fr"$\alpha = {alpha}$"` | f-string + raw string. `\a`가 이스케이프로 해석되지 않게 `r` |
| 왼쪽 그림 | 단순 선형 + 릿지. α가 클수록 직선이 수평에 가까워짐 |
| 오른쪽 그림 | 10차 다항 + 릿지. α=0이면 구불구불(과대적합), α=1이면 매끄러운 곡선 |

> 💡 직접 확인: 10차 다항 + 스케일링 후 가중치 최댓값이 α=10⁻⁵에서 약 23, α=1에서 약 0.48로 줄어든다. 단, **0이 되지는 않는다** (10개 모두 0이 아님). → 라쏘와의 핵심 차이.

**SGD로 릿지 하기**

```python
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
# array([1.83659707])
```

- `penalty="l2"`: 비용 함수에 ℓ₂ 항 추가 = 릿지
- `alpha=0.1 / m`: `SGDRegressor`의 α는 m으로 나눈 것이 아니라 그대로 쓰는 정의 → `Ridge(alpha=0.1)`과 맞추려면 **m으로 나눔**
- `tol=None`: 조기 중지 없이 1000 에포크를 다 돎

**확률적 평균 GD (`solver="sag"`)**

```python
ridge_reg = Ridge(alpha=0.1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84415109])   ← cholesky(1.84414523)와 거의 같음
```

**닫힌 형태 해 직접 계산**

```
θ̂ = (XᵀX + αA)⁻¹ Xᵀ y      A: 왼쪽 위(편향 자리)만 0인 단위 행렬
```

```python
alpha = 0.1
A = np.array([[0., 0.], [0., 1.]])   # 편향은 규제하지 않으므로 [0,0]=0
X_b = np.c_[np.ones(m), X]           # add_dummy_feature와 같음
np.linalg.inv(X_b.T @ X_b + alpha * A) @ X_b.T @ y
# array([[1.23749481],
#        [0.40443361]])

ridge_reg.intercept_, ridge_reg.coef_
# (array([1.23754037]), array([0.40440715]))   ← 일치
```

- `np.c_[a, b]`: 열 방향으로 붙이기

### 5-2. 라쏘 회귀 (Lasso, ℓ₁ 규제)

**가중치 절댓값 합**을 더한다.

```
J(θ) = MSE(θ) + 2α Σ |θᵢ|
```

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
# array([1.87550211])
```

```python
# 그림 4-18
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 1e-2, 1), random_state=42)
```

**라쏘의 핵심: 덜 중요한 특성의 가중치를 정확히 0으로 만든다**
- 오른쪽 그림 α=0.01: 10차 다항인데 거의 **2차 곡선**처럼 보임 → 고차항 가중치가 0
- 직접 확인: α=0.01에서 0이 아닌 가중치는 10개 중 **2개**, α=1에서는 **0개** (수평선)
- → **자동 특성 선택** + **희소 모델(sparse model)**

**왜 ℓ₁은 0을 만들고 ℓ₂는 안 만들까? (그림 4-19)**

| | ℓ₁ (라쏘) | ℓ₂ (릿지) |
|---|---|---|
| 규제 항 그래디언트 | `sign(θ)` — 크기와 상관없이 **일정한 힘** | `θ` — 0에 가까울수록 **힘이 약해짐** |
| 경로 | 한 축이 먼저 0에 닿고 축을 따라 이동 | 원점을 향해 곧게, 점점 느려짐 |
| 최솟값 | 축 위(일부 θ = 0) | 0 근처지만 0은 아님 |

그림 4-19 코드의 핵심 부분:

```python
def bgd_path(theta, X, y, l1, l2, core=1, eta=0.05, n_iterations=200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = (core * 2 / len(X) * X.T @ (X @ theta - y)   # MSE 그레이디언트
                     + l1 * np.sign(theta)                        # ℓ₁ 그레이디언트
                     + l2 * theta)                                # ℓ₂ 그레이디언트
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)
```

- `core=0`으로 호출하면 MSE 항이 사라져 **규제 항만으로** 움직이는 경로(노란 점선)를 그림
- `np.meshgrid` + `contourf`: θ₁·θ₂ 평면 전체의 비용을 등고선으로 칠함

> ⚠️ ℓ₁의 |θ|는 θ=0에서 **미분 불가능**. 대신 **서브그레이디언트 벡터**(`sign(θ)`, 0에서는 0)를 쓰면 경사 하강법이 잘 동작한다.
>
> ⚠️ 라쏘는 최솟값 근처에서 요동하므로 SGD로 훈련할 때는 **학습률을 점점 줄여야** 한다.

### 5-3. 엘라스틱넷 (Elastic Net)

릿지와 라쏘를 **혼합 비율 r**로 섞는다.

```
J(θ) = MSE(θ) + r·(2α Σ|θᵢ|) + (1−r)·(α/m Σθᵢ²)
```

- r = 0 → 릿지, r = 1 → 라쏘

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)   # l1_ratio = r
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
# array([1.8645014])
```

**무엇을 쓸까?**

| 상황 | 추천 |
|---|---|
| 기본 | 규제 없는 선형 회귀는 피하고 **릿지** |
| 쓸모 있는 특성이 몇 개뿐일 것 같을 때 | **라쏘** 또는 **엘라스틱넷** (특성 선택) |
| 특성 수 > 샘플 수, 또는 특성끼리 강한 상관 | 라쏘가 불안정 → **엘라스틱넷** |

### 5-4. 조기 종료 (Early Stopping)

반복 학습 알고리즘(경사 하강법 등)에서 **검증 오차가 최솟값에 도달하면 훈련을 멈추는** 규제 방법. 제프리 힌턴은 "아름다운 공짜 점심"이라 불렀다.

```python
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# 2차 데이터를 다시 만들고 앞/뒤 절반으로 나눔
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))
X_train, y_train = X[: m // 2], y[: m // 2, 0]
X_valid, y_valid = X[m // 2 :], y[m // 2 :, 0]

preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False),
                              StandardScaler())
X_train_prep = preprocessing.fit_transform(X_train)
X_valid_prep = preprocessing.transform(X_valid)

sgd_reg = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
n_epochs = 500
best_valid_rmse = float('inf')
train_errors, val_errors = [], []

for epoch in range(n_epochs):
    sgd_reg.partial_fit(X_train_prep, y_train)
    y_valid_predict = sgd_reg.predict(X_valid_prep)
    val_error = root_mean_squared_error(y_valid, y_valid_predict)
    if val_error < best_valid_rmse:
        best_valid_rmse = val_error
        best_model = deepcopy(sgd_reg)

    y_train_predict = sgd_reg.predict(X_train_prep)   # 그림용
    train_error = root_mean_squared_error(y_train, y_train_predict)
    val_errors.append(val_error)
    train_errors.append(train_error)
```

| 줄 | 의미 |
|---|---|
| `m // 2` | 정수 나눗셈 → 100 |
| `y[: m // 2, 0]` | `(100, 1)`의 0번째 열 → 1차원 `(100,)` |
| `degree=90` | 일부러 과대적합하기 쉬운 고차 모델 |
| `fit_transform` / `transform` | 스케일러는 **훈련 세트로만 fit**, 검증 세트엔 transform만 (데이터 누설 방지) |
| `float('inf')` | 무한대. 첫 비교에서 무조건 갱신되게 |
| `partial_fit()` | **1 에포크만** 훈련하고 이전 상태에서 이어감 (`fit()`은 매번 처음부터) |
| `deepcopy(sgd_reg)` | 모델을 **통째로 복사**. 그냥 대입하면 같은 객체를 가리켜 계속 바뀜 |

**그림 4-20**

```python
best_epoch = np.argmin(val_errors)
plt.annotate('Best model',
             xy=(best_epoch, best_valid_rmse),
             xytext=(best_epoch, best_valid_rmse + 0.5),
             ha="center",
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.plot([0, n_epochs], [best_valid_rmse, best_valid_rmse], "k:", linewidth=2)
plt.plot(val_errors, "b-", linewidth=3, label="Validation set")
plt.plot(best_epoch, best_valid_rmse, "bo")
plt.plot(train_errors, "r--", linewidth=2, label="Training set")
```

- `np.argmin`: 최솟값의 인덱스
- `plt.annotate(텍스트, xy=화살표 끝, xytext=텍스트 위치)`: 화살표 주석
- 직접 돌려 보면 **184번째 에포크**에서 검증 RMSE 최소(약 1.105) → 이후 훈련 오차는 계속 내려가지만 검증 오차는 다시 올라감 (500 에포크에서 약 1.318) = 과대적합 시작

> 💡 SGD·미니배치 GD는 곡선이 매끄럽지 않아 최솟값인지 판단하기 어렵다. 그래서 위 코드처럼 **최고 모델을 저장해 두고**, 한동안 갱신이 없으면 멈춘 뒤 저장된 모델로 되돌린다.

---

## 6. 로지스틱 회귀 (4.6)

회귀 알고리즘을 **분류**에 쓸 수 있다. 로지스틱 회귀는 샘플이 특정 클래스에 속할 **확률**을 추정한다. 확률이 50% 이상이면 양성 클래스(1), 아니면 음성(0) → **이진 분류기**.

### 6-1. 확률 추정

선형 회귀처럼 가중치 합을 계산하지만, 결과를 그대로 내지 않고 **시그모이드(로지스틱) 함수**에 넣는다.

```
p̂ = σ(θᵀx),     σ(t) = 1 / (1 + e⁻ᵗ)
ŷ = 1 (p̂ ≥ 0.5)  /  0 (p̂ < 0.5)
```

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
```

| 개념 | 의미 |
|---|---|
| 시그모이드 σ(t) | 어떤 실수든 **0~1 사이**로 변환하는 S자 곡선 |
| t ≥ 0 ↔ σ(t) ≥ 0.5 | 그래서 **θᵀx ≥ 0이면 양성**으로 예측 |
| **로짓(logit)** | t = θᵀx. 로짓 = log(p / (1−p)) = **로그 오즈**. 시그모이드의 역함수 |

### 6-2. 훈련과 비용 함수

목표: 양성 샘플엔 높은 확률, 음성 샘플엔 낮은 확률.

```
샘플 하나의 비용:  c(θ) = −log(p̂)       (y = 1일 때)
                         −log(1 − p̂)    (y = 0일 때)

전체 비용 (로그 손실):
J(θ) = −(1/m) Σ [ y⁽ⁱ⁾ log(p̂⁽ⁱ⁾) + (1 − y⁽ⁱ⁾) log(1 − p̂⁽ⁱ⁾) ]
```

- 양성인데 p̂ → 0이면 −log(p̂) → ∞ (아주 큰 벌점) / p̂ → 1이면 0
- **닫힌 형태 해가 없다** (정규 방정식 같은 공식 X)
- 하지만 **볼록 함수** → 경사 하강법으로 전역 최솟값을 반드시 찾음

그레이디언트 (선형 회귀와 모양이 거의 같다):

```
∂J/∂θⱼ = (1/m) Σ (σ(θᵀx⁽ⁱ⁾) − y⁽ⁱ⁾) xⱼ⁽ⁱ⁾       ← 예측 오차 × 특성값의 평균
```

### 6-3. 결정 경계 — 붓꽃 데이터

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

iris.target.head(3)    # 0 0 0  ← 섞여 있지 않음 (클래스 순 정렬)
iris.target_names      # array(['setosa', 'versicolor', 'virginica'])
```

| 항목 | 내용 |
|---|---|
| 샘플 | 150개 (품종별 50개) |
| 특성 | 꽃받침(sepal) 길이·너비, 꽃잎(petal) 길이·너비 (cm) |
| 타깃 | 0 setosa / 1 versicolor / 2 virginica |
| `as_frame=True` | 데이터를 pandas DataFrame/Series로 |

> ⚠️ 타깃이 정렬돼 있다 → 앞부분만 자르면 한 품종만 들어감. `train_test_split`(기본 섞기)를 쓴다.

**특성 1개(꽃잎 너비)로 virginica 감지**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = iris.data[["petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
```

| 줄 | 의미 |
|---|---|
| `iris.data[["..."]]` | **대괄호 2개** → DataFrame(2차원) 유지. 1개면 Series(1차원) |
| `.values` | NumPy 배열로 변환 → `(150, 1)` |
| `iris.target_names[iris.target]` | 0/1/2 → 품종 이름 배열로 바꾸는 **팬시 인덱싱** |
| `== 'virginica'` | virginica면 True → 이진 타깃 |
| `train_test_split(..., random_state=42)` | 기본 75% 훈련 / 25% 테스트, 섞어서 분할 |

**확률 곡선과 결정 경계 (그림 4-23)**

```python
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)   # 꽃잎 너비 0~3cm
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
plt.plot(X_train[y_train == 0], y_train[y_train == 0], "bs")   # 음성 샘플 (y=0 위치)
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")   # 양성 샘플 (y=1 위치)

decision_boundary
# np.float64(1.6516516516516517)
```

| 줄 | 의미 |
|---|---|
| `reshape(-1, 1)` | `-1` = "나머지는 알아서" → `(1000, 1)` |
| `predict_proba()` | 모양 `(1000, 2)` — [음성 확률, 양성 확률] |
| `y_proba[:, 1] >= 0.5` | 양성 확률 50% 이상인 위치 (불리언 마스크) |
| `X_new[마스크][0, 0]` | 처음으로 50%를 넘는 x값 = **결정 경계** |

- 꽃잎 너비 **약 1.65cm**가 경계. 그보다 넓으면 virginica로 예측
- 2cm 이상은 확실히 virginica, 1cm 이하는 확실히 아님, 그 사이는 불확실 (두 클래스가 겹침)

```python
log_reg.predict([[1.7], [1.5]])
# array([ True, False])   ← 1.7 > 1.65 → True, 1.5 < 1.65 → False
```

**특성 2개(꽃잎 길이·너비) — 그림 4-24**

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)

# 등고선용 격자
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]      # 격자점 10만 개를 샘플로
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)       # 다시 격자 모양으로

# 결정 경계 직선: θ0 + θ1·x0 + θ2·x1 = 0  →  x1 = −(θ1·x0 + θ0) / θ2
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])

contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)              # 등고선에 확률값 라벨
plt.plot(left_right, boundary, "k--", linewidth=3)
```

| 줄 | 의미 |
|---|---|
| `np.meshgrid(a, b)` | 두 축 값으로 2차원 격자 좌표 행렬 2개 생성 `(200, 500)` |
| `.ravel()` | 2차원 → 1차원으로 펼침 |
| `np.c_[...]` | 두 열을 붙여 `(100000, 2)` 샘플 행렬로 |
| `reshape(x0.shape)` | 예측 결과를 다시 격자 모양으로 → 등고선 그리기 가능 |
| `C=2` | **규제 강도의 역수**. C가 작을수록 규제가 강함 (릿지의 α와 반대) |

- 결정 경계(점선)는 확률 50% 지점들을 잇는 **직선** → 로지스틱 회귀는 **선형 결정 경계**
- 다른 평행선들은 15%, 30%, … 90% 확률 등고선

> 💡 `LogisticRegression`은 기본으로 **ℓ₂ 규제**가 걸려 있다. 하이퍼파라미터 이름이 α가 아니라 **C(= 1/α)**인 점에 주의.

### 6-4. 소프트맥스 회귀 (다항 로지스틱 회귀)

로지스틱 회귀를 **여러 클래스**로 일반화 (OvR처럼 이진 분류기 여러 개를 묶지 않고 하나의 모델로).

1. 클래스 k마다 점수 계산: **sₖ(x) = (θ⁽ᵏ⁾)ᵀ x** — 클래스마다 별도의 파라미터 벡터
2. **소프트맥스 함수**로 확률 변환:

```
p̂ₖ = σ(s(x))ₖ = exp(sₖ(x)) / Σⱼ exp(sⱼ(x))
ŷ = argmaxₖ p̂ₖ          ← 확률이 가장 높은 클래스
```

- 지수를 취한 뒤 합으로 나눠 **정규화** → 모든 클래스 확률의 합 = 1
- 점수(로짓)가 가장 높은 클래스가 확률도 가장 높음

**비용 함수: 크로스 엔트로피**

```
J(Θ) = −(1/m) Σᵢ Σₖ yₖ⁽ⁱ⁾ log(p̂ₖ⁽ⁱ⁾)
```

- 정답 클래스의 확률이 낮으면 큰 벌점. 클래스가 2개면 로지스틱 회귀의 로그 손실과 같다.

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]                      # 0, 1, 2 그대로 → 3 클래스
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5, 2]])
# array([2])            ← 꽃잎 길이 5cm, 너비 2cm → virginica

softmax_reg.predict_proba([[5, 2]]).round(2)
# array([[0.  , 0.04, 0.96]])   ← setosa 0%, versicolor 4%, virginica 96%
```

- 타깃이 3개 이상이면 `LogisticRegression`이 **자동으로 소프트맥스 회귀**로 동작
- `C=30`: 규제를 약하게 (C가 클수록 약함)

**결정 경계 (그림 4-25)**

```python
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])   # 클래스별 배경색

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)    # versicolor 확률 → 등고선
zz = y_predict.reshape(x0.shape)         # 예측 클래스 → 배경 색칠

plt.contourf(x0, x1, zz, cmap=custom_cmap)   # 영역 색칠
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
```

- `contourf`(채움) vs `contour`(선만)
- 클래스 사이 결정 경계가 모두 **직선** (선형 모델이므로)
- 세 경계가 한 점에서 만나는 곳은 세 클래스 확률이 모두 33%

---

## 7. 연습문제

### 7-1. 개념 문제 (1~11) 요약

| # | 질문 | 답 |
|---|---|---|
| 1 | 특성이 수백만 개면? | SGD·미니배치 GD (메모리에 들어가면 배치 GD도). 정규 방정식·SVD는 특성 수에 대해 너무 느림 |
| 2 | 특성 스케일이 많이 다르면? | 비용 함수가 길쭉한 그릇 → GD 수렴이 느림. **스케일링** 필요. 규제 모델도 작은 값 특성을 무시하게 됨. 정규 방정식·SVD는 괜찮음 |
| 3 | 로지스틱 회귀에서 GD가 지역 최솟값에 갇히나? | 아니오. 비용 함수가 **볼록** |
| 4 | GD 알고리즘들은 같은 모델에 도달하나? | 볼록 + 적절한 학습률이면 비슷함. 단, 학습률을 줄이지 않으면 SGD·미니배치는 최솟값 근처에서 계속 맴돌아 약간씩 다름 |
| 5 | 배치 GD에서 검증 오차가 계속 올라가면? | 훈련 오차도 오르면 학습률이 너무 커서 **발산** → 학습률 ↓. 훈련 오차는 그대로면 **과대적합** → 훈련 중지 |
| 6 | 미니배치 GD에서 검증 오차가 오르면 즉시 멈출까? | 아니오. 무작위성 때문에 매번 개선되지 않음. 주기적으로 저장하고, 오래 개선 없으면 최고 모델로 복귀 |
| 7 | 가장 빨리 최적점 근처에 가는 것 / 실제로 수렴하는 것은? | 가장 빠름: **SGD**. 실제 수렴: **배치 GD**만 (나머지는 학습률을 줄여야 수렴) |
| 8 | 다항 회귀에서 검증 오차가 훈련 오차보다 훨씬 크면? | 과대적합. ① 차수 ↓ ② 규제(릿지·라쏘) ③ 훈련 데이터 ↑ |
| 9 | 릿지에서 훈련·검증 오차가 비슷하게 높으면? | 과소적합(높은 편향) → **α를 줄인다** |
| 10 | 릿지 vs 선형 회귀 / 라쏘 vs 릿지 / 엘라스틱넷 vs 라쏘 | 약간의 규제가 거의 항상 낫다 → 릿지. 중요 특성이 적을 것 같으면 라쏘. 라쏘는 상관 높은 특성·n>m에서 불안정 → 엘라스틱넷 (`l1_ratio`를 1에 가깝게) |
| 11 | 사진을 실외/실내, 낮/밤으로 분류 | 배타적이지 않은 클래스(4가지 조합 가능) → **로지스틱 회귀 분류기 2개** |

### 7-2. 12번: NumPy만으로 소프트맥스 회귀 + 조기 종료 구현

사이킷런 없이 앞에서 배운 수식을 그대로 코드로 옮겨 보는 문제.

**① 데이터 준비**

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"].values

X_with_bias = np.c_[np.ones(len(X)), X]     # 편향 열 추가 (x0 = 1)
```

**② 훈련 / 검증 / 테스트 = 60 / 20 / 20 직접 분할**

```python
test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)              # 30
validation_size = int(total_size * validation_ratio)  # 30
train_size = total_size - test_size - validation_size # 90

rng = np.random.default_rng(seed=42)
rnd_indices = rng.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]
```

- 섞은 인덱스를 앞 90 / 가운데 30 / 뒤 30으로 슬라이싱
- `[-test_size:]`: 뒤에서 30개

**③ 원-핫 인코딩**

```python
def to_one_hot(y):
    return np.diag(np.ones(y.max() + 1))[y]

y_train[:10]               # array([1, 1, 2, 0, 2, 2, 1, 2, 2, 0])
to_one_hot(y_train[:10])
# array([[0., 1., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.],
#        [1., 0., 0.], ...])

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)
```

| 코드 | 의미 |
|---|---|
| `np.ones(3)` | `[1, 1, 1]` |
| `np.diag(...)` | 대각선이 1인 3×3 **단위 행렬** |
| `단위행렬[y]` | 팬시 인덱싱 → 각 레이블 번호의 행을 꺼냄 = 원-핫 벡터 |

- 크로스 엔트로피는 **정답 확률 벡터**가 필요하므로 클래스 번호를 원-핫으로 바꿈

**④ 스케일링 (편향 열 제외)**

```python
mean = X_train[:, 1:].mean(axis=0)
std = X_train[:, 1:].std(axis=0)
X_train[:, 1:] = (X_train[:, 1:] - mean) / std
X_valid[:, 1:] = (X_valid[:, 1:] - mean) / std
X_test[:, 1:] = (X_test[:, 1:] - mean) / std
```

- `[:, 1:]`: 0번 열(편향 = 1)은 빼고 스케일링
- 평균·표준편차는 **훈련 세트 값**으로 세 세트 모두 변환 (`StandardScaler`를 직접 구현한 것)

**⑤ 소프트맥스 함수**

```python
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = exps.sum(axis=1, keepdims=True)
    return exps / exp_sums

n_inputs = X_train.shape[1]          # 3 (특성 2 + 편향)
n_outputs = len(np.unique(y_train))  # 3 (클래스 3개)
```

- `axis=1`: 각 샘플(행)별로 클래스 방향 합
- `keepdims=True`: 결과를 `(m,)`이 아니라 `(m, 1)`로 유지 → 나눗셈이 **브로드캐스팅**되어 행마다 나뉨

**⑥ 배치 경사 하강법으로 훈련**

크로스 엔트로피 그레이디언트:

```
∇θ⁽ᵏ⁾ J(Θ) = (1/m) Σ (p̂ₖ⁽ⁱ⁾ − yₖ⁽ⁱ⁾) x⁽ⁱ⁾       → 행렬로: (1/m) Xᵀ (P̂ − Y)
```

```python
eta = 0.5
n_epochs = 5001
m = len(X_train)
epsilon = 1e-5

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))   # (3, 3): 특성 × 클래스

for epoch in range(n_epochs):
    logits = X_train @ Theta                         # (90, 3) 클래스별 점수
    Y_proba = softmax(logits)                        # (90, 3) 클래스별 확률
    if epoch % 1000 == 0:
        Y_proba_valid = softmax(X_valid @ Theta)
        xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
        print(epoch, xentropy_losses.sum(axis=1).mean())
    error = Y_proba - Y_train_one_hot                # 확률 오차
    gradients = 1 / m * X_train.T @ error            # (3, 3)
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

| 줄 | 의미 |
|---|---|
| `Theta` `(3, 3)` | 열 하나가 클래스 하나의 파라미터 벡터 θ⁽ᵏ⁾ |
| `+ epsilon` | 확률이 0이면 log(0) = −∞ → `nan` 방지 |
| `Y_one_hot * log(...)` | 원-핫이라 **정답 클래스 항만 남음** |
| `.sum(axis=1).mean()` | 샘플별 합 → 전체 평균 = 크로스 엔트로피 |

- 3000 에포크 이후 검증 손실이 조금씩 **다시 오름** → 약한 과대적합 신호

```python
logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)      # 확률 최대 클래스
accuracy_score = (y_predict == y_valid).mean()
accuracy_score
# np.float64(0.9333333333333333)
```

**⑦ ℓ₂ 규제 추가**

```python
alpha = 0.01   # 규제 하이퍼파라미터
...
for epoch in range(n_epochs):
    ...
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()                  # 편향 행 제외
    total_loss = xentropy_losses.sum(axis=1).mean() + alpha * l2_loss
    ...
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]   # 규제 그레이디언트
    Theta = Theta - eta * gradients
```

```
0 3.0065
1000 0.2711
...
5000 0.2711
```

- `Theta[1:]`: 0번 행(편향)은 규제하지 않음
- `np.r_[a, b]`: 행 방향으로 붙이기 → 편향 자리엔 0, 나머지는 `α·θ` (½θ²의 미분 = θ)
- 손실값은 규제 항이 더해져 커 보이지만, **검증 정확도는 0.933 → 0.967로 향상**

**⑧ 조기 종료 추가**

```python
eta = 0.5
n_epochs = 50_001
C = 100        # 규제 (α = 1/C)
best_loss = np.inf

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    Y_proba_valid = softmax(X_valid @ Theta)
    xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()
    total_loss = xentropy_losses.sum(axis=1).mean() + 1 / C * l2_loss
    if epoch % 1000 == 0:
        print(epoch, total_loss.round(4))
    if total_loss < best_loss:
        best_loss = total_loss
    else:                                   # 검증 손실이 더 이상 줄지 않으면
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

- `50_001`: 숫자 사이 `_`는 가독성용 (50001과 같음)
- **매 에포크** 검증 손실을 계산해야 조기 종료 판단 가능
- 5000 에포크 대신 **403 에포크**에서 멈춤. 정확도는 그대로 0.967 → 훈련 시간만 단축

> 💡 여기서는 손실이 한 번만 올라가도 바로 멈춘다. 배치 GD라 곡선이 매끄러워서 가능한 방식. SGD라면 위 6번 답처럼 "인내(patience)" 기간을 둬야 한다.

**⑨ 최종 평가**

```python
# 전체 격자에 대한 예측 그림 — 새 데이터도 같은 mean·std로 스케일링 + 편향 추가
X_new = (X_new - mean) / std
X_new_with_bias = np.c_[np.ones(len(X_new)), X_new]

# 테스트 세트
logits = X_test @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)
accuracy_score = (y_predict == y_test).mean()
accuracy_score
# np.float64(0.9666666666666667)
```

- 테스트 정확도 **96.7%** (검증과 동일). 데이터가 작아 시드에 따라 결과가 달라질 수 있음

---

## 8. 전체 흐름 요약

```
y = 4 + 3x + 잡음 (선형 데이터)
   ↓
정규 방정식 (XᵀX)⁻¹Xᵀy / lstsq / pinv / LinearRegression   ← 4.1 닫힌 형태 해
   ↓ 특성이 많거나 데이터가 크면
배치 GD → SGD(+학습 스케줄) → 미니배치 GD / SGDRegressor     ← 4.2 반복 최적화
   ↓ 데이터가 곡선이면
PolynomialFeatures + LinearRegression                      ← 4.3 다항 회귀
   ↓ 차수를 어떻게 정하나?
learning_curve → 과소적합(둘 다 높고 붙음) / 과대적합(간격)   ← 4.4 편향-분산
   ↓ 과대적합을 줄이려면
Ridge(ℓ₂) / Lasso(ℓ₁, 특성 선택) / ElasticNet / 조기 종료   ← 4.5 규제
   ↓ 분류에 쓰려면
LogisticRegression (시그모이드, 로그 손실)                    ← 4.6 이진 분류
LogisticRegression 다중 클래스 = 소프트맥스 (크로스 엔트로피)  ← 4.6 다중 분류
```

### 모델·비용 함수 정리

| 모델 | 출력 | 비용 함수 | 닫힌 형태 해 | 사이킷런 |
|---|---|---|---|---|
| 선형 회귀 | θᵀx | MSE | O | `LinearRegression`, `SGDRegressor` |
| 다항 회귀 | θᵀ·poly(x) | MSE | O | `PolynomialFeatures` + `LinearRegression` |
| 릿지 | θᵀx | MSE + α/m·Σθ² | O | `Ridge`, `SGDRegressor(penalty="l2")` |
| 라쏘 | θᵀx | MSE + 2α·Σ\|θ\| | X | `Lasso`, `SGDRegressor(penalty="l1")` |
| 엘라스틱넷 | θᵀx | MSE + r·ℓ₁ + (1−r)·ℓ₂ | X | `ElasticNet(l1_ratio=r)` |
| 로지스틱 회귀 | σ(θᵀx) | 로그 손실 | X (볼록) | `LogisticRegression` |
| 소프트맥스 회귀 | softmax(Θᵀx) | 크로스 엔트로피 | X (볼록) | `LogisticRegression` (클래스 ≥ 3) |

### 주요 결과

| 실험 | 결과 |
|---|---|
| 정규 방정식 / `LinearRegression` / 배치 GD | θ₀ = 3.691, θ₁ = 3.330 (진짜 4, 3) — 세 방법 동일 |
| SGD (50 에포크, 직접 구현) | θ₀ = 3.698, θ₁ = 3.307 |
| `SGDRegressor` | θ₀ = 3.689, θ₁ = 3.331 |
| 2차 다항 회귀 | 0.505x² + 1.110x + 2.005 (진짜 0.5x² + x + 2) |
| 학습 곡선 RMSE (훈련/검증, 160개) | 선형 ≈ 1.62 / 1.64 (과소적합), 10차 ≈ 0.97 / 1.03 |
| 릿지 / 라쏘 / 엘라스틱넷 `predict([[1.5]])` | 1.844 / 1.876 / 1.865 |
| 조기 종료 (90차 다항 + SGD) | 184 에포크에서 최고 검증 RMSE ≈ 1.105 |
| 로지스틱 회귀 결정 경계 (꽃잎 너비) | ≈ 1.65cm |
| 소프트맥스 `predict_proba([[5, 2]])` | [0.00, 0.04, 0.96] → virginica |
| NumPy 소프트맥스 (규제 + 조기 종료) | 403 에포크 종료, 검증·테스트 정확도 0.967 |

### 하이퍼파라미터 방향 정리

| 하이퍼파라미터 | 값 ↑ 이면 | 주의 |
|---|---|---|
| 학습률 η (`eta0`) | 빨라지지만 너무 크면 발산 | — |
| 다항 차수 `degree` | 복잡도 ↑ → 과대적합 위험 | 특성 수 폭발 |
| `alpha` (Ridge·Lasso·ElasticNet) | 규제 **강해짐** | `SGDRegressor`는 α 정의가 다름 (m으로 나눠 맞춤) |
| `C` (LogisticRegression) | 규제 **약해짐** | α의 역수 |
| `l1_ratio` | 라쏘에 가까워짐 | 0 = 릿지, 1 = 라쏘 |

### 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `add_dummy_feature(X)` / `np.c_[np.ones(m), X]` | 편향 열(x₀=1) 추가 | 1-3, 5-1, 7-2 |
| `A @ B`, `A.T` | 행렬 곱, 전치 | 전체 |
| `theta - eta * gradients` | 경사 하강 스텝 | 2장 전체 |
| `X[i : i+1]` | 샘플 1개를 2차원으로 꺼내기 | 2-3 |
| `rng.permutation(m)` | 매 에포크 섞기 / 직접 분할 | 2-4, 7-2 |
| `y.ravel()` | `(m, 1)` → `(m,)` | 2-3, 5-1 |
| `np.linspace(a, b, n).reshape(-1, 1)` | 예측 곡선용 입력 | 3, 6 |
| `learning_curve(..., scoring="neg_root_mean_squared_error")` | 학습 곡선 | 4-2 |
| `partial_fit()` + `deepcopy()` | 1 에포크씩 훈련 + 최고 모델 저장 | 5-4 |
| `np.meshgrid` → `np.c_[x0.ravel(), x1.ravel()]` → `reshape` | 결정 경계·등고선 그리기 | 6-3, 6-4 |
| `np.diag(np.ones(K))[y]` | 원-핫 인코딩 | 7-2 |
| `sum(axis=1, keepdims=True)` | 행별 합을 브로드캐스팅 가능하게 | 7-2 |
