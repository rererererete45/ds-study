# 4장. 모델 훈련

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `04_training_linear_models.ipynb`
>
> `fit()` 내부에서 모델이 **어떻게 학습하는지** 보는 장. 선형 회귀를 두 방식(닫힌 형태 해 / 경사 하강법)으로 풀고 → 다항 회귀 → 학습 곡선으로 진단 → 규제 → 로지스틱·소프트맥스 회귀로 분류까지 확장. 경사 하강법은 이후 신경망 학습의 기초.
>
> **정리 방식**: 각 소절을 ① 개념 → ② 수식 → ③ 코드 → ④ 결과 해석 → ⑤ 핵심 정리 순서로 통일.

### 실습 데이터

| 데이터 | 생성식 / 출처 | 사용 절 |
|---|---|---|
| 선형 데이터 (m=200) | `y = 4 + 3x + 잡음`, x ∈ [0, 2) | 4.1, 4.2 |
| 2차 데이터 (m=200) | `y = 0.5x² + x + 2 + 잡음`, x ∈ [−3, 3) | 4.3, 4.4, 4.5.4 |
| 작은 선형 데이터 (m=20) | `y = 1 + 0.5x + 잡음/1.5`, x ∈ [0, 3) | 4.5.1 ~ 4.5.3 |
| 붓꽃(iris) | `load_iris()` — 150개, 3품종, 특성 4개 | 4.6 |

### 기호

| 기호 | 의미 | 기호 | 의미 |
|---|---|---|---|
| m | 샘플 수 | θ | 모델 파라미터 (θ₀ 편향, θ₁… 가중치) |
| n | 특성 수 | η | 학습률 |
| ŷ | 예측값 | α | 규제 강도 |
| `X_b` | 맨 앞에 x₀=1 열을 붙인 X | J(θ) | 비용 함수 |

## 목차

- [4.0 환경 설정](#40-환경-설정)
- [4.1 선형 회귀](#41-선형-회귀)
  - [4.1.1 정규 방정식](#411-정규-방정식)
  - [4.1.2 계산 복잡도](#412-계산-복잡도)
- [4.2 경사 하강법](#42-경사-하강법)
  - [4.2.1 배치 경사 하강법](#421-배치-경사-하강법)
  - [4.2.2 확률적 경사 하강법](#422-확률적-경사-하강법)
  - [4.2.3 미니배치 경사 하강법](#423-미니배치-경사-하강법)
- [4.3 다항 회귀](#43-다항-회귀)
- [4.4 학습 곡선](#44-학습-곡선)
- [4.5 규제가 있는 선형 모델](#45-규제가-있는-선형-모델)
  - [4.5.1 릿지 회귀](#451-릿지-회귀)
  - [4.5.2 라쏘 회귀](#452-라쏘-회귀)
  - [4.5.3 엘라스틱넷](#453-엘라스틱넷)
  - [4.5.4 조기 종료](#454-조기-종료)
- [4.6 로지스틱 회귀](#46-로지스틱-회귀)
  - [4.6.1 확률 추정](#461-확률-추정)
  - [4.6.2 훈련과 비용 함수](#462-훈련과-비용-함수)
  - [4.6.3 결정 경계](#463-결정-경계)
  - [4.6.4 소프트맥스 회귀](#464-소프트맥스-회귀)
- [4.7 연습문제](#47-연습문제)
- [4.8 전체 요약](#48-전체-요약)

---

## 4.0 환경 설정

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

- 2·3장과 동일. `root_mean_squared_error`(1.4+) 사용 때문에 버전 확인

---

## 4.1 선형 회귀

### 4.1 개요

**① 개념**
- 입력 특성의 **가중치 합 + 편향**으로 예측하는 모델
- 훈련 = 비용 함수(MSE)를 최소화하는 θ 찾기
- 풀이 방법 두 가지: **닫힌 형태 해**(공식으로 한 번에, 4.1.1) / **경사 하강법**(반복 최적화, 4.2)

**② 수식**

```
예측:   ŷ = θ₀ + θ₁x₁ + … + θₙxₙ = θᵀx      (x₀ = 1로 두면 편향도 같은 형태)
비용:   MSE(θ) = (1/m) Σᵢ (θᵀx⁽ⁱ⁾ − y⁽ⁱ⁾)²
```

| 항목 | 설명 |
|---|---|
| x₀ = 1 | 편향 θ₀을 가중치처럼 다루기 위한 더미 특성 → 행렬 곱 `X_b @ θ` 한 번으로 예측 |
| MSE 사용 이유 | RMSE와 최솟값 위치가 같음(√는 단조 증가) + 미분이 더 쉬움 |
| 제곱 사용 이유 | 부호 상쇄 방지, 큰 오차에 큰 벌점 |

**③ 코드 — 데이터 생성**

```python
import numpy as np

rng = np.random.default_rng(seed=42)
m = 200  # 샘플 수
X = 2 * rng.random((m, 1))                     # 열 벡터
y = 4 + 3 * X + rng.standard_normal((m, 1))    # 열 벡터
```

| 줄 | 의미 |
|---|---|
| `default_rng(seed=42)` | 난수 생성기. 시드 고정 → 재현 가능 |
| `rng.random((m, 1))` | [0, 1) 균등분포, 모양 `(200, 1)` → ×2 하면 [0, 2) |
| `rng.standard_normal((m, 1))` | 표준정규분포 잡음 |
| `(m, 1)` 모양 | 사이킷런은 X를 2차원(샘플 × 특성)으로 기대 |

**④ 결과 해석**
- 진짜 파라미터: θ₀ = 4, θ₁ = 3 → 모델이 이 값에 얼마나 가까운지로 검증

**⑤ 핵심 정리**
- 선형 회귀 = 가중치 합 모델, MSE 최소화로 훈련
- x₀ = 1 트릭으로 편향과 가중치를 하나의 벡터 θ로 처리

---

### 4.1.1 정규 방정식

**① 개념**
- MSE를 최소화하는 θ를 **공식 한 번으로** 계산 (MSE 미분 = 0을 풀어서 얻은 식)
- 사이킷런 `LinearRegression`은 같은 답을 **유사역행렬(SVD)**로 계산

**② 수식**

```
정규 방정식:   θ̂ = (XᵀX)⁻¹ Xᵀ y
유사역행렬:    θ̂ = X⁺ y        (X⁺: 무어-펜로즈 유사역행렬, SVD로 계산)
```

**③ 코드**

```python
from sklearn.preprocessing import add_dummy_feature

X_b = add_dummy_feature(X)  # 각 샘플에 x0 = 1 추가 → (200, 2)
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
theta_best
# array([[3.69084138],
#        [3.32960458]])
```

| 줄 | 의미 |
|---|---|
| `add_dummy_feature(X)` | 맨 앞에 1로 채운 열 추가 |
| `X_b.T` | 전치 |
| `@` | 행렬 곱 |
| `np.linalg.inv()` | 역행렬 |

예측:

```python
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new)
y_predict = X_new_b @ theta_best
y_predict
# array([[ 3.69084138],
#        [10.35005055]])

plt.figure(figsize=(6, 4))
plt.plot(X_new, y_predict, "r-", label="Predictions")   # 예측 직선
plt.plot(X, y, "b.")                                     # 데이터
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")
plt.show()
```

- 직선은 두 점이면 충분 → x = 0, 2만 예측 (그림 4-2)

사이킷런:

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

- 편향(`intercept_`)과 가중치(`coef_`)를 따로 저장 → `add_dummy_feature` 불필요
- `_`로 끝나는 속성 = 학습으로 얻은 값

내부 구현 확인:

```python
theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
theta_best_svd
# array([[3.69084138],
#        [3.32960458]])

np.linalg.pinv(X_b) @ y     # 유사역행렬 직접 계산 → 동일
```

- `LinearRegression`은 `scipy.linalg.lstsq()`(least squares) 기반

**④ 결과 해석**

| 방법 | θ₀ | θ₁ |
|---|---|---|
| 진짜 값 | 4 | 3 |
| 정규 방정식 | 3.691 | 3.330 |
| `LinearRegression` | 3.691 | 3.330 |
| `lstsq` / `pinv` | 3.691 | 3.330 |

- 세 방법 결과 **완전히 동일**
- 진짜 값과의 차이 = **잡음** 때문 (데이터 200개로는 정확히 복원 불가)

**⑤ 핵심 정리**
- 정규 방정식 = 반복 없이 한 번에 최적해
- 유사역행렬이 더 나은 이유: `XᵀX`의 역행렬이 없을 때(m < n, 중복 특성)도 계산 가능 + 더 빠름
- 정규 방정식·SVD는 **특성 스케일링 불필요**

---

### 4.1.2 계산 복잡도

**① 개념**
- 닫힌 형태 해는 정확하지만 **특성 수 n**이 커지면 급격히 느려짐

**② 수식**

| 방법 | 특성 수 n에 대한 복잡도 | n이 2배일 때 | 샘플 수 m에 대한 복잡도 |
|---|---|---|---|
| 정규 방정식 | O(n²·⁴) ~ O(n³) | 약 5.3 ~ 8배 | O(m) |
| SVD | O(n²) | 약 4배 | O(m) |
| 예측 | O(n) | 2배 | O(m) |

**③ 코드** — 없음 (이론)

**④ 결과 해석**
- 특성 수십만 개 → 사실상 사용 불가
- 샘플 수에는 선형 → 메모리에만 들어가면 대용량도 가능
- 전체 데이터를 **한 번에 메모리에** 올려야 함

**⑤ 핵심 정리**
- 특성이 아주 많거나 데이터가 메모리에 안 들어가면 → **경사 하강법(4.2)** 필요

---

## 4.2 경사 하강법

### 4.2 개요

**① 개념**
- 비용 함수의 **기울기(그레이디언트) 반대 방향**으로 θ를 조금씩 이동시켜 최솟값을 찾는 반복 최적화
- 비유: 안개 낀 산에서 발밑의 가장 가파른 내리막 방향으로 한 걸음씩

| 절차 | 내용 |
|---|---|
| 1 | θ를 무작위로 초기화 |
| 2 | 현재 θ에서 그레이디언트 계산 |
| 3 | 그레이디언트 반대 방향으로 η만큼 이동 |
| 4 | 그레이디언트 ≈ 0 (수렴)까지 2~3 반복 |

**② 수식**

```
θ ← θ − η · ∇θ J(θ)
```

| 요소 | 내용 |
|---|---|
| 학습률 η | 너무 작음 → 느림 / 너무 큼 → 발산 |
| 지역 최솟값 | 일반적으로 위험. 단 **선형 회귀 MSE는 볼록 함수** → 전역 최솟값 하나뿐 |
| 특성 스케일 | 스케일이 다르면 비용 함수가 길쭉한 그릇 → 수렴 매우 느림 |

**③ 코드** — 4.2.1 ~ 4.2.3

**④ 결과 해석** — 4.2.1 ~ 4.2.3

**⑤ 핵심 정리**
- 경사 하강법 사용 시 **반드시 특성 스케일링** (`StandardScaler`)
- 매 스텝 그레이디언트를 몇 개의 샘플로 계산하느냐에 따라 3종류: **배치(전체) / 확률적(1개) / 미니배치(일부)**

---

### 4.2.1 배치 경사 하강법

**① 개념**
- 매 스텝 **훈련 세트 전체**로 그레이디언트 계산 → "배치"
- 정확하고 매끄럽게 수렴하지만 데이터가 크면 느림

**② 수식**

```
∇θ MSE(θ) = (2/m) Xᵀ (Xθ − y)
θ ← θ − η · ∇θ MSE(θ)
```

- 해석: "샘플별 오차 × 특성값"의 평균. 오차가 0이면 그레이디언트도 0

**③ 코드**

```python
eta = 0.1        # 학습률
n_epochs = 1000
m = len(X_b)     # 샘플 수

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))   # 무작위 초기화

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

theta
# array([[3.69084138],
#        [3.32960458]])
```

| 줄 | 의미 |
|---|---|
| `rng.standard_normal((2, 1))` | θ₀, θ₁ 무작위 시작 (0.305, −1.040) |
| `X_b @ theta - y` | 샘플별 오차 `(200, 1)` |
| `X_b.T @ (오차)` | 파라미터별 오차 기여도 합 `(2, 1)` |
| `2 / m * ...` | MSE 미분 계수 및 평균 |
| `theta - eta * gradients` | 기울기 반대 방향으로 이동 |
| 에포크 | 훈련 세트 전체를 한 번 사용하는 단위 |

학습률 비교 (그림 4-8):

```python
import matplotlib as mpl

def plot_gradient_descent(theta, eta):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_epochs = 1000
    n_shown = 20
    theta_path = []
    for epoch in range(n_epochs):
        if epoch < n_shown:                     # 처음 20스텝 직선만 그림
            y_predict = X_new_b @ theta
            color = mpl.colors.rgb2hex(plt.cm.OrRd(epoch / n_shown + 0.15))
            plt.plot(X_new, y_predict, linestyle="solid", color=color)
        gradients = 2 / m * X_b.T @ (X_b @ theta - y)
        theta = theta - eta * gradients
        theta_path.append(theta)                # 그림 4-11용 경로 저장
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

- `plt.cm.OrRd(값)`: 0~1 → 주황~빨강. 스텝이 진행될수록 진해짐
- `plt.subplot(131)`: 1행 3열 중 1번째 칸

**④ 결과 해석**

처음 3스텝 (η = 0.1, 직접 계산):

| 스텝 | 그레이디언트 | θ₀, θ₁ | MSE |
|---|---|---|---|
| 0 | — | 0.305, −1.040 | 66.38 |
| 1 | −15.40, −17.99 | 1.845, 0.759 | 22.32 |
| 2 | −8.77, −10.29 | 2.722, 1.788 | 7.96 |
| 3 | −4.98, −5.90 | 3.220, 2.378 | 3.27 |

- 그레이디언트 음수 → θ 증가 방향으로 이동
- 최솟값에 가까울수록 그레이디언트가 작아져 스텝도 자동으로 작아짐
- 1000 에포크 후 정규 방정식과 **소수점까지 동일** (3.691, 3.330)

학습률별 결과 (직접 계산):

| η | 20스텝 후 θ | 1000스텝 후 θ | 판정 |
|---|---|---|---|
| 0.02 | 3.29, 2.47 | 3.69, 3.33 | 느림 |
| 0.1 | 3.81, 3.23 | 3.69, 3.33 | 적절 |
| 0.5 | −50.3, −59.3 | −2×10⁵⁹ | **발산** |

**⑤ 핵심 정리**
- 볼록 함수 + 적절한 η → 전역 최솟값 수렴 보장
- η는 그리드 서치로 탐색, 반복 횟수는 크게 잡고 그레이디언트가 허용오차 ε 이하면 중지
- 단점: 매 스텝 전체 데이터 사용 → 대용량에서 느림

---

### 4.2.2 확률적 경사 하강법

**① 개념**
- 매 스텝 **샘플 1개를 무작위로** 골라 그레이디언트 계산 (Stochastic GD)
- 빠르지만 경로가 불규칙 → 최솟값 근처에서 계속 요동
- 해결책: **학습 스케줄** — 학습률을 처음엔 크게, 점점 작게 (담금질 기법과 유사)

| | 배치 GD | SGD |
|---|---|---|
| 스텝당 사용 샘플 | m개 | 1개 |
| 속도 | 느림 | 매우 빠름 |
| 경로 | 매끄러움 | 지그재그 |
| 최솟값 근처 | 멈춤 | 요동 (학습률 감소 필요) |
| 지역 최솟값 | 갇힐 수 있음 | 요동으로 탈출 가능 |
| 외부 메모리 학습 | 불가 | 가능 |

**② 수식**

```
샘플 i 하나의 그레이디언트:   ∇θ = 2 x⁽ⁱ⁾ᵀ (x⁽ⁱ⁾θ − y⁽ⁱ⁾)       (m으로 나누지 않음)
학습 스케줄:                 η(t) = t₀ / (t + t₁),   t = 누적 스텝 수
```

**③ 코드**

```python
theta_path_sgd = []   # 그림 4-11용

n_epochs = 50
t0, t1 = 5, 50        # 학습 스케줄 하이퍼파라미터

def learning_schedule(t):
    return t0 / (t + t1)

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = rng.integers(m)
        xi = X_b[random_index : random_index + 1]
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
| `rng.integers(m)` | 0 ~ m−1 중 무작위 인덱스 (**복원 추출** → 한 에포크에 중복·누락 가능) |
| `X_b[i : i+1]` | 슬라이싱으로 `(1, 2)` 2차원 유지 (`X_b[i]`는 1차원 `(2,)`) |
| `2 * xi.T @ ...` | 샘플 1개라 m=1 → 나누기 생략 |
| `epoch * m + iteration` | 누적 스텝 수 t |

사이킷런 `SGDRegressor`:

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
| `max_iter=1000` | 최대 에포크 |
| `tol=1e-5`, `n_iter_no_change=100` | 100 에포크 동안 개선이 1e-5 미만이면 중지 |
| `penalty=None` | 규제 없음 (4.5에서 사용) |
| `eta0=0.01` | 초기 학습률 (기본 스케줄로 감소) |
| `y.ravel()` | `(200, 1)` → `(200,)` |

**④ 결과 해석**

학습 스케줄에 따른 η 변화:

| t | 0 | 50 | 500 | 5,000 | 9,999 (마지막) |
|---|---|---|---|---|---|
| η | 0.1 | 0.05 | 0.009 | 0.001 | 0.0005 |

| 방법 | 에포크 | θ₀ | θ₁ |
|---|---|---|---|
| 배치 GD | 1000 | 3.691 | 3.330 |
| SGD (직접 구현) | **50** | 3.698 | 3.307 |
| `SGDRegressor` | ≤1000 | 3.689 | 3.331 |

- 50 에포크만으로 거의 같은 위치 도달. 다만 정확히 같지는 않음 (요동)
- 그림 4-10: 첫 에포크 20스텝 직선이 불규칙하게 움직임

**⑤ 핵심 정리**
- SGD = 1개 샘플 / 빠름 / 요동 / 학습 스케줄로 안정화
- 샘플은 **IID**여야 함 → 정렬된 데이터는 반드시 섞을 것 (`SGDRegressor`는 기본으로 매 에포크 섞음)
- 학습률을 너무 빨리 줄이면 지역 최솟값에 갇히거나 중간에 멈춤, 너무 느리게 줄이면 계속 요동

---

### 4.2.3 미니배치 경사 하강법

**① 개념**
- 매 스텝 **작은 무작위 묶음(미니배치)**으로 그레이디언트 계산 → 배치와 SGD의 중간
- 행렬 연산 최적화(특히 **GPU**)로 SGD보다 빠르고 덜 요동 → 신경망 학습의 표준

**② 수식**

```
∇θ = (2 / b) X_Bᵀ (X_B θ − y_B)      (B: 크기 b인 미니배치)
```

**③ 코드**

```python
from math import ceil

n_epochs = 50
minibatch_size = 20
n_batches_per_epoch = ceil(m / minibatch_size)   # 10

rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1))

t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

theta_path_mgd = []
for epoch in range(n_epochs):
    shuffled_indices = rng.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for iteration in range(0, n_batches_per_epoch):
        idx = iteration * minibatch_size
        xi = X_b_shuffled[idx : idx + minibatch_size]
        yi = y_shuffled[idx : idx + minibatch_size]
        gradients = 2 / minibatch_size * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * n_batches_per_epoch + iteration)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)
```

| 줄 | 의미 |
|---|---|
| `rng.permutation(m)` | 0 ~ m−1 무작위 순열 (**비복원** → 한 에포크에 모든 샘플 정확히 1번) |
| `ceil(m / size)` | 올림. 나누어떨어지지 않으면 마지막 배치가 작아짐 |
| `2 / minibatch_size` | 배치 크기로 평균 |

경로 비교 (그림 4-11):

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

- `[:, 0]`, `[:, 1]`: 경로 배열에서 θ₀, θ₁ 추출 → 파라미터 공간 위 이동 경로

**④ 결과 해석**

| 방법 | 경로 |
|---|---|
| 배치 (파랑) | 최솟값으로 직진 후 정지 |
| SGD (빨강) | 크게 요동 |
| 미니배치 (초록) | 중간 수준 요동 |

**⑤ 핵심 정리 — 선형 회귀 알고리즘 비교 (교재 표 4-1)**

| 알고리즘 | m이 클 때 | 외부 메모리 학습 | n이 클 때 | 하이퍼파라미터 수 | 스케일 조정 | 사이킷런 |
|---|---|---|---|---|---|---|
| 정규 방정식 | 빠름 | X | 느림 | 0 | 불필요 | — |
| SVD | 빠름 | X | 느림 | 0 | 불필요 | `LinearRegression` |
| 배치 GD | 느림 | X | 빠름 | 2 | 필요 | — |
| SGD | 빠름 | O | 빠름 | ≥2 | 필요 | `SGDRegressor` |
| 미니배치 GD | 빠름 | O | 빠름 | ≥2 | 필요 | — |

- 훈련 방법만 다를 뿐 **최종 모델은 동일** (예측 방식 같음)

---

## 4.3 다항 회귀

**① 개념**
- 비선형 데이터에 대해 **각 특성의 거듭제곱을 새 특성으로 추가**한 뒤 선형 모델 훈련
- θ에 대해서는 여전히 선형 → "선형 모델"로 분류

**② 수식**

```
ŷ = θ₀ + θ₁x + θ₂x²          (x² 를 새로운 특성으로 취급)

특성 n개, 차수 d → 생성 특성 수 = (n + d)! / (d! · n!)   (편향 포함)
```

**③ 코드**

데이터 생성:

```python
rng = np.random.default_rng(seed=42)
m = 200
X = 6 * rng.random((m, 1)) - 3                          # [-3, 3)
y = 0.5 * X ** 2 + X + 2 + rng.standard_normal((m, 1))  # 진짜: 0.5x² + x + 2
```

특성 변환 + 훈련:

```python
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]        # array([1.64373629])
X_poly[0]   # array([1.64373629, 2.701869  ])   ← [x, x²]

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
# (array([2.00540719]), array([[1.11022126, 0.50526985]]))
```

| 매개변수 | 의미 |
|---|---|
| `degree=2` | 2차항까지 생성 |
| `include_bias=False` | x⁰ = 1 열 생성 안 함 (`LinearRegression`이 절편 별도 학습) |

예측 곡선 (그림 4-13):

```python
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)     # 새 데이터는 transform만
y_new = lin_reg.predict(X_new_poly)

plt.plot(X, y, "b.")
plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
```

- `np.linspace(a, b, n)`: 균등 간격 n개 → 매끄러운 곡선용 입력

**④ 결과 해석**

| | 절편 | x 계수 | x² 계수 |
|---|---|---|---|
| 진짜 | 2 | 1 | 0.5 |
| 모델 | 2.005 | 1.110 | 0.505 |

- 생성식을 거의 정확히 복원

**⑤ 핵심 정리**
- 다항 회귀 = `PolynomialFeatures` + 선형 회귀
- 특성이 여러 개면 **교차항**(ab, a²b, ab² …)까지 생성 → 특성 수 폭발 (예: n=10, d=3 → 285개)
- 차수가 높을수록 과대적합 위험 → 4.4에서 진단

---

## 4.4 학습 곡선

**① 개념**
- 모델의 **과소적합 / 과대적합**을 진단하는 도구
- 훈련 세트 크기를 늘려 가며 **훈련 오차와 검증 오차**를 그래프로 비교
- 다른 진단 방법: 교차 검증 (2장) — 훈련 점수만 좋으면 과대적합, 둘 다 나쁘면 과소적합

**② 수식 — 편향/분산 분해**

```
일반화 오차 = 편향² + 분산 + 줄일 수 없는 오차
```

| 오차 | 원인 | 결과 | 해결 |
|---|---|---|---|
| 편향 | 잘못된 가정 (2차 데이터를 직선으로) | 과소적합 | 모델 복잡도 ↑ |
| 분산 | 작은 변동에 과민 (고차 다항) | 과대적합 | 복잡도 ↓, 데이터 ↑, 규제 |
| 줄일 수 없는 오차 | 데이터 잡음 | — | 데이터 정제 |

- 복잡도 ↑ → 편향 ↓, 분산 ↑ → **트레이드오프**

**③ 코드**

고차 다항 비교 (그림 4-14):

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

- 300차 특성은 값의 크기 차이가 극단적 → `StandardScaler`로 수치 안정화

선형 회귀 학습 곡선 (그림 4-15):

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
| `train_sizes=np.linspace(0.01, 1.0, 40)` | 훈련 폴드의 1% ~ 100%, 40단계 |
| `cv=5` | 5-폴드 → 훈련 폴드 최대 160개 (x축 범위) |
| `scoring="neg_root_mean_squared_error"` | 음수 RMSE ("클수록 좋음" 규칙) |
| `train_scores` | `(40, 5)` — 크기별 × 폴드별 |
| `-...mean(axis=1)` | 폴드 평균 후 부호 반전 → RMSE |

10차 다항 학습 곡선 (그림 4-16):

```python
polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    LinearRegression())

train_sizes, train_scores, valid_scores = learning_curve(
    polynomial_regression, X, y, train_sizes=np.linspace(0.01, 1.0, 40), cv=5,
    scoring="neg_root_mean_squared_error")
```

**④ 결과 해석**

그림 4-14 (차수 비교):

| 차수 | 결과 |
|---|---|
| 1 | 과소적합 — 포물선 무시 |
| 2 | 적절 — 생성식과 같은 차수 |
| 300 | 과대적합 — 훈련 샘플을 지나가려 심하게 구불거림 |

학습 곡선 (훈련 160개 기준, 직접 계산):

| 모델 | 훈련 RMSE | 검증 RMSE | 곡선 모양 | 진단 |
|---|---|---|---|---|
| 선형 회귀 | ≈ 1.62 | ≈ 1.64 | 높은 곳에서 붙어 평평 | **과소적합** |
| 10차 다항 | ≈ 0.97 | ≈ 1.03 | 훈련 오차 낮음, 두 곡선 사이 간격 | **과대적합** 경향 |

- 선형 회귀 훈련 오차: 샘플 1~2개는 완벽히 맞춰 0 → 샘플 증가에 따라 상승 후 평평
- 선형 회귀 검증 오차: 샘플이 적으면 큼(≈ 3.6) → 감소 후 평평

**⑤ 핵심 정리**

| 학습 곡선 모양 | 진단 | 대응 |
|---|---|---|
| 두 곡선이 높은 값에서 붙어 평평 | 과소적합 | 더 복잡한 모델·더 나은 특성 (데이터 추가는 **무효**) |
| 훈련 오차 낮고 두 곡선 간격 큼 | 과대적합 | 데이터 추가 (간격이 줄어들 때까지), 모델 단순화, 규제 |

---

## 4.5 규제가 있는 선형 모델

### 4.5 개요

**① 개념**
- 과대적합 감소 방법 = 모델 **규제**(자유도 제한)
- 다항 모델 → 차수 감소 / 선형 모델 → **가중치 크기 제한**
- 비용 함수에 **가중치 벌점 항** 추가 → "오차 최소화"와 "가중치 작게 유지" 사이 타협

**② 수식**

```
J(θ) = MSE(θ) + (규제 항)
```

| 규칙 | 이유 |
|---|---|
| 편향 θ₀은 규제하지 않음 | 절편을 줄이면 예측 전체가 내려감 |
| 규제 항은 **훈련 시에만** 사용 | 평가는 규제 없는 지표(RMSE 등)로 |
| **스케일링 필수** | 규제는 가중치 크기에 민감 → 특성 단위가 다르면 불공평한 벌점 |

**③ 코드 — 실습 데이터**

```python
rng = np.random.default_rng(seed=42)
m = 20
X = 3 * rng.random((m, 1))
y = 1 + 0.5 * X + rng.standard_normal((m, 1)) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)
```

- 샘플 20개 → 과대적합이 쉬운 상황을 의도적으로 구성

공통 그림 함수 (그림 4-17, 4-18에서 사용):

```python
def plot_model(model_class, polynomial, alphas, **model_kwargs):
    plt.plot(X, y, "b.", linewidth=3)
    for alpha, style in zip(alphas, ("b:", "g--", "r-")):
        if alpha > 0:
            model = model_class(alpha, **model_kwargs)
        else:
            model = LinearRegression()           # α=0 = 규제 없음
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
```

| 줄 | 의미 |
|---|---|
| `model_class` | `Ridge`, `Lasso` 등 클래스 자체를 인자로 받음 → 재사용 |
| `**model_kwargs` | 추가 키워드 인자를 모델에 그대로 전달 |
| `zip(alphas, styles)` | α와 선 스타일을 짝지어 반복 |
| `polynomial=True` | 10차 다항 + 스케일링 파이프라인 |
| `fr"$\alpha = {alpha}$"` | f-string + raw string |

**④ 결과 해석** — 4.5.1 ~ 4.5.4

**⑤ 핵심 정리** — 4.5.3 끝의 선택 기준 참고

---

### 4.5.1 릿지 회귀

**① 개념**
- **가중치 제곱합**을 비용에 추가 (ℓ₂ 규제, 티호노프 규제)
- 가중치를 0에 가깝게 **줄이지만 0으로 만들지는 않음**
- α = 0 → 선형 회귀, α 큼 → 가중치 ≈ 0 → 데이터 평균을 지나는 수평선

**② 수식**

```
J(θ) = MSE(θ) + (α / m) Σᵢ₌₁ⁿ θᵢ²

닫힌 형태 해:  θ̂ = (XᵀX + αA)⁻¹ Xᵀ y     (A: 왼쪽 위(편향 자리)만 0인 (n+1)×(n+1) 단위행렬)
```

**③ 코드**

```python
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=0.1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84414523])
```

- `solver="cholesky"`: 숄레스키 분해로 닫힌 형태 해 계산

α 비교 (그림 4-17):

```python
plt.figure(figsize=(9, 3.5))
plt.subplot(121)
plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
plt.subplot(122)
plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
plt.show()
```

SGD로 릿지:

```python
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1 / m, tol=None,
                       max_iter=1000, eta0=0.01, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict([[1.5]])
# array([1.83659707])
```

| 매개변수 | 의미 |
|---|---|
| `penalty="l2"` | ℓ₂ 규제 항 추가 = 릿지 |
| `alpha=0.1 / m` | `SGDRegressor`의 α 정의가 다름 → `Ridge(alpha=0.1)`과 맞추려면 m으로 나눔 |
| `tol=None` | 조기 중지 없이 1000 에포크 전부 |

확률적 평균 GD:

```python
ridge_reg = Ridge(alpha=0.1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
# array([1.84415109])
```

닫힌 형태 해 직접 계산:

```python
alpha = 0.1
A = np.array([[0., 0.], [0., 1.]])   # 편향 자리 0
X_b = np.c_[np.ones(m), X]           # 1 열 추가
np.linalg.inv(X_b.T @ X_b + alpha * A) @ X_b.T @ y
# array([[1.23749481],
#        [0.40443361]])

ridge_reg.intercept_, ridge_reg.coef_
# (array([1.23754037]), array([0.40440715]))
```

- `np.c_[a, b]`: 열 방향 결합

**④ 결과 해석**

| 방법 | `predict([[1.5]])` |
|---|---|
| `Ridge(solver="cholesky")` | 1.8441 |
| `Ridge(solver="sag")` | 1.8442 |
| `SGDRegressor(penalty="l2")` | 1.8366 |

- 풀이 방법이 달라도 거의 같은 결과. 닫힌 형태 해 직접 계산값과 `sag` 결과도 일치

그림 4-17:

| | 왼쪽: 선형 + 릿지 (α = 0, 10, 100) | 오른쪽: 10차 다항 + 릿지 (α = 0, 10⁻⁵, 1) |
|---|---|---|
| α = 0 | 일반 직선 | 심하게 구불거림 (과대적합) |
| α 증가 | 기울기 감소 → 수평선에 가까워짐 | 점점 매끄러운 곡선 |

10차 다항 가중치 (직접 계산):

| α | 최대 가중치 크기 | 0이 아닌 가중치 수 |
|---|---|---|
| 10⁻⁵ | ≈ 23.25 | 10 / 10 |
| 1 | ≈ 0.48 | 10 / 10 |

**⑤ 핵심 정리**
- 릿지 = ℓ₂ 규제, 가중치를 **줄이되 0으로 만들지 않음**
- α ↑ → 분산 ↓, 편향 ↑
- 닫힌 형태 해(`cholesky`)와 경사 하강법(`sag`, `SGDRegressor`) 모두 사용 가능
- 규제 없는 선형 회귀보다 대부분 좋음 → **기본 선택**

---

### 4.5.2 라쏘 회귀

**① 개념**
- **가중치 절댓값 합**을 비용에 추가 (ℓ₁ 규제, Least Absolute Shrinkage and Selection Operator)
- 덜 중요한 특성의 가중치를 **정확히 0**으로 만듦 → **자동 특성 선택**, **희소 모델**

**② 수식**

```
J(θ) = MSE(θ) + 2α Σᵢ₌₁ⁿ |θᵢ|

서브그레이디언트:  g(θ, J) = ∇θ MSE(θ) + 2α · sign(θ)
                  sign(θᵢ) = −1 (θᵢ < 0), 0 (θᵢ = 0), +1 (θᵢ > 0)
```

- |θ|는 θ = 0에서 미분 불가 → **서브그레이디언트**로 경사 하강법 사용

ℓ₁ vs ℓ₂ 그레이디언트 비교:

| | ℓ₁ (라쏘) | ℓ₂ (릿지) |
|---|---|---|
| 규제 항 그레이디언트 | `sign(θ)` — 크기 무관 **일정** | `θ` — 0에 가까울수록 **약해짐** |
| 이동 경로 | 한 축이 먼저 0에 도달 후 축을 따라 이동 | 원점을 향해 직진하며 감속 |
| 최솟값 | 축 위 (일부 θ = 0) | 0 근처, 0은 아님 |

**③ 코드**

```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict([[1.5]])
# array([1.87550211])
```

α 비교 (그림 4-18):

```python
plt.subplot(121)
plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
plt.subplot(122)
plot_model(Lasso, polynomial=True, alphas=(0, 1e-2, 1), random_state=42)
```

ℓ₁ / ℓ₂ 경로 시각화 (그림 4-19) 핵심 함수:

```python
def bgd_path(theta, X, y, l1, l2, core=1, eta=0.05, n_iterations=200):
    path = [theta]
    for iteration in range(n_iterations):
        gradients = (core * 2 / len(X) * X.T @ (X @ theta - y)   # MSE 그레이디언트
                     + l1 * np.sign(theta)                        # ℓ₁ 서브그레이디언트
                     + l2 * theta)                                # ℓ₂ 그레이디언트
        theta = theta - eta * gradients
        path.append(theta)
    return np.array(path)
```

| 줄 | 의미 |
|---|---|
| `np.sign(theta)` | 부호만 반환 → ℓ₁ 그레이디언트는 크기 일정 |
| `l2 * theta` | θ에 비례 → ℓ₂ 그레이디언트는 θ가 작아지면 약해짐 |
| `core=0` | MSE 항 제거 → 규제 항만의 경로 (그림의 노란 점선) |

- 나머지 그림 코드: `np.meshgrid`로 θ₁·θ₂ 격자 생성 → `contourf`로 비용 등고선

**④ 결과 해석**

- 그림 4-18 오른쪽 α = 0.01: 10차 다항인데 거의 2차 곡선 형태 → 고차항 가중치 = 0

10차 다항 가중치 (직접 계산):

| α | 0이 아닌 가중치 수 |
|---|---|
| 0.01 | **2** / 10 |
| 1 | **0** / 10 (수평선) |

- 그림 4-19: ℓ₁ 경로는 θ₂ = 0 축에 먼저 닿은 뒤 축을 따라 이동, ℓ₂ 경로는 원점 방향으로 매끄럽게 감속

**⑤ 핵심 정리**
- 라쏘 = ℓ₁ 규제, 가중치를 **정확히 0**으로 → 특성 선택 효과
- 최솟값 근처에서 요동 → SGD 사용 시 **학습률 점진적 감소** 필요
- 특성 수 > 샘플 수, 또는 특성 간 강한 상관이 있으면 불안정

---

### 4.5.3 엘라스틱넷

**① 개념**
- 릿지(ℓ₂)와 라쏘(ℓ₁)를 **혼합 비율 r**로 결합
- 라쏘의 특성 선택 + 릿지의 안정성

**② 수식**

```
J(θ) = MSE(θ) + r · (2α Σ|θᵢ|) + (1 − r) · (α/m Σθᵢ²)

r = 0 → 릿지,   r = 1 → 라쏘
```

**③ 코드**

```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)   # l1_ratio = r
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
# array([1.8645014])
```

**④ 결과 해석**

| 모델 (α = 0.1) | `predict([[1.5]])` |
|---|---|
| 릿지 | 1.844 |
| 엘라스틱넷 (r = 0.5) | 1.865 |
| 라쏘 | 1.876 |

- 엘라스틱넷 결과가 릿지와 라쏘 사이

**⑤ 핵심 정리 — 규제 모델 선택 기준**

| 상황 | 선택 |
|---|---|
| 기본 | 규제 없는 선형 회귀 대신 **릿지** |
| 쓸모 있는 특성이 일부뿐일 것으로 예상 | **라쏘** 또는 **엘라스틱넷** |
| 특성 수 > 샘플 수 / 특성 간 강한 상관 | **엘라스틱넷** (라쏘가 불안정) |
| 라쏘처럼 쓰되 안정적으로 | 엘라스틱넷 + `l1_ratio` ≈ 1 |

---

### 4.5.4 조기 종료

**① 개념**
- 반복 학습에서 **검증 오차가 최소일 때 훈련을 중지**하는 규제 방법
- 훈련이 진행되면 훈련 오차는 계속 감소, 검증 오차는 감소 후 다시 증가(과대적합 시작) → 그 꺾이는 지점에서 중지
- 힌턴: "아름다운 공짜 점심 (beautiful free lunch)"

**② 수식**

```
best_epoch = argmin_epoch  RMSE_valid(epoch)
```

**③ 코드**

```python
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler

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
| `X[: m // 2]` / `X[m // 2 :]` | 앞 100개 훈련 / 뒤 100개 검증 |
| `y[: m // 2, 0]` | `(100, 1)` → `(100,)` |
| `degree=90` | 과대적합하기 쉬운 고차 모델을 의도적으로 사용 |
| `fit_transform` / `transform` | 스케일러는 훈련 세트로만 학습 (데이터 누설 방지) |
| `float('inf')` | 초기 최고 기록 = 무한대 |
| `partial_fit()` | **1 에포크만** 훈련하고 상태 유지 (`fit()`은 매번 초기화) |
| `deepcopy()` | 모델 객체 전체 복사. 단순 대입은 같은 객체를 참조해 이후 훈련에 같이 바뀜 |

그림 4-20:

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

- `np.argmin`: 최솟값 인덱스
- `plt.annotate(텍스트, xy=화살표 끝, xytext=텍스트 위치)`: 화살표 주석

**④ 결과 해석** (직접 계산)

| 시점 | 검증 RMSE |
|---|---|
| **184 에포크 (최적)** | **≈ 1.105** |
| 500 에포크 | ≈ 1.318 |

- 184 에포크 이후 훈련 오차는 계속 감소, 검증 오차는 증가 → 과대적합 시작
- `best_model`에 184 에포크 시점 모델이 저장됨

**⑤ 핵심 정리**
- 조기 종료 = 검증 오차 최소 시점의 모델 사용
- SGD·미니배치는 곡선이 울퉁불퉁 → 즉시 중지하지 말고 **최고 모델 저장 + 일정 기간 개선 없으면 중지 + 저장 모델로 복귀**
- 구현 요소: `partial_fit()`(이어서 훈련) + `deepcopy()`(최고 모델 보존)

---

## 4.6 로지스틱 회귀

### 4.6.1 확률 추정

**① 개념**
- 회귀 알고리즘을 **분류**에 사용. 샘플이 양성 클래스에 속할 **확률** 추정
- 선형 회귀처럼 가중치 합을 계산 → **시그모이드 함수**로 0~1 확률로 변환 → 0.5 기준으로 분류
- 이진 분류기

**② 수식**

```
확률:       p̂ = σ(θᵀx)
시그모이드:  σ(t) = 1 / (1 + e⁻ᵗ)
예측:       ŷ = 1 (p̂ ≥ 0.5),  0 (p̂ < 0.5)
           ⇔ ŷ = 1 (θᵀx ≥ 0),  0 (θᵀx < 0)

로짓:       t = θᵀx = log(p / (1 − p))      (로그 오즈, 시그모이드의 역함수)
```

**③ 코드** (그림 4-21)

```python
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

**④ 결과 해석**

| t | −4 | −2 | 0 | 2 | 4 |
|---|---|---|---|---|---|
| σ(t) | 0.018 | 0.119 | **0.5** | 0.881 | 0.982 |

- S자 곡선, 출력 범위 (0, 1), t = 0에서 정확히 0.5

**⑤ 핵심 정리**
- 로지스틱 회귀 = 선형 결합 → 시그모이드 → 확률
- θᵀx의 부호만으로 클래스 결정 가능
- 로짓 = 시그모이드 입력값 (신경망에서도 사용하는 용어)

---

### 4.6.2 훈련과 비용 함수

**① 개념**
- 목표: 양성 샘플에는 높은 확률, 음성 샘플에는 낮은 확률
- **확신을 갖고 틀릴수록 큰 벌점**을 주는 로그 손실 사용

**② 수식**

```
샘플 1개 비용:   c(θ) = −log(p̂)        (y = 1)
                       −log(1 − p̂)    (y = 0)

로그 손실:       J(θ) = −(1/m) Σᵢ [ y⁽ⁱ⁾ log(p̂⁽ⁱ⁾) + (1 − y⁽ⁱ⁾) log(1 − p̂⁽ⁱ⁾) ]

편도함수:        ∂J/∂θⱼ = (1/m) Σᵢ (σ(θᵀx⁽ⁱ⁾) − y⁽ⁱ⁾) xⱼ⁽ⁱ⁾
```

| 특성 | 내용 |
|---|---|
| 닫힌 형태 해 | **없음** |
| 볼록성 | **볼록** → 경사 하강법으로 전역 최솟값 보장 |
| 그레이디언트 형태 | 선형 회귀와 동일한 구조: (예측 − 정답) × 특성값의 평균 |

**③ 코드** — `LogisticRegression.fit()`이 내부적으로 수행 (4.6.3)

**④ 결과 해석**
- y = 1인데 p̂ → 0 이면 −log(p̂) → ∞, p̂ → 1 이면 0

**⑤ 핵심 정리**
- 로지스틱 회귀 비용 = 로그 손실, 볼록, 경사 하강법으로 최적화
- `LogisticRegression`은 기본으로 **ℓ₂ 규제** 포함, 강도는 **C = 1/α** (C ↑ → 규제 ↓)

---

### 4.6.3 결정 경계

**① 개념**
- **결정 경계**: 두 클래스 확률이 50%로 같은 지점 (θᵀx = 0)
- 붓꽃 데이터로 "virginica인가?" 이진 분류

**② 수식**

```
특성 1개:  θ₀ + θ₁x₁ = 0            →  x₁ = −θ₀ / θ₁           (점)
특성 2개:  θ₀ + θ₁x₁ + θ₂x₂ = 0      →  x₂ = −(θ₁x₁ + θ₀) / θ₂  (직선)
```

**③ 코드**

데이터 확인:

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

iris.target.head(3)    # 0 0 0  ← 클래스 순 정렬 (섞이지 않음)
iris.target_names      # array(['setosa', 'versicolor', 'virginica'])
```

| 항목 | 내용 |
|---|---|
| 샘플 | 150 (품종별 50) |
| 특성 | 꽃받침 길이·너비, 꽃잎 길이·너비 (cm) |
| 타깃 | 0 setosa / 1 versicolor / 2 virginica |

특성 1개 (꽃잎 너비):

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
| `iris.data[["..."]]` | 대괄호 2개 → DataFrame(2차원) 유지 |
| `.values` | NumPy 배열 `(150, 1)` |
| `iris.target_names[iris.target]` | 0/1/2 → 품종 이름 (팬시 인덱싱) |
| `== 'virginica'` | 이진 타깃 (True/False) |
| `train_test_split` | 기본 75/25, 섞어서 분할 (정렬된 데이터라 필수) |

확률 곡선 (그림 4-23):

```python
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0, 0]

plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris virginica proba")
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris virginica proba")
plt.plot([decision_boundary, decision_boundary], [0, 1], "k:", linewidth=2,
         label="Decision boundary")
plt.plot(X_train[y_train == 0], y_train[y_train == 0], "bs")
plt.plot(X_train[y_train == 1], y_train[y_train == 1], "g^")

decision_boundary
# np.float64(1.6516516516516517)

log_reg.predict([[1.7], [1.5]])
# array([ True, False])
```

| 줄 | 의미 |
|---|---|
| `reshape(-1, 1)` | `-1` = 자동 계산 → `(1000, 1)` |
| `predict_proba()` | `(1000, 2)` — [음성 확률, 양성 확률] |
| `y_proba[:, 1] >= 0.5` | 양성 확률 50% 이상 마스크 |
| `X_new[마스크][0, 0]` | 처음 50%를 넘는 x = 결정 경계 |

특성 2개 (꽃잎 길이·너비, 그림 4-24):

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris.target_names[iris.target] == 'virginica'
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log_reg = LogisticRegression(C=2, random_state=42)
log_reg.fit(X_train, y_train)

# 등고선용 격자
x0, x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1),
                     np.linspace(0.8, 2.7, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict_proba(X_new)
zz = y_proba[:, 1].reshape(x0.shape)

# 결정 경계 직선
left_right = np.array([2.9, 7])
boundary = -((log_reg.coef_[0, 0] * left_right + log_reg.intercept_[0])
             / log_reg.coef_[0, 1])

contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
plt.clabel(contour, inline=1)
plt.plot(left_right, boundary, "k--", linewidth=3)
```

격자 그리기 3단계 (이후 장에서도 반복 사용):

| 단계 | 코드 | 결과 |
|---|---|---|
| 1. 격자 생성 | `np.meshgrid(a, b)` | 좌표 행렬 2개 `(200, 500)` |
| 2. 샘플 목록화 | `np.c_[x0.ravel(), x1.ravel()]` | `(100000, 2)` → 모델 입력 |
| 3. 격자로 복원 | `y_proba[:, 1].reshape(x0.shape)` | `(200, 500)` → 등고선 |

**④ 결과 해석**

꽃잎 너비별 virginica 확률 (직접 계산):

| 꽃잎 너비 | 1.0 | 1.5 | **1.65** | 1.7 | 2.0 |
|---|---|---|---|---|---|
| 확률 | 0.076 | 0.360 | **≈ 0.50** | 0.547 | 0.792 |

- 결정 경계 ≈ **1.65cm**. 1.7 → True, 1.5 → False
- 2cm 이상 확실한 virginica, 1cm 이하 확실히 아님, 그 사이는 두 클래스가 겹치는 구간
- 특성 2개: 결정 경계 = **직선**, 평행선들은 15% ~ 90% 확률 등고선

**⑤ 핵심 정리**
- 결정 경계 = 확률 50% = θᵀx = 0
- 로지스틱 회귀의 결정 경계는 **선형**
- `C`: 규제 강도의 역수 (`Ridge`의 α와 반대 방향)

---

### 4.6.4 소프트맥스 회귀

**① 개념**
- 로지스틱 회귀를 **다중 클래스**로 일반화 (다항 로지스틱 회귀)
- 이진 분류기 여러 개(OvR)를 묶지 않고 **하나의 모델**로 직접 처리
- 클래스별 점수 → 소프트맥스로 확률 변환 → 최대 확률 클래스 선택
- 다중 클래스(상호 배타적)용. 다중 출력(한 샘플이 여러 클래스)에는 사용 불가

**② 수식**

```
클래스별 점수:   sₖ(x) = (θ⁽ᵏ⁾)ᵀ x                (클래스마다 별도 파라미터 벡터)
소프트맥스:      p̂ₖ = exp(sₖ(x)) / Σⱼ exp(sⱼ(x))
예측:           ŷ = argmaxₖ p̂ₖ

크로스 엔트로피: J(Θ) = −(1/m) Σᵢ Σₖ yₖ⁽ⁱ⁾ log(p̂ₖ⁽ⁱ⁾)
그레이디언트:    ∇θ⁽ᵏ⁾ J(Θ) = (1/m) Σᵢ (p̂ₖ⁽ⁱ⁾ − yₖ⁽ⁱ⁾) x⁽ⁱ⁾
```

| 항목 | 내용 |
|---|---|
| `exp` 사용 이유 | 모든 점수를 양수로 + 큰 점수를 더 크게 벌림 |
| 합으로 나누는 이유 | 확률 합 = 1 정규화 |
| 클래스 2개일 때 | 크로스 엔트로피 = 로지스틱 회귀의 로그 손실 |

**③ 코드**

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"]                      # 0, 1, 2 → 3 클래스
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

softmax_reg = LogisticRegression(C=30, random_state=42)
softmax_reg.fit(X_train, y_train)

softmax_reg.predict([[5, 2]])
# array([2])

softmax_reg.predict_proba([[5, 2]]).round(2)
# array([[0.  , 0.04, 0.96]])
```

- 타깃 클래스가 3개 이상이면 `LogisticRegression`이 **자동으로 소프트맥스 회귀**로 동작
- `C=30`: 약한 규제

결정 경계 (그림 4-25):

```python
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])

x0, x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1),
                     np.linspace(0, 3.5, 200).reshape(-1, 1))
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

zz1 = y_proba[:, 1].reshape(x0.shape)    # versicolor 확률 → 등고선
zz = y_predict.reshape(x0.shape)         # 예측 클래스 → 영역 색칠

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap="hot")
plt.clabel(contour, inline=1)
```

- `contourf`: 영역 채우기 / `contour`: 등고선만

**④ 결과 해석**

꽃잎 길이 5cm, 너비 2cm의 계산 과정 (직접 계산):

| 단계 | setosa | versicolor | virginica |
|---|---|---|---|
| 점수 sₖ | −10.39 | 3.61 | 6.77 |
| exp(sₖ) | ≈ 0.00003 | ≈ 37 | ≈ 871 |
| ÷ 합(≈ 908) = p̂ₖ | **0.00** | **0.04** | **0.96** |

- 예측: virginica (클래스 2)
- 그림 4-25: 클래스 간 결정 경계가 모두 **직선**, 세 경계가 만나는 점에서 세 확률이 모두 33%

**⑤ 핵심 정리**
- 소프트맥스 회귀 = 클래스별 선형 점수 → 소프트맥스 → argmax
- 비용 = 크로스 엔트로피, 그레이디언트는 로지스틱 회귀와 같은 형태
- 사이킷런에서는 `LogisticRegression`에 다중 클래스 타깃을 넣으면 됨

---

## 4.7 연습문제

### 4.7.1 개념 문제 (1 ~ 11)

| # | 질문 | 답 |
|---|---|---|
| 1 | 특성이 수백만 개일 때 사용할 알고리즘 | SGD, 미니배치 GD (메모리에 들어가면 배치 GD). 정규 방정식·SVD는 n에 대해 너무 느림 |
| 2 | 특성 스케일이 크게 다를 때 문제와 해결 | 비용 함수가 길쭉해져 GD 수렴 지연 → **스케일링**. 규제 모델은 작은 값 특성을 무시하게 됨. 정규 방정식·SVD는 영향 없음 |
| 3 | 로지스틱 회귀에서 GD가 지역 최솟값에 갇히는가 | 아니오. 비용 함수가 **볼록** |
| 4 | GD 알고리즘들이 같은 모델에 도달하는가 | 볼록 + 적절한 η면 비슷. 학습률을 줄이지 않으면 SGD·미니배치는 최적점 주변 요동으로 약간씩 다름 |
| 5 | 배치 GD에서 검증 오차가 계속 증가 | 훈련 오차도 증가 → η 과대로 **발산** → η 감소. 훈련 오차는 유지 → **과대적합** → 훈련 중지 |
| 6 | 미니배치 GD에서 검증 오차 증가 즉시 중지? | 아니오. 무작위성으로 일시 증가 가능. 주기적 저장 → 오래 개선 없으면 최고 모델로 복귀 |
| 7 | 최적점 근처 도달이 가장 빠른 것 / 실제 수렴하는 것 | 가장 빠름: SGD. 실제 수렴: 배치 GD (나머지는 학습률 감소 필요) |
| 8 | 다항 회귀에서 검증 오차 ≫ 훈련 오차 | 과대적합 → ① 차수 감소 ② 규제(ℓ₂, ℓ₁) ③ 훈련 데이터 증가 |
| 9 | 릿지에서 훈련·검증 오차가 비슷하게 높음 | 과소적합(높은 편향) → **α 감소** |
| 10 | 릿지 vs 선형 / 라쏘 vs 릿지 / 엘라스틱넷 vs 라쏘 | 약간의 규제가 대체로 나음 → 릿지. 소수 특성만 중요하면 라쏘. 라쏘가 불안정한 경우(n > m, 강한 상관) 엘라스틱넷 (`l1_ratio` ≈ 1) |
| 11 | 실외/실내 + 낮/밤 분류 | 상호 배타적이지 않음 → 소프트맥스 1개가 아니라 **로지스틱 회귀 2개** |

### 4.7.2 문제 12 — NumPy로 소프트맥스 회귀 + 조기 종료 구현

**① 개념**
- 사이킷런 없이 4.6.4의 수식을 직접 코드로 구현
- 단계: 데이터 분할 → 원-핫 인코딩 → 스케일링 → 소프트맥스 → 배치 GD → ℓ₂ 규제 → 조기 종료 → 평가

**② 수식**

```
소프트맥스:       p̂ₖ = exp(sₖ) / Σⱼ exp(sⱼ)
크로스 엔트로피:   J = −(1/m) Σᵢ Σₖ yₖ⁽ⁱ⁾ log(p̂ₖ⁽ⁱ⁾ + ε)
그레이디언트:      ∇Θ J = (1/m) Xᵀ (P̂ − Y)
ℓ₂ 규제 추가:     J += α · ½ Σ Θ[1:]²,    ∇ += α · Θ[1:]  (편향 행 제외)
```

**③ 코드**

데이터 준비 + 분할 (60 / 20 / 20):

```python
X = iris.data[["petal length (cm)", "petal width (cm)"]].values
y = iris["target"].values

X_with_bias = np.c_[np.ones(len(X)), X]     # 편향 열 추가

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

원-핫 인코딩:

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
| `np.diag(np.ones(3))` | 3×3 단위 행렬 |
| `단위행렬[y]` | 팬시 인덱싱 → 레이블 번호의 행 = 원-핫 벡터 |

스케일링 (편향 열 제외):

```python
mean = X_train[:, 1:].mean(axis=0)
std = X_train[:, 1:].std(axis=0)
X_train[:, 1:] = (X_train[:, 1:] - mean) / std
X_valid[:, 1:] = (X_valid[:, 1:] - mean) / std
X_test[:, 1:] = (X_test[:, 1:] - mean) / std
```

- 평균·표준편차는 **훈련 세트 기준**으로 세 세트에 적용 (`StandardScaler` 직접 구현)

소프트맥스 함수:

```python
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = exps.sum(axis=1, keepdims=True)
    return exps / exp_sums

n_inputs = X_train.shape[1]          # 3 (특성 2 + 편향)
n_outputs = len(np.unique(y_train))  # 3 (클래스 3)
```

- `keepdims=True`: 합을 `(m, 1)`로 유지 → 행별 나눗셈 브로드캐스팅

(a) 기본 배치 GD:

```python
eta = 0.5
n_epochs = 5001
m = len(X_train)
epsilon = 1e-5

rng = np.random.default_rng(seed=42)
Theta = rng.standard_normal((n_inputs, n_outputs))   # (3, 3)

for epoch in range(n_epochs):
    logits = X_train @ Theta
    Y_proba = softmax(logits)
    if epoch % 1000 == 0:
        Y_proba_valid = softmax(X_valid @ Theta)
        xentropy_losses = -(Y_valid_one_hot * np.log(Y_proba_valid + epsilon))
        print(epoch, xentropy_losses.sum(axis=1).mean())
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    Theta = Theta - eta * gradients
```

| 줄 | 의미 |
|---|---|
| `Theta` `(3, 3)` | 행 = 특성, 열 = 클래스 (열 하나 = θ⁽ᵏ⁾) |
| `+ epsilon` | log(0) = −∞ 방지 |
| `Y_one_hot * log(...)` | 정답 클래스 항만 남김 |
| `.sum(axis=1).mean()` | 샘플별 합 → 평균 = 크로스 엔트로피 |

정확도:

```python
logits = X_valid @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)
accuracy_score = (y_predict == y_valid).mean()
```

(b) ℓ₂ 규제 추가:

```python
alpha = 0.01
# ... 루프 내부
    l2_loss = 1 / 2 * (Theta[1:] ** 2).sum()
    total_loss = xentropy_losses.sum(axis=1).mean() + alpha * l2_loss
    ...
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients
```

- `Theta[1:]`: 편향 행 제외
- `np.r_[a, b]`: 행 방향 결합 → 편향 자리 0, 나머지 α·θ

(c) 조기 종료 추가:

```python
eta = 0.5
n_epochs = 50_001
C = 100        # α = 1/C
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
    else:
        print(epoch - 1, best_loss.round(4))
        print(epoch, total_loss.round(4), "early stopping!")
        break
    error = Y_proba - Y_train_one_hot
    gradients = 1 / m * X_train.T @ error
    gradients += np.r_[np.zeros([1, n_outputs]), 1 / C * Theta[1:]]
    Theta = Theta - eta * gradients
```

- 매 에포크 검증 손실 계산 → 증가하면 `break`

(d) 최종 평가:

```python
# 새 데이터 예측 시 동일 전처리: 스케일링 → 편향 열
X_new = (X_new - mean) / std
X_new_with_bias = np.c_[np.ones(len(X_new)), X_new]

logits = X_test @ Theta
Y_proba = softmax(logits)
y_predict = Y_proba.argmax(axis=1)
accuracy_score = (y_predict == y_test).mean()
```

**④ 결과 해석**

| 단계 | 검증 손실 (에포크 0 → 1000 → … → 5000) | 종료 에포크 | 검증 정확도 | 테스트 정확도 |
|---|---|---|---|---|
| (a) 기본 | 2.974 → 0.093 → 0.089 → 0.088 → 0.089 → 0.089 | 5000 | 0.933 | — |
| (b) + ℓ₂ (α = 0.01) | 3.007 → 0.271 → … → 0.271 | 5000 | **0.967** | — |
| (c) + 조기 종료 (C = 100) | 3.007 → 0.271 | **403** | 0.967 | **0.967** |

- (a) 3000 에포크 이후 검증 손실 소폭 상승 → 약한 과대적합
- (b) 손실 값은 규제 항 때문에 커졌지만 정확도는 향상
- (c) 같은 정확도를 약 1/12 에포크로 달성

**⑤ 핵심 정리**
- 소프트맥스 회귀의 핵심 계산 = `softmax(X @ Θ)` → `Xᵀ(P̂ − Y) / m` → `Θ −= η·∇`
- 규제는 편향 행 제외, 조기 종료는 매 에포크 검증 손실 확인
- 배치 GD는 손실 곡선이 매끄러워 1회 증가 즉시 중지 가능 (SGD는 인내 기간 필요)

---

## 4.8 전체 요약

### 4.8.1 흐름도

```
선형 데이터
   │
   ├─ 닫힌 형태 해: 정규 방정식 / lstsq·pinv / LinearRegression     ← 4.1
   │     특성 수 n이 크면 느림, 스케일링 불필요
   │
   └─ 반복 최적화: 배치 GD → SGD(+학습 스케줄) → 미니배치 GD          ← 4.2
         대용량·외부 메모리 가능, 스케일링 필수
   ↓ 비선형 데이터
PolynomialFeatures + LinearRegression                              ← 4.3
   ↓ 차수 선택
learning_curve: 과소적합(높고 붙음) / 과대적합(간격)                 ← 4.4
   ↓ 과대적합 감소
Ridge(ℓ₂) / Lasso(ℓ₁, 특성 선택) / ElasticNet / 조기 종료           ← 4.5
   ↓ 분류
LogisticRegression: 시그모이드 + 로그 손실 (이진)                    ← 4.6
LogisticRegression: 소프트맥스 + 크로스 엔트로피 (다중 클래스)
```

### 4.8.2 모델·비용 함수 비교

| 모델 | 출력 | 비용 함수 | 닫힌 형태 해 | 사이킷런 |
|---|---|---|---|---|
| 선형 회귀 | θᵀx | MSE | O | `LinearRegression`, `SGDRegressor` |
| 다항 회귀 | θᵀ·poly(x) | MSE | O | `PolynomialFeatures` + `LinearRegression` |
| 릿지 | θᵀx | MSE + (α/m)Σθ² | O | `Ridge`, `SGDRegressor(penalty="l2")` |
| 라쏘 | θᵀx | MSE + 2αΣ\|θ\| | X | `Lasso`, `SGDRegressor(penalty="l1")` |
| 엘라스틱넷 | θᵀx | MSE + r·ℓ₁ + (1−r)·ℓ₂ | X | `ElasticNet(l1_ratio=r)` |
| 로지스틱 회귀 | σ(θᵀx) | 로그 손실 | X (볼록) | `LogisticRegression` |
| 소프트맥스 회귀 | softmax(Θᵀx) | 크로스 엔트로피 | X (볼록) | `LogisticRegression` (클래스 ≥ 3) |

### 4.8.3 주요 결과

| 절 | 실험 | 결과 |
|---|---|---|
| 4.1.1 | 정규 방정식 / `LinearRegression` / `lstsq` | θ = (3.691, 3.330), 진짜 (4, 3) |
| 4.2.1 | 배치 GD (η = 0.1, 1000 에포크) | θ = (3.691, 3.330) — 정규 방정식과 동일 |
| 4.2.1 | 배치 GD (η = 0.5) | 발산 |
| 4.2.2 | SGD 직접 구현 (50 에포크) | θ = (3.698, 3.307) |
| 4.2.2 | `SGDRegressor` | θ = (3.689, 3.331) |
| 4.3 | 2차 다항 회귀 | 0.505x² + 1.110x + 2.005 (진짜 0.5x² + x + 2) |
| 4.4 | 학습 곡선 RMSE (훈련 / 검증) | 선형 1.62 / 1.64, 10차 0.97 / 1.03 |
| 4.5.1 | 릿지 `predict([[1.5]])` | 1.844 |
| 4.5.2 | 라쏘 `predict([[1.5]])` / 10차 α=0.01 비영 가중치 | 1.876 / 2개 |
| 4.5.3 | 엘라스틱넷 `predict([[1.5]])` | 1.865 |
| 4.5.4 | 조기 종료 (90차 + SGD) | 184 에포크, 검증 RMSE 1.105 |
| 4.6.3 | 로지스틱 회귀 결정 경계 | 꽃잎 너비 1.65cm |
| 4.6.4 | 소프트맥스 `predict_proba([[5, 2]])` | [0.00, 0.04, 0.96] |
| 4.7.2 | NumPy 소프트맥스 (규제 + 조기 종료) | 403 에포크, 검증·테스트 정확도 0.967 |

### 4.8.4 하이퍼파라미터 방향

| 하이퍼파라미터 | 값 ↑ | 주의 |
|---|---|---|
| 학습률 η (`eta0`) | 빠름, 과하면 발산 | — |
| 다항 차수 `degree` | 복잡도 ↑, 과대적합 위험 | 특성 수 폭발 |
| `alpha` (Ridge·Lasso·ElasticNet) | 규제 **강화** | `SGDRegressor`는 α/m으로 맞춤 |
| `C` (LogisticRegression) | 규제 **약화** | C = 1/α |
| `l1_ratio` | 라쏘에 가까워짐 | 0 = 릿지, 1 = 라쏘 |

### 4.8.5 코드 패턴

| 패턴 | 용도 | 절 |
|---|---|---|
| `add_dummy_feature(X)` / `np.c_[np.ones(m), X]` | 편향 열 추가 | 4.1.1, 4.5.1, 4.7.2 |
| `A @ B`, `A.T` | 행렬 곱, 전치 | 전체 |
| `theta - eta * gradients` | 경사 하강 스텝 | 4.2 |
| `X[i : i+1]` | 샘플 1개를 2차원으로 | 4.2.2 |
| `rng.permutation(m)` | 섞기 / 직접 분할 | 4.2.3, 4.7.2 |
| `y.ravel()` | `(m, 1)` → `(m,)` | 4.2.2, 4.5.1 |
| `np.linspace(a, b, n).reshape(-1, 1)` | 예측 곡선 입력 | 4.3, 4.6 |
| `learning_curve(..., scoring="neg_root_mean_squared_error")` | 학습 곡선 | 4.4 |
| `partial_fit()` + `deepcopy()` | 에포크별 훈련 + 최고 모델 보존 | 4.5.4 |
| `meshgrid` → `np.c_[ravel, ravel]` → `reshape` | 결정 경계·등고선 | 4.6.3, 4.6.4 |
| `np.diag(np.ones(K))[y]` | 원-핫 인코딩 | 4.7.2 |
| `sum(axis=1, keepdims=True)` | 행별 합 브로드캐스팅 | 4.7.2 |
| `(pred == y).mean()` | 정확도 | 4.7.2 |
