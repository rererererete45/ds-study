# 7장. 차원 축소

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `07_dimensionality_reduction.ipynb`
>
> 특성이 수천~수백만 개면 훈련이 느려지고 좋은 해를 찾기도 어려워진다(**차원의 저주**). 이 장은 정보를 최대한 남기면서 특성 수를 줄이는 방법을 다룬다. 큰 줄기는 **투영**(PCA, 랜덤 투영)과 **매니폴드 학습**(LLE, Isomap, t-SNE) 두 가지.
>
> 실습 데이터
>
> | 데이터 | 형태 | 용도 |
> |---|---|---|
> | 작은 3D 타원 데이터 | `(60, 3)` 직접 생성 | PCA 원리 (SVD, 투영) |
> | 스위스 롤 `make_swiss_roll` | `(1000, 3)` + 위치값 `t` | 매니폴드 학습 (LLE, MDS, Isomap, t-SNE) |
> | MNIST | `(70000, 784)` | 차원 수 선택, 압축, 랜덤·점진적 PCA, 연습문제 |
> | 가짜 고차원 데이터 | `(5000, 20000)` 정규분포 | 랜덤 투영 |

## 목차

1. [차원의 저주 (7.1)](#1-차원의-저주-71)
2. [차원 축소 접근법 (7.2)](#2-차원-축소-접근법-72)
3. [주성분 분석 (7.3)](#3-주성분-분석-73)
4. [랜덤 투영 (7.4)](#4-랜덤-투영-74)
5. [지역 선형 임베딩 (7.5)](#5-지역-선형-임베딩-75)
6. [다른 차원 축소 기법 (7.6)](#6-다른-차원-축소-기법-76)
7. [연습문제](#7-연습문제)
8. [전체 흐름 요약](#8-전체-흐름-요약)

---

## 0. 준비

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

앞 장들과 같은 환경 확인 + 그래프 글꼴 설정.

---

## 1. 차원의 저주 (7.1)

노트북에는 코드가 없고 책 본문에서 설명하는 절. 핵심은 **고차원 공간은 직관과 다르게 행동한다**는 것.

| 현상 | 의미 |
|---|---|
| 점이 경계에 몰림 | 단위 정사각형에서 경계 0.001 이내에 있을 확률은 약 0.4%지만, 10,000차원 초입방체에선 99.999999% 이상 |
| 점끼리 멀어짐 | 무작위 두 점 사이 평균 거리가 차원이 커질수록 계속 커짐 (아래 실험) |
| 데이터가 희박해짐 | 새 샘플이 훈련 샘플과 멀리 떨어짐 → 예측이 **외삽**에 의존 → 과대적합 위험 ↑ |
| 데이터로 메우기 불가 | 밀도를 유지하려면 필요한 샘플 수가 차원에 따라 **지수적으로** 증가 |

직접 확인해 보는 코드 (노트북에 없는 추가 예시):

```python
import numpy as np

rng = np.random.default_rng(42)
for n in (2, 3, 10, 100, 1000, 10000):
    a = rng.random((2000, n))     # 단위 초입방체 안의 무작위 점 2,000개
    b = rng.random((2000, n))
    print(n, np.linalg.norm(a - b, axis=1).mean().round(3))
```

| 차원 n | 평균 거리 | 이론값 ≈ √(n/6) |
|---|---|---|
| 2 | 0.528 | 0.577 |
| 3 | 0.663 | 0.707 |
| 10 | 1.272 | 1.291 |
| 100 | 4.067 | 4.082 |
| 1,000 | 12.904 | 12.910 |
| 10,000 | 40.825 | 40.825 |

- 모든 점이 **같은 크기의 상자 안**에 있는데도 차원이 늘면 서로 점점 멀어진다
- 거리 기반 알고리즘(KNN, 군집 등)이 고차원에서 약해지는 이유

> 💡 해결책은 두 가지: **데이터를 늘리거나**(현실적으로 불가능한 경우가 많음) **차원을 줄이거나**.

---

## 2. 차원 축소 접근법 (7.2)

### 2-1. 투영 (Projection)

실제 데이터는 모든 차원에 고르게 퍼져 있지 않고 **저차원 부분 공간 근처**에 몰려 있는 경우가 많다. 그 부분 공간에 수직으로 내리꽂으면 된다.

먼저 실습용 3D 데이터를 만든다.

```python
# extra code
import numpy as np
from scipy.spatial.transform import Rotation

m = 60
X = np.zeros((m, 3))                                   # 60×3 영행렬
rng = np.random.default_rng(seed=42)
angles = (rng.random(m) ** 3 + 0.5) * 2 * np.pi        # 고르지 않은 각도 분포
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5  # 타원 (x축 1, y축 0.5)
X += 0.28 * rng.standard_normal((m, 3))                # 3축 모두에 노이즈
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)  # 3D 회전
X += [0.2, 0, 0.2]                                     # 살짝 평행 이동
```

| 줄 | 의미 |
|---|---|
| `rng.random(m) ** 3` | 0~1 균등분포를 세제곱 → 0 근처로 쏠림 → 각도가 **불균등**하게 분포 |
| `np.cos`, `np.sin * 0.5` | 반지름 1×0.5인 **타원** 위의 점 |
| `0.28 * standard_normal` | z축에도 노이즈 → 완전한 평면이 아니라 **평면 근처**에 분포 |
| `Rotation.from_rotvec(...)` | 회전 벡터로 3D 회전. 축과 나란한 평면이면 너무 쉬우니 비스듬히 기울임 |

그림 7-2는 이 데이터와 PCA가 찾은 투영 평면을, 그림 7-3은 그 평면에 투영한 2D 결과(z₁, z₂ 축)를 그린다. 그리는 코드 자체는 `plot_surface`, `meshgrid` 같은 Matplotlib 3D 기능 위주라 생략. 그 안에서 쓰인 핵심 세 줄만:

```python
pca = PCA(n_components=2)
X2D = pca.fit_transform(X)              # 3D → 2D
X3D_inv = pca.inverse_transform(X2D)    # 2D 좌표를 다시 3D 평면 위 위치로
```

- `X3D_inv`는 **평면 위로 떨어진 점의 3D 위치**. 원본 `X`와 연결한 점선이 "투영선"
- 원래 점이 평면 위인지 아래인지(`X[:, 2] >= X3D_inv[:, 2]`)를 나눠 그리는 건 3D 그림에서 가려짐을 자연스럽게 하기 위함

### 2-2. 매니폴드 학습 (Manifold Learning)

투영이 항상 통하지는 않는다. 대표 반례가 **스위스 롤**.

```python
from sklearn.datasets import make_swiss_roll

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
```

| 반환값 | 의미 |
|---|---|
| `X_swiss` | `(1000, 3)` 롤 케이크처럼 말린 2D 평면 |
| `t` | 각 점이 **롤을 펼쳤을 때 가로 위치**. 그림에서 색으로 표시. 정답 좌표 역할 |

```python
# 그림 7-5 핵심
plt.subplot(121)
plt.scatter(X_swiss[:, 0], X_swiss[:, 1], c=t, cmap=darker_hot)  # x3 버리고 투영 → 층이 뭉개짐
plt.subplot(122)
plt.scatter(t, X_swiss[:, 1], c=t, cmap=darker_hot)              # 롤을 펼친 모습 → 깔끔
```

- 왼쪽: 그냥 x₃ 축을 버리면 서로 다른 층이 **겹쳐 버린다**
- 오른쪽: `t`를 가로축으로 쓰면 롤이 **펼쳐진** 2D 평면이 된다 → 이게 매니폴드 학습의 목표

> **d차원 매니폴드** = 국부적으로는 d차원 평면처럼 보이는 n차원 공간의 일부 (d < n). 스위스 롤은 3D 공간 속의 2D 매니폴드.
>
> **매니폴드 가정**: 실제 고차원 데이터 대부분은 훨씬 낮은 차원의 매니폴드 근처에 있다. 예: MNIST의 "숫자처럼 보이는 이미지"는 784차원 공간 전체에서 아주 작은 영역.

그림 7-6은 **매니폴드로 펼친다고 늘 쉬워지지는 않는다**는 걸 보여 준다.

```python
positive_class = X_swiss[:, 0] > 5            # 경계: 3D에선 평면 x1 = 5 (단순)
# → 펼친 2D 공간에선 경계가 여러 조각으로 쪼개짐 (복잡)

positive_class = 2 * (t[:] - 4) > X_swiss[:, 1]  # 경계: 펼친 2D에선 직선 (단순)
# → 3D 원본 공간에선 복잡한 곡면
```

> ⚠️ 차원 축소는 **훈련 속도는 거의 항상 올려 주지만**, 성능이 좋아질지는 데이터에 달렸다. 연습문제 9에서 직접 확인.

### 2-3. 투영이 데이터를 잘 보존하는 방향 (그림 7-7)

```python
angle = np.pi / 5
stretch = 5
m = 200

rng = np.random.default_rng(seed=3)
X_line = rng.standard_normal((m, 2)) / 10
X_line = X_line @ np.array([[stretch, 0], [0, 1]])            # x축으로 5배 늘림
X_line = X_line @ [[np.cos(angle), np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]]            # 회전

u1 = np.array([np.cos(angle), np.sin(angle)])                       # 실선: 길게 퍼진 방향
u2 = np.array([np.cos(angle - 2 * np.pi / 6), np.sin(angle - 2 * np.pi / 6)])  # 파선: 중간
u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])  # 점선: 수직 방향

X_proj1 = X_line @ u1.reshape(-1, 1)   # 각 방향으로 투영한 1D 값
X_proj2 = X_line @ u2.reshape(-1, 1)
X_proj3 = X_line @ u3.reshape(-1, 1)
```

- 길쭉한 2D 데이터를 세 방향 직선에 투영해 비교
- **u1(데이터가 가장 길게 퍼진 방향)** 에 투영하면 점들이 넓게 퍼진 채 남음 → 분산 최대 → 정보 손실 최소
- u3에 투영하면 점들이 한곳에 뭉침 → 정보 대부분 소실
- 동치 표현: 분산을 최대로 보존하는 축 = 원본과 투영 사이 **평균 제곱 거리를 최소화**하는 축. 이게 PCA의 아이디어

---

## 3. 주성분 분석 (7.3)

PCA(Principal Component Analysis): **데이터에 가장 가까운 초평면**을 찾아 그 위로 투영.

### 3-1. 주성분 찾기 — SVD

```python
import numpy as np

X_centered = X - X.mean(axis=0)        # 1) 평균을 0으로 (중앙 정렬)
U, s, Vt = np.linalg.svd(X_centered)   # 2) 특잇값 분해
c1 = Vt[0]                             # 첫 번째 주성분 (단위 벡터)
c2 = Vt[1]                             # 두 번째 주성분
```

| 줄 | 의미 |
|---|---|
| `X.mean(axis=0)` | 열(특성)별 평균 → shape `(3,)`. 브로드캐스팅으로 모든 행에서 뺌 |
| 중앙 정렬 | PCA는 데이터가 원점 중심이라고 가정. 직접 SVD할 땐 **반드시** 해야 함 |
| `np.linalg.svd` | X = U Σ Vᵀ 로 분해. **Vᵀ의 각 행이 주성분** |
| `Vt[0]` | 분산이 가장 큰 방향. `Vt[1]`은 그와 **직교**하면서 남은 분산이 가장 큰 방향 |

**`s`와 Σ의 관계**

```python
# extra code – s로부터 Σ 만들기
m, n = X.shape                      # 60, 3
Σ = np.zeros_like(X_centered)       # (60, 3) 영행렬
Σ[:n, :n] = np.diag(s)              # 위쪽 3×3 대각선에만 특잇값
assert np.allclose(X_centered, U @ Σ @ Vt)   # 원래대로 복원되는지 검증
```

- 이론상 Σ는 `m × n` 행렬이지만 대각선 말고는 전부 0 → NumPy는 대각 값만 1차원 `s`로 돌려줌
- `np.allclose`: 부동소수점 오차를 허용하는 비교

> 💡 주성분의 **부호**는 실행할 때마다 뒤집힐 수 있다(c₁ 대신 −c₁). 방향만 같으면 같은 축. 결과를 비교할 때 주의.

### 3-2. d차원으로 투영

```python
W2 = Vt[:2].T          # 처음 두 주성분을 열로 → shape (3, 2)
X2D = X_centered @ W2  # (60, 3) @ (3, 2) = (60, 2)
```

- 수식: **X_d-proj = X W_d**. W_d는 처음 d개 주성분을 열로 쌓은 행렬
- 행렬 곱 한 번 = 각 샘플을 각 주성분 방향에 내적 = 그 축 위의 좌표

### 3-3. 사이킷런 사용하기

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)   # 중앙 정렬까지 자동

pca.components_
# array([[ 0.66824153,  0.73208333,  0.13231495],
#        [ 0.74374636, -0.66151587, -0.09611511]])
```

| 속성 | 의미 |
|---|---|
| `components_` | 주성분 행렬. shape `(n_components, n_features)` → **Vt[:2]와 같음** (행 = 주성분) |
| `mean_` | 학습 때 뺀 평균. `transform`·`inverse_transform`에서 자동 사용 |
| `fit_transform` | 평균 계산 → SVD → 투영을 한 번에 |

> 💡 첫 주성분 `[0.67, 0.73, 0.13]`은 x₁, x₂가 비슷한 비중, x₃는 작음 → 데이터를 x₁·x₂ 쪽으로 45°쯤 돌려 놓았기 때문(앞에서 `π/4` 회전).

### 3-4. 설명된 분산의 비율

```python
pca.explained_variance_ratio_
# array([0.82279334, 0.10821224])

1 - pca.explained_variance_ratio_.sum()
# 0.06899442237323172
```

- 1번 축이 전체 분산의 **약 82%**, 2번 축이 **약 11%**를 설명
- 2D로 줄이며 잃은 분산은 **약 6.9%** → 3번째 축(z 방향 노이즈)은 정보가 거의 없음

> ⚠️ 노트북 본문 설명은 "76%, 15%, 약 9% 손실"이라고 적혀 있지만, 실제 출력값은 82% / 11% / 6.9%. 데이터 생성 코드가 바뀐 뒤 설명이 갱신되지 않은 것으로 보인다. **출력값 기준으로 이해**하면 된다.

### 3-5. 적절한 차원 수 선택

**방법 1 — 누적 분산을 직접 계산**

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train = mnist.data[:60_000], mnist.target[:60_000]
X_test, y_test = mnist.data[60_000:], mnist.target[60_000:]

pca = PCA()                                    # n_components 미지정 → 784개 전부
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)  # 누적합
d = np.argmax(cumsum >= 0.95) + 1              # 95%를 처음 넘는 지점
d   # 154
```

| 줄 | 의미 |
|---|---|
| `np.cumsum` | `[a, b, c]` → `[a, a+b, a+b+c]`. i번째 값 = 처음 i+1개 축이 설명하는 분산 |
| `cumsum >= 0.95` | 불리언 배열 |
| `np.argmax(불리언)` | **처음 True가 나오는 인덱스** (True=1이 최댓값이므로). 3장에서도 쓴 패턴 |
| `+ 1` | 인덱스는 0부터, 차원 수는 1부터 |

→ 784차원 중 **154차원(약 20%)** 만으로 분산 95% 보존.

**방법 2 — 비율을 바로 넘기기 (권장)**

```python
pca = PCA(n_components=0.95)        # 0~1 실수 → "이만큼 분산을 보존하라"
X_reduced = pca.fit_transform(X_train)

pca.n_components_                   # 154  (실제로 고른 차원 수)
pca.explained_variance_ratio_.sum() # 0.9501960192613031
```

- `n_components`가 **정수**면 차원 수, **0~1 실수**면 보존할 분산 비율
- `n_components_`(끝에 밑줄) = 학습 후 결정된 값

**방법 3 — 누적 분산 그래프의 엘보** (그림 7-8)

```python
plt.plot(cumsum, linewidth=3)
plt.plot([d, d], [0, 0.95], "k:")       # d=154 세로 점선
plt.plot([0, d], [0.95, 0.95], "k:")    # 0.95 가로 점선
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"))
```

- 곡선이 꺾이는 지점(약 65차원, 85%) 이후로는 차원을 늘려도 분산이 별로 안 늘어난다

**방법 4 — 차원 수를 하이퍼파라미터로 튜닝** (지도 학습 전처리일 때)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline

clf = make_pipeline(PCA(random_state=42),
                    RandomForestClassifier(random_state=42))
param_distrib = {
    "pca__n_components": np.arange(10, 80),
    "randomforestclassifier__n_estimators": np.arange(50, 500)
}
rnd_search = RandomizedSearchCV(clf, param_distrib, n_iter=10, cv=3,
                                random_state=42)
rnd_search.fit(X_train[:1000], y_train[:1000])

print(rnd_search.best_params_)
# {'randomforestclassifier__n_estimators': 475, 'pca__n_components': 57}
```

| 줄 | 의미 |
|---|---|
| `make_pipeline(PCA, RF)` | 단계 이름이 클래스 소문자로 자동 생성 → `pca`, `randomforestclassifier` |
| `"pca__n_components"` | `단계이름__파라미터` 형식 (2장과 동일) |
| `np.arange(10, 80)` | 10~79 정수 중에서 무작위 선택 |
| `n_iter=10, cv=3` | 조합 10개 × 폴드 3개 = 30번 학습 |
| `X_train[:1000]` | 시간 절약용으로 1,000개만 |

→ 784차원을 **57차원**으로 줄여도 분류가 잘 된다. 95% 기준(154)보다 훨씬 적음 = 분류에 필요한 정보는 분산의 95%보다 적을 수 있다.

```python
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

clf = make_pipeline(PCA(random_state=42), SGDClassifier())
param_grid = {"pca__n_components": np.arange(10, 80)}
grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X_train[:1000], y_train[:1000])

grid_search.best_params_   # {'pca__n_components': 75}
```

- 선형 모델(SGD)은 탐색 범위의 최댓값 근처인 **75**를 골랐다 → 범위를 더 넓혀 볼 여지가 있음
- 같은 데이터라도 **뒤에 붙는 모델에 따라 최적 차원이 다르다**

### 3-6. 압축을 위한 PCA

```python
pca = PCA(0.95)
X_reduced = pca.fit_transform(X_train, y_train)   # y는 무시됨 (비지도)

X_recovered = pca.inverse_transform(X_reduced)    # 154차원 → 784차원 복원
```

| 줄 | 의미 |
|---|---|
| `PCA(0.95)` | 첫 위치 인자가 `n_components` |
| `fit_transform(X, y)` | 파이프라인 호환을 위해 y를 받지만 사용하지 않음 |
| `inverse_transform` | **X_recovered = X_d-proj W_dᵀ + 평균**. 버린 5% 분산은 돌아오지 않음 |

- 데이터 크기: 784 → 154로 **약 20%** (원본의 80% 절약)
- 원본과 복원본의 평균 제곱 거리 = **재구성 오차**

```python
# 그림 7-9: 원본 vs 압축 후 복원
for idx, X in enumerate((X_train[::2100], X_recovered[::2100])):  # 2,100장마다 하나씩
    plt.subplot(1, 2, idx + 1)
    for row in range(5):
        for col in range(5):
            plt.imshow(X[row * 5 + col].reshape(28, 28), cmap="binary",
                       vmin=0, vmax=255, extent=(row, row + 1, col, col + 1))
```

- `[::2100]` → 60,000장에서 약 29장 추출, 그중 25장을 5×5로 배치
- `extent`로 각 이미지를 격자 칸에 직접 배치 (subplot 25개를 만들지 않는 방법)
- 결과: 약간 흐려지지만 숫자는 그대로 알아볼 수 있음

### 3-7. 랜덤 PCA

```python
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
X_reduced = rnd_pca.fit_transform(X_train)
```

| `svd_solver` | 동작 |
|---|---|
| `"full"` | 완전한 SVD. 정확하지만 O(m·n²) + O(n³) |
| `"randomized"` | 확률적 알고리즘으로 **처음 d개 주성분의 근삿값**만 구함. O(m·d²) + O(d³) |
| `"auto"` (기본값) | 데이터가 크고(500×500 이상) d가 작으면(80% 미만) 자동으로 randomized 선택 |
| `"covariance_eigh"` | 샘플이 특성보다 훨씬 많을 때 빠름 (사이킷런 1.5+) |

> 💡 d가 n보다 훨씬 작을 때 randomized가 압도적으로 빠르다. 결과도 거의 같다.

### 3-8. 점진적 PCA

전체 데이터가 메모리에 안 들어갈 때 → **미니배치로 나눠 학습**.

```python
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):  # 60,000 → 600개씩 100묶음
    inc_pca.partial_fit(X_batch)                    # fit 대신 partial_fit

X_reduced = inc_pca.transform(X_train)
```

| 줄 | 의미 |
|---|---|
| `np.array_split` | 배열을 n개로 나눔. `np.split`과 달리 **나누어 떨어지지 않아도** 됨 |
| `partial_fit` | 이전 학습 결과를 유지한 채 이 배치로 **추가 학습**. `fit`은 매번 처음부터 다시 함 |

**`np.memmap` — 디스크 파일을 메모리처럼 쓰기**

```python
# 프로그램 1: 데이터를 디스크에 저장
filename = "my_mnist.mmap"
X_mmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_mmap[:] = X_train   # 실제로는 조금씩 나눠 쓰는 루프여도 됨
X_mmap.flush()        # 캐시에 남은 내용까지 디스크에 기록
```

```python
# 프로그램 2: 불러와서 학습
X_mmap = np.memmap(filename, dtype="float32", mode="readonly").reshape(-1, 784)
batch_size = X_mmap.shape[0] // n_batches        # 600
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mmap)
# IncrementalPCA(batch_size=600, n_components=154)
```

| 포인트 | 설명 |
|---|---|
| `memmap` | 배열처럼 인덱싱하면 **필요한 부분만** 디스크에서 읽어 옴 |
| 불러올 때 shape 정보 없음 | 파일엔 숫자만 저장됨 → `reshape(-1, 784)`로 직접 복원. dtype도 저장 때와 같아야 함 |
| `fit` 사용 가능 | `IncrementalPCA.fit`은 내부적으로 `batch_size`씩 잘라 `partial_fit`을 반복 → memmap과 궁합이 좋음 |

### PCA 세 가지 비교

| 종류 | 언제 | 특징 |
|---|---|---|
| `PCA()` | 기본. 데이터가 메모리에 들어갈 때 | 정확 |
| `PCA(svd_solver="randomized")` | d ≪ n, 메모리에 들어갈 때 | 훨씬 빠름, 근사 |
| `IncrementalPCA` | 메모리 초과, 온라인 학습 | 느리지만 미니배치 가능 |

---

## 4. 랜덤 투영 (7.4)

아주 고차원(수만 이상)일 땐 PCA조차 느리다. 랜덤 투영은 **무작위 선형 투영 행렬**을 곱하기만 한다. 놀랍게도 거리 정보가 꽤 잘 보존된다.

### 4-1. 존슨–린덴슈트라우스 보조정리 — 몇 차원이 필요한가

```python
from sklearn.random_projection import johnson_lindenstrauss_min_dim

m, ε = 5_000, 0.1
d = johnson_lindenstrauss_min_dim(m, eps=ε)
d   # 7300
```

```python
# extra code – 내부 공식
d = int(4 * np.log(m) / (ε ** 2 / 2 - ε ** 3 / 3))
d   # 7300
```

| 기호 | 의미 |
|---|---|
| `m` | 샘플 수 (5,000) |
| `ε` | 허용 왜곡. 0.1이면 임의의 두 점 사이 제곱 거리가 **±10% 이상 변하지 않음** |
| `d` | 필요한 최소 차원 = **4 log(m) / (ε²/2 − ε³/3)** |

> 💡 공식에 **원래 차원 n이 없다**. 샘플 수와 허용 오차만으로 결정. 원본이 2만 차원이든 200만 차원이든 7,300차원이면 충분.

### 4-2. 직접 구현

```python
n = 20_000
rng = np.random.default_rng(seed=42)
P = rng.standard_normal((d, n)) / np.sqrt(d)   # 투영 행렬 (7300, 20000)

X = rng.standard_normal((m, n))                 # 가짜 데이터 (5000, 20000)
X_reduced = X @ P.T                             # (5000, 20000) @ (20000, 7300)
```

| 줄 | 의미 |
|---|---|
| `standard_normal / sqrt(d)` | 평균 0, **분산 1/d**인 가우스 난수. 나누는 건 표준편차(√분산)이므로 `sqrt(d)` |
| `P` | **데이터를 전혀 보지 않고** 만든 행렬. 학습이 필요 없음 |
| `X @ P.T` | 각 샘플을 7,300개 무작위 방향에 투영 |

> ⚠️ 이 절은 메모리를 약 2.5 GB 쓴다. 부족하면 `m`, `n`을 줄일 것.

### 4-3. 사이킷런

```python
from sklearn.random_projection import GaussianRandomProjection

gaussian_rnd_proj = GaussianRandomProjection(eps=ε, random_state=42)
X_reduced = gaussian_rnd_proj.fit_transform(X)   # 위와 같은 결과
```

- `fit`이 하는 일: 데이터 shape만 보고 d 계산 + 무작위 행렬 생성 (`components_`에 저장)
- `n_components`를 직접 지정해도 됨

**역변환 (느림)**

```python
components_pinv = np.linalg.pinv(gaussian_rnd_proj.components_)  # 유사역행렬
X_recovered = X_reduced @ components_pinv.T
```

- `P`는 정사각 행렬이 아니라 역행렬이 없음 → **유사역행렬**(`pinv`)로 근사 복원
- 계산량이 커서 몇 분 걸릴 수 있음

### 4-4. 희소 랜덤 투영

```python
from sklearn.random_projection import SparseRandomProjection

%timeit GaussianRandomProjection(random_state=42).fit(X)
%timeit SparseRandomProjection(random_state=42).fit(X)
gaussian_rnd_proj = GaussianRandomProjection(random_state=42).fit(X)
sparse_rnd_proj = SparseRandomProjection(random_state=42).fit(X)
%timeit gaussian_rnd_proj.transform(X)
%timeit sparse_rnd_proj.transform(X)
```

| 단계 | Gaussian | Sparse |
|---|---|---|
| `fit` | 5.04 s | 3.62 s |
| `transform` | **43.2 s** | **5.77 s** (약 7.5배 빠름) |

- 희소 행렬: 대부분 0이고 일부만 ±값 → **메모리 적게, 곱셈 빠르게**
- 0이 아닌 비율 기본값 = 1/√n → n = 20,000이면 약 0.7%만 0이 아님
- 거리 보존 품질은 가우스 버전과 거의 같다 → **실무에선 Sparse를 기본으로**

> 💡 `%timeit`은 주피터 매직 명령. 여러 번 반복 실행해 평균 ± 표준편차를 보여 준다. 일반 파이썬 스크립트에선 동작하지 않음.

---

## 5. 지역 선형 임베딩 (7.5)

LLE(Locally Linear Embedding): 투영이 아닌 **매니폴드 학습** 기법.

**알고리즘 두 단계**

1. 각 샘플 x⁽ⁱ⁾를 가장 가까운 k개 이웃의 **선형 결합**으로 표현하는 가중치 w를 찾는다 (가중치 합 = 1, 이웃이 아니면 0)
2. 그 **가중치 관계가 최대한 유지되도록** 저차원 좌표 z를 찾는다

→ "가까운 이웃과의 관계"만 보존하므로 말려 있는 롤도 펼칠 수 있다.

```python
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_unrolled = lle.fit_transform(X_swiss)
```

| 파라미터 | 의미 |
|---|---|
| `n_components=2` | 결과 차원 |
| `n_neighbors=10` | 1단계에서 볼 이웃 수 k. 너무 작으면 국부 구조 불안정, 너무 크면 롤의 다른 층까지 이웃이 됨 |
| `random_state` | 고유값 계산의 무작위 초기화 고정 |

```python
# 그림 7-10
plt.scatter(X_unrolled[:, 0], X_unrolled[:, 1], c=t, cmap=darker_hot)

# extra: z1과 t의 관계
plt.scatter(X_unrolled[:, 0], t, c=t, cmap=darker_hot)
```

- 색(`t`)이 한쪽에서 다른 쪽으로 **순서대로 변함** → 롤이 펼쳐졌다
- 두 번째 그림에서 z₁과 t가 거의 단조 관계 → LLE가 롤의 "펼친 가로 위치"를 찾아냄
- 다만 펼친 결과는 **국부적으로만** 잘 보존되고 큰 규모에선 늘어나거나 휘어 있다

**계산 복잡도**

| 단계 | 복잡도 |
|---|---|
| k개 이웃 찾기 | O(m log(m) n log(k)) |
| 가중치 최적화 | O(m n k³) |
| 저차원 표현 | **O(d m²)** ← m²이 병목 |

> ⚠️ 샘플 수의 **제곱**에 비례 → 대규모 데이터엔 부적합. 연습문제 10에서 5,000장에 약 3초.

---

## 6. 다른 차원 축소 기법 (7.6)

```python
from sklearn.manifold import MDS, Isomap, TSNE

mds = MDS(n_components=2, normalized_stress=False, random_state=42)
X_reduced_mds = mds.fit_transform(X_swiss)

isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X_swiss)

tsne = TSNE(n_components=2, init="random", learning_rate="auto",
            random_state=42)
X_reduced_tsne = tsne.fit_transform(X_swiss)
```

| 기법 | 보존하려는 것 | 파라미터 메모 | 스위스 롤 결과 (그림 7-11) |
|---|---|---|---|
| **MDS** (다차원 스케일링) | 모든 샘플 쌍의 **거리** | `normalized_stress=False`: 기본값 변경 예고 경고를 끄려고 명시 | 롤을 펼치지 못하고 납작하게 누름 (큰 거리까지 보존하려 하므로) |
| **Isomap** | 이웃 그래프 위의 **측지 거리**(그래프를 따라가는 최단 경로) | 기본 `n_neighbors=5` | 잘 펼침 |
| **t-SNE** | 가까운 점은 가깝게, 먼 점은 멀게 (**국부 구조** 강조) | `init="random"`, `learning_rate="auto"`: 버전 간 기본값 변경 대응 | 롤의 연속성은 깨지지만 군집은 뚜렷 → **시각화용** |

**추가로 책에서 소개하는 기법**

| 기법 | 핵심 | 사이킷런 |
|---|---|---|
| **LDA** (선형 판별 분석) | **레이블을 사용**. 클래스를 가장 잘 구분하는 축으로 투영. 분류 전처리로 좋음 | `LinearDiscriminantAnalysis` |
| **랜덤 트리 임베딩** | 랜덤 트리의 리프 인덱스로 희소 표현 | `RandomTreesEmbedding` |
| **커널 PCA** | 커널 트릭으로 비선형 투영 (아래) | `KernelPCA` |
| **UMAP** | t-SNE와 비슷하지만 더 빠르고 전역 구조도 꽤 보존 | 별도 패키지 `umap-learn` |

> ⚠️ **t-SNE, MDS, Isomap, LLE에는 새 데이터용 `transform`이 없거나 제한적**이다(t-SNE는 아예 없음). 학습한 데이터만 변환 가능 → 주로 시각화·탐색용.

### 6-1. 커널 PCA (Extra Material)

```python
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04, random_state=42)
X_reduced = rbf_pca.fit_transform(X_swiss)
```

- 데이터를 (암묵적으로) 아주 고차원 특성 공간에 매핑한 뒤 그곳에서 PCA → 원래 공간 기준으로는 **비선형** 투영
- 부록 C(SVM)의 커널 트릭과 같은 원리

```python
lin_pca = KernelPCA(kernel="linear")                    # = 일반 PCA
rbf_pca = KernelPCA(kernel="rbf", gamma=0.002)          # 가우스 RBF
sig_pca = KernelPCA(kernel="sigmoid", gamma=0.002, coef0=1)

kernel_pcas = ((lin_pca, "Linear kernel"),
               (rbf_pca, rf"RBF kernel, $\gamma={rbf_pca.gamma}$"),
               (sig_pca, rf"Sigmoid kernel, $\gamma={sig_pca.gamma}, r={sig_pca.coef0}$"))

for idx, (kpca, title) in enumerate(kernel_pcas):
    kpca.n_components = 2          # 생성 후 속성으로 파라미터 지정도 가능
    kpca.random_state = 42
    X_reduced = kpca.fit_transform(X_swiss)
    plt.subplot(1, 3, idx + 1)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=darker_hot)
```

| 줄 | 의미 |
|---|---|
| `rf"..."` | raw + f-string. `\gamma` 같은 LaTeX 백슬래시를 그대로 두면서 `{}` 값도 넣음 |
| `kpca.n_components = 2` | 세 모델에 공통 설정을 루프 안에서 한 번에 |

- 커널 종류와 `gamma`에 따라 결과가 완전히 달라짐
- **비지도**라 좋은 커널을 고르는 기준이 없음 → 두 가지 방법:
  1. 지도 학습 전처리라면 파이프라인에 넣고 `GridSearchCV`로 `kpca__gamma`, `kpca__kernel` 탐색
  2. `fit_inverse_transform=True`로 **재구성 오차**를 계산해 가장 작은 조합 선택

---

## 7. 연습문제

### 1~8번 개념 문제

| 번호 | 질문 | 답 요약 |
|---|---|---|
| 1 | 차원 축소의 목적과 단점 | **목적**: 훈련 속도 향상(노이즈 제거로 성능 향상도 가능), 시각화, 압축. **단점**: 정보 손실, 계산 비용, 파이프라인 복잡도 증가, 변환된 특성 해석이 어려움 |
| 2 | 차원의 저주란? | 저차원엔 없던 문제가 고차원에서 생김. 무작위 고차원 벡터끼리 멀리 떨어져 → 과대적합 위험 ↑, 많은 데이터 없이는 패턴을 찾기 어려움 |
| 3 | 차원 축소를 되돌릴 수 있나? | 완벽한 복원은 거의 불가능(정보 손실). PCA는 역변환이 쉽지만 t-SNE 같은 기법은 역변환 자체가 없음 |
| 4 | 매우 비선형적인 데이터에 PCA를 써도 되나? | 쓸모없는 차원은 제거할 수 있어서 대부분 유효. 하지만 스위스 롤처럼 쓸모없는 차원이 없으면 정보를 너무 많이 잃음 → 펼쳐야지 **눌러 버리면 안 됨** |
| 5 | 1,000차원 데이터에 분산 95% PCA → 몇 차원? | **데이터에 따라 다름**. 거의 한 줄로 늘어선 데이터면 1차원, 완전 무작위면 약 950차원. 누적 분산 그래프가 내재 차원을 가늠하는 방법 |
| 6 | 일반·점진적·랜덤 PCA·랜덤 투영 선택 | 일반: 기본(메모리에 들어갈 때). 점진적: 메모리 초과·온라인. 랜덤 PCA: 메모리에 들어가고 차원을 크게 줄일 때. 랜덤 투영: 매우 고차원 |
| 7 | 차원 축소 성능 평가 방법 | 역변환 후 **재구성 오차** 측정. 역변환이 없으면 뒤에 붙는 모델의 성능으로 판단 |
| 8 | 두 기법을 연결해도 되나? | 된다. **PCA나 랜덤 투영으로 쓸모없는 차원을 빠르게 제거 → LLE 같은 느린 기법** 적용. 성능은 비슷하고 시간은 훨씬 적게 듦 |

### 9번 — PCA가 훈련을 빠르게 하는가?

```python
X_train = mnist.data[:60000]
y_train = mnist.target[:60000]
X_test = mnist.data[60000:]
y_test = mnist.target[60000:]
```

**(1) 랜덤 포레스트, 원본 784차원**

```python
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
%time rnd_clf.fit(X_train, y_train)          # Wall time: 57.7 s

from sklearn.metrics import accuracy_score
y_pred = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred)               # 0.9705
```

**(2) PCA 95% → 랜덤 포레스트**

```python
pca = PCA(n_components=0.95)
X_train_reduced = pca.fit_transform(X_train)   # 154차원

rnd_clf_with_pca = RandomForestClassifier(n_estimators=100, random_state=42)
%time rnd_clf_with_pca.fit(X_train_reduced, y_train)   # Wall time: 3min 4s  ← 더 느림!

X_test_reduced = pca.transform(X_test)          # 테스트는 fit 없이 transform만
y_pred = rnd_clf_with_pca.predict(X_test_reduced)
accuracy_score(y_test, y_pred)                  # 0.9488  ← 더 나쁨
```

- **약 3배 느려지고 정확도도 2%p 하락** 😭
- 이유: 원본 MNIST는 가장자리 픽셀이 거의 항상 0 → 트리가 쉽게 무시함. PCA 후엔 모든 특성이 **밀집된 실수값**이 되고, 주성분 축은 트리가 자르기 좋은 축 정렬 경계가 아님 → 분할 탐색이 더 어려움
- 2-2절(그림 7-6)에서 본 "차원 축소가 늘 문제를 쉽게 만들지는 않는다"의 실례

> ⚠️ `pca.transform(X_test)` — 테스트 세트에 `fit_transform`을 쓰면 안 된다. 훈련 세트에서 학습한 주성분을 그대로 써야 한다(2장의 데이터 누수 원칙).

**(3) SGD 분류기로 다시**

```python
sgd_clf = SGDClassifier(random_state=42)
%time sgd_clf.fit(X_train, y_train)          # Wall time: 3min 8s
accuracy_score(y_test, sgd_clf.predict(X_test))   # 0.874

sgd_clf_with_pca = SGDClassifier(random_state=42)
%time sgd_clf_with_pca.fit(X_train_reduced, y_train)   # Wall time: 44.5 s
accuracy_score(y_test, sgd_clf_with_pca.predict(X_test_reduced))   # 0.8959
```

| 모델 | 원본 시간 | PCA 후 시간 | 원본 정확도 | PCA 후 정확도 |
|---|---|---|---|---|
| 랜덤 포레스트 | 57.7 s | 3 min 4 s (**느려짐**) | **0.9705** | 0.9488 |
| SGD | 3 min 8 s | 44.5 s (**약 4배 빨라짐**) | 0.874 | **0.8959** |

> 💡 결론: PCA는 **속도와 성능 모두 좋아질 수도, 둘 다 나빠질 수도** 있다. 모델과 데이터에 따라 다르므로 반드시 직접 비교해야 한다.
>
> (노트북 설명은 SGD 속도 향상을 "약 6배", "5배"라고 적었지만 출력된 시간 기준으로는 약 4배)

### 10번 — t-SNE로 MNIST 시각화

```python
X_sample, y_sample = X_train[:5000], y_train[:5000]   # 속도를 위해 5,000장만

tsne = TSNE(n_components=2, init="random", learning_rate="auto",
            random_state=42)
%time X_reduced = tsne.fit_transform(X_sample)        # Wall time: 1min 1s
```

```python
plt.figure(figsize=(13, 10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
            c=y_sample.astype(np.int8), cmap="jet", alpha=0.5)
plt.axis('off')
plt.colorbar()
plt.show()
```

| 줄 | 의미 |
|---|---|
| `y_sample.astype(np.int8)` | 레이블이 문자열 `'0'`~`'9'` → 색으로 쓰려면 숫자로 변환 |
| `cmap="jet"` | 숫자 0~9를 무지개색에 대응 |
| `alpha=0.5` | 점이 겹치는 부분을 보이게 반투명 |

- t-SNE는 **레이블을 전혀 보지 않았는데도** 숫자별로 군집이 잘 분리됨
- 다만 **3과 5**, **4와 9**는 많이 겹친다

**4와 9만 따로 다시 t-SNE**

```python
idx = (y_sample == '4') | (y_sample == '9')   # 불리언 OR
X_subset = X_sample[idx]
y_subset = y_sample[idx]

tsne_subset = TSNE(n_components=2, init="random", learning_rate="auto",
                   random_state=42)
X_subset_reduced = tsne_subset.fit_transform(X_subset)
```

- 두 숫자만 넣고 다시 돌리면 분리가 더 좋아짐 (공간을 두 클래스에 다 쓸 수 있으므로)

**점 대신 숫자·이미지로 그리는 함수**

```python
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.04, images=None, figsize=(13, 10)):
    X_normalized = MinMaxScaler().fit_transform(X)   # 좌표를 0~1로
    neighbors = np.array([[10., 10.]])               # 이미 그린 위치 목록 (가짜 점 1개로 시작)
    plt.figure(figsize=figsize)
    cmap = plt.cm.jet
    digits = np.unique(y)
    for digit in digits:                             # 1) 배경 산점도
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1],
                    c=[cmap(float(digit) / 9)], alpha=0.5)
    plt.axis("off")
    ax = plt.gca()
    for index, image_coord in enumerate(X_normalized):   # 2) 겹치지 않게 라벨 얹기
        closest_distance = np.linalg.norm(neighbors - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(float(y[index]) / 9),
                         fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"),
                                          image_coord)
                ax.add_artist(imagebox)
```

| 줄 | 의미 |
|---|---|
| `MinMaxScaler` | 기법마다 좌표 범위가 다름 → 0~1로 맞춰야 `min_distance=0.04`가 같은 의미 |
| `neighbors = [[10, 10]]` | 화면 밖의 가짜 점. 처음부터 배열이 비어 있지 않게 해서 `if` 분기를 없앰 |
| `cmap(float(digit) / 9)` | 0~9를 0~1로 바꿔 컬러맵에서 색 하나를 꺼냄. `[...]`로 감싸야 단색으로 인식 |
| `np.linalg.norm(..., axis=1).min()` | 이미 그린 모든 위치와의 거리 중 최솟값 |
| `np.r_[a, [b]]` | 행 방향으로 이어 붙이기 (`np.vstack`과 같음) |
| `AnnotationBbox(OffsetImage(...), 좌표)` | 그래프의 특정 좌표에 작은 이미지를 붙임 |

```python
plot_digits(X_reduced, y_sample)                                   # 숫자로
plot_digits(X_reduced, y_sample, images=X_sample, figsize=(35, 25)) # 이미지로
plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))  # 4·9만
```

- 4·9 그림에서 **기울어진 4와 9가 위쪽에 모임** → 일부 4는 실제로 9처럼 생겼다

**다른 기법과 비교 (같은 5,000장)**

```python
pca = PCA(n_components=2, random_state=42)
X_pca_reduced = pca.fit_transform(X_sample)

lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_lle_reduced = lle.fit_transform(X_sample)

pca_lle = make_pipeline(PCA(n_components=0.95),
                        LocallyLinearEmbedding(n_components=2, random_state=42))
X_pca_lle_reduced = pca_lle.fit_transform(X_sample)

X_mds_reduced = MDS(n_components=2, normalized_stress=False,
                    random_state=42).fit_transform(X_sample)

pca_mds = make_pipeline(PCA(n_components=0.95, random_state=42),
                        MDS(n_components=2, normalized_stress=False, random_state=42))
X_pca_mds_reduced = pca_mds.fit_transform(X_sample)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda_reduced = lda.fit_transform(X_sample, y_sample)   # 레이블 사용!
```

| 기법 | 시간 (Wall) | 결과 |
|---|---|---|
| PCA | 0.37 s | 엄청 빠르지만 군집이 심하게 겹침 |
| LLE | 3.3 s | 느리고 결과도 나쁨 |
| PCA → LLE | 4.2 s | 결과는 비슷하게 나쁨, 조금 빨라짐 |
| MDS | **13 min 15 s** | 군집이 너무 겹침 |
| PCA → MDS | 13 min 13 s | 같은 결과, 빨라지지도 않음 |
| LDA | 3.3 s | 빠르고 처음엔 좋아 보이지만 여러 군집이 크게 겹침 |
| **t-SNE** | 1 min 1 s | **가장 잘 분리됨** 🏆 |

> 💡 **시각화 목적이면 t-SNE**(또는 UMAP). PCA는 빠른 전처리·압축용. LDA는 레이블이 있을 때 분류 전처리로 고려.

---

## 8. 전체 흐름 요약

```
고차원 데이터 (차원의 저주: 희박, 멀리 떨어짐)      ← 7.1
   │
   ├─ 투영 (부분 공간에 수직으로)                    ← 7.2
   │    ├─ PCA: SVD → Vt[:d] → X_centered @ W_d     ← 7.3
   │    │    ├─ n_components=0.95 / 엘보 / 하이퍼파라미터 튜닝
   │    │    ├─ inverse_transform → 압축·재구성 오차
   │    │    ├─ svd_solver="randomized"  (빠름)
   │    │    └─ IncrementalPCA + memmap  (메모리 초과)
   │    └─ 랜덤 투영: JL 보조정리로 d 결정           ← 7.4
   │         └─ Gaussian / Sparse(더 빠름)
   │
   └─ 매니폴드 학습 (말린 것을 펼치기)               ← 7.2
        ├─ LLE: 이웃 선형 관계 보존                  ← 7.5
        └─ MDS · Isomap · t-SNE · LDA · 커널 PCA     ← 7.6
```

### 기법 비교

| 기법 | 종류 | 새 데이터 `transform` | 역변환 | 주 용도 |
|---|---|---|---|---|
| PCA | 선형 투영 | ✅ | ✅ | 전처리, 압축, 노이즈 제거 |
| 랜덤 PCA / 점진적 PCA | 선형 투영 | ✅ | ✅ | 대규모 데이터 |
| 랜덤 투영 | 선형 투영 | ✅ | △ (`pinv`, 느림) | 초고차원 데이터 |
| 커널 PCA | 비선형 | ✅ | △ (`fit_inverse_transform`) | 비선형 구조 |
| LDA | 선형 (지도) | ✅ | ❌ | 분류 전처리 |
| LLE | 매니폴드 | ✅ (제한적) | ❌ | 매니폴드 펼치기 |
| Isomap | 매니폴드 | ✅ | ❌ | 매니폴드 펼치기 |
| MDS | 거리 보존 | ❌ | ❌ | 시각화 (느림) |
| t-SNE | 국부 구조 | ❌ | ❌ | **시각화** |

### 주요 결과

| 실험 | 값 |
|---|---|
| 3D 타원 → 2D PCA, 설명된 분산 | 82.3% + 10.8% (손실 6.9%) |
| MNIST 분산 95% 보존 차원 | **154** / 784 |
| PCA + 랜덤 포레스트 최적 차원 (랜덤 서치) | 57 |
| PCA + SGD 최적 차원 (그리드 서치) | 75 |
| JL 최소 차원 (m=5,000, ε=0.1) | 7,300 |
| 랜덤 투영 transform: Gaussian vs Sparse | 43.2 s vs 5.77 s |
| RF 원본 → PCA | 0.9705 → 0.9488, 57.7 s → 3 min 4 s |
| SGD 원본 → PCA | 0.874 → 0.8959, 3 min 8 s → 44.5 s |

### 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `X - X.mean(axis=0)` | 중앙 정렬 | 3-1 |
| `np.linalg.svd(X)` → `Vt[:d].T` | 주성분 행렬 W_d | 3-1, 3-2 |
| `np.argmax(np.cumsum(r) >= 0.95) + 1` | 누적 비율이 기준을 처음 넘는 개수 | 3-5 |
| `PCA(n_components=0.95)` | 분산 비율로 차원 자동 선택 | 3-5 |
| `make_pipeline(PCA(), 모델)` + `"pca__n_components"` | 차원 수를 하이퍼파라미터로 | 3-5 |
| `pca.inverse_transform` | 압축 후 복원 | 3-6 |
| `np.array_split` + `partial_fit` | 미니배치 학습 | 3-8 |
| `np.memmap(..., mode="readonly").reshape(-1, n)` | 디스크 데이터를 배열처럼 | 3-8 |
| `pca.transform(X_test)` | 테스트는 `fit` 없이 변환만 | 7 (9번) |
| `(y == 'a') \| (y == 'b')` | 불리언 OR 마스크 | 7 (10번) |
