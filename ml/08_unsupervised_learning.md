# 8장. 비지도 학습

[← 7장. 차원 축소](07_dimensionality_reduction.md) · [📚 전체 목차](../README.md) · 9장. 인공 신경망 (정리 예정) →

> 📘 **교재** 『핸즈온 머신러닝 with 사이킷런, 파이토치』 8장 · 노트북 `08_unsupervised_learning.ipynb`
>
> 지금까지는 정답(레이블) `y`가 있는 **지도 학습**이었다. 현실 데이터 대부분은 레이블이 없다. 8장은 `X`만으로 구조를 찾는 **비지도 학습** 중 두 가지를 다룬다.
>
> - **군집(clustering)**: 비슷한 샘플끼리 묶기 — k-평균, DBSCAN
> - **가우스 혼합(Gaussian mixture)**: 데이터가 여러 개의 정규분포에서 나왔다고 보고 그 분포를 추정 — 군집 + 밀도 추정 + 이상치 탐지
>
> ⬅️ **앞에서** 1장 "머신러닝 시스템의 종류"에서 이름만 나온 군집·이상치 탐지를 직접 구현한다. 7장 차원 축소에 이어 비지도 학습의 두 번째 축.  
> ➡️ **다음으로** 사이킷런 파트를 마치고 **인공 신경망과 파이토치**로 넘어간다. 4장의 경사 하강법이 신경망 학습의 출발점이 된다.

## 이 장의 흐름

| 단계 | 절 | 핵심 질문 | 결과물 |
|---|---|---|---|
| 1. 군집 기초 | 8.1 | 레이블 없이 비슷한 샘플을 어떻게 묶는가 | `KMeans`, 이너셔, k-평균++, 미니배치 |
| 2. 군집 수·한계 | 8.1 | 군집은 몇 개가 적당하고, 언제 실패하는가 | 엘보, 실루엣 점수, k-평균의 한계 |
| 3. 군집 활용 | 8.1 | 군집을 어디에 쓰는가 | 이미지 분할, 준지도 학습(레이블 전파) |
| 4. 밀도 기반 군집 | 8.1 | 모양이 제각각인 군집은 | `DBSCAN` |
| 5. 가우스 혼합 | 8.2 | 데이터를 만든 분포를 추정할 수 있는가 | `GaussianMixture`(EM), `covariance_type` |
| 6. 이상치·모델 선택 | 8.2 | 밀도로 이상치를 찾고 군집 수를 고르려면 | 밀도 임곗값, BIC·AIC, 베이즈 가우스 혼합 |

## 데이터

| 데이터 | 형태 | 사용 절 | 비고 |
|---|---|---|---|
| `load_iris()` | `(150, 4)` | 8.1 | 분류 vs 군집 비교 |
| `make_blobs()` 5개 덩어리 | `(2000, 2)` | 8.1 | k-평균 기본 실습 |
| `make_blobs()` 길쭉한 덩어리 3개 | `(1250, 2)` | 8.1, 8.2 | k-평균의 한계, 가우스 혼합 |
| `ladybug.png` | `(533, 800, 3)` | 8.1 | 이미지 색상 분할 |
| `load_digits()` | `(1797, 64)` — 8×8 손글씨 | 8.1 | 준지도 학습 |
| `make_moons()` | `(1000, 2)` | 8.1, 8.2 | DBSCAN, 베이즈 가우스 혼합 |

## 목차

- [0. 준비](#0-준비)
- [1. 군집 (8.1)](#1-군집-81)
   - [1-1. 분류 vs 군집](#1-1-분류-vs-군집)
   - [1-2. k-평균 훈련과 예측](#1-2-k-평균-훈련과-예측)
   - [1-3. k-평균 알고리즘의 동작](#1-3-k-평균-알고리즘의-동작)
   - [1-4. 이너셔와 다중 초기화](#1-4-이너셔와-다중-초기화)
   - [1-5. k-평균++ · 엘칸 · 미니배치](#1-5-k-평균--엘칸--미니배치)
   - [1-6. 최적의 군집 수 찾기](#1-6-최적의-군집-수-찾기)
   - [1-7. k-평균의 한계](#1-7-k-평균의-한계)
   - [1-8. 활용 ① 이미지 분할](#1-8-활용--이미지-분할)
   - [1-9. 활용 ② 준지도 학습](#1-9-활용--준지도-학습)
   - [1-10. DBSCAN](#1-10-dbscan)
   - [1-11. 다른 군집 알고리즘 (보너스)](#1-11-다른-군집-알고리즘-보너스)
- [2. 가우스 혼합 (8.2)](#2-가우스-혼합-82)
   - [2-1. GaussianMixture 훈련](#2-1-gaussianmixture-훈련)
   - [2-2. 예측 · 샘플링 · 밀도 추정](#2-2-예측--샘플링--밀도-추정)
   - [2-3. 공분산 제약 (covariance_type)](#2-3-공분산-제약-covariance_type)
   - [2-4. 가우스 혼합으로 이상치 탐지](#2-4-가우스-혼합으로-이상치-탐지)
   - [2-5. 군집 수 선택: BIC · AIC](#2-5-군집-수-선택-bic--aic)
   - [2-6. 베이즈 가우스 혼합](#2-6-베이즈-가우스-혼합)
- [3. 정리](#3-정리)
   - [3-1. 전체 흐름](#3-1-전체-흐름)
   - [3-2. 개념과 코드 대응](#3-2-개념과-코드-대응)
   - [3-3. 알고리즘 비교](#3-3-알고리즘-비교)
   - [3-4. 주요 결과](#3-4-주요-결과)
   - [3-5. 자주 쓴 코드 패턴](#3-5-자주-쓴-코드-패턴)
   - [3-6. 핵심 원칙](#3-6-핵심-원칙)

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

모든 장 노트북 맨 앞에 반복되는 환경 확인 + 그래프 글꼴 설정. 각 줄의 의미는 [1장 0. 준비](01_the_machine_learning_landscape.md#0-준비) 참고.

---

## 1. 군집 (8.1)

**군집** = 레이블 없이 비슷한 샘플을 같은 그룹(클러스터)으로 묶는 작업.

| 활용 | 예 |
|---|---|
| 고객 세분화 | 구매 이력이 비슷한 고객끼리 묶어 마케팅·추천 |
| 데이터 분석 | 새 데이터셋을 군집으로 나눠 군집별로 분석 |
| 차원 축소 | 각 샘플을 "군집별 친밀도" 벡터로 바꿈 (k차원) |
| 특성 공학 | 2장의 `ClusterSimilarity` — 위경도 군집 유사도를 특성으로 추가 |
| 이상치 탐지 | 어느 군집과도 친밀도가 낮은 샘플 = 이상치 |
| 준지도 학습 | 레이블 몇 개를 같은 군집 전체에 전파 |
| 검색 엔진 | 비슷한 이미지 찾기 — 같은 군집 이미지 반환 |
| 이미지 분할 | 비슷한 색 픽셀끼리 묶기 |

"비슷하다"의 정의는 알고리즘마다 다르다. k-평균은 **중심점(센트로이드)까지의 거리**, DBSCAN은 **밀집 정도**, 가우스 혼합은 **확률 분포**로 판단한다.

### 1-1. 분류 vs 군집

```python
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

plt.figure(figsize=(9, 3.5))

plt.subplot(121)                       # 왼쪽: 레이블이 있는 경우 (분류)
plt.plot(X[y==0, 2], X[y==0, 3], "yo", label="Iris setosa")
plt.plot(X[y==1, 2], X[y==1, 3], "bs", label="Iris versicolor")
plt.plot(X[y==2, 2], X[y==2, 3], "g^", label="Iris virginica")
plt.xlabel("Petal length"); plt.ylabel("Petal width")
plt.grid(); plt.legend()

plt.subplot(122)                       # 오른쪽: 레이블이 없는 경우 (군집)
plt.scatter(X[:, 2], X[:, 3], c="k", marker=".")
plt.xlabel("Petal length")
plt.tick_params(labelleft=False)
plt.show()
```

| 코드 | 의미 |
|---|---|
| `X[y==0, 2]` | 불리언 마스크로 setosa 행만 고르고, 2번 열(꽃잎 길이)만 꺼냄 |
| `"yo"`, `"bs"`, `"g^"` | 색+마커 단축 표기: 노란 원, 파란 사각형, 초록 삼각형 |
| `c="k"` | 모든 점을 검정으로 — 레이블 정보를 버린 상태 |

- 왼쪽(분류): 세 품종이 색으로 구분됨
- 오른쪽(군집): 같은 데이터인데 레이블이 없다. 왼쪽 아래 덩어리(setosa)는 쉽게 보이지만 오른쪽 위는 두 품종이 섞여 하나처럼 보인다

그런데 꽃잎 2개가 아니라 **특성 4개를 모두** 쓰면 군집으로도 꽤 잘 나눌 수 있다.

```python
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture

y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)

mapping = {}
for class_id in np.unique(y):
    mode, _ = stats.mode(y_pred[y==class_id])   # 이 품종 샘플들이 가장 많이 배정된 군집 번호
    mapping[mode] = class_id                    # 군집 번호 → 품종 번호

y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

(y_pred==y).sum() / len(y_pred)
# np.float64(0.9666666666666667)
```

| 코드 | 의미 |
|---|---|
| `GaussianMixture(n_components=3)` | 가우스 혼합으로 군집 3개 (2절에서 자세히) |
| `.fit(X).predict(X)` | 메서드 연결. `fit()`이 자기 자신을 반환하므로 바로 `predict()` 가능 |
| `stats.mode(배열)` | 최빈값과 그 개수를 반환. `mode, _`로 최빈값만 받음 |
| `mapping` | **군집 번호는 임의**다. 군집 0이 setosa라는 보장이 없으므로, 품종별로 가장 많이 몰린 군집 번호를 찾아 번호를 맞춰 줌 |

> 💡 레이블을 **한 번도 보지 않고** 묶었는데 96.7%가 실제 품종과 일치. 군집 결과를 정답과 비교하려면 이렇게 **번호 매핑**이 먼저 필요하다.

### 1-2. k-평균 훈련과 예측

**k-평균(k-means)**: 군집 중심 k개를 찾고, 각 샘플을 가장 가까운 중심의 군집에 배정한다. 가장 빠르고 단순한 군집 알고리즘.

#### 데이터 만들기와 훈련

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

blob_centers = np.array([[ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
                         [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers, cluster_std=blob_std,
                  random_state=7)

k = 5
kmeans = KMeans(n_clusters=k, random_state=43)
y_pred = kmeans.fit_predict(X)
```

| 코드 | 의미 |
|---|---|
| `make_blobs(...)` | 지정한 중심 주변에 정규분포로 점을 뿌려 덩어리(blob)를 만드는 함수 |
| `centers=blob_centers` | 덩어리 5개의 중심 좌표 |
| `cluster_std=blob_std` | 덩어리별 표준편차. 0.4 · 0.3은 넓게, 0.1 세 개는 촘촘하게 |
| `n_clusters=k` | **찾을 군집 수를 사람이 정해야 함** (k-평균의 큰 단점) |
| `fit_predict(X)` | 훈련 + 훈련 데이터의 군집 번호 반환을 한 번에 |

`y`(진짜 덩어리 번호)는 만들기만 하고 훈련에는 쓰지 않는다 — 비지도 학습이니까.

```python
def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$", rotation=0)

plt.figure(figsize=(8, 4))
plot_clusters(X)       # y를 안 넘기면 한 색으로 그림 → 레이블 없는 데이터
plt.show()
```

- `c=y` : 점 색을 배열 값으로 지정. `None`이면 기본색
- `"$x_1$"` : 달러 기호로 감싸면 수식(아래첨자)으로 표시

#### 결과 확인

```python
y_pred
# array([4, 0, 1, ..., 2, 1, 0], dtype=int32)

y_pred is kmeans.labels_
# True

kmeans.cluster_centers_
# array([[-2.80389616,  1.80117999],
#        [ 0.20876306,  2.25551336],
#        [-2.79290307,  2.79641063],
#        [-1.46679593,  2.28585348],
#        [-2.80037642,  1.30082566]])
```

| 속성 | 의미 |
|---|---|
| `labels_` | 훈련 샘플마다 배정된 **군집 번호**. 여기서 "레이블"은 정답이 아니라 **예측 결과**라는 점에 주의 |
| `y_pred is kmeans.labels_` | `is` = 같은 객체인지 비교. `fit_predict()`가 `labels_`를 그대로 반환했다는 뜻 |
| `cluster_centers_` | 찾아낸 중심 5개의 좌표 `(k, 특성 수)` |

찾은 중심이 처음 만든 `blob_centers`와 거의 같다 (순서만 다름). 예: `[-2.80, 1.80]` ≈ `[-2.8, 1.8]`.

#### 새 샘플 예측

```python
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)
# array([1, 1, 2, 2], dtype=int32)
```

- 각 샘플을 **가장 가까운 중심**의 군집으로 배정
- `[3, 2]`는 어느 덩어리와도 멀지만 그래도 가장 가까운 군집 1로 배정된다 → 이것이 **하드 군집**의 한계

#### 결정 경계 그리기 (보로노이 다이어그램)

```python
def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)

def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]   # 가중치 작은 중심은 생략 (2절에서 사용)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=35, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)      # 흰 원
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=2, linewidths=12,
                color=cross_color, zorder=11, alpha=1)         # 그 위에 검은 X

def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
    # (축 레이블 처리 생략)

plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()
```

`plot_decision_boundaries()`는 이 장 내내 재사용하는 핵심 그림 함수다.

| 단계 | 코드 | 의미 |
|---|---|---|
| ① 범위 | `X.min(axis=0) - 0.1` | 열(특성)별 최솟값에서 여유 0.1 |
| ② 격자 | `np.meshgrid(linspace, linspace)` | 1000×1000 = 100만 개 격자점의 x좌표 판, y좌표 판 |
| ③ 펼치기 | `np.c_[xx.ravel(), yy.ravel()]` | 2차원 판을 1차원으로 펴서 열로 붙임 → `(1000000, 2)` 샘플 행렬 |
| ④ 예측 | `clusterer.predict(...)` | 격자점마다 군집 번호 |
| ⑤ 되접기 | `Z.reshape(xx.shape)` | 다시 1000×1000 판으로 |
| ⑥ 칠하기 | `contourf` / `contour` | 영역을 색으로 채우고(f = filled), 경계선을 검정으로 |

- `zorder` : 그리는 순서(클수록 위). 중심 표시가 점들에 가리지 않게 10, 11
- 결과 그림은 **보로노이 다이어그램**: 평면을 "어느 중심이 가장 가까운가"로 나눈 것. 경계선은 두 중심을 잇는 선분의 수직이등분선이다
- 경계 근처 몇몇 점은 잘못 배정됐을 수 있다. k-평균은 **군집 크기(퍼진 정도)가 다르면** 잘 못 맞춘다 — 거리만 보기 때문

#### 하드 군집 vs 소프트 군집

```python
kmeans.transform(X_new).round(2)
# array([[2.81, 0.33, 2.9 , 1.49, 2.89],
#        [5.81, 2.8 , 5.85, 4.48, 5.84],
#        [1.21, 3.29, 0.29, 1.69, 1.71],
#        [0.73, 3.22, 0.36, 1.55, 1.22]])
```

| 방식 | 출력 | 메서드 |
|---|---|---|
| **하드 군집** | 샘플마다 군집 번호 1개 | `predict()` |
| **소프트 군집** | 샘플마다 군집별 점수 k개 | `transform()` — 중심까지의 거리 |

- 행 = 샘플 4개, 열 = 중심 5개까지의 **유클리드 거리**
- 첫 샘플 `[0, 2]`는 중심 1까지 0.33으로 가장 가까움 → `predict()` 결과 1과 일치
- 두 번째 샘플 `[3, 2]`는 가장 가까운 중심도 2.8 → 어느 군집에도 잘 속하지 않는다는 정보가 남는다

> 💡 `transform()`은 **차원 축소**로도 쓸 수 있다. 특성이 몇 개든 결과는 k차원. 고차원 데이터를 k개 "군집 거리" 특성으로 바꿔 다른 모델의 입력으로 쓰는 식.

정말 유클리드 거리인지 직접 계산해서 확인:

```python
np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2)
               - kmeans.cluster_centers_, axis=2).round(2)
# 위와 같은 결과
```

| 단계 | 형태 | 의미 |
|---|---|---|
| `np.tile(X_new, (1, k))` | `(4, 10)` | 각 행을 가로로 5번 복사 |
| `.reshape(-1, k, 2)` | `(4, 5, 2)` | "샘플 4 × 중심 5 × 좌표 2" 3차원으로 |
| `- cluster_centers_` | `(4, 5, 2)` | `(5, 2)`가 브로드캐스팅되어 샘플마다 중심 5개를 뺌 |
| `np.linalg.norm(..., axis=2)` | `(4, 5)` | 마지막 축(좌표)으로 √(Δx² + Δy²) |

### 1-3. k-평균 알고리즘의 동작

레이블도 중심도 모르는데 어떻게 찾을까? 둘 중 하나를 알면 나머지는 쉽다는 점을 이용한다.

1. 중심 k개를 **무작위로** 놓는다 (예: 샘플 k개를 골라 그 위치에)
2. 수렴할 때까지(중심이 더 이상 안 움직일 때까지) 반복
   1. 각 샘플을 **가장 가까운 중심**에 배정 (레이블 갱신)
   2. 각 중심을 **배정된 샘플들의 평균**으로 이동 (중심 갱신)

- 이 과정은 반복할 때마다 이너셔(뒤에서 설명)가 줄어들기 때문에 **반드시 수렴**한다
- 계산 복잡도는 샘플 수 m, 군집 수 k, 특성 수 n에 대해 대체로 **선형**. 단, 데이터에 군집 구조가 있을 때 얘기이고 최악의 경우 기하급수적일 수 있지만 실제로는 드물다

```python
kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=1, random_state=18)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=2, random_state=18)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1, max_iter=3, random_state=18)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)
# → 3행 2열 그림: 왼쪽 = 중심 갱신, 오른쪽 = 레이블 갱신
```

| 하이퍼파라미터 | 의미 |
|---|---|
| `init="random"` | 원래 k-평균처럼 중심을 무작위로 초기화 (기본값은 k-평균++) |
| `n_init=1` | 초기화를 1번만 |
| `max_iter=1, 2, 3` | 반복 횟수를 일부러 제한 → 중심이 이동하는 과정을 단계별로 관찰 |
| `random_state=18` | 세 모델이 **같은 초기 중심**에서 출발하도록 고정 |

그림을 보면 1회차에 엉망이던 중심이 2~3회차 만에 거의 제자리를 찾는다.

#### 초기화에 따른 변동성

```python
def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)
    plt.figure(figsize=(10, 3.2))
    plt.subplot(121); plot_decision_boundaries(clusterer1, X); plt.title(title1)
    plt.subplot(122); plot_decision_boundaries(clusterer2, X, show_ylabels=False); plt.title(title2)

kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1, random_state=2)
kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1, random_state=8)

plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
                          "Solution 1", "Solution 2 (with a different random init)")
```

- 수렴은 보장되지만 **최적해(전역 최소)로의 수렴은 보장되지 않는다**
- 초기 중심 위치에 따라 덩어리 하나를 둘로 쪼개고, 두 덩어리를 하나로 합친 **지역 최적해**에 빠질 수 있다

해결책 ① — 중심의 대략적 위치를 알면 직접 지정:

```python
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, random_state=42)
kmeans.fit(X)
```

- `init`에 `(k, 특성 수)` 배열을 넘기면 그 위치에서 시작
- 예: 다른 군집 알고리즘으로 대략 돌린 결과를 초기값으로 쓰는 식

### 1-4. 이너셔와 다중 초기화

정답이 없으니 정확도는 못 잰다. 대신 **각 샘플과 자기 중심 사이의 거리**를 본다.

> **이너셔(inertia)** = 각 샘플과 가장 가까운 중심 사이 거리의 **제곱합**. 작을수록 군집이 촘촘하다.

```python
kmeans.inertia_             # 좋은 초기값 지정
# 211.59853725816828
kmeans_rnd_init1.inertia_   # 무작위 초기화, Solution 1
# 219.58201503602285
kmeans_rnd_init2.inertia_   # 무작위 초기화, Solution 2 (나쁜 해)
# 600.3600713094229
```

그림에서 나쁘게 보였던 Solution 2는 이너셔가 약 3배. 이너셔로 해의 좋고 나쁨을 비교할 수 있다.

직접 계산해서 확인:

```python
X_dist = kmeans.transform(X)
(X_dist[np.arange(len(X_dist)), kmeans.labels_] ** 2).sum()
# np.float64(211.59853725816882)
```

| 코드 | 의미 |
|---|---|
| `kmeans.transform(X)` | `(2000, 5)` — 모든 샘플과 모든 중심 사이 거리 |
| `X_dist[np.arange(2000), labels_]` | **팬시 인덱싱**. i번째 행에서 `labels_[i]`번째 열 → 자기 군집 중심까지 거리만 뽑음 `(2000,)` |
| `** 2).sum()` | 제곱해서 합 = 이너셔 |

끝자리가 조금 다른 건 부동소수점 계산 순서 차이.

```python
kmeans.score(X)
# -211.59853725816828
```

- `score()`는 **음수 이너셔**를 반환. 사이킷런은 "점수는 클수록 좋다" 규칙을 지키기 때문 (2장의 `neg_root_mean_squared_error`와 같은 이유)

#### 다중 초기화

해결책 ② — 무작위 초기화로 여러 번 돌려서 **이너셔가 가장 작은 해**를 고른다.

```python
kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10, random_state=2)
kmeans_rnd_10_inits.fit(X)

kmeans_rnd_10_inits.inertia_
# 211.59853725816828
```

- `n_init=10` : 알고리즘 전체를 10번 실행하고 최선을 선택
- 결과가 처음에 좋은 초기값을 직접 줬을 때와 **똑같은 211.60** → k=5일 때의 최적해로 보인다

| `init` 설정 | `n_init` 기본값 |
|---|---|
| `"k-means++"` (기본) 또는 배열 | 1 |
| `"random"` 또는 함수 | 10 |

### 1-5. k-평균++ · 엘칸 · 미니배치

#### k-평균++ 초기화 (기본값)

중심을 완전히 무작위로 두지 않고 **서로 멀리 떨어지게** 고르는 방법 (Arthur & Vassilvitskii, 2006).

1. 첫 중심 c₁을 데이터에서 무작위로 하나 고른다
2. 다음 중심은 샘플 xᵢ를 확률 **D(xᵢ)² / Σⱼ D(xⱼ)²** 로 고른다
   - D(xᵢ) = xᵢ와 **이미 고른 중심 중 가장 가까운 것**까지의 거리
   - 기존 중심에서 먼 샘플일수록 뽑힐 확률이 높다
3. k개를 고를 때까지 2를 반복
4. 이후는 일반 k-평균과 같음

- 초기화가 조금 복잡해지지만 지역 최적해에 빠질 확률이 크게 줄어 `n_init`을 1로 줄일 수 있다 → 전체적으로 이득
- `KMeans(init="k-means++")` — **기본값이라 따로 안 써도 됨**

#### 엘칸(Elkan) 알고리즘

- **삼각 부등식**(AC ≤ AB + BC)과 거리의 상한·하한을 추적해 불필요한 거리 계산을 건너뜀
- `KMeans(algorithm="elkan")`. 기본값은 `"lloyd"`(일반 k-평균)
- 데이터에 따라 오히려 느려질 수도 있어 기본값은 아님

#### 미니배치 k-평균

```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)

minibatch_kmeans.inertia_
# 211.65945105712612
```

- 반복마다 전체 데이터가 아니라 **미니배치**로 중심을 조금씩 이동 (Sculley, 2010)
- 일반 k-평균보다 몇 배 빠르고, 메모리에 안 들어가는 대용량 데이터도 가능
- 이너셔는 211.66 vs 211.60으로 **약간 나쁘다**. k가 커질수록 차이가 벌어진다
- `partial_fit()`으로 데이터를 조각조각 넣을 수도 있다

#### memmap과 함께 쓰기 (교재 외 보너스)

데이터가 메모리에 안 들어갈 때 7장 점진적 PCA처럼 `memmap`을 쓴다.

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train = mnist.data[:60000], mnist.target[:60000]
X_test, y_test = mnist.data[60000:], mnist.target[60000:]

filename = "my_mnist.mmap"
X_memmap = np.memmap(filename, dtype='float32', mode='write', shape=X_train.shape)
X_memmap[:] = X_train          # 디스크 파일에 기록
X_memmap.flush()               # 버퍼를 디스크로 확실히 내보냄

minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
minibatch_kmeans.fit(X_memmap)
```

| 코드 | 의미 |
|---|---|
| `np.memmap(...)` | 디스크 파일을 배열처럼 다루는 객체. 필요한 부분만 메모리로 읽음 |
| `mode='write'` | 새 파일 생성 (`'w+'`와 같음) |
| `X_memmap[:] = X_train` | `[:]`로 **내용을 복사**. `X_memmap = X_train`이면 변수만 바뀌고 파일엔 안 써짐 |
| `batch_size=10` | 미니배치 크기 |

### 1-6. 최적의 군집 수 찾기

지금은 데이터를 직접 만들어서 k=5인 걸 알지만, 보통은 모른다. k를 잘못 고르면:

```python
kmeans_k3 = KMeans(n_clusters=3, random_state=42)
kmeans_k8 = KMeans(n_clusters=8, random_state=42)
plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")

kmeans_k3.inertia_   # 653.2167190021554
kmeans_k8.inertia_   # 127.13141880461835
```

- k=3: 덩어리 여러 개가 합쳐짐 / k=8: 덩어리가 쪼개짐. 둘 다 나쁘다
- 그런데 이너셔는 k=8이 훨씬 작다 → **이너셔를 최소화하는 k를 고르면 안 된다**
- k가 커질수록 각 샘플이 중심에 가까워지니 이너셔는 계속 줄어든다 (극단적으로 k = 샘플 수이면 0)

#### 방법 ① 엘보(elbow)

```python
kmeans_per_k = [KMeans(n_clusters=k, random_state=43).fit(X)
                for k in range(1, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]

plt.figure(figsize=(8, 3.5))
plt.plot(range(1, 10), inertias, "bo-")
plt.xlabel("$k$"); plt.ylabel("Inertia")
plt.annotate("", xy=(4, inertias[3]), xytext=(4.45, 650),
             arrowprops=dict(facecolor='black', shrink=0.1))   # 화살표만 그림
plt.text(4.5, 650, "Elbow", horizontalalignment="center")
plt.axis([1, 8.5, 0, 1300])
plt.show()
```

| 코드 | 의미 |
|---|---|
| 리스트 컴프리헨션 + `.fit(X)` | k=1~9 모델 9개를 한 줄로 훈련. `fit()`이 모델을 반환하므로 리스트에 바로 들어감 |
| `inertias[3]` | 인덱스 3 = k=4 (k가 1부터 시작하므로 인덱스 = k − 1) |
| `plt.annotate("", xy, xytext, arrowprops)` | 텍스트 없이 xytext → xy 화살표만 |

- 이너셔가 급격히 떨어지다가 **k=4에서 꺾인다** (팔꿈치)
- k=4보다 작으면 나쁘고, 크면 큰 도움이 안 된다는 뜻 → k=4가 괜찮은 선택
- 단, 실제로는 왼쪽 아래 덩어리 두 개가 하나로 묶인다. 엘보는 **대략적인** 방법

```python
plot_decision_boundaries(kmeans_per_k[4 - 1], X)   # k=4 결과 확인
```

#### 방법 ② 실루엣 점수

**실루엣 계수** = (b − a) / max(a, b)

| 기호 | 의미 |
|---|---|
| a | 같은 군집 다른 샘플까지의 **평균 거리** (군집 내 거리) |
| b | **가장 가까운 다른 군집** 샘플까지의 평균 거리 |

| 값 | 해석 |
|---|---|
| +1에 가까움 | 자기 군집 안에 잘 있고 다른 군집과 멀다 |
| 0 근처 | 군집 경계에 있다 |
| −1에 가까움 | 잘못된 군집에 배정됐을 가능성 |

**실루엣 점수** = 모든 샘플의 실루엣 계수 평균.

```python
from sklearn.metrics import silhouette_score

silhouette_score(X, kmeans.labels_)
# np.float64(0.655517642572828)        ← k=5 모델

silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]       # k=1은 제외
plt.plot(range(2, 10), silhouette_scores, "bo-")
```

- `kmeans_per_k[1:]` : 군집이 1개면 b를 정의할 수 없어 실루엣 계산 불가 → k=2부터
- 그래프: **k=4가 가장 높고(약 0.69) k=5도 0.66으로** 높다. 엘보 방법보다 정보가 많다

  | k | 2 | 3 | **4** | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|
  | 실루엣 점수 | 0.595 | 0.572 | **0.689** | 0.656 | 0.594 | 0.550 | 0.562 | 0.553 |
- 단점: 모든 샘플 쌍 거리가 필요해 계산 비용이 크다 (O(m²))

#### 방법 ③ 실루엣 다이어그램

모든 샘플의 실루엣 계수를 **군집별로 정렬해서** 칼 모양으로 그린 것.

```python
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter

plt.figure(figsize=(11, 9))
for k in (3, 4, 5, 6):
    plt.subplot(2, 2, k - 2)

    y_pred = kmeans_per_k[k - 1].labels_
    silhouette_coefficients = silhouette_samples(X, y_pred)   # 샘플별 계수 (2000,)

    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]   # 군집 i의 계수만
        coeffs.sort()                                   # 오름차순 정렬 → 칼 모양
        color = plt.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)            # y축 눈금 = 칼 가운데
        pos += len(coeffs) + padding                    # 다음 칼 시작 위치

    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    plt.axvline(x=silhouette_scores[k - 2], color="red", linestyle="--")   # 평균 = 실루엣 점수
    plt.title(f"$k={k}$")
plt.show()
```

| 코드 | 의미 |
|---|---|
| `silhouette_samples()` | 평균 내기 전, **샘플별** 실루엣 계수 |
| `fill_betweenx(y, x1, x2)` | y 위치마다 x1~x2 사이를 가로로 칠함 → 가로 막대 수천 개가 모여 칼 모양 |
| `plt.cm.Spectral(i / k)` | 0~1 값을 컬러맵 색으로 |
| `FixedLocator` / `FixedFormatter` | y축 눈금 위치와 글자를 직접 지정 |
| `silhouette_scores[k - 2]` | 리스트가 k=2부터 시작하므로 인덱스 = k − 2 |

읽는 법:
- **칼의 높이** = 군집 크기, **칼의 너비** = 계수 크기
- 빨간 점선(평균)에 **못 미치는 칼**이 있으면 그 군집은 나쁜 군집
- k=4: 군집 하나(두 덩어리가 합쳐진 것)가 유난히 크다
- k=5: 모든 칼이 점선을 넘고 크기도 비슷 → **k=5가 가장 좋다**
- 실루엣 점수만 보면 k=4였는데, 다이어그램이 더 정확한 판단을 도와준다

### 1-7. k-평균의 한계

```python
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))   # 선형 변환 → 길쭉하게 기울임
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]                                          # 평행 이동
X = np.r_[X1, X2]                                          # 행 방향으로 이어 붙임
y = np.r_[y1, y2]

kmeans_good = KMeans(n_clusters=3,
                     init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]),
                     random_state=42)
kmeans_bad = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans_good.fit(X)
kmeans_bad.fit(X)
```

| 코드 | 의미 |
|---|---|
| `X1.dot(2×2 행렬)` | 모든 점에 같은 선형 변환 → 원형 덩어리가 **기울어진 타원**이 됨 |
| `np.r_[A, B]` | 세로로 쌓기 (`np.vstack`과 같은 효과) |

- 크기·밀도·방향이 다른 **타원형 군집 3개**
- 좋은 초기값을 줘도, `n_init=10`을 줘도 k-평균은 제대로 못 나눈다
- 심지어 **잘못 나눈 쪽의 이너셔가 더 낮다** — 좋은 초기값 쪽 약 2,242.6, 나쁜 쪽 약 2,179.6 (그림 제목에 표시됨) → 이너셔가 낮다고 좋은 군집이 아니다

| k-평균의 한계 | 이유 |
|---|---|
| 최적해 보장 없음 | 초기화에 따라 지역 최적해 → 여러 번 실행 필요 |
| k를 지정해야 함 | 엘보·실루엣으로 추정 |
| 크기·밀도가 다른 군집 | 거리만 보므로 경계가 중심들의 중간에 그어짐 |
| 원형이 아닌 군집 | 보로노이 경계는 직선 → 타원·초승달 모양 불가 |

> ⚠️ **k-평균 전에 특성 스케일을 맞춰야 한다.** 스케일이 다르면 군집이 한 방향으로 늘어나 결과가 나빠진다. 스케일링이 모든 문제를 해결하진 않지만 대체로 도움이 된다.
>
> 타원형 군집에는 **가우스 혼합**(2절)이 잘 맞는다.

### 1-8. 활용 ① 이미지 분할

**이미지 분할**: 이미지를 여러 영역(세그먼트)으로 나누는 작업.

| 종류 | 설명 |
|---|---|
| 색상 분할 | 비슷한 색 픽셀끼리 묶음 ← **여기서 하는 것** |
| 시맨틱 분할 | 같은 **종류**의 물체 픽셀끼리 (예: 보행자 전체) |
| 인스턴스 분할 | 같은 **개별 물체** 픽셀끼리 (보행자 각각) |

시맨틱·인스턴스 분할은 12장 CNN에서 다룬다. 여기선 k-평균으로 색상 분할.

```python
from pathlib import Path
import urllib.request

homlp_root = "https://github.com/ageron/handson-mlp/raw/main/"
filename = "ladybug.png"
filepath = Path(f"my_{filename}")
if not filepath.is_file():
    url = f"{homlp_root}/images/unsupervised_learning/{filename}"
    urllib.request.urlretrieve(url, filepath)

import PIL

image = np.asarray(PIL.Image.open(filepath))
image.shape
# (533, 800, 3)
```

- `PIL.Image.open()` → 이미지 객체, `np.asarray()` → 넘파이 배열
- `(높이, 너비, 채널)` = 533 × 800 픽셀, 채널 3개(R, G, B), 값 0~255

```python
X = image.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
segmented_img = kmeans.cluster_centers_[kmeans.labels_]
segmented_img = segmented_img.reshape(image.shape)
```

| 단계 | 형태 | 의미 |
|---|---|---|
| `image.reshape(-1, 3)` | `(426400, 3)` | 픽셀 하나 = RGB 3특성 샘플 하나. 위치 정보는 버리고 **색만** 봄 |
| `KMeans(n_clusters=8)` | — | 색 공간(3차원)에서 군집 8개 = 대표 색 8개 |
| `cluster_centers_` | `(8, 3)` | 대표 색 8개 (군집 내 평균 색) |
| `cluster_centers_[labels_]` | `(426400, 3)` | **팬시 인덱싱**: 픽셀마다 자기 군집의 대표 색으로 교체 |
| `.reshape(image.shape)` | `(533, 800, 3)` | 다시 이미지 모양으로 |

군집 수를 바꿔 가며 비교:

```python
segmented_imgs = []
n_colors = (10, 8, 6, 4, 2)
for n_clusters in n_colors:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

plt.subplot(2, 3, 1); plt.imshow(image); plt.title("Original image")
for idx, n_clusters in enumerate(n_colors):
    plt.subplot(2, 3, 2 + idx)
    plt.imshow(segmented_imgs[idx] / 255)   # 실수 배열은 0~1 범위여야 imshow가 제대로 그림
    plt.title(f"{n_clusters} colors")
```

- 색이 적을수록 단순해진다
- **군집 6개 미만**이 되면 무당벌레의 빨간색이 사라진다 → 빨간 픽셀은 이미지에서 차지하는 비율이 작아서 독립 군집을 얻지 못하고 주변 색과 합쳐짐
- k-평균은 **비슷한 크기의 군집**을 선호한다는 특성이 여기서도 드러난다
- `/ 255` : 중심 좌표는 실수(float)인데, `imshow`는 실수면 0~1, 정수면 0~255로 해석

### 1-9. 활용 ② 준지도 학습

**준지도 학습**: 레이블 없는 샘플은 많고 레이블 있는 샘플은 적을 때. 레이블 작업은 비싸므로(특히 전문가가 해야 하면) **어떤 샘플에 레이블을 달지**가 중요하다.

#### 기준선: 무작위 50개만 레이블

```python
from sklearn.datasets import load_digits

X_digits, y_digits = load_digits(return_X_y=True)
X_train, y_train = X_digits[:1400], y_digits[:1400]
X_test, y_test = X_digits[1400:], y_digits[1400:]
```

- `load_digits` : MNIST 축소판. 8×8 흑백 이미지 1,797장, 픽셀값 0~16
- `return_X_y=True` : `Bunch` 대신 `(X, y)` 튜플로 바로 받음

```python
from sklearn.linear_model import LogisticRegression

n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])   # 앞 50개만 사용
log_reg.score(X_test, y_test)
# 0.7581863979848866

log_reg_full = LogisticRegression(max_iter=10_000)
log_reg_full.fit(X_train, y_train)                      # 1,400개 전부 사용 (참고용 상한)
log_reg_full.score(X_test, y_test)
# 0.9093198992443325
```

- `max_iter=10_000` : 기본 100회로는 수렴 경고가 나서 넉넉히. `10_000`의 `_`는 자릿수 구분용 (값은 10000)
- 50개 → 75.8%, 1,400개 → 90.9%

#### 1단계: 대표 이미지 50개에 레이블

```python
k = 50
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)                  # (1400, 50) 거리
representative_digit_idx = X_digits_dist.argmin(axis=0)        # (50,)
X_representative_digits = X_train[representative_digit_idx]   # (50, 64)
```

| 코드 | 의미 |
|---|---|
| `fit_transform(X_train)` | 훈련 + 모든 샘플과 50개 중심 사이 거리 `(1400, 50)` |
| `.argmin(axis=0)` | **열(군집)마다** 거리가 가장 작은 **행(샘플) 인덱스** → 군집별로 중심에 가장 가까운 샘플 |
| `X_train[인덱스 배열]` | 그 50개 이미지 = **대표 이미지** |

> ⚠️ `axis=0`이 헷갈리기 쉬움. `axis=0`은 "행 방향으로 훑어 내려가며" 비교 → 결과가 열 개수(50)만큼. 샘플마다 가장 가까운 군집이 필요하면 `axis=1`(= `kmeans.labels_`와 같음).

대표 이미지를 그려 보고 **사람이 직접** 레이블을 단다:

```python
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)                      # 5행 10열
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary",
               interpolation="bilinear")                    # 8×8을 부드럽게 보간
    plt.axis('off')
plt.show()

y_representative_digits = np.array([
    8, 0, 1, 3, 6, 7, 5, 4, 2, 8,
    2, 3, 9, 5, 3, 9, 1, 7, 9, 1,
    4, 6, 9, 7, 5, 2, 2, 1, 3, 3,
    6, 0, 4, 9, 8, 1, 8, 4, 2, 4,
    2, 3, 9, 7, 8, 9, 6, 5, 6, 4,
])
```

> ⚠️ 이 레이블 배열은 **그림을 보고 손으로 입력한 값**이다. 사이킷런 버전이나 `random_state`가 달라 군집 순서가 바뀌면 이 배열도 다시 봐야 한다.

```python
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)
# 0.8337531486146096
```

- 똑같이 **50개만** 레이블했는데 75.8% → **83.4%**
- 무작위 샘플 대신 **대표 샘플**에 레이블을 다는 게 훨씬 효율적

#### 2단계: 레이블 전파

대표 이미지의 레이블을 **같은 군집의 모든 샘플**에 복사한다.

```python
y_train_propagated = np.empty(len(X_train), dtype=np.int64)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train, y_train_propagated)
log_reg.score(X_test, y_test)
# 0.8690176322418136
```

| 코드 | 의미 |
|---|---|
| `np.empty(1400, dtype=np.int64)` | 값을 초기화하지 않은 빈 배열 (어차피 전부 덮어씀) |
| `kmeans.labels_ == i` | 군집 i에 속한 샘플 마스크 |
| `= y_representative_digits[i]` | 그 샘플 전부에 군집 i 대표의 레이블 대입 |

- 레이블 1,400개로 훈련 → **86.9%**
- 문제: 군집 경계 근처 샘플은 잘못된 레이블을 받았을 가능성이 높다

#### 3단계: 부분 전파 — 중심에서 가까운 50%만

```python
percentile_closest = 50

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]   # 자기 중심까지 거리 (1400,)
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)     # 군집 i 거리의 중앙값
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1                       # 먼 절반은 -1로 표시

partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
```

| 코드 | 의미 |
|---|---|
| `X_digits_dist[np.arange(1400), labels_]` | 1-4절 이너셔 계산과 같은 팬시 인덱싱 — 샘플마다 자기 중심까지 거리 |
| `np.percentile(배열, 50)` | 50번째 백분위수 = 중앙값. **군집마다 따로** 구함 |
| `in_cluster & above_cutoff` | "군집 i 소속이면서 기준보다 먼" 샘플 (`&` = 원소별 AND) |
| `= -1` | 거리는 음수가 될 수 없으니 −1을 **"제외" 표시**로 사용 |
| `partially_propagated` | 남길 샘플 마스크 (약 절반) |

```python
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
log_reg.score(X_test, y_test)
# 0.8841309823677582

(y_train_partially_propagated == y_train[partially_propagated]).mean()
# np.float64(0.9887798036465638)
```

- 데이터는 줄었지만 **88.4%**로 더 올랐다 → 틀린 레이블(경계 샘플)을 걷어낸 효과
- 전파된 레이블의 정확도가 **98.9%** — 사람이 단 레이블 50개만으로 이 정도
- 전체 1,400개 정답 레이블(90.9%)에 꽤 근접

| 단계 | 사람이 단 레이블 | 테스트 정확도 |
|---|---|---|
| 무작위 50개 | 50 | 75.8% |
| 대표 이미지 50개 | 50 | 83.4% |
| 군집 전체 전파 | 50 | 86.9% |
| 중심 가까운 50%만 전파 | 50 | **88.4%** |
| (참고) 정답 레이블 1,400개 | 1,400 | 90.9% |

> 💡 **능동 학습(active learning)**: 더 개선하려면 반복한다. ① 모델이 가장 확신하지 못하는 샘플을 사람이 레이블링(가능하면 서로 다른 군집에서) ② 추가 레이블로 재훈련 ③ 성능 향상이 멈출 때까지 반복. 이 방식을 **불확실성 샘플링**이라 한다.
>
> 사이킷런에는 `LabelSpreading`, `LabelPropagation`(`sklearn.semi_supervised`)처럼 레이블 전파를 자동으로 해 주는 클래스와, 모델의 자신 있는 예측을 레이블로 추가하며 반복 훈련하는 `SelfTrainingClassifier`도 있다.

### 1-10. DBSCAN

**DBSCAN**(Density-Based Spatial Clustering of Applications with Noise): **밀집된 영역**이 이어진 곳을 하나의 군집으로 본다. 군집 모양에 제약이 없다.

동작 원리:

1. 샘플마다 반경 **ε(eps)** 안에 몇 개의 샘플이 있는지 센다 (= ε-이웃)
2. ε-이웃에 (자기 자신 포함) **min_samples**개 이상이 있으면 **핵심 샘플** — 밀집 지역 안에 있다는 뜻
3. 핵심 샘플의 이웃은 모두 같은 군집. 이웃 중 핵심 샘플이 있으면 그 이웃도 같은 군집 → **연쇄적으로 확장**
4. 핵심 샘플도 아니고 핵심 샘플의 이웃도 아닌 샘플은 **이상치** (레이블 −1)

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
```

- `make_moons` : 초승달 두 개가 맞물린 모양. k-평균이 절대 못 나누는 모양
- `noise=0.05` : 점에 섞는 가우스 잡음의 표준편차

```python
dbscan.labels_[:10]
# array([ 0,  2, -1, -1,  1,  0,  0,  0,  2,  5])

dbscan.core_sample_indices_[:10]
# array([ 0,  4,  5,  6,  7,  8, 10, 11, 12, 13])

dbscan.components_
# array([[-0.02137124,  0.40618608],
#        [-0.84192557,  0.53058695],
#        ...,
#        [ 0.79419406,  0.60777171]])
```

| 속성 | 의미 |
|---|---|
| `labels_` | 샘플별 군집 번호. **−1 = 이상치** |
| `core_sample_indices_` | 핵심 샘플의 **인덱스** |
| `components_` | 핵심 샘플의 **좌표** = `X[core_sample_indices_]` |

- 인덱스 1, 2, 3, 9 같은 샘플은 `core_sample_indices_`에 없음 → 핵심 샘플이 아님
- 레이블 5까지 나온다 → 초승달 2개인데 군집이 여러 개로 쪼개졌다는 신호

#### ε에 따른 결과 비교

```python
def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True        # 핵심 샘플 마스크
    anomalies_mask = dbscan.labels_ == -1                # 이상치 마스크
    non_core_mask = ~(core_mask | anomalies_mask)        # 둘 다 아닌 것 = 경계 샘플

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")   # 큰 원 = ε 반경 느낌
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20,
                c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)      # 빨간 X = 이상치
    plt.scatter(non_cores[:, 0], non_cores[:, 1],
                c=dbscan.labels_[non_core_mask], marker=".")                      # 작은 점 = 경계 샘플
    plt.title(f"eps={dbscan.eps:.2f}, min_samples={dbscan.min_samples}")

dbscan2 = DBSCAN(eps=0.2)
dbscan2.fit(X)

plt.subplot(121); plot_dbscan(dbscan, X, size=100)
plt.subplot(122); plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
```

| 코드 | 의미 |
|---|---|
| `np.zeros_like(a, dtype=bool)` | `a`와 같은 모양의 False 배열 |
| `core_mask[인덱스] = True` | 인덱스 배열 → 불리언 마스크로 변환 |
| `~(A \| B)` | NOT(A OR B). 핵심도 이상치도 아닌 샘플 |

| 설정 | 결과 |
|---|---|
| `eps=0.05` | 이웃 반경이 너무 작아 군집이 **7개**로 쪼개지고 이상치 77개 (약 8%) |
| `eps=0.2` (`min_samples` 기본 5) | 초승달 **2개**를 정확히 찾고 이상치 없음 |

→ DBSCAN은 하이퍼파라미터 두 개(ε, min_samples)만으로 **어떤 모양의 군집이든** 찾고, 이상치에도 강하다.

#### 새 샘플 예측: DBSCAN + 분류기

```python
dbscan = dbscan2      # 이후 eps=0.2 모델 사용
```

DBSCAN에는 `predict()`가 **없다** (`fit_predict()`만 있음). 새 샘플에 군집을 배정하려면 **분류기를 따로 훈련**한다.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
```

- 훈련 데이터 = **핵심 샘플만** (좌표 `components_`, 레이블 `labels_[core_sample_indices_]`)
- 이상치나 경계 샘플까지 넣을지는 선택. 핵심 샘플만 쓰면 더 깔끔한 경계가 나온다

```python
X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
knn.predict(X_new)
# array([1, 0, 1, 0])

knn.predict_proba(X_new)
# array([[0.18, 0.82],
#        [1.  , 0.  ],
#        [0.12, 0.88],
#        [1.  , 0.  ]])
```

- `predict_proba` : 이웃 50개 중 각 군집 비율. 첫 샘플은 이웃 50개 중 41개(82%)가 군집 1
- 그런데 **이상치가 없는 데이터로 훈련**했기 때문에 `[2, 1]`처럼 멀리 떨어진 샘플도 무조건 군집 0 또는 1로 분류된다

거리 기준으로 이상치를 직접 판정:

```python
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > 0.2] = -1
y_pred.ravel()
# array([-1,  0,  1, -1])
```

| 코드 | 의미 |
|---|---|
| `knn.kneighbors(X_new, n_neighbors=1)` | 가장 가까운 훈련 샘플 1개의 **거리**와 **인덱스** — 각 `(4, 1)` |
| `labels_[core_sample_indices_][y_pred_idx]` | 핵심 샘플 레이블 배열에서 그 인덱스의 레이블 |
| `y_pred[y_dist > 0.2] = -1` | 가장 가까운 핵심 샘플도 ε(0.2)보다 멀면 이상치 |
| `.ravel()` | `(4, 1)` → `(4,)` 1차원으로 펼침 |

- `[-0.5, 0]`(두 초승달 사이 빈 공간)과 `[2, 1]`(바깥)이 이상치로 판정됨

| 장점 | 단점 |
|---|---|
| 군집 수를 지정할 필요 없음 | 군집 간 **밀도 차이가 크면** ε 하나로 다 잡기 어려움 |
| 임의 모양의 군집 | 대규모 데이터에서 메모리 사용 큼 (최대 O(m²)) |
| 이상치에 강함 · 이상치 탐지 가능 | `predict()` 없음 → 분류기 필요 |
| 하이퍼파라미터 2개 | |

> 💡 밀도가 다양한 군집에는 **HDBSCAN**(`sklearn.cluster.HDBSCAN`)이 더 낫다. ε 대신 여러 밀도 수준을 탐색해서 `min_samples`(와 최소 군집 크기)만 정하면 된다.

### 1-11. 다른 군집 알고리즘 (보너스)

노트북 보너스 코드에 나오는 두 개와 교재가 소개하는 나머지를 정리.

#### 스펙트럴 군집

```python
from sklearn.cluster import SpectralClustering

sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
sc1.fit(X)
sc1.affinity_matrix_.round(2)
# array([[1. , 0. , 0. , ..., 0. , 0. , 0. ],
#        [0. , 1. , 0.3, ..., 0. , 0. , 0. ],
#        ...

sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
sc2.fit(X)
```

- 샘플 간 **유사도 행렬**(`affinity_matrix_`, 1000×1000)을 만들고 → 저차원 임베딩으로 차원을 줄인 뒤 → 그 공간에서 k-평균 같은 군집 알고리즘 적용
- 유사도 = RBF 커널 exp(−γ‖xᵢ − xⱼ‖²). 대각선은 자기 자신이라 1
- `gamma`가 크면 가까운 점만 유사(γ=100 → 초승달을 잘 나눔), 작으면 멀리 있는 점까지 유사(γ=1 → 잘 못 나눔)
- 복잡한 구조를 잘 잡지만 샘플이 많으면 느리고, 군집 크기가 많이 다르면 약함

#### 병합 군집 (agglomerative)

```python
from sklearn.cluster import AgglomerativeClustering

X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)     # 1차원 점 4개
agg = AgglomerativeClustering(linkage="complete").fit(X)

def learned_parameters(estimator):
    return [attrib for attrib in dir(estimator)
            if attrib.endswith("_") and not attrib.startswith("_")]

learned_parameters(agg)
# ['children_', 'labels_', 'n_clusters_', 'n_connected_components_',
#  'n_features_in_', 'n_leaves_']

agg.children_
# array([[0, 1],
#        [2, 3],
#        [4, 5]])
```

| 코드 | 의미 |
|---|---|
| `reshape(-1, 1)` | 1차원 배열 → `(4, 1)`. 사이킷런 입력은 2차원이어야 함 |
| `linkage="complete"` | 두 군집 사이 거리 = **가장 먼** 샘플 쌍의 거리 |
| `learned_parameters()` | `dir()`로 속성 이름을 다 꺼내 **끝이 `_`이고 시작이 `_`가 아닌 것** = 학습된 파라미터만 추림 (사이킷런 관례) |

`children_` 읽는 법 — 아래에서 위로 **가장 가까운 두 군집을 반복해서 합침**:

| 단계 | 병합 | 새 노드 번호 | 이유 |
|---|---|---|---|
| 1 | 0(값 0) + 1(값 2) | 4 | 거리 2로 가장 가까움 |
| 2 | 2(값 5) + 3(값 8.5) | 5 | 거리 3.5 ({0,2}와 5는 complete 기준 5) |
| 3 | 4 + 5 | 6 (루트) | 남은 두 군집 |

- 원래 샘플은 0~3번, 새로 생기는 노드는 4번부터 번호가 붙는다
- 결과는 **이진 트리(덴드로그램)**. 원하는 높이에서 잘라 군집 수를 정함 (기본 `n_clusters=2`)
- 다양한 모양의 군집을 잡고, 연결 행렬(`connectivity`)을 주면 대규모 데이터에도 적용 가능

#### 교재가 소개하는 그 밖의 알고리즘

| 알고리즘 | 특징 |
|---|---|
| **BIRCH** | 대규모 데이터용. 특성 수가 많지 않을 때(20개 이하) 미니배치 k-평균보다 빠를 수 있음. 트리 구조로 데이터를 요약 |
| **평균-이동(mean-shift)** | 각 샘플 주변 원의 평균 쪽으로 원을 계속 이동 → 수렴한 지점(밀도 최고점)이 같으면 같은 군집. 군집 수 자동, 대규모엔 부적합 (O(m²)) |
| **유사도 전파(affinity propagation)** | 샘플끼리 메시지를 주고받으며 각자 자신을 대표할 샘플(exemplar)을 선출. 군집 수 자동, 대규모엔 부적합 |

---

## 2. 가우스 혼합 (8.2)

**가우스 혼합 모델(GMM)**: 데이터가 **파라미터를 모르는 가우스(정규) 분포 여러 개의 혼합**에서 생성됐다고 가정하는 **확률 모델**.

- 하나의 가우스 분포에서 나온 샘플들은 보통 **타원형 군집**을 이룬다
- 군집마다 모양·크기·밀도·방향이 다를 수 있다 → 1-7절에서 k-평균이 실패한 데이터에 딱 맞음

데이터가 생성되는 과정 (가장 기본적인 `GaussianMixture` 기준):

1. 군집 k개 중 하나를 **가중치 φ⁽ʲ⁾의 확률로** 고른다
2. 고른 군집 j의 가우스 분포(평균 μ⁽ʲ⁾, 공분산 Σ⁽ʲ⁾)에서 샘플 x를 뽑는다

우리는 x만 보고 **φ, μ, Σ를 추정**해야 한다. 어느 군집에서 나왔는지는 관측되지 않는 **잠재 변수**.

| 파라미터 | 의미 | 사이킷런 속성 |
|---|---|---|
| φ (가중치) | 각 군집의 비율. 합 = 1 | `weights_` |
| μ (평균) | 각 군집의 중심 | `means_` |
| Σ (공분산 행렬) | 각 군집의 모양·크기·방향 | `covariances_` |

### 2-1. GaussianMixture 훈련

1-7절과 같은 데이터(길쭉한 타원 3개)를 다시 만든다.

```python
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]
```

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
gm.fit(X)
```

| 하이퍼파라미터 | 의미 |
|---|---|
| `n_components=3` | 가우스 분포(군집) 개수 |
| `n_init=10` | 초기화 10번 후 최선 선택. **기본값은 1**이라 직접 지정하는 게 좋음 |

#### 기댓값-최대화(EM) 알고리즘

`fit()` 내부에서 쓰는 알고리즘. k-평균과 구조가 거의 같다.

| 단계 | k-평균 | EM |
|---|---|---|
| 초기화 | 중심 무작위 | 파라미터 무작위 (기본은 k-평균 결과로 초기화) |
| 배정 단계 | **기댓값 단계(E)**: 샘플마다 각 군집에 속할 **확률(책임, responsibility)** 계산 | 가장 가까운 중심 **하나**에 배정 |
| 갱신 단계 | **최대화 단계(M)**: **모든 샘플**을 책임만큼 가중해서 φ, μ, Σ 갱신 | 배정된 샘플의 평균으로 중심 이동 |

- EM은 **소프트 배정을 하는 k-평균의 일반화**라고 볼 수 있다. 각 샘플은 확률만큼 모든 군집 갱신에 기여한다
- k-평균처럼 나쁜 해에 수렴할 수 있으므로 `n_init`을 크게

#### 추정된 파라미터

```python
gm.weights_
# array([0.40005972, 0.20961444, 0.39032584])

gm.means_
# array([[-1.40764129,  1.42712848],
#        [ 3.39947665,  1.05931088],
#        [ 0.05145113,  0.07534576]])

gm.covariances_
# array([[[ 0.63478217,  0.72970097],
#         [ 0.72970097,  1.16094925]],
#
#        [[ 1.14740131, -0.03271106],
#         [-0.03271106,  0.95498333]],
#
#        [[ 0.68825143,  0.79617956],
#         [ 0.79617956,  1.21242183]]])
```

- **가중치 검증**: 데이터를 만들 때 X1 1,000개(500 + 500)와 X2 250개, 총 1,250개 → 실제 비율 0.4 / 0.4 / 0.2. 추정값 0.400 / 0.390 / 0.210으로 거의 정확
- 공분산 `(3, 2, 2)` = 군집 3개 × 2×2 행렬
  - 대각선 = 각 축의 분산, 비대각선 = 두 축의 공분산
  - 군집 0, 2는 비대각선이 +0.73, +0.80 → **오른쪽 위로 기울어진 타원** (선형 변환으로 기울인 X1)
  - 군집 1은 비대각선 ≈ −0.03 → **거의 원형** (변환 안 한 X2)

```python
gm.converged_   # True — 수렴했는지
gm.n_iter_      # 4   — EM 반복 횟수
```

### 2-2. 예측 · 샘플링 · 밀도 추정

#### 하드 군집 / 소프트 군집

```python
gm.predict(X)
# array([2, 2, 0, ..., 1, 1, 1])

gm.predict_proba(X).round(3)
# array([[0.   , 0.023, 0.977],
#        [0.001, 0.016, 0.983],
#        [1.   , 0.   , 0.   ],
#        ...,
#        [0.   , 1.   , 0.   ]])
```

| 메서드 | 결과 | 비교 |
|---|---|---|
| `predict()` | 가장 확률이 높은 군집 번호 | k-평균의 `predict()`와 같은 역할 |
| `predict_proba()` | 군집별 **확률** (행 합 = 1) | k-평균 `transform()`(거리)보다 해석이 쉬움 |

첫 샘플은 군집 2일 확률 97.7%, 군집 1일 확률 2.3%.

#### 생성 모델: 새 샘플 만들기

```python
X_new, y_new = gm.sample(6)
X_new
# array([[-2.32491052,  1.04752548],
#        [-1.16654983,  1.62795173],
#        [ 1.84860618,  2.07374016],
#        [ 3.98304484,  1.49869936],
#        [ 3.8163406 ,  0.53038367],
#        [ 0.38079484, -0.56239369]])

y_new
# array([0, 0, 1, 1, 1, 2])
```

- 가우스 혼합은 **생성 모델** → 학습한 분포에서 새 데이터를 뽑을 수 있다
- `y_new` = 각 샘플이 뽑힌 군집. **군집 순서대로** 묶여서 나온다 (섞여 나오지 않음)

#### 밀도 추정

```python
gm.score_samples(X).round(2)
# array([-2.61, -3.57, -3.33, ..., -3.51, -4.4 , -3.81])
```

- 각 위치에서 **확률 밀도 함수(PDF)의 로그값**
- 값이 클수록 밀도가 높은 곳 (= 이런 샘플이 나올 법한 곳)
- 이 값은 **확률이 아니라 밀도**다. 확률은 영역에 대해 적분해야 나온다

PDF를 전체 공간에서 적분하면 1이 되는지 확인 (보너스):

```python
resolution = 100
grid = np.arange(-10, 10, 1 / resolution)          # -10~10을 0.01 간격으로 → 2,000개
xx, yy = np.meshgrid(grid, grid)                   # 2000×2000 격자
X_full = np.vstack([xx.ravel(), yy.ravel()]).T     # (4000000, 2) — np.c_[...]와 같은 결과

pdf = np.exp(gm.score_samples(X_full))             # 로그 → 실제 밀도
pdf_probas = pdf * (1 / resolution) ** 2           # 밀도 × 작은 사각형 넓이(0.01²) = 그 칸의 확률
pdf_probas.sum()
# np.float64(0.9999999999225088)
```

- 적분을 "작은 사각형 넓이 × 밀도의 합"으로 근사 (리만 합)
- 결과 ≈ 1 → 제대로 된 확률 분포

#### 결정 경계와 밀도 등고선 그리기

```python
from matplotlib.colors import LogNorm

def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))

    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])   # 음의 로그 밀도
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))                    # 밀도 색칠
    plt.contour(xx, yy, Z, norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12), linewidths=1, colors='k')

    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, linewidths=2, colors='r', linestyles='dashed')   # 결정 경계

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)
    # (축 레이블 생략)

plt.figure(figsize=(8, 4))
plot_gaussian_mixture(gm, X)
plt.show()
```

| 코드 | 의미 |
|---|---|
| `-score_samples()` | 음의 로그 밀도. 밀도가 높을수록 값이 작음 |
| `LogNorm(vmin, vmax)` | 색 매핑을 로그 스케일로 |
| `np.logspace(0, 2, 12)` | 10⁰~10²를 로그 간격 12단계 → 등고선 높이 |
| 빨간 점선 | `predict()` 결과가 바뀌는 곳 = 결정 경계 |
| `plot_centroids(means_, weights_)` | 1-2절에서 만든 함수. 가중치가 최대의 1/10 미만인 중심은 생략 |

- 세 타원의 **평균, 모양, 경계를 모두 정확히** 찾았다. k-평균이 실패했던 데이터

### 2-3. 공분산 제약 (covariance_type)

특성(차원)이나 군집이 많고 샘플이 적으면 EM이 최적해로 수렴하기 어렵다. 공분산 행렬에 **제약**을 걸어 학습할 파라미터 수를 줄일 수 있다.

| `covariance_type` | 군집 모양 | 그림으로 보면 |
|---|---|---|
| `"full"` (기본) | 제약 없음. 군집마다 크기·모양·방향 자유 | 제각각 기울어진 타원 |
| `"tied"` | 모든 군집이 **같은 공분산 행렬** 공유 | 같은 모양·크기·방향의 타원 (위치만 다름) |
| `"diag"` | 대각 공분산 → 타원 축이 **좌표축과 평행** | 눕거나 선 타원 (기울지 않음) |
| `"spherical"` | 군집마다 분산 하나 → **원형**, 지름은 군집마다 다를 수 있음 | 크기가 다른 원 |

```python
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X); gm_tied.fit(X); gm_spherical.fit(X); gm_diag.fit(X)

def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))
    plt.subplot(121); plot_gaussian_mixture(gm1, X)
    plt.title(f'covariance_type="{gm1.covariance_type}"')
    plt.subplot(122); plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title(f'covariance_type="{gm2.covariance_type}"')

compare_gaussian_mixtures(gm_tied, gm_spherical, X)   # 교재 그림
compare_gaussian_mixtures(gm_full, gm_diag, X)        # 보너스
```

- `tied` : 원형이던 군집 1도 다른 두 군집처럼 기울어진 타원으로 표현됨 → 이 데이터엔 약간 안 맞음
- `spherical` : 기울어진 타원을 원으로밖에 표현 못 함 → 경계가 어긋남
- 이 데이터는 `full`이 가장 정확

계산 복잡도 (m = 샘플 수, n = 특성 수, k = 군집 수):

| 타입 | 훈련 복잡도 |
|---|---|
| `spherical`, `diag` | O(k·m·n) — 데이터가 커도 OK |
| `tied`, `full` | O(k·m·n² + k·n³) — **특성이 많으면 감당 못 함** |

### 2-4. 가우스 혼합으로 이상치 탐지

**이상치 탐지**: 보통과 많이 다른 샘플 찾기 (사기 거래, 불량품, 데이터 정제 단계에서 이상치 제거 등).

가우스 혼합에서는 간단하다 — **밀도가 낮은 곳에 있는 샘플 = 이상치**. 밀도 임곗값만 정하면 된다.

```python
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 2)
anomalies = X[densities < density_threshold]
```

| 코드 | 의미 |
|---|---|
| `score_samples(X)` | 샘플별 로그 밀도 |
| `np.percentile(densities, 2)` | 하위 2% 지점의 밀도 |
| `densities < threshold` | 그보다 낮은 샘플 → 전체의 약 2%를 이상치로 |

```python
plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')   # 빨간 별 = 이상치
plt.ylim(top=5.1)
```

- 예: 불량률이 약 2%로 알려진 공장이라면 임곗값을 하위 2%로
- 임곗값 조정 = **정밀도/재현율 트레이드오프** (3장)
  - 거짓 양성(정상인데 불량 판정)이 너무 많으면 임곗값을 낮춤
  - 거짓 음성(불량을 놓침)이 너무 많으면 임곗값을 높임

> ⚠️ 가우스 혼합은 이상치까지 포함해서 학습하므로 이상치가 너무 많으면 "정상" 분포가 왜곡된다. 그럴 땐 **한 번 학습 → 이상치 제거 → 다시 학습**. 또는 이상치에 강한 공분산 추정(`EllipticEnvelope`)을 쓴다.

| 용어 | 차이 |
|---|---|
| **이상치 탐지** (outlier/anomaly detection) | 훈련 데이터에 이상치가 섞여 있다고 가정 |
| **특이치 탐지** (novelty detection) | 훈련 데이터는 **깨끗하다**고 가정하고, 새 샘플이 이상한지 판단 |

교재가 소개하는 다른 이상치·특이치 탐지 알고리즘:

| 알고리즘 | 사이킷런 | 아이디어 |
|---|---|---|
| PCA (재구성 오차) | `PCA` + `inverse_transform` | 정상 샘플보다 재구성 오차가 훨씬 큼 (7장) |
| Fast-MCD | `EllipticEnvelope` | 정상 샘플이 가우스 분포 하나에서 나왔다고 보고 이상치를 무시하며 타원 추정 |
| 아이솔레이션 포레스트 | `IsolationForest` | 무작위 분할을 반복하면 이상치는 **금방 고립됨**. 고차원에서 효율적 (2장에서도 등장) |
| LOF | `LocalOutlierFactor` | 주변 이웃보다 밀도가 훨씬 낮으면 이상치 |
| 단일 클래스 SVM | `OneClassSVM` | 원본 공간을 감싸는 경계를 학습. 특이치 탐지에 적합, 대규모엔 부적합 |

### 2-5. 군집 수 선택: BIC · AIC

가우스 혼합에서는 이너셔나 실루엣 점수를 쓸 수 없다. 둘 다 **군집이 원형**이라고 가정하기 때문 (타원형 군집에선 엉뚱한 답을 줌).

대신 **이론적 정보 기준**이 가장 작은 모델을 고른다.

- **BIC** (Bayesian information criterion) = log(m)·p − 2·log(L̂)
- **AIC** (Akaike information criterion) = 2p − 2·log(L̂)

| 기호 | 의미 |
|---|---|
| m | 샘플 수 |
| p | 모델이 학습하는 파라미터 수 |
| L̂ | 모델 **가능도 함수의 최댓값** |

- **−2·log(L̂)** : 데이터를 잘 설명할수록 작아짐 (적합도)
- **p 항** : 파라미터가 많을수록 커짐 (복잡도 벌점)
- → 데이터를 잘 설명하면서도 **단순한** 모델을 고르게 함
- BIC는 m이 클수록 벌점이 커서 더 **단순한** 모델을, AIC는 조금 더 복잡한 모델을 고르는 경향. 둘이 다르게 나오면 BIC 쪽이 더 단순한 모델

#### 확률 vs 가능도

같은 함수 f(x; θ)를 어느 쪽에서 보느냐의 차이.

| 구분 | 고정 | 변수 | 질문 |
|---|---|---|---|
| **확률** | 파라미터 θ | 결과 x | θ인 모델에서 x가 나올 확률은? |
| **가능도** | 관측된 x | 파라미터 θ | x를 관측했는데, θ가 얼마일 때 가장 그럴듯한가? |

- 확률 분포(PDF)는 x에 대해 적분하면 1. **가능도 함수는 θ에 대해 적분해도 1이 아니다**
- 관측 데이터가 주어졌을 때 가능도를 최대화하는 θ를 찾는 게 **최대 가능도 추정(MLE)**
- 실제로는 곱셈이 덧셈이 되고 계산이 안정적인 **로그 가능도**를 최대화 (로그는 단조증가라 최댓값 위치가 같음)

노트북 그림 8-18은 f(x; θ)를 등고선으로 그리고, θ 고정(가로선 → PDF)과 x 고정(세로선 → 가능도 함수)으로 단면을 잘라 이 차이를 보여 준다. 그림 생성 코드는 길어서 생략.

#### 계산

```python
gm.bic(X)   # np.float64(8189.733705221636)
gm.aic(X)   # np.float64(8102.508425106598)
```

직접 계산 (보너스):

```python
n_clusters = 3
n_dims = 2
n_params_for_weights = n_clusters - 1                              # 2
n_params_for_means = n_clusters * n_dims                           # 6
n_params_for_covariance = n_clusters * n_dims * (n_dims + 1) // 2  # 9
n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
max_log_likelihood = gm.score(X) * len(X)                          # log(L̂)
bic = np.log(len(X)) * n_params - 2 * max_log_likelihood
aic = 2 * n_params - 2 * max_log_likelihood
# bic = 8189.733705221636
# aic = 8102.508425106598
# n_params = 17
```

| 파라미터 | 개수 | 이유 |
|---|---|---|
| 가중치 | k − 1 = 2 | 3개지만 **합이 1**이라 2개만 정하면 나머지가 결정됨 (자유도 −1) |
| 평균 | k × n = 6 | 군집마다 2차원 좌표 |
| 공분산 | k × n(n+1)/2 = 9 | n×n 대칭 행렬은 대각선 포함 위쪽 절반만 독립 → 1 + 2 + … + n = n(n+1)/2 = 3 |
| **합계** | **17** | |

- `gm.score(X)` = 샘플당 **평균** 로그 가능도 → `× len(X)`로 전체 로그 가능도
- `gm.bic()` 결과와 정확히 일치

#### k에 따른 BIC · AIC

```python
gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
             for k in range(1, 11)]
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]

plt.figure(figsize=(8, 3))
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$"); plt.ylabel("Information Criterion")
plt.axis([1, 9.5, min(aics) - 50, max(aics) + 50])
plt.annotate("", xy=(3, bics[2]), xytext=(3.4, 8650),
             arrowprops=dict(facecolor='black', shrink=0.1))
plt.text(3.5, 8660, "Minimum", horizontalalignment="center")
plt.legend(); plt.grid()
plt.show()
```

- BIC와 AIC 모두 **k=3에서 최소** → 정답
- `covariance_type`도 함께 바꿔 가며 BIC를 비교하면 최적의 공분산 형태까지 고를 수 있다

### 2-6. 베이즈 가우스 혼합

최적의 군집 수를 일일이 찾는 대신, **불필요한 군집의 가중치를 0(또는 0에 가깝게)으로** 만들어 주는 모델.

```python
from sklearn.mixture import BayesianGaussianMixture

bgm = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=500,
                              random_state=42)
bgm.fit(X)
bgm.weights_.round(2)
# array([0.4 , 0.21, 0.39, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])
```

| 하이퍼파라미터 | 의미 |
|---|---|
| `n_components=10` | 최적 군집 수보다 **충분히 크게** 설정 |
| `max_iter=500` | 기본 100회로는 수렴하지 않을 수 있어 늘림 |

- 군집 10개 중 **3개만 살아남았다** — 자동으로 k=3을 찾음
- 가중치 0.40 / 0.21 / 0.39는 `GaussianMixture` 결과와 거의 같다

```python
plt.figure(figsize=(8, 5))
plot_gaussian_mixture(bgm, X)     # 가중치 작은 7개 중심은 plot_centroids가 생략
```

> 💡 원리: 군집 가중치 등 파라미터를 고정값이 아니라 **확률 변수**로 보고 사전 분포를 둔다. `weight_concentration_prior`가 작으면 적은 수의 군집을, 크면 많은 군집을 선호하도록 사전 믿음을 조절할 수 있다.

#### 한계: 타원이 아닌 군집

```python
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)

bgm = BayesianGaussianMixture(n_components=10, n_init=10, max_iter=500, random_state=42)
bgm.fit(X_moons)

plt.subplot(121); plot_data(X_moons)                          # 원본
plt.subplot(122); plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)
```

- 초승달 2개를 찾지 못하고 **타원 8개**로 덮어 버림
- 가우스 혼합은 군집이 타원형이라고 가정하기 때문 → 모양이 다르면 실패
- 다만 **밀도 추정**은 꽤 그럴듯해서 이상치 탐지 용도로는 쓸 만하다

> 💡 초승달처럼 복잡한 모양: 군집 목적이면 DBSCAN·HDBSCAN·스펙트럴 군집, 밀도 추정 목적이면 가우스 혼합도 가능.

---

## 3. 정리

### 3-1. 전체 흐름

```
[8.1 군집]
KMeans(n_clusters=k)                    ← 중심 k개, 거리 기반, 하드/소프트(transform)
  ├ init: k-means++ (기본) / random + n_init
  ├ 평가: inertia_ (작을수록, 단 k 선택엔 부적합)
  ├ k 선택: 엘보 → 실루엣 점수 → 실루엣 다이어그램
  ├ 변형: MiniBatchKMeans (대용량, memmap)
  └ 한계: 크기·밀도·모양이 다른 군집 ✗ → 스케일링 필수
  활용: 이미지 색상 분할 / 준지도 학습(대표 샘플 + 레이블 전파)
DBSCAN(eps, min_samples)                ← 밀도 기반, 임의 모양, -1 = 이상치
  └ predict 없음 → KNN 분류기로 새 샘플 배정

[8.2 가우스 혼합]
GaussianMixture(n_components, n_init)   ← EM 알고리즘, 타원형 군집
  ├ predict / predict_proba / sample / score_samples(로그 밀도)
  ├ covariance_type: full / tied / diag / spherical
  ├ 이상치 탐지: score_samples < 하위 N% 임곗값
  ├ k 선택: bic() / aic() 최소
  └ BayesianGaussianMixture             ← 불필요한 군집 가중치 → 0
```

### 3-2. 개념과 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| k-평균 | `KMeans(n_clusters=k)` | 거리 기반 → 특성 스케일링 필수 |
| 군집 평가 | `inertia_`, `silhouette_score` | 이너셔는 k가 커질수록 줄기만 → k 선택엔 실루엣 |
| 대용량 | `MiniBatchKMeans` | 미니배치로 빠르게, 이너셔는 약간 손해 |
| 이미지 분할 | `kmeans.cluster_centers_[kmeans.labels_]` | 화소를 군집 중심 색으로 교체 |
| 준지도 학습 | `fit_transform` → `argmin` → 레이블 전파 | 대표 샘플만 레이블링 |
| DBSCAN | `DBSCAN(eps, min_samples)` + `KNeighborsClassifier` | 밀도 기반, 임의 모양, `-1` = 이상치 |
| 가우스 혼합 | `GaussianMixture(n_components, covariance_type)` | EM 알고리즘, 소프트 군집·밀도 추정 |
| 이상치 탐지 | `score_samples` + `np.percentile` | 밀도가 낮은 샘플 = 이상치 |
| 군집 수 선택 | `bic()`, `aic()`, `BayesianGaussianMixture` | 정보 기준 최소 / 불필요한 군집 가중치 → 0 |

### 3-3. 알고리즘 비교

| 알고리즘 | 군집 수 지정 | 군집 모양 | 이상치 처리 | 새 샘플 예측 | 대규모 데이터 |
|---|---|---|---|---|---|
| k-평균 | 필요 | 원형, 비슷한 크기 | ✗ (모두 배정) | `predict()` | ○ (미니배치) |
| DBSCAN | 불필요 | 임의 모양 | ○ (레이블 −1) | ✗ (분류기 필요) | △ |
| 가우스 혼합 | 필요 (BIC/AIC) | 타원형 | ○ (밀도 임곗값) | `predict()`, `predict_proba()` | △ (full은 특성 수에 민감) |
| 베이즈 가우스 혼합 | 상한만 | 타원형 | ○ | `predict()` | △ |

### 3-4. 주요 결과

| 실험 | 지표 | 값 |
|---|---|---|
| 붓꽃, 가우스 혼합 (특성 4개) | 품종 일치율 | 0.967 |
| 5개 덩어리, k-평균 좋은 해 | 이너셔 | 211.60 |
| 무작위 초기화 나쁜 해 | 이너셔 | 600.36 |
| 미니배치 k-평균 | 이너셔 | 211.66 |
| k=3 / k=8 | 이너셔 | 653.22 / 127.13 |
| k=5 | 실루엣 점수 | 0.656 |
| 숫자, 무작위 50개 | 테스트 정확도 | 0.758 |
| 숫자, 대표 50개 | 테스트 정확도 | 0.834 |
| 숫자, 전체 전파 | 테스트 정확도 | 0.869 |
| 숫자, 50% 부분 전파 | 테스트 정확도 | **0.884** (전파 레이블 정확도 0.989) |
| 숫자, 정답 1,400개 (참고) | 테스트 정확도 | 0.909 |
| 가우스 혼합 k=3 | BIC / AIC / 파라미터 수 | 8189.7 / 8102.5 / 17 |
| 베이즈 가우스 혼합 (10개 설정) | 남은 군집 | 3개 (0.40 / 0.21 / 0.39) |

### 3-5. 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `model.fit_predict(X)` | 훈련 + 훈련 데이터 군집 번호 | 1-2, 1-10 |
| `model.fit_transform(X)` | 훈련 + 중심까지 거리 행렬 | 1-9 |
| `np.meshgrid` + `np.c_[xx.ravel(), yy.ravel()]` + `reshape` | 결정 경계·등고선용 격자 예측 | 1-2, 2-2 |
| `D[np.arange(m), labels_]` | 행마다 다른 열 하나씩 뽑기 (자기 군집 거리) | 1-4, 1-9 |
| `centers[labels_]` | 샘플마다 자기 군집 중심값으로 교체 | 1-8 |
| `dist.argmin(axis=0)` | 열(군집)마다 가장 가까운 샘플 인덱스 | 1-9 |
| `np.percentile(값, q)` | 백분위수로 임곗값 정하기 | 1-9, 2-4 |
| `[Model(k=k).fit(X) for k in range(...)]` | k별 모델 한 번에 훈련 | 1-6, 2-5 |
| `stats.mode()` + 매핑 딕셔너리 | 군집 번호 ↔ 실제 클래스 맞추기 | 1-1 |
| `knn.kneighbors(X, n_neighbors=1)` + 거리 임곗값 | 거리 기반 이상치 판정 | 1-10 |
| `np.exp(score_samples())` | 로그 밀도 → 밀도 | 2-2 |

### 3-6. 핵심 원칙

1. 비지도 학습은 레이블 없이 `X`만으로 구조(군집·밀도)를 찾는다.
2. k-평균은 빠르고 단순하지만 원형·비슷한 크기의 군집을 가정한다. 특성 스케일링이 필수다.
3. 이너셔는 k가 커질수록 줄기만 하므로, k는 실루엣 점수·실루엣 다이어그램으로 고른다.
4. 군집은 그 자체보다 이미지 분할·준지도 학습(대표 샘플 레이블 전파)·전처리 같은 **활용**에서 가치가 크다.
5. 모양이 불규칙하고 이상치가 섞여 있으면 DBSCAN, 확률·밀도가 필요하면 가우스 혼합을 쓴다.
6. 가우스 혼합의 군집 수는 BIC·AIC가 최소인 값으로 고르거나, 베이즈 가우스 혼합으로 자동으로 줄인다.

---

> ➡️ **다음 장 — 9장. 인공 신경망 (정리 예정)**  
> 사이킷런 파트를 마치고 **인공 신경망과 파이토치**로 넘어간다. 4장의 경사 하강법이 신경망 학습의 출발점이 된다.

[← 7장. 차원 축소](07_dimensionality_reduction.md) · [📚 전체 목차](../README.md) · 9장. 인공 신경망 (정리 예정) →
