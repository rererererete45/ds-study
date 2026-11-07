# 6장. 앙상블 학습과 랜덤 포레스트

[← 5장. 결정 트리](05_decision_trees.md) · [📚 전체 목차](../README.md) · [7장. 차원 축소 →](07_dimensionality_reduction.md)

> 📘 **교재** 『핸즈온 머신러닝 with 사이킷런, 파이토치』 6장 · 노트북 `06_ensemble_learning_and_random_forests.ipynb`
>
> 5장의 결정 트리는 혼자서는 **분산이 크다**(훈련 데이터가 조금만 바뀌어도 모델이 크게 바뀜). 6장은 이런 예측기를 **여러 개 모아서** 하나보다 나은 예측을 만드는 방법을 다룬다. "군중의 지혜"를 머신러닝에 옮긴 것.
>
> ⬅️ **앞에서** 5장 마지막의 "여러 트리의 다수결" 실험을 체계화한다. 2장에서 이미 써 본 `RandomForestRegressor`의 원리도 여기서 풀린다.  
> ➡️ **다음으로** 모델을 바꾸는 대신 **데이터의 특성 수를 줄이는** 쪽으로 — 차원 축소. 5장의 PCA 축 정렬, 6장의 특성 중요도·랜덤 서브스페이스와 이어진다.

## 이 장의 흐름

| 단계 | 절 | 핵심 질문 | 결과물 |
|---|---|---|---|
| 1. 원리·투표 | 6.1 | 여러 예측기를 모으면 왜 좋아지는가 | 큰 수의 법칙, 직접·간접 투표 |
| 2. 배깅·페이스팅 | 6.2 | 같은 알고리즘을 다른 데이터로 학습하면 | `BaggingClassifier`, OOB 평가 |
| 3. 랜덤 패치·서브스페이스 | 6.3 | 특성까지 무작위로 고르면 | `max_features`, `bootstrap_features` |
| 4. 랜덤 포레스트 | 6.4 | 트리 배깅을 더 무작위하게 하면 | `RandomForestClassifier`, 엑스트라 트리, 특성 중요도 |
| 5. 부스팅 | 6.5 | 앞 모델의 실수를 순차적으로 고치면 | AdaBoost, 그레이디언트 부스팅, HGB |
| 6. 스태킹 | 6.6 | 집계 방식 자체를 학습하면 | `StackingClassifier` |

**앙상블을 만드는 네 갈래**

| 갈래 | 아이디어 | 주로 줄이는 것 | 이 장의 절 |
|---|---|---|---|
| **투표** | 서로 **다른 알고리즘**을 같은 데이터로 학습 → 다수결/확률 평균 | 개별 모델의 오류 | 6.1 |
| **배깅·페이스팅** (+ 랜덤 포레스트) | **같은 알고리즘**을 서로 다른 **데이터 부분집합**으로 **병렬** 학습 | **분산** | 6.2 ~ 6.4 |
| **부스팅** | 약한 학습기를 **순차적**으로, 앞 모델의 실수를 고치도록 학습 | **편향** (+ 분산) | 6.5 |
| **스태킹** | 예측기들의 예측을 입력으로 받는 **메타 모델(블렌더)** 을 학습 | 집계 방식 자체를 학습 | 6.6 |

## 데이터

| 데이터 | 형태 | 사용 절 | 비고 |
|---|---|---|---|
| `make_moons(n_samples=500, noise=0.30)` | `(500, 2)` | 6.1, 6.2, 6.4, 6.5, 6.6 | 이진 분류, 375 훈련 / 125 테스트 |
| 붓꽃(iris) | `(150, 4)` | 6.4 | 3클래스, 특성 중요도 |
| MNIST | `(70000, 784)` | 6.4, 연습문제 | 10클래스, 특성 중요도 히트맵 |
| 2차식 + 잡음 | `(100, 1)` | 6.5 | $`y = 3x^2 + \epsilon`$, 그레이디언트 부스팅 |
| 캘리포니아 주택 가격 | 2장과 동일 | 6.5 | 히스토그램 기반 그레이디언트 부스팅 |

## 목차

- [0. 준비](#0-준비)
- [1. 투표 기반 분류기 (6.1)](#1-투표-기반-분류기-61)
   - [1-0. 왜 모으면 좋아지나? — 큰 수의 법칙 (그림 6-3)](#1-0-왜-모으면-좋아지나--큰-수의-법칙-그림-6-3)
   - [1-1. 직접 투표 (hard voting)](#1-1-직접-투표-hard-voting)
   - [1-2. 간접 투표 (soft voting)](#1-2-간접-투표-soft-voting)
- [2. 배깅과 페이스팅 (6.2)](#2-배깅과-페이스팅-62)
   - [2-1. 사이킷런의 배깅](#2-1-사이킷런의-배깅)
   - [2-2. 결정 경계 비교 (그림 6-5)](#2-2-결정-경계-비교-그림-6-5)
   - [2-3. OOB 평가 (out-of-bag)](#2-3-oob-평가-out-of-bag)
- [3. 랜덤 패치와 랜덤 서브스페이스 (6.3)](#3-랜덤-패치와-랜덤-서브스페이스-63)
   - [3-1. 보충 코드: 50개 특성 데이터로 비교](#3-1-보충-코드-50개-특성-데이터로-비교)
- [4. 랜덤 포레스트 (6.4)](#4-랜덤-포레스트-64)
   - [4-1. RandomForestClassifier](#4-1-randomforestclassifier)
   - [4-2. 랜덤 포레스트 ≈ 배깅 + `max_features="sqrt"`](#4-2-랜덤-포레스트--배깅--max_featuressqrt)
   - [4-3. 엑스트라 트리 (Extremely Randomized Trees)](#4-3-엑스트라-트리-extremely-randomized-trees)
   - [4-4. 특성 중요도](#4-4-특성-중요도)
- [5. 부스팅 (6.5)](#5-부스팅-65)
   - [5-1. AdaBoost (Adaptive Boosting)](#5-1-adaboost-adaptive-boosting)
   - [5-2. 그레이디언트 부스팅 (Gradient Boosting)](#5-2-그레이디언트-부스팅-gradient-boosting)
   - [5-3. 히스토그램 기반 그레이디언트 부스팅 (HGB)](#5-3-히스토그램-기반-그레이디언트-부스팅-hgb)
- [6. 스태킹 (6.6)](#6-스태킹-66)
- [7. 연습문제 (6.7)](#7-연습문제-67)
   - [7-1. 8번: 투표 분류기](#7-1-8번-투표-분류기)
   - [7-2. 9번: 스태킹 앙상블](#7-2-9번-스태킹-앙상블)
- [8. 정리](#8-정리)
   - [8-1. 전체 흐름](#8-1-전체-흐름)
   - [8-2. 개념과 코드 대응](#8-2-개념과-코드-대응)
   - [8-3. 앙상블 방법 비교](#8-3-앙상블-방법-비교)
   - [8-4. 주요 결과](#8-4-주요-결과)
   - [8-5. 자주 쓴 코드 패턴](#8-5-자주-쓴-코드-패턴)
   - [8-6. 핵심 원칙](#8-6-핵심-원칙)

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

> ⚠️ **버전 메모** (scikit-learn 1.9 기준)
> - `AdaBoostClassifier(algorithm="SAMME")` → `algorithm` 인자가 **1.8에서 삭제**. 지우고 쓰면 된다 (SAMME가 유일한 방식이 됨).
> - `make_column_transformer(..., force_int_remainder_cols=False)` → 이 인자도 **1.9에서 삭제**. 지우고 쓰면 된다.
> - 노트북은 `sklearn >= 1.6.1`을 요구. 출력값은 노트북에 저장된 결과를 그대로 옮겼다.

---

## 1. 투표 기반 분류기 (6.1)

### 1-0. 왜 모으면 좋아지나? — 큰 수의 법칙 (그림 6-3)

앞면이 나올 확률이 **51%** 인 살짝 기울어진 동전을 생각하자. 한 번 던지면 거의 반반이지만, 많이 던질수록 앞면 비율은 51%에 수렴하고 "앞면이 과반"일 확률은 1에 가까워진다.

```python
import matplotlib.pyplot as plt
import numpy as np

heads_proba = 0.51
rng = np.random.default_rng(seed=42)
coin_tosses = (rng.random((10000, 10)) < heads_proba).astype(np.int32)
cumulative_heads = coin_tosses.cumsum(axis=0)
cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
plt.grid()
plt.show()
```

| 줄 | 의미 |
|---|---|
| `rng.random((10000, 10))` | 0~1 균등분포 난수를 10,000행 × 10열로. **열 하나 = 동전 던지기 시리즈 하나** (총 10개 시리즈) |
| `< heads_proba` | 0.51보다 작으면 True(앞면). 51% 확률로 앞면이 되는 동전을 흉내 냄 |
| `.astype(np.int32)` | True/False → 1/0 |
| `.cumsum(axis=0)` | 행 방향 누적합 → n번째 행 = n번 던질 때까지 나온 앞면 수 |
| `np.arange(1, 10001).reshape(-1, 1)` | 던진 횟수 1~10000을 **열 벡터**로. 브로드캐스팅으로 10개 열을 한 번에 나눔 |
| `cumulative_heads_ratio` | 지금까지의 앞면 비율. 10개 선이 모두 0.51 근처로 모여 든다 |

**앙상블과의 연결**: 정확도 51%짜리 **약한 학습기**(weak learner)를 1,000개 모아 다수결하면 정확도가 75% 가까이 오를 수 있다. 단, 조건이 있다.

> 💡 **핵심 전제: 예측기들이 서로 독립이어야 한다.** 같은 데이터로 학습하면 같은 실수를 하기 쉬워 효과가 떨어진다. 그래서 앙상블에서는 **최대한 다른** 예측기를 모으는 것이 중요하다 → 다른 알고리즘(6.1), 다른 데이터(6.2), 다른 특성(6.3)을 쓰는 이유.

### 1-1. 직접 투표 (hard voting)

각 분류기의 예측 클래스를 모아 **다수결**.

```python
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))
    ]
)
voting_clf.fit(X_train, y_train)
```

| 줄 | 의미 |
|---|---|
| `make_moons(noise=0.30)` | 반달 2개가 맞물린 2차원 데이터. 선형으로 안 나뉘어서 비선형 모델 비교에 좋음 |
| `train_test_split(...)` | 기본 `test_size=0.25` → 375개 훈련, 125개 테스트 |
| `estimators=[(이름, 모델), ...]` | **(이름, 추정기) 튜플의 리스트**. 이름은 나중에 `named_estimators_["lr"]`처럼 꺼낼 때 사용 |
| 세 모델의 선택 | 선형(로지스틱), 트리 앙상블(RF), 커널(SVC) → **서로 실수하는 방식이 다름** = 다양성 확보 |
| `voting` (기본값 `"hard"`) | 직접 투표 |
| `fit` | 내부적으로 각 모델을 **복제(clone)** 해서 학습. 원본 객체는 학습되지 않는다 |

#### 개별 모델 성능 확인

```python
for name, clf in voting_clf.named_estimators_.items():
    print(name, "=", clf.score(X_test, y_test))
# lr = 0.864
# rf = 0.896
# svc = 0.896
```

- `named_estimators_` (끝에 `_`): **학습이 끝난** 복제본들이 담긴 딕셔너리 같은 객체 (`Bunch`)
- `named_estimators` (끝에 `_` 없음): 학습 전 원본. 사이킷런 관례상 **`_`로 끝나는 속성은 `fit` 이후에 생긴다**

#### 투표 과정 들여다보기

```python
voting_clf.predict(X_test[:1])
# array([1])

[clf.predict(X_test[:1]) for clf in voting_clf.estimators_]
# [array([1]), array([1]), array([0])]
```

- 첫 번째 테스트 샘플에 대해 lr=1, rf=1, svc=0 → **2 대 1로 클래스 1**
- `estimators_`: 학습된 복제본의 **리스트** (`named_estimators_`의 값들만)

```python
voting_clf.score(X_test, y_test)
# 0.912
```

> ✅ 개별 최고 0.896 → 앙상블 **0.912**. 가장 좋은 모델보다도 낫다.

### 1-2. 간접 투표 (soft voting)

모든 분류기가 `predict_proba()`를 지원하면, **클래스 확률의 평균**이 가장 높은 클래스를 고른다. 자신 있는 예측(확률 0.99)에 더 큰 비중을 주는 셈이라 보통 직접 투표보다 낫다.

```python
voting_clf.voting = "soft"
voting_clf.named_estimators["svc"].probability = True
voting_clf.fit(X_train, y_train)
voting_clf.score(X_test, y_test)
# 0.92
```

| 줄 | 의미 |
|---|---|
| `voting = "soft"` | 하이퍼파라미터를 속성으로 바꾸고 **다시 `fit`** |
| `named_estimators["svc"]` | **원본**(`_` 없음)을 수정해야 다음 `fit` 때 복제본에 반영됨 |
| `probability = True` | `SVC`는 기본적으로 `predict_proba`가 없다. 켜면 내부에서 **5-겹 교차 검증으로 확률을 보정**(Platt scaling)하므로 학습이 느려진다 |

**직접 vs 간접 예시** (한 샘플, 클래스 1의 확률)

| 모델 | P(클래스 1) | 직접 투표 표 |
|---|---|---|
| lr | 0.55 | 1 |
| rf | 0.52 | 1 |
| svc | 0.05 | 0 |
| **결과** | 평균 0.373 → **0** | 2:1 → **1** |

→ 간신히 1이라고 한 두 모델보다, 강하게 0이라고 확신한 svc의 의견이 반영된다.

---

## 2. 배깅과 페이스팅 (6.2)

6.1은 **다른 알고리즘**을 썼다. 6.2는 **같은 알고리즘**을 쓰되, 훈련 세트에서 **무작위로 뽑은 서로 다른 부분집합**으로 각각 학습시킨다.

| 방식 | 샘플링 | 같은 샘플이 한 예측기에 여러 번? | 사이킷런 |
|---|---|---|---|
| **배깅** (bagging = **b**ootstrap **agg**regat**ing**) | **중복 허용** (복원 추출) | 가능 | `bootstrap=True` (기본) |
| **페이스팅** (pasting) | **중복 없음** (비복원 추출) | 불가 | `bootstrap=False` |

- 집계: 분류는 **최빈값**(통계적 최빈값, 즉 다수결), 회귀는 **평균**. `predict_proba`가 있으면 확률 평균(간접 투표)을 쓴다
- 개별 예측기는 원본 전체로 학습한 것보다 **편향이 크지만**, 모으면 **편향은 비슷하고 분산은 줄어든다**
- 각 예측기를 **병렬로 학습·예측** 가능 → CPU 코어 여러 개를 쓰기 좋다 (부스팅은 순차라서 불가)

### 2-1. 사이킷런의 배깅

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            max_samples=100, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
```

| 인자 | 의미 |
|---|---|
| `DecisionTreeClassifier()` | 기반 예측기(base estimator). 아무 분류기나 가능. 규제 없는 트리 = 분산 큰 모델 → 배깅 효과가 가장 큼 |
| `n_estimators=500` | 트리 500개 |
| `max_samples=100` | 트리마다 훈련 샘플 **100개**를 뽑음. 정수 = 개수, 실수(0~1) = 비율. 기본값 1.0 (훈련 세트 크기만큼) |
| `bootstrap` (기본 `True`) | 복원 추출 → 배깅. 페이스팅은 `False` |
| `n_jobs=-1` | 사용 가능한 **모든 CPU 코어**로 병렬 학습 |
| `random_state=42` | 샘플링 무작위성 고정 |

**페이스팅으로 바꾸려면** (보충 코드):

```python
paste_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                              max_samples=100, bootstrap=False,
                              n_jobs=-1, random_state=42)
paste_clf.fit(X_train, y_train)
paste_clf.score(X_test, y_test)   # 0.92   (같은 조건 배깅: 0.904, 트리 1개: 0.856)
```

### 2-2. 결정 경계 비교 (그림 6-5)

```python
def plot_decision_boundary(clf, X, y, alpha=1.0):
    axes=[-1.5, 2.4, -1, 1.5]
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8 * alpha)
    colors = ["#78785c", "#c47b27"]
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf, X_train, y_train)
plt.title("Decision Tree")
plt.sca(axes[1])
plot_decision_boundary(bag_clf, X_train, y_train)
plt.title("Decision Trees with Bagging")
plt.ylabel("")
plt.show()
```

`plot_decision_boundary`는 5장부터 계속 쓰는 **결정 경계 그리기 패턴**이다.

| 단계 | 코드 | 의미 |
|---|---|---|
| ① 격자 만들기 | `np.meshgrid(linspace, linspace)` | 그림 영역을 100×100 = 10,000개 점으로 촘촘히 덮음 |
| ② 모델 입력 형태로 | `np.c_[x1.ravel(), x2.ravel()]` | 2차원 격자 2개를 펼쳐서 `(10000, 2)` 행렬로 |
| ③ 전부 예측 | `clf.predict(X_new).reshape(x1.shape)` | 예측 결과를 다시 100×100으로 접음 |
| ④ 색칠 | `contourf` (채우기) + `contour` (경계선) | 예측 클래스별로 영역 색칠 |
| ⑤ 데이터 점 | `X[:, 0][y == idx]` | 클래스별로 다른 색·모양 |

- `alpha` 인자: 나중에 AdaBoost 그림에서 경계를 여러 개 겹쳐 그릴 때 반투명하게 쓰려고 넣어 둔 것
- `plt.sca(ax)`: "set current axes" → 이후 `plt.xxx` 호출이 그 서브플롯에 그려짐

**그림 해석**

| | 트리 1개 | 배깅 500개 |
|---|---|---|
| 경계 | 들쭉날쭉, 잡음 샘플 하나하나에 맞춤 | 훨씬 부드러움 |
| 훈련 오류 | 거의 0 | 약간 있음 (편향 비슷) |
| 일반화 | 과대적합 | **분산 감소** → 더 잘 일반화 |

> 💡 배깅 vs 페이스팅: 부트스트래핑은 부분집합에 **다양성을 더 주므로** 배깅이 편향은 조금 더 높지만, 예측기 간 상관이 줄어 **분산이 더 감소**한다. 일반적으로 배깅을 더 선호. 시간 여유가 있으면 교차 검증으로 둘 다 평가해 보면 된다.

### 2-3. OOB 평가 (out-of-bag)

배깅은 복원 추출이라 훈련 세트 크기 m만큼 뽑으면 각 예측기는 평균적으로 **약 63%** 의 샘플만 본다. 한 번도 안 뽑힌 나머지 **약 37%** 를 **OOB 샘플**이라 하고, **별도 검증 세트 없이** 이걸로 평가할 수 있다.

**왜 63%?** — 특정 샘플 하나가 m번 뽑는 동안 한 번도 안 뽑힐 확률은 `(1 − 1/m)^m`. m이 크면 `e^{-1} ≈ 0.368`.

```python
print(1 - (1 - 1 / 1000) ** 1000)   # 0.6323045752290363  (m=1000일 때 뽑힐 확률)
print(1 - np.exp(-1))               # 0.6321205588285577  (m→∞ 극한)
```

```python
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                            oob_score=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_
# 0.896
```

| 줄 | 의미 |
|---|---|
| `oob_score=True` | 학습이 끝나면 자동으로 OOB 평가 수행 |
| `max_samples` 생략 | 기본값 1.0 → 훈련 세트 크기(375개)만큼 복원 추출. 그래야 63/37 공식이 맞음 |
| `oob_score_` | 각 훈련 샘플을, **그 샘플을 안 본 트리들만으로** 예측했을 때의 정확도 |

```python
bag_clf.oob_decision_function_[:3]  # 처음 3개 훈련 샘플의 OOB 클래스 확률
# array([[0.32352941, 0.67647059],
#        [0.3375    , 0.6625    ],
#        [1.        , 0.        ]])
```

- 첫 샘플: 이 샘플을 안 본 트리들 중 **67.6%가 양성(1)** 으로 예측
- `oob_decision_function_`은 기반 예측기에 `predict_proba`가 있을 때 확률을 반환

OOB 점수가 테스트 점수를 잘 추정했는지 확인:

```python
from sklearn.metrics import accuracy_score

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.92
```

> OOB 0.896 vs 테스트 0.92 → 같은 범위. 테스트 세트가 125개로 작아서 차이가 좀 있을 뿐, OOB는 **검증 세트를 따로 떼지 않고도** 쓸 만한 추정치를 준다. 데이터가 적을 때 특히 유용.

---

## 3. 랜덤 패치와 랜덤 서브스페이스 (6.3)

> 📌 이 절은 교재 본문에만 있고 **노트북에 코드가 없다**. 아래 코드는 개념 확인용으로 직접 작성·실행한 보충 코드.

`BaggingClassifier`는 **샘플**뿐 아니라 **특성**도 무작위로 뽑을 수 있다. 각 예측기가 **특성의 일부만** 보고 학습 → 고차원 데이터(이미지 등)에서 유용하고 학습도 빨라진다.

| 인자 | 샘플링 대상 |
|---|---|
| `max_samples`, `bootstrap` | **샘플(행)** |
| `max_features`, `bootstrap_features` | **특성(열)** — 사용법은 위와 동일 |

| 방법 | 샘플 | 특성 | 설정 |
|---|---|---|---|
| 배깅 | 샘플링 O | 전부 사용 | 기본 |
| **랜덤 패치** (random patches) | **샘플링 O** | **샘플링 O** | `max_samples<1.0` 또는 `bootstrap=True` + `max_features<1.0` 또는 `bootstrap_features=True` |
| **랜덤 서브스페이스** (random subspaces) | **전부 사용** | **샘플링 O** | `bootstrap=False, max_samples=1.0` + `max_features<1.0` 또는 `bootstrap_features=True` |

- 특성 샘플링 → 예측기 다양성 ↑ → **편향 약간 ↑, 분산 ↓**

### 3-1. 보충 코드: 50개 특성 데이터로 비교

```python
from sklearn.datasets import make_classification

X_hd, y_hd = make_classification(n_samples=1000, n_features=50,
                                 n_informative=10, random_state=42)
X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(
    X_hd, y_hd, random_state=42)

base = dict(estimator=DecisionTreeClassifier(), n_estimators=300,
            n_jobs=-1, random_state=42)

bagging   = BaggingClassifier(**base)
patches   = BaggingClassifier(**base, max_samples=0.7,
                              max_features=0.5, bootstrap_features=True)
subspaces = BaggingClassifier(**base, bootstrap=False, max_samples=1.0,
                              max_features=0.5, bootstrap_features=True)

for name, model in [("bagging", bagging), ("patches", patches),
                    ("subspaces", subspaces)]:
    model.fit(X_hd_train, y_hd_train)
    print(name, model.score(X_hd_test, y_hd_test))
# bagging   0.864
# patches   0.904
# subspaces 0.876
```

| 줄 | 의미 |
|---|---|
| `make_classification(n_features=50, n_informative=10)` | 50개 특성 중 **10개만 쓸모 있는** 인공 분류 데이터. 나머지는 잡음·중복 |
| `base = dict(...)` + `**base` | 공통 인자를 딕셔너리로 묶어 세 모델에 풀어 넣기 |
| `max_features=0.5` | 트리마다 특성 **50%(25개)** 만 사용 |
| `bootstrap_features=True` | 특성을 **중복 허용**으로 뽑음 (같은 특성이 두 번 뽑힐 수 있음) |
| `max_samples=0.7` | 랜덤 패치: 샘플도 70%만 |

어떤 트리가 어떤 특성·샘플을 봤는지 확인:

```python
len(patches.estimators_features_[0])   # 25   ← 첫 트리가 쓴 특성 인덱스 25개
patches.estimators_features_[0][:10]   # array([27, 41, 10, 12, 2, 9, 39, 28, 7, 24])
len(patches.estimators_samples_[0])    # 525  ← 750 × 0.7
```

> 💡 특성이 많고 그중 일부만 유용할 때, 특성 샘플링이 트리들끼리의 상관을 낮춰 효과가 크다. 이 아이디어를 트리 **노드 단위**로 적용한 것이 바로 다음 절의 랜덤 포레스트.

---

## 4. 랜덤 포레스트 (6.4)

**랜덤 포레스트 = 결정 트리의 배깅** (보통 `max_samples`=훈련 세트 크기) **+ 노드 분할 시 특성 무작위 선택**.

### 4-1. RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

| 인자 | 의미 |
|---|---|
| `n_estimators=500` | 트리 500개 |
| `max_leaf_nodes=16` | 각 트리의 리프 노드 최대 16개 → 트리 크기 규제. `DecisionTreeClassifier`의 하이퍼파라미터를 **거의 전부** 그대로 받는다 |
| `n_jobs=-1` | 모든 코어 사용 |

**트리 하나와 무엇이 다른가?**
- 일반 결정 트리: 노드를 나눌 때 **모든 특성** 중 최선을 찾음
- 랜덤 포레스트: 노드마다 **무작위로 고른 특성 일부(기본 √n개)** 중에서만 최선을 찾음
- → 트리들이 더 다양해짐 → 편향 ↑ 조금, 분산 ↓ 많이 → 전체적으로 더 좋은 모델

### 4-2. 랜덤 포레스트 ≈ 배깅 + `max_features="sqrt"`

```python
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(max_features="sqrt", max_leaf_nodes=16),
    n_estimators=500, n_jobs=-1, random_state=42)

bag_clf.fit(X_train, y_train)
y_pred_bag = bag_clf.predict(X_test)
np.all(y_pred_bag == y_pred_rf)  # np.True_  ← 예측이 완전히 같다
```

| 줄 | 의미 |
|---|---|
| `DecisionTreeClassifier(max_features="sqrt")` | **노드 분할마다** √(특성 수)개만 후보로 → 랜덤 포레스트의 핵심 |
| `np.all(a == b)` | 두 배열이 모든 원소에서 같은지 |

> ⚠️ 6.3의 `BaggingClassifier(max_features=...)`는 **트리 하나당** 특성 부분집합을 고정한다. 랜덤 포레스트의 `max_features`는 **노드마다 새로** 뽑는다. 이름은 같지만 적용 단위가 다르다.

> 💡 같은 결과라면 `RandomForestClassifier`를 쓰는 게 낫다 — 결정 트리에 **최적화**되어 있고 코드도 간단. 회귀는 `RandomForestRegressor`.

### 4-3. 엑스트라 트리 (Extremely Randomized Trees)

노드 분할 시 **특성뿐 아니라 임곗값까지 무작위로** 정한다 (특성마다 무작위 임곗값 하나씩 → 그중 최선 선택).

| | 랜덤 포레스트 | 엑스트라 트리 |
|---|---|---|
| 특성 후보 | 무작위 일부 | 무작위 일부 |
| 임곗값 | 특성마다 **최적값 탐색** | **무작위** |
| 편향 / 분산 | — | 편향 ↑, 분산 ↓ |
| 학습 속도 | 느림 (최적 임곗값 찾는 게 트리 학습에서 가장 오래 걸림) | **훨씬 빠름** |
| 기본 `bootstrap` | `True` | `False` |
| 사이킷런 | `RandomForestClassifier` | `ExtraTreesClassifier` (사용법 동일) |

```python
# 보충 코드 (노트북의 연습문제 8번에서 사용됨)
from sklearn.ensemble import ExtraTreesClassifier

extra_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16,
                                 n_jobs=-1, random_state=42)
extra_clf.fit(X_train, y_train)
extra_clf.score(X_test, y_test)   # 0.912  (같은 설정 랜덤 포레스트도 0.912)
```

> 어느 쪽이 나을지는 미리 알기 어렵다 → **둘 다 교차 검증으로** 비교하는 것이 정석.

### 4-4. 특성 중요도

랜덤 포레스트는 각 특성이 **불순도를 평균적으로 얼마나 줄였는지**(가중 평균, 가중치 = 노드의 훈련 샘플 수)로 특성 중요도를 계산한다. 전체 합이 1이 되도록 정규화.

```python
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
rnd_clf = RandomForestClassifier(n_estimators=500, random_state=42)
rnd_clf.fit(iris.data, iris.target)
for score, name in zip(rnd_clf.feature_importances_, iris.data.columns):
    print(round(score, 2), name)
# 0.11 sepal length (cm)
# 0.02 sepal width (cm)
# 0.44 petal length (cm)
# 0.42 petal width (cm)
```

| 줄 | 의미 |
|---|---|
| `load_iris(as_frame=True)` | DataFrame으로 받아서 `.columns`로 특성 이름을 쉽게 꺼냄 |
| `feature_importances_` | 특성별 중요도 배열, 합 = 1 |
| `zip(중요도, 이름)` | 두 리스트를 짝지어 순회 |

→ **꽃잎(petal)** 길이·너비가 86%. 꽃받침 너비는 거의 쓸모없다. **특성 선택**할 때 빠르게 감 잡는 용도로 유용.

#### MNIST 픽셀 중요도 히트맵 (그림 6-6)

```python
from sklearn.datasets import fetch_openml

X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False,
                                parser='auto')

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_mnist, y_mnist)

heatmap_image = rnd_clf.feature_importances_.reshape(28, 28)
plt.imshow(heatmap_image, cmap="hot")
cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(),
                           rnd_clf.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
plt.axis("off")
plt.show()
```

| 줄 | 의미 |
|---|---|
| `return_X_y=True` | `Bunch` 대신 `(X, y)` 튜플로 바로 받음 (3장은 `mnist.data`, `mnist.target`으로 꺼냈음) |
| `parser='auto'` | ARFF 파서 자동 선택 (pandas 있으면 빠른 파서) |
| `feature_importances_.reshape(28, 28)` | 픽셀 784개의 중요도를 이미지 모양으로 다시 접음 |
| `cmap="hot"` | 검정 → 빨강 → 노랑 → 흰색 순으로 값이 큼 |
| `colorbar(ticks=[min, max])` | 컬러바 눈금을 양 끝 2개만 두고 글자로 바꿈 |

→ 가운데(숫자가 쓰이는 곳)가 밝고, 가장자리는 어둡다. 가장자리 픽셀은 늘 0이라 분류에 쓸모가 없다.

---

## 5. 부스팅 (6.5)

**약한 학습기 여러 개를 연결해 강한 학습기**를 만드는 앙상블. 앞 모델을 보완하도록 **순차적으로** 학습하는 것이 배깅과의 결정적 차이.

| | 배깅 / 랜덤 포레스트 | 부스팅 |
|---|---|---|
| 학습 순서 | **병렬** (독립) | **순차** (앞 모델에 의존) |
| 각 모델이 보는 것 | 무작위 부분집합 | 앞 모델이 **틀린 부분** |
| 줄이는 것 | 분산 | 편향 (+ 분산) |
| 단점 | — | 병렬화 불가 → 확장성 떨어짐 |

### 5-1. AdaBoost (Adaptive Boosting)

**아이디어**: 앞 예측기가 **틀린 샘플의 가중치를 높여서** 다음 예측기가 그 샘플에 더 집중하게 한다.

#### 알고리즘 (교재 식 6-1 ~ 6-4)

처음 모든 샘플 가중치 `w⁽ⁱ⁾ = 1/m`. 예측기 j를 학습할 때마다:

| 단계 | 식 | 의미 |
|---|---|---|
| ① 가중 오류율 | `r_j = Σ(틀린 샘플의 w⁽ⁱ⁾) / Σ(모든 w⁽ⁱ⁾)` | 가중치 기준으로 몇 %를 틀렸나 |
| ② 예측기 가중치 | `α_j = η · log((1 − r_j) / r_j)` | 정확할수록 α 큼. 무작위 추측(r=0.5)이면 0, 무작위보다 나쁘면 음수 |
| ③ 샘플 가중치 갱신 | 틀린 샘플만 `w⁽ⁱ⁾ ← w⁽ⁱ⁾ · exp(α_j)` | 틀린 샘플을 키움 |
| ④ 정규화 | `w⁽ⁱ⁾ ← w⁽ⁱ⁾ / Σw` | 합을 1로 |
| 예측 | `ŷ(x) = argmax_k Σ_{ŷ_j(x)=k} α_j` | 각 예측기가 **α만큼의 표**를 던지는 가중 다수결 |

`η`는 학습률. 정해진 예측기 수에 도달하거나 완벽한 예측기가 나오면 멈춘다.

#### 직접 구현해서 그려 보기 (그림 6-8)

```python
m = len(X_train)  # 샘플 수 (375)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    sample_weights = np.ones(m) / m
    plt.sca(axes[subplot])
    for i in range(5):
        svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
        y_pred = svm_clf.predict(X_train)

        error_weights = sample_weights[y_pred != y_train].sum()
        r = error_weights / sample_weights.sum()                # 식 6-1
        alpha = learning_rate * np.log((1 - r) / r)             # 식 6-2
        sample_weights[y_pred != y_train] *= np.exp(alpha)      # 식 6-3
        sample_weights /= sample_weights.sum()                  # 정규화

        plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)
        plt.title(f"learning_rate = {learning_rate}")
    if subplot == 0:
        plt.text(-0.75, -0.95, "1", fontsize=16)
        plt.text(-1.05, -0.95, "2", fontsize=16)
        plt.text(1.0, -0.95, "3", fontsize=16)
        plt.text(-1.45, -0.5, "4", fontsize=16)
        plt.text(1.36,  -0.95, "5", fontsize=16)
    else:
        plt.ylabel("")
plt.show()
```

| 줄 | 의미 |
|---|---|
| `np.ones(m) / m` | 처음엔 모든 샘플 가중치 1/m |
| `SVC(C=0.2, gamma=0.6)` | 일부러 **규제를 강하게** 건 약한 SVM (C 작음) |
| `sample_weight=sample_weights * m` | `fit`에 샘플 가중치 전달. ×m 해서 평균이 1이 되게 (SVM의 C와 스케일이 맞도록) |
| `y_pred != y_train` | 틀린 샘플 마스크 |
| `sample_weights[마스크] *= np.exp(alpha)` | **틀린 샘플만** 가중치를 e^α배 |
| `plot_decision_boundary(..., alpha=0.4)` | 경계 5개를 반투명으로 겹쳐 그림 (여기 `alpha`는 투명도, AdaBoost의 α와 다름) |
| `plt.text(...)` | 왼쪽 그림에 경계 번호 1~5 표시 |

> ⚠️ 노트북 주석은 `equation 7-1` 등으로 되어 있는데, 이전 판 장 번호가 남은 것. 이번 판에선 6장 식이다.

**그림 해석**
- 왼쪽(η=1): 1번 경계가 틀린 샘플들에 가중치가 실려 2번 경계는 그쪽을 더 잘 맞춤 → 경계가 점점 바뀌어 감
- 오른쪽(η=0.5): 가중치 증가폭이 절반이라 변화가 더 완만
- 이 방식은 경사 하강법과 비슷하다: 파라미터 하나를 조정하는 대신 **예측기를 하나씩 추가**해 앙상블을 개선

> ⚠️ 순차 학습이라 **병렬화 불가**. 배깅보다 확장성이 낮다.

#### 사이킷런 AdaBoostClassifier

```python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=30,
    learning_rate=0.5, random_state=42)    # 원본: algorithm="SAMME" 추가 (1.8에서 삭제됨)
ada_clf.fit(X_train, y_train)
plot_decision_boundary(ada_clf, X_train, y_train)
```

| 인자 | 의미 |
|---|---|
| `DecisionTreeClassifier(max_depth=1)` | **결정 스텀프**(decision stump): 노드 1개 + 리프 2개. AdaBoost의 기본 기반 예측기 |
| `n_estimators=30` | 스텀프 30개 |
| `learning_rate=0.5` | 위 식의 η |
| SAMME | **S**tagewise **A**dditive **M**odeling using a **M**ulticlass **E**xponential loss — AdaBoost의 다중 클래스 버전. 클래스 2개면 원래 AdaBoost와 같다 |

- 보충 실행: 테스트 정확도 **0.88**
- 과대적합되면 → `n_estimators` 줄이기 또는 기반 예측기 규제 강화

### 5-2. 그레이디언트 부스팅 (Gradient Boosting)

AdaBoost처럼 앞의 오차를 보정하지만, 샘플 가중치를 바꾸는 대신 **앞 예측기가 만든 잔여 오차(residual)에 새 예측기를 학습**시킨다.

#### 직접 구현: 트리 3개로 잔차 학습

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

m = 100
rng = np.random.default_rng(seed=42)
X = rng.random((m, 1)) - 0.5                 # -0.5 ~ 0.5
noise = 0.05 * rng.standard_normal(m)
y = 3 * X[:, 0] ** 2 + noise                 # y = 3x² + 가우스 잡음

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)
```

```python
y2 = y - tree_reg1.predict(X)                # 1번 트리의 잔차
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
tree_reg2.fit(X, y2)                         # 2번 트리는 "잔차"를 맞히도록 학습

y3 = y2 - tree_reg2.predict(X)               # 1+2번으로도 남은 잔차
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
tree_reg3.fit(X, y3)
```

```python
X_new = np.array([[-0.4], [0.], [0.5]])
sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
# array([0.57356534, 0.0405142 , 0.66914249])
```

| 줄 | 의미 |
|---|---|
| `X[:, 0] ** 2` | X는 `(100, 1)` → 첫 열만 꺼내 1차원으로 제곱 |
| `y - tree_reg1.predict(X)` | **잔차** = 정답 − 지금까지의 예측 |
| `fit(X, y2)` | 입력은 그대로, **타깃만 잔차로 바뀜** |
| `sum(... for tree in ...)` | 앙상블 예측 = 모든 트리 예측의 **합** (평균 아님!) |

참값과 비교: x=−0.4 → 3×0.16=0.48, x=0 → 0, x=0.5 → 0.75. 트리 3개로 꽤 가까워졌다.

#### 그림 6-9: 잔차와 앙상블 예측

```python
def plot_predictions(regressors, X, y, axes, style,
                     label=None, data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1))
                 for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center")
    plt.axis(axes)

plt.figure(figsize=(11, 11))

plt.subplot(3, 2, 1)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="g-",
                 label="$h_1(x_1)$", data_label="Training set")
plt.ylabel("$y$  ", rotation=0)
plt.title("Residuals and tree predictions")

plt.subplot(3, 2, 2)
plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1)$", data_label="Training set")
plt.title("Ensemble predictions")

plt.subplot(3, 2, 3)
plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_2(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1)$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 4)
plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.2, 0.8],
                  style="r-", label="$h(x_1) = h_1(x_1) + h_2(x_1)$")

plt.subplot(3, 2, 5)
plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.4, 0.6], style="g-",
                 label="$h_3(x_1)$", data_style="k+",
                 data_label="Residuals: $y - h_1(x_1) - h_2(x_1)$")
plt.xlabel("$x_1$")
plt.ylabel("$y$  ", rotation=0)

plt.subplot(3, 2, 6)
plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y,
                 axes=[-0.5, 0.5, -0.2, 0.8], style="r-",
                 label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
plt.xlabel("$x_1$")
plt.show()
```

- `plot_predictions`: 회귀 모델 **리스트**를 받아 예측을 **합산**해 곡선으로 그림 → 앙상블 그대로 시각화
- `x1.reshape(-1, 1)`: 1차원 → `(500, 1)` 2차원 (모델 입력 형태)

| 행 | 왼쪽 (초록, 개별 트리) | 오른쪽 (빨강, 누적 앙상블) |
|---|---|---|
| 1 | `h₁`: 원본 y에 학습 | `h₁` 하나 — 계단 4개, 거침 |
| 2 | `h₂`: **잔차** `y − h₁`에 학습 (검은 +) | `h₁ + h₂` — 더 좋아짐 |
| 3 | `h₃`: 잔차 `y − h₁ − h₂`에 학습 | `h₁ + h₂ + h₃` — 2차곡선에 가까움 |

→ 트리가 추가될수록 **잔차는 0으로 줄고, 앙상블 예측은 정답에 가까워진다.**

#### 사이킷런 GradientBoostingRegressor

```python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3,
                                 learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
```

- 위의 수동 구현과 **같은 앙상블** (깊이 2 트리 3개, 축소 없음)
- 트리 관련 하이퍼파라미터(`max_depth`, `min_samples_leaf` 등) + 앙상블 하이퍼파라미터(`n_estimators`, `learning_rate`)

**`learning_rate` = 축소(shrinkage)**: 각 트리의 기여도를 이 값만큼 곱한다.

| `learning_rate` | 필요한 트리 수 | 특징 |
|---|---|---|
| 큼 (1.0) | 적음 | 빨리 맞추지만 과대적합·과소적합 조절이 거침 |
| 작음 (0.05) | **많음** | 일반화가 더 잘 됨 — **축소**라는 규제 기법 |

#### 조기 종료로 최적 트리 수 찾기

```python
gbrt_best = GradientBoostingRegressor(
    max_depth=2, learning_rate=0.05, n_estimators=500,
    n_iter_no_change=10, random_state=42)
gbrt_best.fit(X, y)

gbrt_best.n_estimators_
# 53
```

| 인자 / 속성 | 의미 |
|---|---|
| `n_estimators=500` | 상한. 넉넉하게 |
| `n_iter_no_change=10` | 검증 점수가 **10번 연속 개선되지 않으면** 멈춤 (조기 종료) |
| `validation_fraction` (기본 0.1) | `n_iter_no_change`를 켜면 훈련 세트의 10%를 자동으로 떼어 검증용으로 씀 |
| `tol` (기본 1e-4) | "개선"으로 인정할 최소 폭 |
| `n_estimators_` (끝에 `_`) | **실제로 만든 트리 수** → 53개에서 멈춤 |

> 💡 `n_iter_no_change`가 너무 작으면 잡음 때문에 너무 일찍 멈출 수 있고(과소적합), 너무 크면 과대적합. 10 정도가 무난.

#### 그림 6-10: 트리 수 비교

```python
fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

plt.sca(axes[0])
plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], style="r-",
                 label="Ensemble predictions")
plt.title(f"learning_rate={gbrt.learning_rate}, "
          f"n_estimators={gbrt.n_estimators_}")
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)

plt.sca(axes[1])
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8], style="r-")
plt.title(f"learning_rate={gbrt_best.learning_rate}, "
          f"n_estimators={gbrt_best.n_estimators_}")
plt.xlabel("$x_1$")
plt.show()
```

- `plot_predictions([gbrt], ...)`: GBRT 모델 하나도 리스트로 감싸면 그대로 동작
- 왼쪽(η=1.0, 3개): 트리가 부족해 **과소적합**
- 오른쪽(η=0.05, 53개): 부드럽게 잘 맞음

#### 확률적 그레이디언트 부스팅 (교재 설명)

`subsample=0.25`처럼 지정하면 각 트리가 **무작위로 고른 훈련 샘플 25%** 로 학습 → 편향 ↑, 분산 ↓, **학습 속도 ↑**.

### 5-3. 히스토그램 기반 그레이디언트 부스팅 (HGB)

대용량 데이터용 GB. 입력 특성을 **구간(bin)으로 나눠 정수로 바꿔서** 학습한다.

| 특징 | 내용 |
|---|---|
| 구간 수 | `max_bins` (기본·최대 255) |
| 속도 | 일반 GB 대비 **수백 배** 빠를 수 있음. 복잡도 O(b×m) (b=구간 수) |
| 정밀도 손실 | 구간화로 약간 손실 → 규제 효과도 있음 (데이터에 따라 득/실) |
| 결측치 | **그대로 지원** (Imputer 불필요) |
| 범주형 특성 | `categorical_features`로 지정하면 **직접 지원** (원-핫 불필요, 단 정수 0~max_bins−1로 인코딩돼 있어야 함) |
| 조기 종료 | 샘플 10,000개 넘으면 **자동으로 켜짐** (`early_stopping="auto"`) |
| 클래스 | `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` |

#### 캘리포니아 주택 데이터 (2장 데이터 재사용)

```python
from pathlib import Path
import tarfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing_labels = train_set["median_house_value"]
housing = train_set.drop("median_house_value", axis=1)
```

- 2장의 `load_housing_data()`와 같다. 이번엔 계층 샘플링 없이 단순 분할
- 타깃(`median_house_value`)과 특성을 분리

```python
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder

hgb_reg = make_pipeline(
    make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                            remainder="passthrough"),
                            # 원본: force_int_remainder_cols=False 추가 (1.9에서 삭제됨)
    HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
)
hgb_reg.fit(housing, housing_labels)
```

| 줄 | 의미 |
|---|---|
| `OrdinalEncoder()` on `ocean_proximity` | 범주 5개(`<1H OCEAN`, `INLAND` …)를 정수 0~4로. 원-핫이 필요 없다 |
| `remainder="passthrough"` | 나머지 수치 특성은 **그대로 통과**. 결측치(`total_bedrooms`)도 HGB가 직접 처리하므로 Imputer 없음 |
| `make_column_transformer` 출력 순서 | 변환한 열이 **앞**, passthrough 열이 뒤 → `ocean_proximity`가 **0번 열** |
| `categorical_features=[0]` | 0번 열을 범주형으로 취급하라고 HGB에 알림 |

**2장과 비교**: 2장은 Imputer + 스케일러 + 원-핫 + 클러스터 유사도 + 로그 변환 등 긴 파이프라인이 필요했다. HGB는 **전처리 거의 없이** 비슷한 성능을 낸다 — 스케일링도 필요 없다(트리 기반이라).

```python
from sklearn.model_selection import cross_val_score

hgb_rmses = -cross_val_score(hgb_reg, housing, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10)
pd.Series(hgb_rmses).describe()
# count       10.000000
# mean     47613.307194
# std       1295.422509
# min      44963.213061
# ...
# max      49176.368465
```

- `scoring="neg_root_mean_squared_error"`: 사이킷런 점수는 "클수록 좋음"이라 RMSE에 음수를 붙여 반환 → 앞에 `-`를 붙여 되돌림
- 10-겹 CV 평균 RMSE **약 47,600달러** — 2장에서 긴 전처리 파이프라인과 함께 만든 랜덤 포레스트(CV RMSE 평균 47,038달러)와 거의 같은데, 전처리도 하이퍼파라미터 튜닝도 거의 안 한 상태

> 💡 **파이썬 생태계의 GB 라이브러리**: XGBoost, LightGBM, CatBoost — 대용량·대회에서 많이 쓰인다. 사이킷런 HGB는 LightGBM에서 영감을 받았다.

---

## 6. 스태킹 (6.6)

투표처럼 **단순한 함수(다수결, 평균)** 로 집계하는 대신, **집계 자체를 모델에 학습**시키자는 아이디어. 이 마지막 모델을 **블렌더**(blender) 또는 **메타 학습기**(meta learner)라고 한다.

**블렌더 훈련 세트를 만드는 법**
1. 각 예측기에 `cross_val_predict()`를 써서 훈련 샘플마다 **표본 외(out-of-sample) 예측**을 얻는다 (자기가 학습한 샘플을 예측하면 과하게 정확해서 블렌더가 속는다)
2. 이 예측들을 **입력 특성**으로, 원래 타깃을 **타깃**으로 블렌더를 학습
3. 블렌더 학습 후, 기반 예측기들은 **전체 훈련 세트로 한 번 더 학습**

```python
from sklearn.ensemble import StackingClassifier

stacking_clf = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    final_estimator=RandomForestClassifier(random_state=43),
    cv=5  # 교차 검증 폴드 수
)
stacking_clf.fit(X_train, y_train)
stacking_clf.score(X_test, y_test)
# 0.928
```

| 인자 | 의미 |
|---|---|
| `estimators` | 1층 기반 예측기들. `VotingClassifier`와 같은 형식 |
| `final_estimator` | **블렌더**. 지정 안 하면 `LogisticRegression` |
| `cv=5` | 블렌더용 예측을 5-겹 `cross_val_predict`로 생성 |
| `SVC(probability=True)` | 블렌더 입력으로 무엇을 쓸지는 `stack_method="auto"`가 정함: `predict_proba` → `decision_function` → `predict` 순으로 있는 것 사용 |
| `random_state=43` | 블렌더 RF를 1층 RF와 다른 시드로 |

**결과 비교 (moons 테스트 세트)**

| 모델 | 정확도 |
|---|---|
| 로지스틱 회귀 | 0.864 |
| 랜덤 포레스트 / SVC | 0.896 |
| 직접 투표 | 0.912 |
| 간접 투표 | 0.920 |
| **스태킹** | **0.928** |

> 💡 블렌더 층을 여러 개 쌓을 수도 있다 (블렌더들의 예측을 다시 블렌더에 → 다층 스태킹). 성능은 조금 더 오르지만 학습 시간과 복잡도가 크게 늘어난다.

> ⚠️ 스태킹이 **항상** 이기는 건 아니다. 학습 비용이 크므로 투표와 비교해서 고른다 (아래 연습문제가 그런 사례).

---

## 7. 연습문제 (6.7)

> 1~7번은 개념 문제(교재 부록). 노트북에는 8, 9번 코드 풀이만 있다.

### 7-1. 8번: 투표 분류기

**① 데이터 분할** (훈련 50,000 / 검증 10,000 / 테스트 10,000)

```python
X_train, y_train = X_mnist[:50_000], y_mnist[:50_000]
X_valid, y_valid = X_mnist[50_000:60_000], y_mnist[50_000:60_000]
X_test, y_test = X_mnist[60_000:], y_mnist[60_000:]
```

- `50_000`: 파이썬 숫자 리터럴의 `_`는 자릿수 구분용 (50000과 같다)

**② 개별 분류기 4종 학습**

```python
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)

estimators = [random_forest_clf, extra_trees_clf, svm_clf, mlp_clf]
for estimator in estimators:
    print("Training the", estimator)
    estimator.fit(X_train, y_train)

[estimator.score(X_valid, y_valid) for estimator in estimators]
# [0.9736, 0.9743, 0.8662, 0.9613]
```

| 모델 | 메모 | 검증 정확도 |
|---|---|---|
| 랜덤 포레스트 | | 0.9736 |
| 엑스트라 트리 | | 0.9743 |
| `LinearSVC(max_iter=100, tol=20)` | 일부러 빨리 끝나게 설정 → 성능 낮음 | 0.8662 |
| `MLPClassifier` | 다층 퍼셉트론(9장에서 본격 등장) | 0.9613 |

**③ 투표 분류기로 묶기**

```python
from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]
voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
voting_clf.score(X_valid, y_valid)
# 0.975
```

**④ 함정: 복제본은 클래스 "인덱스"로 학습됐다**

```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_valid_encoded = encoder.fit_transform(y_valid)
# MNIST는 숫자 = 인덱스이므로 그냥 정수로 바꿔도 된다
y_valid_encoded = y_valid.astype(np.int64)

[estimator.score(X_valid, y_valid_encoded)
 for estimator in voting_clf.estimators_]
# [0.9736, 0.9743, 0.8662, 0.9613]
```

- `VotingClassifier`는 내부에서 레이블을 `LabelEncoder`로 **0, 1, 2 … 인덱스로 바꿔서** 복제본을 학습시킨다
- 그래서 `estimators_`의 복제본을 직접 평가할 때 `y_valid`(문자열 `'0'`~`'9'`)를 주면 전부 틀린 것으로 나온다 → **인덱스로 변환해서** 넣어야 함

**⑤ 약한 SVM 빼기**

```python
voting_clf.set_params(svm_clf="drop")
voting_clf.estimators          # svm_clf 자리가 'drop'으로 바뀜 (학습 전 목록)
voting_clf.estimators_         # 학습된 목록엔 아직 LinearSVC가 남아 있음!

svm_clf_trained = voting_clf.named_estimators_.pop("svm_clf")
voting_clf.estimators_.remove(svm_clf_trained)

voting_clf.score(X_valid, y_valid)
# 0.9761   ← 0.975에서 조금 개선
```

| 줄 | 의미 |
|---|---|
| `set_params(svm_clf="drop")` | 이름으로 추정기를 `"drop"`(제외)으로 지정. **다음 `fit`부터** 반영 |
| `named_estimators_.pop(...)` | 재학습 대신 학습된 딕셔너리에서 직접 꺼내 제거 |
| `estimators_.remove(...)` | 리스트에서도 같은 객체 제거 (둘 다 지워야 일관됨) |

**⑥ 간접 투표 시도**

```python
voting_clf.voting = "soft"         # 재학습 불필요 (집계 방식만 바뀜)
voting_clf.score(X_valid, y_valid)
# 0.9703   ← 이번엔 직접 투표가 더 좋음
```

**⑦ 테스트 세트 최종 평가**

```python
voting_clf.voting = "hard"
voting_clf.score(X_test, y_test)
# 0.9733

[estimator.score(X_test, y_test.astype(np.int64))
 for estimator in voting_clf.estimators_]
# [0.968, 0.9703, 0.9618]
```

→ 최고 단일 모델 오류율 약 **3% → 2.7%**, 오류가 약 **10% 감소**.

### 7-2. 9번: 스태킹 앙상블

**① 수동 스태킹 — 검증 세트 예측으로 블렌더 학습**

```python
X_valid_predictions = np.empty((len(X_valid), len(estimators)), dtype=object)

for index, estimator in enumerate(estimators):
    X_valid_predictions[:, index] = estimator.predict(X_valid)

X_valid_predictions
# array([['3', '3', '3', '3'],
#        ['8', '8', '8', '8'],
#        ...], dtype=object)
```

| 줄 | 의미 |
|---|---|
| `np.empty((10000, 4), dtype=object)` | 빈 배열. 예측이 문자열이라 `dtype=object` |
| `[:, index] = ...` | 각 열 = 한 모델의 예측 → **새 훈련 세트의 특성 4개** |

- 검증 세트는 4개 모델이 **학습에 쓰지 않은 데이터**라서 블렌더용으로 쓸 수 있다 (= 홀드아웃 방식 블렌딩)

```python
rnd_forest_blender = RandomForestClassifier(n_estimators=200, oob_score=True,
                                            random_state=42)
rnd_forest_blender.fit(X_valid_predictions, y_valid)
rnd_forest_blender.oob_score_
# 0.9738
```

- 블렌더도 랜덤 포레스트. `oob_score=True`로 별도 검증 없이 성능 추정

**② 테스트 세트 평가**

```python
X_test_predictions = np.empty((len(X_test), len(estimators)), dtype=object)
for index, estimator in enumerate(estimators):
    X_test_predictions[:, index] = estimator.predict(X_test)

y_pred = rnd_forest_blender.predict(X_test_predictions)
accuracy_score(y_test, y_pred)
# 0.9688   ← 투표 분류기(0.9733)보다 낮음
```

**③ StackingClassifier로 다시**

```python
X_train_full, y_train_full = X_mnist[:60_000], y_mnist[:60_000]

stack_clf = StackingClassifier(named_estimators,
                               final_estimator=rnd_forest_blender)
stack_clf.fit(X_train_full, y_train_full)
stack_clf.score(X_test, y_test)
# 0.9795   ← 가장 좋음
```

**왜 더 좋아졌나?**

| | 수동 스태킹 | `StackingClassifier` |
|---|---|---|
| 블렌더 학습 데이터 | 검증 세트 10,000개 | **K-겹 CV**로 훈련 세트 전체 60,000개의 표본 외 예측 |
| 기반 모델 학습 데이터 | 50,000개 | 60,000개 (훈련+검증 합침) |
| 블렌더 입력 | `predict` (클래스 레이블) | **`predict_proba`** (클래스별 확률 → 정보량 많음) |
| 결과 | 0.9688 | **0.9795** |

---

## 8. 정리

### 8-1. 전체 흐름

```
약한 학습기 여러 개 + 큰 수의 법칙                   ← 6.1 앙상블의 원리 (단, 예측기 다양성 필수)
   ↓
VotingClassifier (hard / soft)                       ← 6.1 다른 알고리즘 + 다수결/확률 평균
   ↓
BaggingClassifier (bootstrap=True/False)             ← 6.2 같은 알고리즘 + 다른 샘플 → 분산 ↓
   ├ oob_score=True → 검증 세트 없이 평가 (~37%)
   └ max_features, bootstrap_features               ← 6.3 랜덤 패치 / 서브스페이스 (특성 샘플링)
   ↓
RandomForestClassifier (노드마다 √n개 특성)          ← 6.4 = 배깅 + max_features="sqrt"
   ├ ExtraTreesClassifier (임곗값까지 무작위)
   └ feature_importances_ (특성 중요도)
   ↓
AdaBoostClassifier (틀린 샘플 가중치 ↑)              ← 6.5 순차 학습, 결정 스텀프
GradientBoostingRegressor (잔차에 학습)              ← 6.5 learning_rate(축소) + n_iter_no_change
HistGradientBoostingRegressor (구간화)               ← 6.5 대용량, 결측치·범주형 직접 지원
   ↓
StackingClassifier (블렌더가 집계를 학습)             ← 6.6 cross_val_predict로 블렌더 훈련 세트
```

### 8-2. 개념과 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| 투표 | `VotingClassifier(estimators=[...], voting="soft")` | 서로 다른 알고리즘으로 다양성 확보 |
| 배깅·페이스팅 | `BaggingClassifier(bootstrap=True/False, n_jobs=-1)` | 같은 알고리즘 + 다른 샘플 → 분산 ↓ |
| OOB 평가 | `oob_score=True` → `oob_score_` | 검증 세트 없이 일반화 성능 추정 |
| 특성 샘플링 | `max_features`, `bootstrap_features` | 고차원 데이터에서 다양성 ↑ |
| 랜덤 포레스트 | `RandomForestClassifier`, `ExtraTreesClassifier` | 노드마다 특성을 무작위로 골라 분할 |
| 특성 중요도 | `feature_importances_` | 불순도 감소량의 가중 평균 |
| AdaBoost | `AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))` | 틀린 샘플의 가중치 ↑, 순차 학습 |
| 그레이디언트 부스팅 | `GradientBoostingRegressor(learning_rate, n_iter_no_change)` | 잔차 학습, 축소 + 조기 종료 |
| HGB | `HistGradientBoostingRegressor` | 구간화로 대용량, 결측치·범주형 직접 지원 |
| 스태킹 | `StackingClassifier(final_estimator=...)` | 블렌더가 집계 방식을 학습 |

### 8-3. 앙상블 방법 비교

| 방법 | 예측기 종류 | 데이터 | 학습 | 집계 | 사이킷런 |
|---|---|---|---|---|---|
| 투표 | **서로 다름** | 같음 | 병렬 | 다수결 / 확률 평균 | `VotingClassifier`, `VotingRegressor` |
| 배깅 | 같음 | 샘플 복원 추출 | 병렬 | 다수결 / 평균 | `BaggingClassifier(bootstrap=True)` |
| 페이스팅 | 같음 | 샘플 비복원 추출 | 병렬 | 다수결 / 평균 | `BaggingClassifier(bootstrap=False)` |
| 랜덤 패치 | 같음 | 샘플 + **특성** 샘플링 | 병렬 | 다수결 / 평균 | `BaggingClassifier(max_samples, max_features)` |
| 랜덤 서브스페이스 | 같음 | **특성만** 샘플링 | 병렬 | 다수결 / 평균 | `BaggingClassifier(bootstrap=False, max_features)` |
| 랜덤 포레스트 | 트리 | 배깅 + 노드별 특성 샘플링 | 병렬 | 다수결 / 평균 | `RandomForestClassifier` |
| 엑스트라 트리 | 트리 | + 무작위 임곗값 | 병렬 | 다수결 / 평균 | `ExtraTreesClassifier` |
| AdaBoost | 같음 (스텀프) | 샘플 **가중치** 조정 | **순차** | α 가중 다수결 | `AdaBoostClassifier` |
| 그레이디언트 부스팅 | 트리 | 앞 모델의 **잔차** | **순차** | **합** | `GradientBoostingRegressor`, `HistGradientBoosting*` |
| 스태킹 | 서로 다름 | 같음 (+ CV) | 병렬 → 블렌더 | **학습된 모델** | `StackingClassifier`, `StackingRegressor` |

### 8-4. 주요 결과

| 데이터 | 모델 / 설정 | 지표 | 값 |
|---|---|---|---|
| moons | 로지스틱 / RF / SVC | 테스트 정확도 | 0.864 / 0.896 / 0.896 |
| moons | 직접 투표 → 간접 투표 | 테스트 정확도 | 0.912 → 0.920 |
| moons | 트리 1개 (보충) | 테스트 정확도 | 0.856 |
| moons | 배깅 500 (max_samples=100) / 페이스팅 (보충) | 테스트 정확도 | 0.904 / 0.920 |
| moons | 배깅 500 (전체 샘플) | OOB / 테스트 | 0.896 / 0.920 |
| moons | 랜덤 포레스트 = 배깅(`max_features="sqrt"`) | 예측 일치 | True |
| moons | AdaBoost 스텀프 30개 (보충) | 테스트 정확도 | 0.88 |
| moons | **스태킹** | 테스트 정확도 | **0.928** |
| 50특성 인공 데이터 | 배깅 / 랜덤 패치 / 서브스페이스 (보충) | 테스트 정확도 | 0.864 / 0.904 / 0.876 |
| iris | 랜덤 포레스트 | 꽃잎 길이·너비 중요도 | 0.44 / 0.42 |
| 2차식 | GB 조기 종료 (η=0.05) | 최종 트리 수 | 53 |
| 주택 | HGB 파이프라인 | 10-겹 CV RMSE 평균 | 약 47,613 |
| MNIST | RF / ET / LinearSVC / MLP | 검증 정확도 | 0.9736 / 0.9743 / 0.8662 / 0.9613 |
| MNIST | 투표 (SVM 제외, hard) | 테스트 정확도 | 0.9733 |
| MNIST | 수동 블렌딩 (RF 블렌더) | 테스트 정확도 | 0.9688 |
| MNIST | **`StackingClassifier`** | 테스트 정확도 | **0.9795** |

### 8-5. 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `estimators=[("이름", 모델), ...]` | 투표·스태킹의 기반 예측기 지정 | 1, 6, 7 |
| `named_estimators_` / `estimators_` | **학습된 복제본** 꺼내기 (`_` 없는 쪽은 원본) | 1-1, 7-1 |
| `ensemble.named_estimators["x"].param = ...` → `fit` | 기반 예측기 설정 바꾸고 재학습 | 1-2 |
| `n_jobs=-1` | 모든 CPU 코어로 병렬 학습 | 2 ~ 4 |
| `oob_score=True` → `oob_score_` | 검증 세트 없이 일반화 성능 추정 | 2-3, 7-2 |
| `np.meshgrid` + `np.c_[ravel, ravel]` + `contourf` | 2차원 결정 경계 그리기 | 2-2, 5-1 |
| `feature_importances_.reshape(28, 28)` | 픽셀 중요도를 이미지로 | 4-4 |
| `fit(X, y, sample_weight=w)` | 샘플 가중치 학습 (AdaBoost 원리) | 5-1 |
| `y - model.predict(X)` → 새 모델의 타깃 | 잔차 학습 (GB 원리) | 5-2 |
| `sum(m.predict(X) for m in models)` | 부스팅 앙상블 예측 = 합 | 5-2 |
| `n_iter_no_change=10` → `n_estimators_` | 조기 종료와 실제 트리 수 확인 | 5-2 |
| `make_column_transformer(..., remainder="passthrough")` | 일부 열만 변환, 나머지 통과 | 5-3 |
| `-cross_val_score(scoring="neg_root_mean_squared_error")` | CV RMSE | 5-3 |
| `set_params(이름="drop")` | 앙상블에서 예측기 제외 | 7-1 |

### 8-6. 핵심 원칙

1. 앙상블은 예측기가 **다양할수록**(오류가 서로 독립일수록) 효과가 크다.
2. 모든 예측기가 확률을 낼 수 있으면 직접 투표보다 간접 투표가 대체로 낫다.
3. 배깅·랜덤 포레스트는 분산을 줄이고 병렬화가 쉽다. OOB 점수로 별도 검증 세트 없이 평가할 수 있다.
4. 랜덤 포레스트는 특성 중요도를 함께 제공하므로 특성 선택·데이터 이해에도 유용하다.
5. 부스팅은 순차 학습이라 병렬화가 어렵다. 학습률을 낮추고(축소) 조기 종료로 트리 수를 정한다.
6. 대용량 표 데이터에는 히스토그램 기반 그레이디언트 부스팅(HGB)이 빠르고 강력한 기본 선택이다.
7. 스태킹의 블렌더는 반드시 홀드아웃(교차 검증) 예측으로 훈련해야 과대적합을 피한다.

---

> ➡️ **다음 장 — [7장. 차원 축소](07_dimensionality_reduction.md)**  
> 모델을 바꾸는 대신 **데이터의 특성 수를 줄이는** 쪽으로 — 차원 축소. 5장의 PCA 축 정렬, 6장의 특성 중요도·랜덤 서브스페이스와 이어진다.

[← 5장. 결정 트리](05_decision_trees.md) · [📚 전체 목차](../README.md) · [7장. 차원 축소 →](07_dimensionality_reduction.md)
