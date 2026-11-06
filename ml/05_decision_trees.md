# 5장. 결정 트리

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `05_decision_trees.ipynb`
>
> 결정 트리는 **"질문 하나로 데이터를 두 갈래로 나누는 일"을 반복**하는 모델. 분류·회귀 모두 가능하고, 스케일링 같은 전처리가 거의 필요 없으며, 왜 그렇게 예측했는지 사람이 그대로 읽을 수 있다(**화이트박스** 모델). 대신 **훈련 데이터에 쉽게 과대적합**되고, 데이터가 조금만 바뀌어도 **모양이 크게 바뀌는(분산이 큰)** 약점이 있다 → 6장 랜덤 포레스트로 이어지는 이유.
>
> 실습 데이터
>
> | 데이터 | 사용 절 | 내용 |
> |---|---|---|
> | 붓꽃(iris) | 5.1~5.3, 5.9, 5.10 | 150개, 3품종(setosa·versicolor·virginica). **꽃잎 길이·너비 2개 특성만** 사용 |
> | moons | 5.7, 연습문제 | 반달 2개가 맞물린 이진 분류용 가짜 데이터 (`make_moons`) |
> | 2차 함수 + 잡음 | 5.8 | `y = x² + 잡음`, 200개 (회귀용) |
>
> 실행 환경: 노트북 요구사항은 scikit-learn ≥ 1.6.1. 이 노트의 출력값은 **scikit-learn 1.9.1**로 직접 실행한 결과.

## 목차

1. [결정 트리 학습과 시각화 (5.1)](#1-결정-트리-학습과-시각화-51)
2. [예측 (5.2)](#2-예측-52)
3. [클래스 확률 추정 (5.3)](#3-클래스-확률-추정-53)
4. [CART 훈련 알고리즘 (5.4)](#4-cart-훈련-알고리즘-54)
5. [계산 복잡도 (5.5)](#5-계산-복잡도-55)
6. [지니 불순도 또는 엔트로피? (5.6)](#6-지니-불순도-또는-엔트로피-56)
7. [규제 하이퍼파라미터 (5.7)](#7-규제-하이퍼파라미터-57)
8. [회귀 (5.8)](#8-회귀-58)
9. [축 방향에 대한 민감성 (5.9)](#9-축-방향에-대한-민감성-59)
10. [결정 트리의 분산 문제 (5.10)](#10-결정-트리의-분산-문제-510)
11. [부록: `tree_`로 트리 구조 직접 다루기](#11-부록-tree_로-트리-구조-직접-다루기)
12. [연습문제 풀이](#12-연습문제-풀이)
13. [전체 흐름 요약](#13-전체-흐름-요약)

---

## 1. 결정 트리 학습과 시각화 (5.1)

### 1-1. 훈련

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
```

| 줄 | 의미 |
|---|---|
| `load_iris(as_frame=True)` | 특성은 DataFrame, 타깃은 Series로 받음 → 열 이름으로 고르기 편함 |
| `[["petal length (cm)", "petal width (cm)"]]` | 4개 특성 중 **꽃잎 길이·너비 2개만**. 2차원이라 결정 경계를 평면에 그릴 수 있음 |
| `.values` | NumPy 배열로 변환 (특성 이름 없이 학습 → 예측 때 리스트를 넣어도 경고 없음) |
| `y_iris` | 0=setosa, 1=versicolor, 2=virginica |
| `max_depth=2` | 질문을 **최대 2단계**까지만. 그림으로 보기 좋게 일부러 작게 제한 |
| `random_state=42` | 결정 트리 훈련에도 무작위성이 있음 (10절에서 자세히) |

> 💡 **스케일링을 안 했다.** 결정 트리는 "특성 하나 ≤ 임곗값" 질문만 하므로 특성의 단위·크기와 무관하다. 4장 경사 하강법 모델들과 가장 다른 점.

### 1-2. 시각화 — `export_graphviz`

```python
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file="my_iris_tree.dot",
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

from graphviz import Source
Source.from_file("my_iris_tree.dot")       # 노트북 안에서 바로 그림으로

!dot -Tpng "my_iris_tree.dot" -o "my_iris_tree.png"   # 명령줄로 PNG 변환
```

| 인자 | 의미 |
|---|---|
| `out_file` | 트리를 **Graphviz `.dot` 텍스트 파일**로 저장 (그림 정보를 글자로 적은 파일) |
| `feature_names` / `class_names` | 노드에 `x[0]` 대신 실제 특성·클래스 이름 표시 |
| `rounded=True` | 모서리가 둥근 상자 |
| `filled=True` | 다수 클래스 색으로 칠하고, **순수할수록 진하게** |
| `Source.from_file` | `graphviz` 파이썬 패키지로 `.dot`을 렌더링 |
| `dot -Tpng` | Graphviz 명령줄 도구로 PNG 변환 (`!`는 노트북에서 셸 명령 실행) |

> 💡 Graphviz 없이 글자로 보고 싶으면 `sklearn.tree.export_text`, 그림이면 `sklearn.tree.plot_tree`(matplotlib)도 있다.

```python
from sklearn.tree import export_text
print(export_text(tree_clf, feature_names=["petal length", "petal width"],
                  show_weights=True))
# |--- petal length <= 2.45
# |   |--- weights: [50.00, 0.00, 0.00] class: 0
# |--- petal length >  2.45
# |   |--- petal width <= 1.75
# |   |   |--- weights: [0.00, 49.00, 5.00] class: 1
# |   |--- petal width >  1.75
# |   |   |--- weights: [0.00, 1.00, 45.00] class: 2
```

### 1-3. 그림 5-1 읽는 법

```
                     [노드 0: 루트]
                petal length ≤ 2.45 ?
                gini=0.667  samples=150
                value=[50, 50, 50]
               True /            \ False
     [노드 1: 리프]               [노드 2]
     gini=0.0                petal width ≤ 1.75 ?
     samples=50              gini=0.5  samples=100
     value=[50, 0, 0]        value=[0, 50, 50]
     → setosa               True /          \ False
                    [노드 3: 리프]         [노드 4: 리프]
                    gini=0.168            gini=0.043
                    samples=54            samples=46
                    value=[0, 49, 5]      value=[0, 1, 45]
                    → versicolor          → virginica
```

| 항목 | 의미 |
|---|---|
| **루트 노드** | 깊이 0, 맨 위 노드. 전체 150개에서 시작 |
| **리프 노드** | 자식이 없는 노드. 여기서 예측값이 정해짐 |
| `samples` | 이 노드에 도달한 **훈련 샘플 수** |
| `value` | 이 노드에 도달한 샘플의 **클래스별 개수** |
| `gini` | 이 노드의 **지니 불순도** (0이면 한 클래스만 있는 **순수** 노드) |
| 왼쪽 가지 | 조건이 **참(True)**일 때. 오른쪽은 거짓 |

**지니 불순도** — 노드 *i*에서 무작위로 두 샘플을 뽑았을 때 클래스가 서로 다를 확률

$$G_i = 1 - \sum_{k=1}^{n} p_{i,k}^2 \qquad (p_{i,k} = \text{노드 } i \text{에서 클래스 } k \text{의 비율})$$

| 노드 | 계산 | 값 |
|---|---|---|
| 루트 [50, 50, 50] | 1 − (1/3)² × 3 | **0.667** (3클래스 최대 불순) |
| 노드 1 [50, 0, 0] | 1 − 1² | **0** (순수) |
| 노드 2 [0, 50, 50] | 1 − 0.5² − 0.5² | **0.5** |
| 노드 3 [0, 49, 5] | 1 − (49/54)² − (5/54)² | **0.168** |
| 노드 4 [0, 1, 45] | 1 − (1/46)² − (45/46)² | **0.043** |

> 💡 사이킷런은 **CART** 알고리즘을 써서 항상 **이진 트리**(자식이 정확히 2개)를 만든다. ID3 같은 다른 알고리즘은 자식이 3개 이상인 노드도 만들 수 있음.

---

## 2. 예측 (5.2)

새 꽃이 들어오면 **루트부터 질문에 답하며 내려가** 리프의 다수 클래스를 출력.

> 예) 꽃잎 길이 5cm, 너비 1.5cm → 루트 "길이 ≤ 2.45?" 아니오 → 오른쪽 → "너비 ≤ 1.75?" 예 → 노드 3 → **versicolor**

### 2-1. 결정 경계 그리기 (그림 5-2)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
plt.figure(figsize=(8, 4))

lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)

for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
             style, label=f"Iris {name}")
```

| 줄 | 의미 |
|---|---|
| `np.meshgrid(a, b)` | x축 100개 × y축 100개 → 평면 위 **격자점 10,000개**의 좌표 행렬 2개 |
| `.ravel()` | 100×100 행렬을 1차원 10,000개로 펼침 |
| `np.c_[a, b]` | 펼친 두 배열을 열로 합쳐 `(10000, 2)` → 모델 입력 형태 |
| `.predict(...).reshape(lengths.shape)` | 격자점마다 예측 후 다시 100×100으로 접음 |
| `plt.contourf` | 예측값이 같은 영역을 **색으로 채움** → 결정 경계 |
| `"yo"`, `"bs"`, `"g^"` | 색+마커: 노랑 원 / 파랑 사각형 / 초록 삼각형 |
| `X_iris[:, 0][y_iris == idx]` | 클래스 `idx`인 샘플의 꽃잎 길이만 골라냄 |

> 💡 meshgrid → ravel → `np.c_` → predict → reshape → contourf 는 **2차원 결정 경계 그리기 표준 패턴**. 이 장에서 계속 반복된다.

### 2-2. 분할선 표시

```python
tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_deeper.fit(X_iris, y_iris)
th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
# th0=2.45, th1=1.75, th2a=4.95, th2b=4.85

plt.plot([th0, th0], [0, 3], "k-", linewidth=2)       # 깊이 0: 세로 실선
plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)  # 깊이 1: 가로 파선
plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)   # 깊이 2: 점선 (아래쪽)
plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)   # 깊이 2: 점선 (위쪽)
```

- `tree_.threshold` → 노드별 분할 임곗값 배열. `[[0, 2, 3, 6]]`처럼 **팬시 인덱싱**으로 원하는 노드만 꺼냄
- 깊이 3 트리를 따로 만든 이유: `max_depth=3`이면 **어디서 더 나눌지**(점선)를 보여 주려고
- 모든 분할선이 **축에 수직** → 결정 경계는 항상 **계단 모양 직사각형 영역**의 조합 (9절의 약점과 연결)

### 2-3. 모델 해석 — 화이트박스 vs 블랙박스

| 화이트박스 | 블랙박스 |
|---|---|
| 결정 트리 | 랜덤 포레스트, 신경망 |
| 예측 이유를 **질문 경로**로 그대로 설명 가능 | 좋은 예측을 하지만 "왜?"를 설명하기 어려움 |

트리 구조는 `tree_clf.tree_`에 있음 → [11절 부록](#11-부록-tree_로-트리-구조-직접-다루기)에서 직접 다룸.

---

## 3. 클래스 확률 추정 (5.3)

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)
# array([[0.   , 0.907, 0.093]])

tree_clf.predict([[5, 1.5]])
# array([1])  → versicolor
```

- 샘플이 도달한 **리프 노드의 클래스 비율**이 곧 확률
- (5, 1.5) → 노드 3 `value=[0, 49, 5]` → `[0/54, 49/54, 5/54]` = `[0, 0.907, 0.093]`
- `predict` = 확률이 가장 높은 클래스

> ⚠️ **같은 리프에 들어가면 확률이 완전히 같다.** (6, 1.5)도, (5, 1.0)도 노드 3이면 똑같이 90.7%. 리프 경계 근처에 있어도 확신도가 낮아지지 않음 → 결정 트리 확률은 거칠다.

---

## 4. CART 훈련 알고리즘 (5.4)

> 노트북에는 이 절의 코드가 없다. 아래 코드는 원리를 확인하려고 **직접 작성한 것**.

### 4-1. 원리

**CART** (Classification And Regression Tree)
1. 특성 하나 *k*와 임곗값 *t_k*를 골라 훈련 세트를 **두 부분집합**으로 나눔
2. 가능한 모든 (*k*, *t_k*) 쌍 중에서 아래 **비용 함수가 최소**인 것을 선택
3. 나뉜 부분집합에 같은 방법을 **재귀적으로** 반복
4. `max_depth` 도달, 또는 불순도를 더 줄이는 분할을 못 찾으면 멈춤

**분류용 CART 비용 함수**

$$J(k, t_k) = \frac{m_\text{left}}{m} G_\text{left} + \frac{m_\text{right}}{m} G_\text{right}$$

- *G* = 왼쪽/오른쪽 자식의 불순도, *m* = 샘플 수
- **샘플 수로 가중 평균**한 자식 불순도 → 작게 나뉜 쪽보다 크게 나뉜 쪽의 불순도가 더 중요

### 4-2. 직접 구현해서 확인

```python
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=3) / len(y)   # 클래스별 비율
    return 1 - (p ** 2).sum()

def best_split(X, y):
    m = len(y)
    best = (None, None, np.inf)                 # (특성, 임곗값, 비용)
    for k in range(X.shape[1]):                 # 모든 특성
        values = np.unique(X[:, k])
        for t in (values[:-1] + values[1:]) / 2:   # 인접 값의 중간점이 후보 임곗값
            left = X[:, k] <= t
            J = left.sum() / m * gini(y[left]) + (~left).sum() / m * gini(y[~left])
            if J < best[2] - 1e-12:
                best = (k, t, J)
    return best

y = y_iris.values
best_split(X_iris, y)
# (0, 2.45, 0.3333)   → 특성 0(꽃잎 길이) ≤ 2.45, 비용 1/3   ← 사이킷런과 동일

right = X_iris[:, 0] > 2.45
best_split(X_iris[right], y[right])
# (1, 1.75, 0.1103)   → 오른쪽 자식은 꽃잎 너비 ≤ 1.75       ← 그림 5-1과 동일
```

- 루트 분할 비용 = 50/150 × 0 + 100/150 × 0.5 = **0.333**
- 임곗값 후보는 **정렬된 고유값 사이의 중간점** → 2.45 = (1.9 + 3.0) / 2 (setosa 최대 길이와 versicolor 최소 길이의 중간)

> ⚠️ **같은 비용을 내는 분할이 하나 더 있다**: 꽃잎 너비 ≤ 0.8도 setosa 50개를 완벽히 떼어내 비용이 똑같이 0.333. 둘 중 무엇이 뽑힐지는 `random_state`가 정함 → 10절의 핵심.

### 4-3. 탐욕 알고리즘

- CART는 **현재 단계에서 가장 좋은 분할**만 고르고, 몇 단계 뒤 결과는 고려하지 않음 (**탐욕적**, greedy)
- 그래서 "꽤 좋은" 트리를 찾지만 **최적 트리를 보장하지는 않음**
- 최적 트리 찾기는 **NP-완전** 문제 → O(exp(*m*)) 시간이 필요해 작은 데이터에서도 현실적으로 불가능

---

## 5. 계산 복잡도 (5.5)

| 단계 | 복잡도 | 이유 |
|---|---|---|
| **예측** | O(log₂ *m*) | 루트→리프까지 노드 하나에 특성 하나만 확인. 대체로 균형 잡힌 트리의 깊이 ≈ log₂ *m* |
| **훈련** | O(*n* × *m* log₂ *m*) | 각 깊이에서 모든 샘플의 모든 특성(*n*개)을 비교, 정렬 비용 *m* log *m* |

- 예측 시간은 **특성 수 *n*과 무관** → 특성이 많아도 예측이 매우 빠름
- 예) *m* = 100만이면 깊이 ≈ log₂(10⁶) ≈ **20** → 질문 약 20번이면 예측 끝

> 💡 연습문제 5번: 100만 개로 1시간 걸렸다면 1,000만 개는? *K* = 10 × log₂(10⁷) / log₂(10⁶) ≈ **11.7시간**. 데이터 10배 → 시간은 10배보다 약간 더.

---

## 6. 지니 불순도 또는 엔트로피? (5.6)

```python
DecisionTreeClassifier(criterion="entropy")   # 기본값은 "gini"
```

**엔트로피** — 정보 이론에서 온 개념. "메시지의 평균 정보량". 한 클래스만 있으면 0.

$$H_i = -\sum_{k=1,\, p_{i,k} \ne 0}^{n} p_{i,k} \log_2 (p_{i,k})$$

| 노드 | 지니 | 엔트로피 |
|---|---|---|
| 루트 [50, 50, 50] | 0.667 | 1.585 (= log₂ 3) |
| 노드 3 [0, 49, 5] | 0.168 | 0.445 (= −49/54·log₂(49/54) − 5/54·log₂(5/54)) |
| 순수 노드 | 0 | 0 |

**직접 확인 — 붓꽃에서 두 기준 비교** (노트북 외 추가 코드)

```python
tree_ent = DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=42)
tree_ent.fit(X_iris, y_iris)
(tree_ent.predict(X_iris) == tree_clf.predict(X_iris)).all()   # True → 완전히 같은 트리
```

| | 지니 | 엔트로피 |
|---|---|---|
| 계산 속도 | **조금 빠름** (로그 계산 없음) → 기본값으로 좋음 | 조금 느림 |
| 경향 | 가장 빈도 높은 클래스를 **한쪽 가지로 고립**시키는 경향 | 조금 더 **균형 잡힌** 트리 |
| 결과 | **대부분 비슷한 트리** | |

> 💡 사이킷런은 `criterion="log_loss"`도 지원하며 `"entropy"`와 같은 기준이다.

---

## 7. 규제 하이퍼파라미터 (5.7)

### 7-1. 왜 규제가 필요한가

| 구분 | 의미 | 예 |
|---|---|---|
| **파라미터 모델** | 파라미터 수가 학습 전에 **정해짐** → 자유도 제한, 과대적합 위험 ↓ (과소적합 위험 ↑) | 선형 회귀 |
| **비파라미터 모델** | 파라미터 수가 학습 전에 **정해지지 않음** → 데이터에 맞춰 구조가 자라남 | 결정 트리 |

제한 없는 결정 트리는 **모든 리프가 순수해질 때까지** 자라 → 훈련 데이터를 외워 버림. 그래서 성장을 제한해야 함.

### 7-2. 주요 하이퍼파라미터

| 하이퍼파라미터 | 의미 | 규제 강화 방향 |
|---|---|---|
| `max_depth` | 최대 깊이 (기본 `None` = 무제한) | ↓ |
| `max_features` | 분할 시 **평가할 특성 수** | ↓ |
| `max_leaf_nodes` | 최대 리프 노드 수 | ↓ |
| `min_samples_split` | 분할하려면 노드에 **최소 몇 개** 있어야 하는지 | ↑ |
| `min_samples_leaf` | 리프 노드에 있어야 할 **최소 샘플 수** | ↑ |
| `min_weight_fraction_leaf` | `min_samples_leaf`와 같지만 **전체 가중치 대비 비율**로 지정 | ↑ |

> 💡 **`min_*`은 올리고, `max_*`는 내리면** 규제가 세진다.
>
> 💡 **가지치기**: 일단 끝까지 키운 뒤 불필요한 노드를 잘라내는 방식도 있다(예: χ² 검정으로 "이 분할이 우연인지" 판정). 사이킷런에서는 `ccp_alpha`(비용 복잡도 가지치기)로 할 수 있다.

### 7-3. moons로 비교 (그림 5-3)

```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier(random_state=42)                      # 규제 없음
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)  # 리프당 최소 5개
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
```

| 줄 | 의미 |
|---|---|
| `make_moons(n_samples, noise)` | 반달 모양 2클래스 데이터. `noise`는 점들을 흩뜨리는 정도 (표준편차) |
| `min_samples_leaf=5` | 리프마다 최소 5개 → 샘플 1~2개짜리 작은 영역을 못 만듦 |

```python
def plot_decision_boundary(clf, X, y, axes, cmap):
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)   # 영역 색칠
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8) # 경계선만 회색으로
    colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[cmap][idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
plt.sca(axes[0])
plot_decision_boundary(tree_clf1, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title("No restrictions")
plt.sca(axes[1])
plot_decision_boundary(tree_clf2, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"min_samples_leaf = {tree_clf2.min_samples_leaf}")
plt.show()
```

- 2-1절의 결정 경계 패턴을 **함수로 묶은 것**. 9절에서도 재사용
- `plt.contour` → `contourf`(채우기)와 달리 **경계선만** 그림
- `plt.sca(ax)` → "set current axes". 이후 `plt.*` 명령이 그 서브플롯에 그려짐
- `sharey=True` → 두 그래프가 y축 공유

### 7-4. 새 데이터로 평가

```python
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2,
                                        random_state=43)   # 다른 시드 = 새 데이터
tree_clf1.score(X_moons_test, y_moons_test)   # 0.898
tree_clf2.score(X_moons_test, y_moons_test)   # 0.92
```

직접 확인한 비교 (노트북 외 추가 확인값)

| | 규제 없음 | `min_samples_leaf=5` |
|---|---|---|
| 깊이 / 리프 수 | 8 / 19 | 6 / 13 |
| 훈련 정확도 | **1.000** (전부 외움) | 0.947 |
| 테스트 정확도 | 0.898 | **0.920** |

→ 훈련 점수를 조금 포기하고 **일반화 성능**을 얻음. 왼쪽 그림은 이상치 하나를 위해 만든 작은 영역이 많고, 오른쪽은 경계가 더 매끈함.

---

## 8. 회귀 (5.8)

### 8-1. 2차 데이터와 회귀 트리

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(seed=42)
X_quad = rng.random((200, 1)) - 0.5   # -0.5 ~ 0.5 사이 200개, 특성 1개
y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))   # y = x² + 잡음

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
```

| 줄 | 의미 |
|---|---|
| `rng.random((200, 1))` | [0, 1) 균등분포 → `- 0.5`로 [−0.5, 0.5) |
| `rng.standard_normal` | 평균 0, 표준편차 1 정규분포. × 0.025로 작은 잡음 |
| `DecisionTreeRegressor` | 회귀용 결정 트리. 사용법은 분류기와 같음 |

### 8-2. 회귀 트리 읽기 (그림 5-4)

```python
export_graphviz(tree_reg, out_file="my_regression_tree.dot",
                feature_names=["x1"], rounded=True, filled=True)
Source.from_file("my_regression_tree.dot")
```

```
                       x1 ≤ 0.343 ?
                squared_error=0.006  samples=200  value=0.080
                   /                                  \
          x1 ≤ -0.302 ?                           x1 ≤ 0.431 ?
     samples=175  value=0.065                 samples=25  value=0.185
       /                \                       /                \
 samples=42        samples=133            samples=14         samples=11
 value=0.151       value=0.038            value=0.150        value=0.229
 mse=0.0028        mse=0.0017             mse=0.0005         mse=0.0006
```

| 분류 트리 | 회귀 트리 |
|---|---|
| 리프가 **클래스**를 예측 | 리프가 **값(숫자)**을 예측 |
| `value` = 클래스별 개수 | `value` = 그 리프 샘플들의 **타깃 평균** |
| `gini` | `squared_error` (MSE) |

> 예) *x*₁ = 0.2 → "≤ 0.343" 예 → "≤ −0.302" 아니오 → 133개 리프 → **ŷ = 0.038** (`tree_reg.predict([[0.2]])`로 확인)

> 💡 이 데이터는 샘플이 무작위라 교재 그림의 숫자와 다를 수 있다. 구조와 읽는 법만 같으면 된다.

### 8-3. 깊이 2 vs 깊이 3 (그림 5-5)

```python
tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg2.fit(X_quad, y_quad)

tree_reg.tree_.threshold
# array([ 0.343 , -0.3018, -2.    , -2.    ,  0.4314, -2.    , -2.    ])
tree_reg2.tree_.threshold
# array([ 0.343 , -0.3018, -0.414 , -2.    , -2.    ,  0.2182, -2.    ,
#        -2.    ,  0.4314,  0.3948, -2.    , -2.    ,  0.4647, -2.    , -2.    ])
```

- **`-2`는 리프 노드** 표시 (분할이 없으니 임곗값도 없음 → 사이킷런이 쓰는 자리 표시값 `TREE_UNDEFINED`)
- 깊이 3 트리도 깊이 0·1의 임곗값(0.343, −0.302, 0.431)은 같고 **아래에 분할만 추가**됨

```python
def plot_regression_predictions(tree_reg, X, y, axes=[-0.5, 0.5, -0.05, 0.25]):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)   # 500개 입력 (2차원으로)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.plot(X, y, "b.")                                     # 훈련 데이터: 파란 점
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")   # 예측: 빨간 선

th0, th1a, th1b = tree_reg.tree_.threshold[[0, 1, 4]]   # 깊이 0, 깊이 1 두 개
th2s = tree_reg2.tree_.threshold[[2, 5, 9, 12]]         # 깊이 2 네 개
```

- `.reshape(-1, 1)` → 1차원 500개를 `(500, 1)`로. 사이킷런은 특성이 1개여도 **2차원 입력**을 요구
- 예측선은 **계단 모양** — 각 구간(리프)마다 평균값 하나로 일정
- 깊이 2 → 계단 4개, 깊이 3 → 계단 8개. 깊을수록 곡선을 잘게 따라감

### 8-4. 회귀용 CART 비용 함수

불순도 대신 **MSE를 최소화**하도록 분할

$$J(k, t_k) = \frac{m_\text{left}}{m} \text{MSE}_\text{left} + \frac{m_\text{right}}{m} \text{MSE}_\text{right}$$

$$\text{MSE}_\text{node} = \frac{\sum_{i \in \text{node}} (\hat{y}_\text{node} - y^{(i)})^2}{m_\text{node}}, \qquad \hat{y}_\text{node} = \frac{\sum_{i \in \text{node}} y^{(i)}}{m_\text{node}}$$

→ 각 영역의 샘플들이 **그 영역 평균에 최대한 가깝게** 모이도록 나눔.

### 8-5. 회귀도 과대적합된다 (그림 5-6)

```python
tree_reg1 = DecisionTreeRegressor(random_state=42)                       # 규제 없음
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)  # 리프당 10개 이상
tree_reg1.fit(X_quad, y_quad)
tree_reg2.fit(X_quad, y_quad)

x1 = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)
```

| | 규제 없음 | `min_samples_leaf=10` |
|---|---|---|
| 리프 수 (직접 확인) | **200** (샘플 1개당 리프 1개) | 15 |
| 예측선 | 잡음까지 따라가며 **들쭉날쭉** | 2차 곡선 모양을 **매끈하게** 근사 |

> ⚠️ 리프 수 = 샘플 수 → 모든 훈련 샘플을 그대로 외운 상태. 회귀 트리도 규제 없이는 심하게 과대적합.

---

## 9. 축 방향에 대한 민감성 (5.9)

결정 트리의 분할은 항상 **축에 수직**(특성 하나 ≤ 임곗값). 그래서 데이터가 **회전**되어 있으면 약해진다.

### 9-1. 같은 데이터를 45° 회전 (그림 5-7)

```python
rng = np.random.default_rng(seed=42)
X_square = rng.random((100, 2)) - 0.5           # -0.5~0.5 정사각형 안의 점 100개
y_square = (X_square[:, 0] > 0).astype(np.int64)   # x₁ > 0이면 1 → 세로선으로 완벽히 나뉨

angle = np.pi / 4  # 45도
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)

tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)

# 두 결과를 7절의 plot_decision_boundary로 나란히 그림 (cmap="Pastel1")
```

| 줄 | 의미 |
|---|---|
| `.astype(np.int64)` | True/False → 1/0 정수 레이블 |
| `rotation_matrix` | 2차원 **회전 행렬**. 점을 원점 기준으로 45° 돌림 |
| `X.dot(R)` | 행렬 곱 → 모든 점을 한 번에 회전 (행 벡터 × R이라 실제 회전 방향은 시계 방향) |

직접 확인한 결과

| | 원래 데이터 | 45° 회전 |
|---|---|---|
| 깊이 / 리프 수 | 1 / **2** | 4 / **6** |
| 결정 경계 | 세로선 **하나** | 대각선을 흉내 내는 **계단** |

- 데이터 자체는 똑같이 선형으로 나뉘는데, 대각선은 **축 수직선 여러 개로 근사**해야 함
- 회전 쪽 경계는 훈련 데이터엔 맞아도 **불필요하게 꼬여서 일반화가 나쁠 가능성**이 큼

### 9-2. 해결책: PCA로 축 맞추기 (그림 5-8)

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)
```

| 줄 | 의미 |
|---|---|
| `StandardScaler()` | PCA 전에 스케일링 (PCA는 분산 기준이라 단위에 민감) |
| `PCA()` | 데이터를 **분산이 가장 큰 방향이 첫 번째 축**이 되도록 회전 (자세한 건 7장) |
| `make_pipeline` | 이름을 자동으로 붙여 주는 파이프라인 (2장) |
| `fit_transform` | 스케일링 → 회전된 새 좌표 *z*₁, *z*₂ |

```python
# 그림 5-8: 회전된 좌표계에서의 결정 경계
axes = [-2.2, 2.4, -0.6, 0.7]
z0s, z1s = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                       np.linspace(axes[2], axes[3], 100))
X_iris_pca_all = np.c_[z0s.ravel(), z1s.ravel()]
y_pred = tree_clf_pca.predict(X_iris_pca_all).reshape(z0s.shape)
plt.contourf(z0s, z1s, y_pred, alpha=0.3, cmap=custom_cmap)
...
th1, th2 = tree_clf_pca.tree_.threshold[[0, 2]]   # -0.911, 0.758
plt.plot([th1, th1], axes[2:], "k-", linewidth=2)
plt.plot([th2, th2], axes[2:], "k--", linewidth=2)
```

직접 확인한 결과

- 두 분할 모두 **같은 특성 *z*₁**만 사용 (`tree_.feature` = `[0, -2, 0, -2, -2]`) → 결정 경계가 **세로선 두 개**로 단순해짐
- *z*₁ 하나가 분산의 **98.1%**를 설명 (꽃잎 길이와 너비가 강하게 상관 → 대각선 방향이 핵심 축)
- 훈련 정확도: 원래 좌표 0.96 → PCA 좌표 **0.98**

> 💡 데이터에 대각선 경계가 있을 것 같으면 **PCA로 먼저 축을 맞추는 것**을 고려. 다만 결과 특성 *z*₁, *z*₂는 해석이 어려워지는 대가가 있다.

---

## 10. 결정 트리의 분산 문제 (5.10)

**분산이 크다** = 훈련 데이터나 하이퍼파라미터가 조금만 바뀌어도 **완전히 다른 모델**이 나올 수 있다.

### 10-1. 시드만 바꿔도 달라진다 (그림 5-9)

```python
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
tree_clf_tweaked.fit(X_iris, y_iris)
```

```python
y_pred = tree_clf_tweaked.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)
...
th0, th1 = tree_clf_tweaked.tree_.threshold[[0, 2]]
plt.plot([0, 7.2], [th0, th0], "k-", linewidth=2)    # 가로선
plt.plot([0, 7.2], [th1, th1], "k--", linewidth=2)   # 가로선
```

```
random_state=42                         random_state=40
|--- petal length <= 2.45               |--- petal width <= 0.80        ← 루트가 다름!
|   |--- class: 0                       |   |--- class: 0
|--- petal length >  2.45               |--- petal width >  0.80
|   |--- petal width <= 1.75            |   |--- petal width <= 1.75
|   |   |--- class: 1                   |   |   |--- class: 1
|   |--- petal width >  1.75            |   |--- petal width >  1.75
|   |   |--- class: 2                   |   |   |--- class: 2
```

- 같은 데이터, 같은 하이퍼파라미터인데 **루트 질문이 바뀜** → 결정 경계가 세로선 대신 **가로선 두 개**
- 원인: 4-2절에서 본 **동점 분할**. 사이킷런은 분할마다 **특성을 무작위 순서로 검토**하고, 비용이 같으면 먼저 찾은 것을 택함 → `random_state`가 동점을 깸

**직접 확인 — 시드 0~99로 100번 훈련했을 때 루트 분할**

| 루트 분할 | 횟수 |
|---|---|
| petal width ≤ 0.80 | 51 |
| petal length ≤ 2.45 | 49 |

> ⚠️ 훈련 정확도는 둘 다 0.96으로 같지만, **훈련 샘플이 없는 영역**(예: 길이 2, 너비 1.5인 꽃)은 두 모델이 다르게 예측한다. "같은 성능, 다른 모델" = 분산이 큰 모델의 전형.

### 10-2. 해결책: 여러 트리의 평균

- 트리 **여러 개를 훈련해 예측을 평균(다수결)** 내면 분산이 크게 줄어듦
- 이런 트리 앙상블이 **랜덤 포레스트** → 6장. 연습문제 8번이 직접 만들어 보는 예고편

---

## 11. 부록: `tree_`로 트리 구조 직접 다루기

훈련된 트리의 모든 정보는 `tree_` 속성(`sklearn.tree._tree.Tree` 객체)에 **NumPy 배열**로 들어 있다. 노드 번호는 0(루트)부터 **깊이 우선 순서**.

### 11-1. 기본 정보

```python
tree = tree_clf.tree_

tree.node_count     # 5   전체 노드 수
tree.max_depth      # 2
tree.max_n_classes  # 3
tree.n_features     # 2
tree.n_outputs      # 1   타깃 열 수 (다중 출력이면 2 이상)
tree.n_leaves       # 3
tree.impurity       # array([0.6667, 0.    , 0.5   , 0.168 , 0.0425])  노드별 지니
```

### 11-2. 부모-자식 관계

```python
tree.children_left[0], tree.children_right[0]   # (1, 2)  루트의 자식
tree.children_left[3], tree.children_right[3]   # (-1, -1)  리프는 둘 다 -1

is_leaf = (tree.children_left == tree.children_right)
np.arange(tree.node_count)[is_leaf]             # array([1, 3, 4])  리프 노드 번호
```

- 리프는 왼쪽·오른쪽 자식 번호가 **같다**(`-1`) → 이 성질로 리프를 골라냄

### 11-3. 분할 정보

```python
tree.feature     # array([ 0, -2,  1, -2, -2])   0=꽃잎 길이, 1=꽃잎 너비
tree.threshold   # array([ 2.45, -2.  ,  1.75, -2.  , -2.  ])
```

- **리프 자리의 `-2`는 의미 없는 값** → 무시

### 11-4. 노드별 클래스 정보

```python
tree.value
# array([[[0.3333, 0.3333, 0.3333]],
#        [[1.    , 0.    , 0.    ]],
#        [[0.    , 0.5   , 0.5   ]],
#        [[0.    , 0.9074, 0.0926]],
#        [[0.    , 0.0217, 0.9783]]])      shape (5, 1, 3) = (노드, 출력, 클래스)

tree.n_node_samples                       # array([150,  50, 100,  54,  46])

np.all(tree.value.sum(axis=(1, 2)) == tree.n_node_samples)
# 노트북 의도: True  /  scikit-learn 1.9.1 실행 결과: False
```

> ⚠️ **버전 차이 주의.** 최근 scikit-learn은 분류 트리의 `tree_.value`에 **개수가 아니라 클래스 비율**을 저장한다(각 행의 합 = 1). 그래서 노트북의 마지막 확인 코드가 `False`가 나온다. 개수가 필요하면 가중 샘플 수를 곱하면 된다.
>
> ```python
> counts = tree.value.squeeze(1) * tree.weighted_n_node_samples[:, None]
> # [[50, 50, 50], [50, 0, 0], [0, 50, 50], [0, 49, 5], [0, 1, 45]]
> np.all(counts.sum(axis=1).round() == tree.n_node_samples)   # True
> ```

### 11-5. 노드별 깊이 계산

```python
def compute_depth(tree_clf):
    tree = tree_clf.tree_
    depth = np.zeros(tree.node_count)
    stack = [(0, 0)]                        # (노드 번호, 깊이) — 루트부터
    while stack:
        node, node_depth = stack.pop()      # 하나 꺼내서
        depth[node] = node_depth            # 깊이 기록
        if tree.children_left[node] != tree.children_right[node]:   # 리프가 아니면
            stack.append((tree.children_left[node], node_depth + 1))
            stack.append((tree.children_right[node], node_depth + 1))
    return depth

depth = compute_depth(tree_clf)
depth   # array([0., 1., 1., 2., 2.])
```

- **스택**을 이용한 **깊이 우선 탐색** (재귀 없이 반복문으로)
- 자식을 넣을 때 깊이 +1

```python
tree_clf.tree_.feature[(depth == 1) & (~is_leaf)]     # array([1])     깊이 1 분할 노드의 특성
tree_clf.tree_.threshold[(depth == 1) & (~is_leaf)]   # array([1.75])  그 임곗값
```

- `~is_leaf` → 불리언 배열 뒤집기 = **분할 노드**(split node)
- 그림 5-2의 "Depth=1" 가로선 위치를 코드로 꺼낸 것

---

## 12. 연습문제 풀이

### 1~6번 (개념)

| 번 | 질문 | 답 |
|---|---|---|
| 1 | 샘플 100만 개, 제한 없이 훈련한 트리의 깊이? | 리프 ≈ 샘플 수, 균형 이진 트리 깊이 ≈ log₂(10⁶) ≈ **20** (실제론 약간 더 깊음) |
| 2 | 자식 노드의 지니 불순도는 부모보다 항상 낮은가? | **대체로 낮지만 항상은 아님.** 아래 예시 참고 |
| 3 | 과대적합이면 `max_depth`를? | **줄인다** (모델 제한 = 규제) |
| 4 | 과소적합이면 스케일링이 도움 되나? | **아니다.** 트리는 스케일과 무관 → 시간 낭비 |
| 5 | 100만 개 훈련에 1시간, 1,000만 개면? | *K* = 10 × log₂(10⁷)/log₂(10⁶) ≈ **11.7시간** |
| 6 | 특성 수가 2배면? | 훈련 시간도 **약 2배** (*n*에 비례) |

**2번 예시** — 1차원 데이터가 A, B, A, A, A 순서로 놓인 노드 (A 4개, B 1개)

- 부모 지니 = 1 − (1/5)² − (4/5)² = **0.32**
- CART는 두 번째 뒤에서 분할 → 왼쪽 [A, B], 오른쪽 [A, A, A]
- 왼쪽 지니 = 1 − 0.5² − 0.5² = **0.5** (부모보다 높음!)
- 하지만 가중 합 = 2/5 × 0.5 + 3/5 × 0 = **0.2** < 0.32 → 전체적으로는 개선이라 선택됨

> 💡 CART가 최소화하는 건 **자식들의 가중 평균** 불순도. 한쪽이 나빠져도 다른 쪽이 그보다 더 좋아지면 OK.

### 7번 — moons 데이터로 트리 튜닝

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV

X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons,
                                                    test_size=0.2,
                                                    random_state=42)

params = {
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': [1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 3, 4]
}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              params, cv=3)
grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_
# DecisionTreeClassifier(max_depth=6, max_leaf_nodes=17, random_state=42)
```

| 부분 | 의미 |
|---|---|
| `noise=0.4` | 5.7절(0.2)보다 훨씬 많이 섞임 → 완벽 분리 불가, 규제가 중요 |
| `test_size=0.2` | 8,000 훈련 / 2,000 테스트 |
| `params` | 98 × 6 × 3 = **1,764 조합** × 3폴드 = 5,292번 훈련 (트리는 빨라서 수십 초) |
| `best_estimator_` | 최고 조합으로 **전체 훈련 세트에 다시 학습된** 모델 (`refit=True` 기본) |

```python
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)   # 0.8595
```

- 교차 검증 최고 점수 0.854 → 테스트 **0.8595** (교재 목표 85~87% 달성)
- `grid_search_cv.predict` = `best_estimator_.predict`

### 8번 — 숲 직접 키우기 (랜덤 포레스트 예고)

**a. 100개짜리 작은 훈련 세트 1,000개 만들기**

```python
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances,
                  random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```

- `ShuffleSplit` → 매번 **새로 섞어서** 훈련/테스트로 나눔. `n_splits=1000`번 반복
- `test_size=len(X_train) - 100` → 나머지를 테스트로 돌리면 **훈련 쪽은 정확히 100개**
- 여기선 훈련 인덱스만 쓰고 `mini_test_index`는 버림

**b. 트리 1,000개 훈련 → 개별 성능**

```python
from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)   # 0.8056605
```

- `clone` → 7번 최적 하이퍼파라미터를 가진 **학습 안 된** 복제본 1,000개
- 100개로만 학습해 개별 트리는 평균 **80.6%** (7번의 86%보다 낮음)

**c. 다수결 투표**

```python
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)   # (1000, 2000)
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)               # 트리마다 한 행

from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)       # 열(테스트 샘플)마다 최빈값
```

| 줄 | 의미 |
|---|---|
| `np.empty(..., dtype=np.uint8)` | 값 초기화 없이 공간만 확보. 0/1이라 `uint8`로 충분 |
| `mode(..., axis=0)` | **세로 방향**(트리 1,000개)의 최빈값 → 샘플마다 다수결 예측 + 득표수 |

**d. 평가**

```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))   # 0.873
```

| 모델 | 테스트 정확도 |
|---|---|
| 작은 트리 1개 (평균) | 0.806 |
| 튜닝한 트리 1개 (전체 데이터) | 0.8595 |
| **작은 트리 1,000개 다수결** | **0.873** |

> 💡 하나하나는 약한 트리지만 **다수결로 모으면 최고의 단일 트리보다 낫다** → 이게 랜덤 포레스트. 10절의 "분산이 크다"는 약점을 평균으로 상쇄한 것. 6장으로 이어짐.
>
> ⚠️ 최근 SciPy의 `mode`는 기본적으로 축을 없애서 결과가 이미 1차원 `(2000,)`이다. `.reshape([-1])`은 옛 버전(2차원 반환) 대비용이라 있어도 무해.

---

## 13. 전체 흐름 요약

```
load_iris → 꽃잎 길이·너비                        ← 데이터 (스케일링 불필요)
   ↓
DecisionTreeClassifier(max_depth=2).fit           ← 5.1 훈련
export_graphviz / export_text                     ←     시각화, 지니 불순도 읽기
   ↓
루트→리프 질문 따라 내려가기 / predict_proba        ← 5.2 예측, 5.3 확률 = 리프 비율
   ↓
CART: J = 가중 평균 불순도 최소 분할, 탐욕적·재귀   ← 5.4  (예측 O(log m), 훈련 O(n·m log m) 5.5)
criterion="gini" | "entropy"                      ← 5.6 대부분 같은 결과
   ↓
max_depth↓ / min_samples_leaf↑ …                  ← 5.7 규제 (비파라미터 모델의 과대적합 방지)
DecisionTreeRegressor (리프 = 평균, MSE 최소)      ← 5.8 회귀
   ↓
약점 ①: 축 수직 분할 → 회전에 민감 → PCA           ← 5.9
약점 ②: 분산이 큼 (동점·데이터 변화에 흔들림)       ← 5.10 → 여러 트리 평균 = 랜덤 포레스트(6장)
```

### 결정 트리 장단점

| 장점 | 단점 |
|---|---|
| 해석 쉬움 (화이트박스) | 훈련 데이터에 **과대적합**되기 쉬움 → 규제 필수 |
| 스케일링·중앙화 불필요 | **축 방향 민감** (경계가 계단 모양) |
| 분류·회귀·다중 출력 모두 가능 | **분산이 큼** (작은 변화에 모델이 크게 바뀜) |
| 예측이 매우 빠름 O(log *m*) | 확률 추정이 거침 (같은 리프 = 같은 확률) |

### 분류 트리 vs 회귀 트리

| | `DecisionTreeClassifier` | `DecisionTreeRegressor` |
|---|---|---|
| 리프 예측 | 다수 클래스 (+ 클래스 비율 = 확률) | 샘플 타깃 **평균** |
| 분할 기준 | 지니 / 엔트로피 | MSE (`squared_error`) |
| CART 비용 | 자식 불순도 가중 평균 | 자식 MSE 가중 평균 |

### 주요 결과

| 실험 | 결과 |
|---|---|
| 붓꽃 깊이 2 트리 | 루트 길이 ≤ 2.45, 다음 너비 ≤ 1.75, 훈련 정확도 0.96 |
| (5, 1.5) 확률 | [0, 0.907, 0.093] → versicolor |
| moons 규제 없음 / `min_samples_leaf=5` | 테스트 0.898 / **0.920** |
| 회귀 리프 수 규제 없음 / `min_samples_leaf=10` | 200 / 15 |
| 정사각형 데이터 리프 수 원본 / 45° 회전 | 2 / 6 |
| 붓꽃 PCA 후 트리 | *z*₁만 사용, 훈련 정확도 0.98 |
| 시드 100개 중 루트 분할 | 길이 2.45 → 49번, 너비 0.8 → 51번 |
| 연습 7: 튜닝한 트리 | `max_depth=6, max_leaf_nodes=17`, 테스트 0.8595 |
| 연습 8: 작은 트리 평균 / 다수결 | 0.806 / **0.873** |

### 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `export_graphviz(..., filled=True)` + `Source.from_file` | 트리 그림 | 1-2, 8-2 |
| `export_text(..., show_weights=True)` | 트리를 글자로 빠르게 확인 | 1-2 |
| `meshgrid` → `ravel` → `np.c_` → `predict` → `reshape` → `contourf` | 2차원 결정 경계 그리기 | 2-1, 7-3, 9 |
| `tree_.threshold[[i, j]]` | 원하는 노드의 임곗값 꺼내기 | 2-2, 8-3, 9-2 |
| `children_left == children_right` | 리프 노드 판별 | 11-2 |
| `X.reshape(-1, 1)` | 특성 1개를 2차원 입력으로 | 8-3 |
| `X.dot(rotation_matrix)` | 데이터 회전 | 9-1 |
| `make_pipeline(StandardScaler(), PCA())` | 축 맞추기 전처리 | 9-2 |
| `ShuffleSplit(test_size=전체-100)` | 작은 무작위 부분집합 여러 개 | 12 (8번) |
| `scipy.stats.mode(Y, axis=0)` | 여러 모델 다수결 | 12 (8번) |
