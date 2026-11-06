# 5장. 결정 트리

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `05_decision_trees.ipynb`
>
> 결정 트리의 구조·학습 원리·규제·한계를 다룬다. 결정 트리는 6장 앙상블(랜덤 포레스트, 부스팅)의 기본 구성 요소다.
>
> 실행 환경: 노트북 요구사항은 scikit-learn ≥ 1.6.1. 이 노트의 출력값과 그림은 scikit-learn 1.9.1로 직접 실행한 결과.

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

| 단계 | 절 | 입력 | 출력 |
|---|---|---|---|
| 1. 학습·시각화 | 5.1 | 훈련 세트 | 학습된 트리, 노드 정보 |
| 2. 예측 | 5.2–5.3 | 학습된 트리, 새 샘플 | 예측 클래스, 클래스 확률, 결정 경계 |
| 3. 훈련 원리 | 5.4–5.6 | 훈련 세트 | 분할 규칙(CART), 계산 비용, 불순도 지표 |
| 4. 규제 | 5.7 | 과대적합된 트리 | 일반화 성능이 좋은 트리 |
| 5. 회귀 | 5.8 | 연속형 타깃 | 회귀 트리 |
| 6. 한계 | 5.9–5.10 | 학습된 트리 | 약점과 대응 방법 (PCA, 앙상블) |

## 데이터

| 데이터 | 사용 절 | 구성 | 비고 |
|---|---|---|---|
| 붓꽃(iris) | 5.1–5.3, 5.9, 5.10 | 150개, 3클래스(setosa·versicolor·virginica 각 50) | 꽃잎 길이·너비 2개 특성만 사용 |
| moons | 5.7, 연습문제 | `make_moons`로 생성한 2클래스 데이터 | `noise`로 겹침 정도 조절 |
| 2차 데이터 | 5.8 | 200개, $`y = x^2 + \text{잡음}`$ | $`x \in [-0.5, 0.5)`$ |

## 목차

1. [결정 트리 학습과 시각화 (5.1)](#1-결정-트리-학습과-시각화-51)
   - [1-1. 모델 학습](#1-1-모델-학습)
   - [1-2. 트리 시각화](#1-2-트리-시각화)
   - [1-3. 노드 정보와 지니 불순도](#1-3-노드-정보와-지니-불순도)
2. [예측 (5.2)](#2-예측-52)
   - [2-1. 예측 경로](#2-1-예측-경로)
   - [2-2. 결정 경계 시각화](#2-2-결정-경계-시각화)
   - [2-3. 모델 해석 가능성](#2-3-모델-해석-가능성)
3. [클래스 확률 추정 (5.3)](#3-클래스-확률-추정-53)
4. [CART 훈련 알고리즘 (5.4)](#4-cart-훈련-알고리즘-54)
   - [4-1. 비용 함수](#4-1-비용-함수)
   - [4-2. 분할 탐색 구현](#4-2-분할-탐색-구현)
   - [4-3. 탐욕 알고리즘의 한계](#4-3-탐욕-알고리즘의-한계)
5. [계산 복잡도 (5.5)](#5-계산-복잡도-55)
6. [지니 불순도 또는 엔트로피? (5.6)](#6-지니-불순도-또는-엔트로피-56)
7. [규제 하이퍼파라미터 (5.7)](#7-규제-하이퍼파라미터-57)
   - [7-1. 파라미터 모델과 비파라미터 모델](#7-1-파라미터-모델과-비파라미터-모델)
   - [7-2. 규제 하이퍼파라미터 종류](#7-2-규제-하이퍼파라미터-종류)
   - [7-3. moons 데이터 실험](#7-3-moons-데이터-실험)
8. [회귀 (5.8)](#8-회귀-58)
   - [8-1. 회귀 트리 학습](#8-1-회귀-트리-학습)
   - [8-2. 회귀용 CART 비용 함수](#8-2-회귀용-cart-비용-함수)
   - [8-3. 깊이에 따른 예측 형태](#8-3-깊이에-따른-예측-형태)
   - [8-4. 회귀 트리의 규제](#8-4-회귀-트리의-규제)
9. [축 방향에 대한 민감성 (5.9)](#9-축-방향에-대한-민감성-59)
   - [9-1. 회전 실험](#9-1-회전-실험)
   - [9-2. PCA를 이용한 축 정렬](#9-2-pca를-이용한-축-정렬)
10. [결정 트리의 분산 문제 (5.10)](#10-결정-트리의-분산-문제-510)
11. [부록: 트리 구조 접근 (`tree_`)](#11-부록-트리-구조-접근-tree_)
12. [연습문제 (5.11)](#12-연습문제-511)
13. [정리](#13-정리)

---

## 1. 결정 트리 학습과 시각화 (5.1)

### 1-1. 모델 학습

**목적**
붓꽃 데이터로 결정 트리 분류기를 학습한다.

**개념·원리**
- **결정 트리**: "특성 $k$ ≤ 임곗값 $t$" 형태의 이진 질문을 계층적으로 적용해 입력 공간을 영역으로 나누고, 각 영역에 예측값을 할당하는 모델
- 분류(`DecisionTreeClassifier`)와 회귀(`DecisionTreeRegressor`)에 모두 사용
- 각 질문은 특성 하나를 임곗값과 비교할 뿐이므로 **특성 스케일링이 필요 없다**

**코드**

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values   # 꽃잎 길이·너비 2개
y_iris = iris.target                                                   # 0, 1, 2

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)       # 최대 깊이 2
tree_clf.fit(X_iris, y_iris)
```

| 코드 | 설명 |
|---|---|
| `load_iris(as_frame=True)` | 특성은 DataFrame, 타깃은 Series로 로드 |
| `[["petal length (cm)", "petal width (cm)"]]` | 특성 2개만 선택 → 결정 경계를 2차원 평면에 표시 가능 |
| `.values` | NumPy 배열로 변환 |
| `max_depth=2` | 트리 깊이 제한 (시각화용 설정, 7절에서 규제로 다시 다룸) |
| `random_state=42` | 훈련 과정의 무작위성 고정 (원인은 10절) |

**정리**
- 결정 트리는 전처리(스케일링) 없이 원본 특성으로 학습한다

---

### 1-2. 트리 시각화

**목적**
학습된 트리의 질문 구조를 그림으로 확인한다.

**코드**

```python
from sklearn.tree import export_graphviz
from graphviz import Source

export_graphviz(
        tree_clf,
        out_file="my_iris_tree.dot",                                # .dot 텍스트 파일로 저장
        feature_names=["petal length (cm)", "petal width (cm)"],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )
Source.from_file("my_iris_tree.dot")                                # 노트북 안에서 렌더링

!dot -Tpng "my_iris_tree.dot" -o "my_iris_tree.png"                # 명령줄로 PNG 변환
```

**원리**

| 코드 | 설명 |
|---|---|
| `export_graphviz` | 트리 구조를 Graphviz의 `.dot` 형식(그래프 기술 언어)으로 출력 |
| `feature_names` / `class_names` | 노드에 특성·클래스를 이름으로 표시 (기본값은 `x[0]`, `y[0]`) |
| `rounded=True` | 노드 모서리를 둥글게 |
| `filled=True` | 다수 클래스의 색으로 채우고, 순수할수록 진하게 |
| `Source.from_file` | `graphviz` 파이썬 패키지로 렌더링 |
| `!dot -Tpng` | Graphviz 명령줄 도구로 PNG 변환. `!`는 노트북에서 셸 명령 실행 |

Graphviz 없이 확인하는 방법:

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

- `sklearn.tree.plot_tree(tree_clf)`: matplotlib으로 그리기

**결과 해석**

![그림 5-1 붓꽃 결정 트리](images/05/fig5-1_iris_tree.png)

| 깊이 | 노드 | 질문 / 결과 |
|---|---|---|
| 0 | 루트 | 꽃잎 길이 ≤ 2.45 |
| 1 | 왼쪽 (참) | 리프 → setosa |
| 1 | 오른쪽 (거짓) | 꽃잎 너비 ≤ 1.75 |
| 2 | 왼쪽 (참) | 리프 → versicolor |
| 2 | 오른쪽 (거짓) | 리프 → virginica |

---

### 1-3. 노드 정보와 지니 불순도

**목적**
노드에 표시된 수치의 의미를 파악한다.

**개념·원리**

| 용어 | 정의 |
|---|---|
| 루트 노드 | 깊이 0의 최상위 노드 |
| 분할 노드 | 질문을 가지고 자식 2개로 나뉘는 노드 |
| 리프 노드 | 자식이 없는 노드. 예측값이 결정됨 |
| 깊이 | 루트에서 해당 노드까지의 분할 횟수 |

| 노드 항목 | 의미 |
|---|---|
| `samples` | 노드에 도달한 훈련 샘플 수 |
| `value` | 노드에 도달한 샘플의 클래스별 개수 |
| `class` | `value`가 최대인 클래스 = 노드의 예측 |
| `gini` | 노드의 지니 불순도 |

**지니 불순도** (Gini impurity)

```math
G_i = 1 - \sum_{k=1}^{n} p_{i,k}^{2}
```

- $`p_{i,k}`$: 노드 $i$의 샘플 중 클래스 $k$의 비율, $n$: 클래스 수
- 의미: 노드에서 샘플 2개를 복원 추출했을 때 두 샘플의 클래스가 다를 확률
- 범위: 0 (한 클래스만 존재, **순수** 노드) ~ $`1 - 1/n`$ (모든 클래스 균등)

**결과 해석**

| 노드 | value | 계산 | 지니 |
|---|---|---|---|
| 루트 | [50, 50, 50] | $`1 - 3 \times (1/3)^2`$ | 0.667 |
| 깊이 1 왼쪽 | [50, 0, 0] | $`1 - 1^2`$ | 0 |
| 깊이 1 오른쪽 | [0, 50, 50] | $`1 - 0.5^2 - 0.5^2`$ | 0.5 |
| 깊이 2 왼쪽 | [0, 49, 5] | $`1 - (49/54)^2 - (5/54)^2 = 1 - 0.823 - 0.009`$ | 0.168 |
| 깊이 2 오른쪽 | [0, 1, 45] | $`1 - (1/46)^2 - (45/46)^2`$ | 0.043 |

- 루트 0.667 = 3클래스 균등 분포의 최댓값 $`1 - 1/3`$
- 깊이가 깊어질수록 불순도 감소 → 분할이 클래스를 분리하고 있음

**정리**
- 결정 트리 학습 = 노드의 불순도를 낮추는 분할을 반복하는 과정
- 사이킷런의 트리는 모든 분할 노드의 자식이 2개인 **이진 트리** (ID3 등은 3개 이상 가능)

---

## 2. 예측 (5.2)

### 2-1. 예측 경로

**목적**
새 샘플을 분류하는 절차를 확인한다.

**개념·원리**
루트에서 시작해 각 노드의 질문에 답하며 리프까지 내려가고, 리프의 클래스를 출력한다.

**결과 해석**

| 입력 (길이, 너비) | 루트: 길이 ≤ 2.45 | 깊이 1: 너비 ≤ 1.75 | 도착 리프 value | 예측 |
|---|---|---|---|---|
| (1.4, 0.2) | 참 | — | [50, 0, 0] | setosa |
| (5.0, 1.5) | 거짓 | 참 | [0, 49, 5] | versicolor |
| (5.5, 2.0) | 거짓 | 거짓 | [0, 1, 45] | virginica |

**정리**
- 예측 시 비교 횟수 = 도착한 리프의 깊이 → 예측이 빠름 (5절)

---

### 2-2. 결정 경계 시각화

**목적**
입력 공간 전체에서 트리가 어떤 클래스로 예측하는지 확인한다.

**개념·원리**
- **결정 경계**: 예측 클래스가 바뀌는 경계
- 각 질문은 특성 하나만 비교 → 경계는 항상 **축에 수직인 직선**
- 결과적으로 입력 공간이 **축에 정렬된 직사각형 영역**들로 나뉜다

그리는 방법: 평면을 격자로 덮고 → 모든 격자점을 예측 → 같은 예측끼리 색칠

| 단계 | 코드 | 결과 형태 |
|---|---|---|
| 1. 격자 생성 | `np.meshgrid(x 100개, y 100개)` | (100, 100) 행렬 2개 |
| 2. 샘플 배열로 변환 | `np.c_[a.ravel(), b.ravel()]` | (10000, 2) |
| 3. 예측 | `predict` | (10000,) |
| 4. 격자 형태 복원 | `.reshape(a.shape)` | (100, 100) |
| 5. 영역 색칠 | `plt.contourf` | — |

```python
xx, yy = np.meshgrid([1, 2, 3], [10, 20])
# xx = [[1, 2, 3], [1, 2, 3]],  yy = [[10, 10, 10], [20, 20, 20]]
np.c_[xx.ravel(), yy.ravel()]
# [[1, 10], [2, 10], [3, 10], [1, 20], [2, 20], [3, 20]]   → 격자점 6개
```

**코드**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
plt.figure(figsize=(8, 4))

lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)    # 영역 색칠
for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],  # 클래스별 샘플
             style, label=f"Iris {name}")

# 분할선: 깊이 3 트리의 임곗값을 꺼내 표시
tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_deeper.fit(X_iris, y_iris)
th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
# th0=2.45, th1=1.75, th2a=4.95, th2b=4.85

plt.plot([th0, th0], [0, 3], "k-", linewidth=2)       # 깊이 0: x = 2.45
plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)  # 깊이 1: y = 1.75
plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)   # 깊이 2 (max_depth=3일 때)
plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)
...
plt.show()
```

- `tree_.threshold`: 노드 번호 순으로 임곗값을 저장한 배열 (11절)
- 깊이 3 트리는 다음 분할 위치(점선)를 참고로 보여 주기 위해 별도 학습

**결과 해석**

![그림 5-2 붓꽃 결정 경계](images/05/fig5-2_iris_boundary.png)

| 선 | 분할 | 효과 |
|---|---|---|
| 실선 (깊이 0) | 길이 = 2.45 | setosa 영역 분리 |
| 파선 (깊이 1) | 너비 = 1.75 | versicolor / virginica 분리 |
| 점선 (깊이 2) | 길이 = 4.95, 4.85 | `max_depth=3`일 때 추가될 분할 (현재 모델에는 없음) |

**정리**
- 결정 경계는 축에 수직인 선분으로만 구성된다 → 9절 한계의 원인
- meshgrid → ravel → `np.c_` → predict → reshape → contourf 는 이 장 전체에서 쓰는 결정 경계 시각화 절차

---

### 2-3. 모델 해석 가능성

**개념·원리**

| 구분 | 모델 | 특징 |
|---|---|---|
| 화이트박스 | 결정 트리 | 예측 근거를 질문 경로로 그대로 설명 가능 |
| 블랙박스 | 랜덤 포레스트, 신경망 | 성능은 좋지만 예측 근거를 설명하기 어려움 |

**정리**
- 판단 근거를 제시해야 하는 문제(대출 심사, 의료 등)에서 결정 트리의 장점

---

## 3. 클래스 확률 추정 (5.3)

**목적**
예측 클래스 외에 클래스별 확률 추정값을 얻는다.

**개념·원리**
샘플이 도달한 리프의 클래스 비율을 확률로 사용한다.

```math
\hat{p}_k = \frac{\text{리프에 있는 클래스 } k \text{ 훈련 샘플 수}}{\text{리프의 전체 훈련 샘플 수}}
```

`predict`는 $`\hat{p}_k`$가 최대인 클래스를 반환한다.

**코드**

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)   # array([[0.   , 0.907, 0.093]])
tree_clf.predict([[5, 1.5]])                  # array([1])
```

- 입력은 (샘플 수, 특성 수) 형태의 2차원 → 샘플 1개도 `[[...]]`로 감쌈

**결과 해석**
- (5, 1.5) → 깊이 2 왼쪽 리프 (value = [0, 49, 5])
- 확률 = [0/54, 49/54, 5/54] = [0, 0.907, 0.093] → versicolor

**한계**
같은 리프에 도달하는 모든 입력은 확률이 같다. (6, 1.5), (5, 1.0)도 [0, 0.907, 0.093]. 경계 근처 입력이라도 확률이 낮아지지 않는다.

**정리**
- 확률 = 리프의 클래스 비율, 리프 단위로 계단식 → 추정이 거칠다

---

## 4. CART 훈련 알고리즘 (5.4)

> 노트북에는 이 절의 코드가 없다. 4-2의 코드는 알고리즘 검증을 위해 직접 작성한 것이다.

### 4-1. 비용 함수

**목적**
트리가 각 노드에서 분할 질문을 고르는 기준을 정의한다.

**개념·원리**

사이킷런은 **CART**(Classification And Regression Tree) 알고리즘을 사용한다.

1. 특성 $k$와 임곗값 $`t_k`$로 훈련 세트를 두 부분집합으로 나눈다
2. 가능한 모든 $`(k, t_k)`$ 중 비용 함수 $J$가 최소인 쌍을 선택한다
3. 나뉜 부분집합에 같은 과정을 재귀적으로 반복한다
4. `max_depth`에 도달하거나 불순도를 줄이는 분할이 없으면 멈춘다

**분류용 CART 비용 함수**

```math
J(k, t_k) = \frac{m_\text{left}}{m} G_\text{left} + \frac{m_\text{right}}{m} G_\text{right}
```

- $m$: 분할 전 샘플 수, $`m_\text{left}`$, $`m_\text{right}`$: 각 자식의 샘플 수
- $`G_\text{left}`$, $`G_\text{right}`$: 각 자식의 불순도
- 자식 불순도의 **샘플 수 가중 평균** → 큰 자식의 불순도가 더 크게 반영됨
- 후보 임곗값: 특성의 정렬된 고유값에서 **인접한 두 값의 중간점**

**결과 해석**

루트 노드의 후보 분할 비교 (전체 후보 63개 중 일부)

| 분할 | 왼쪽 value | $`G_\text{left}`$ | 오른쪽 value | $`G_\text{right}`$ | $J$ |
|---|---|---|---|---|---|
| 길이 ≤ 2.45 | [50, 0, 0] | 0 | [0, 50, 50] | 0.5 | **0.333** |
| 너비 ≤ 0.8 | [50, 0, 0] | 0 | [0, 50, 50] | 0.5 | **0.333** |
| 너비 ≤ 1.75 | [50, 49, 5] | 0.545 | [0, 1, 45] | 0.043 | 0.391 |
| 길이 ≤ 4.75 | [50, 44, 1] | 0.508 | [0, 6, 49] | 0.194 | 0.393 |
| 길이 ≤ 5.05 | [50, 49, 9] | 0.573 | [0, 1, 41] | 0.046 | 0.425 |
| 너비 ≤ 1.35 | [50, 28, 0] | 0.460 | [0, 22, 50] | 0.424 | 0.443 |

- 최소 비용: $`J = \frac{50}{150} \times 0 + \frac{100}{150} \times 0.5 = 0.333`$
- 2.45 = (setosa 최대 길이 1.9 + versicolor 최소 길이 3.0) / 2
- **동점 존재**: 너비 ≤ 0.8도 $J$ = 0.333. 0.8 = (setosa 최대 너비 0.6 + versicolor 최소 너비 1.0) / 2 → 10절 분산 문제의 원인

---

### 4-2. 분할 탐색 구현

**목적**
비용 함수로 최적 분할을 찾는 과정을 직접 구현해 사이킷런 결과와 비교한다.

**코드**

```python
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=3) / len(y)        # 클래스별 비율
    return 1 - (p ** 2).sum()

def best_split(X, y):
    m = len(y)
    best = (None, None, np.inf)                      # (특성, 임곗값, 비용)
    for k in range(X.shape[1]):                      # 모든 특성
        values = np.unique(X[:, k])                  # 정렬된 고유값
        for t in (values[:-1] + values[1:]) / 2:     # 인접값의 중간점 = 후보 임곗값
            left = X[:, k] <= t
            J = left.sum() / m * gini(y[left]) + (~left).sum() / m * gini(y[~left])
            if J < best[2] - 1e-12:                  # 부동소수 오차 허용
                best = (k, t, J)
    return best

y = y_iris.values
best_split(X_iris, y)                        # (0, 2.45, 0.3333)   루트

right = X_iris[:, 0] > 2.45                  # 오른쪽 자식에 재귀 적용
best_split(X_iris[right], y[right])          # (1, 1.75, 0.1103)   깊이 1
```

**결과 해석**
- 루트: 특성 0(길이) ≤ 2.45, $J$ = 0.333 → 사이킷런 루트와 일치
- 깊이 1: 특성 1(너비) ≤ 1.75, $J$ = 0.110 → 사이킷런 깊이 1 분할과 일치

**정리**
- CART = "모든 후보 분할의 비용을 계산 → 최소 선택 → 자식에 재귀 적용"

---

### 4-3. 탐욕 알고리즘의 한계

**개념·원리**
- CART는 각 단계에서 **현재 비용이 최소인 분할**만 선택하고 이후 단계는 고려하지 않는 **탐욕 알고리즘**(greedy algorithm)
- 합리적으로 좋은 해를 찾지만 **전역 최적 트리를 보장하지 않는다**
- 최적 트리 탐색은 **NP-완전** 문제 → $`O(\exp(m))`$ 시간이 필요해 작은 데이터에서도 비현실적

**정리**
- 탐욕적 분할은 최적성 대신 계산 가능성을 택한 타협

---

## 5. 계산 복잡도 (5.5)

**목적**
샘플 수 $m$, 특성 수 $n$에 따른 예측·훈련 비용을 파악한다.

**개념·원리**

| 단계 | 복잡도 | 근거 |
|---|---|---|
| 예측 | $`O(\log_2 m)`$ | 루트→리프 경로 길이 = 트리 깊이. 대체로 균형 잡힌 트리의 깊이 ≈ $`\log_2 m`$. 노드마다 특성 1개만 비교 → $n$과 무관 |
| 훈련 | $`O(n \times m \log_2 m)`$ | 각 깊이 수준에서 모든 특성 $n$개에 대해 샘플 $m$개를 비교 |

**결과 해석**

| $m$ | $`\log_2 m`$ (예측 시 비교 횟수) |
|---|---|
| $`10^3`$ | ≈ 10 |
| $`10^6`$ | ≈ 20 |
| $`10^9`$ | ≈ 30 |

훈련 시간 증가 배율 예 (연습문제 5번): $m$을 $`10^6`$ → $`10^7`$으로 늘리면

```math
K = \frac{n \times 10m \log_2(10m)}{n \times m \log_2 m} = 10 \times \frac{\log_2 10^7}{\log_2 10^6} = 10 \times \frac{23.25}{19.93} \approx 11.7
```

**정리**
- 예측: 샘플 수에 로그로 증가, 특성 수와 무관 → 매우 빠름
- 훈련: 특성 수에 비례, 샘플 수에 대해 선형보다 약간 크게 증가

---

## 6. 지니 불순도 또는 엔트로피? (5.6)

**목적**
지니 외의 불순도 지표인 엔트로피를 이해하고 두 지표를 비교한다.

**개념·원리**

**엔트로피** (entropy): 정보 이론에서 메시지의 평균 정보량. 불순도 지표로 사용 가능

```math
H_i = -\sum_{\substack{k=1 \\ p_{i,k} \neq 0}}^{n} p_{i,k} \log_2 p_{i,k}
```

| 성질 | 지니 | 엔트로피 |
|---|---|---|
| 순수 노드 | 0 | 0 |
| 최댓값 ($n$ 클래스 균등) | $`1 - 1/n`$ | $`\log_2 n`$ |
| 연산 | 제곱 | 로그 |

| 노드 | 지니 | 엔트로피 |
|---|---|---|
| [50, 50, 50] | 0.667 | $`\log_2 3`$ = 1.585 |
| [0, 49, 5] | 0.168 | $`-(0.907 \log_2 0.907 + 0.093 \log_2 0.093)`$ = 0.445 |

**코드**

```python
DecisionTreeClassifier(criterion="entropy")   # 기본값 "gini". "log_loss"는 "entropy"와 동일

# 두 기준 비교 (노트북 외 검증 코드)
tree_ent = DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=42)
tree_ent.fit(X_iris, y_iris)
(tree_ent.predict(X_iris) == tree_clf.predict(X_iris)).all()   # True
```

**결과 해석**

이진 분류 노드에서 클래스 1 비율 $p$에 따른 두 지표 (추가 그림):

![지니 vs 엔트로피](images/05/extra_gini_vs_entropy.png)

- 두 지표 모두 $p$ = 0, 1에서 0, $p$ = 0.5에서 최대
- 엔트로피 / 2 ≈ 지니 → 분할 순위가 대부분 같음
- 붓꽃 데이터에서 두 기준이 동일한 트리 생성

| 구분 | 지니 | 엔트로피 |
|---|---|---|
| 계산 속도 | 약간 빠름 | 약간 느림 |
| 분할 경향 | 가장 빈도 높은 클래스를 한쪽 가지로 고립 | 조금 더 균형 잡힌 트리 |

**정리**
- 대부분 결과가 비슷하므로 계산이 빠른 지니(기본값)를 사용

---

## 7. 규제 하이퍼파라미터 (5.7)

### 7-1. 파라미터 모델과 비파라미터 모델

**목적**
결정 트리가 과대적합되기 쉬운 이유를 이해한다.

**개념·원리**

| 구분 | 정의 | 과대적합 경향 | 예 |
|---|---|---|---|
| 파라미터 모델 | 모델 파라미터 수가 훈련 전에 결정됨 → 자유도 제한 | 낮음 (과소적합 위험 높음) | 선형 모델 |
| 비파라미터 모델 | 파라미터 수가 훈련 전에 결정되지 않음 → 구조가 데이터에 맞춰 커짐 | 높음 | 결정 트리 |

- 제약 없는 결정 트리는 모든 리프가 순수해질 때까지 분할 → 훈련 데이터에 과대적합

**정리**
- 트리의 성장을 제한하는 **규제**가 필요하다

---

### 7-2. 규제 하이퍼파라미터 종류

**개념·원리**

| 하이퍼파라미터 | 의미 | 규제 강화 |
|---|---|---|
| `max_depth` | 최대 깊이 (기본 `None` = 제한 없음) | ↓ |
| `max_features` | 각 노드에서 분할에 고려할 최대 특성 수 | ↓ |
| `max_leaf_nodes` | 최대 리프 노드 수 | ↓ |
| `min_samples_split` | 분할하기 위해 노드에 필요한 최소 샘플 수 | ↑ |
| `min_samples_leaf` | 리프 노드에 필요한 최소 샘플 수 | ↑ |
| `min_weight_fraction_leaf` | `min_samples_leaf`를 전체 가중치 대비 비율로 지정 | ↑ |

- 규칙: `max_*` 감소, `min_*` 증가 → 규제 강화

**가지치기** (pruning)
- 제약 없이 트리를 만든 뒤 불필요한 노드를 제거하는 방식
- 교재: χ² 검정으로 분할이 통계적으로 유의미한지(순전히 우연이 아닌지) 판정해 유의하지 않은 자식 노드 제거
- 사이킷런: `ccp_alpha`(비용 복잡도 가지치기)

---

### 7-3. moons 데이터 실험

**목적**
규제 유무에 따른 결정 경계와 일반화 성능을 비교한다.

**코드: 학습**

```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier(random_state=42)                      # 규제 없음
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)  # 리프당 최소 5개
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
```

- `make_moons(n_samples, noise)`: 두 반달 모양 클래스 생성. `noise` = 가우스 잡음의 표준편차

**코드: 결정 경계 시각화 함수**

```python
def plot_decision_boundary(clf, X, y, axes, cmap):
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)    # 영역 채우기
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)  # 경계선만
    colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[cmap][idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)   # 서브플롯 2개, y축 공유
plt.sca(axes[0])                                                   # 왼쪽 서브플롯 선택
plot_decision_boundary(tree_clf1, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title("No restrictions")
plt.sca(axes[1])                                                   # 오른쪽 서브플롯 선택
plot_decision_boundary(tree_clf2, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"min_samples_leaf = {tree_clf2.min_samples_leaf}")
plt.show()
```

- 함수 본문 앞 4줄 = 2-2의 결정 경계 절차를 함수화. 9절에서 재사용

**코드: 일반화 성능 평가**

```python
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2,
                                        random_state=43)   # 같은 분포의 새 데이터
tree_clf1.score(X_moons_test, y_moons_test)   # 0.898
tree_clf2.score(X_moons_test, y_moons_test)   # 0.92
```

**결과 해석**

![그림 5-3 규제 비교](images/05/fig5-3_moons_regularization.png)

| 모델 | 깊이 | 리프 수 | 훈련 정확도 | 테스트 정확도 | 진단 |
|---|---|---|---|---|---|
| 규제 없음 | 8 | 19 | 1.000 | 0.898 | 과대적합 |
| `min_samples_leaf=5` | 6 | 13 | 0.947 | **0.920** | 일반화 개선 |

(깊이·리프 수·훈련 정확도는 노트북 외 추가 확인값)

- 규제 없음: 반대 클래스 영역에 섞인 개별 샘플까지 분리하는 작은 영역 생성 → 경계가 복잡
- 규제 적용: 훈련 정확도는 낮아졌지만 경계가 단순하고 테스트 정확도 상승

**정리**
- 결정 트리는 규제 없이 훈련 세트를 암기한다
- 규제 효과는 훈련 점수가 아닌 테스트(또는 교차 검증) 점수로 판단한다

---

## 8. 회귀 (5.8)

### 8-1. 회귀 트리 학습

**목적**
결정 트리로 연속값을 예측한다.

**개념·원리**

| 구분 | 분류 트리 | 회귀 트리 |
|---|---|---|
| 클래스 | `DecisionTreeClassifier` | `DecisionTreeRegressor` |
| 리프 예측값 | 다수 클래스 | 리프 샘플들의 타깃 **평균** |
| 분할 기준 | 불순도 (지니/엔트로피) | **MSE** |

**코드**

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(seed=42)
X_quad = rng.random((200, 1)) - 0.5                                  # [-0.5, 0.5), 특성 1개
y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))         # y = x² + N(0, 0.025²)

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)

export_graphviz(tree_reg, out_file="my_regression_tree.dot",
                feature_names=["x1"], rounded=True, filled=True)
Source.from_file("my_regression_tree.dot")
```

**결과 해석**

![그림 5-4 회귀 트리](images/05/fig5-4_regression_tree.png)

| 노드 | 구간 | samples | value (평균) | squared_error |
|---|---|---|---|---|
| 루트 | 전체 | 200 | 0.080 | 0.0056 |
| 리프 | $`x_1 \le -0.302`$ | 42 | 0.151 | 0.0028 |
| 리프 | $`-0.302 < x_1 \le 0.343`$ | 133 | 0.038 | 0.0017 |
| 리프 | $`0.343 < x_1 \le 0.431`$ | 14 | 0.150 | 0.0005 |
| 리프 | $`x_1 > 0.431`$ | 11 | 0.229 | 0.0006 |

- 예측 예: $`x_1 = 0.2`$ → 두 번째 리프 → $`\hat{y}`$ = 0.038 (`tree_reg.predict([[0.2]])` = 0.0376). 실제 함수값 $`0.2^2 = 0.04`$에 근접
- 데이터가 무작위 생성이므로 교재 그림과 수치가 다를 수 있음

---

### 8-2. 회귀용 CART 비용 함수

**개념·원리**

분류용 비용 함수(4-1)의 불순도 $G$를 MSE로 바꾼다.

```math
J(k, t_k) = \frac{m_\text{left}}{m} \text{MSE}_\text{left} + \frac{m_\text{right}}{m} \text{MSE}_\text{right}
```

```math
\text{MSE}_\text{node} = \frac{1}{m_\text{node}} \sum_{i \in \text{node}} \left( \hat{y}_\text{node} - y^{(i)} \right)^2, \qquad
\hat{y}_\text{node} = \frac{1}{m_\text{node}} \sum_{i \in \text{node}} y^{(i)}
```

- 각 자식 노드의 샘플이 그 노드 평균에 최대한 가깝게 모이도록 분할

**정리**
- 리프 예측 = 평균, 분할 기준 = 가중 MSE 최소

---

### 8-3. 깊이에 따른 예측 형태

**목적**
트리 깊이가 회귀 예측 함수에 미치는 영향을 확인한다.

**개념·원리**
- 각 리프가 상수 하나를 출력 → 예측 함수는 **구간별 상수 함수**(계단 함수)
- 깊이 $d$ 트리의 구간 수 ≤ $`2^d`$

**코드**

```python
tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_reg2.fit(X_quad, y_quad)

tree_reg.tree_.threshold
# array([ 0.343 , -0.3018, -2.    , -2.    ,  0.4314, -2.    , -2.    ])
tree_reg2.tree_.threshold
# array([ 0.343 , -0.3018, -0.414 , -2.    , -2.    ,  0.2182, -2.    ,
#        -2.    ,  0.4314,  0.3948, -2.    , -2.    ,  0.4647, -2.    , -2.    ])

def plot_regression_predictions(tree_reg, X, y, axes=[-0.5, 0.5, -0.05, 0.25]):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)   # (500,) → (500, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.plot(X, y, "b.")                                     # 훈련 데이터
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")   # 예측

th0, th1a, th1b = tree_reg.tree_.threshold[[0, 1, 4]]   # 깊이 0, 깊이 1 ×2
th2s = tree_reg2.tree_.threshold[[2, 5, 9, 12]]         # 깊이 2 ×4
```

- `-2`: 리프 노드 표시값 (사이킷런의 `TREE_UNDEFINED`)
- `.reshape(-1, 1)`: 사이킷런은 특성이 1개여도 2차원 입력을 요구

**결과 해석**

![그림 5-5 회귀 트리 깊이 비교](images/05/fig5-5_regression_depth.png)

| 모델 | 구간 수 | 예측 형태 |
|---|---|---|
| `max_depth=2` | 4 | 거친 계단 |
| `max_depth=3` | 8 | 곡선을 더 촘촘히 따라가는 계단 |

- 깊이 3 트리는 깊이 0·1 임곗값(0.343, −0.302, 0.431)을 유지하고 하위 분할만 추가

---

### 8-4. 회귀 트리의 규제

**목적**
규제 없는 회귀 트리의 과대적합을 확인한다.

**코드**

```python
tree_reg1 = DecisionTreeRegressor(random_state=42)                       # 규제 없음
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)  # 리프당 최소 10개
tree_reg1.fit(X_quad, y_quad)
tree_reg2.fit(X_quad, y_quad)

x1 = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)
y_pred1 = tree_reg1.predict(x1)
y_pred2 = tree_reg2.predict(x1)
```

**결과 해석**

![그림 5-6 회귀 규제 비교](images/05/fig5-6_regression_regularization.png)

| 모델 | 리프 수 (추가 확인값) | 예측 형태 | 진단 |
|---|---|---|---|
| 규제 없음 | 200 (= 샘플 수) | 모든 샘플을 통과하는 불규칙한 선 | 과대적합 |
| `min_samples_leaf=10` | 15 | 2차 곡선을 매끄럽게 근사 | 적절 |

**정리**
- 회귀 트리도 분류 트리와 같은 하이퍼파라미터로 규제한다

---

## 9. 축 방향에 대한 민감성 (5.9)

### 9-1. 회전 실험

**목적**
데이터의 방향(회전)이 결정 트리에 미치는 영향을 확인한다.

**개념·원리**
- 분할이 항상 축에 수직(2-2) → 기울어진 경계는 여러 개의 축 수직 분할(계단)로 근사해야 함
- 분할이 많아질수록 모델이 복잡해지고 일반화에 불리

**코드**

```python
rng = np.random.default_rng(seed=42)
X_square = rng.random((100, 2)) - 0.5
y_square = (X_square[:, 0] > 0).astype(np.int64)        # 정답 경계: x₁ = 0 (세로선)

angle = np.pi / 4                                       # 45도
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)        # 전체 샘플 45° 회전 → 경계가 대각선

tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)

# 7-3의 plot_decision_boundary로 두 모델을 나란히 시각화 (cmap="Pastel1")
```

**결과 해석**

![그림 5-7 회전에 대한 민감성](images/05/fig5-7_rotation.png)

| 데이터 | 정답 경계 | 깊이 | 리프 수 | 학습된 경계 |
|---|---|---|---|---|
| 원본 | 세로선 | 1 | 2 | 세로선 1개 |
| 45° 회전 | 대각선 | 4 | 6 | 계단 |

(깊이·리프 수는 노트북 외 추가 확인값)

- 두 데이터는 회전 외에 동일하지만 회전된 데이터에서 트리가 훨씬 복잡해짐

---

### 9-2. PCA를 이용한 축 정렬

**목적**
데이터를 회전시켜 주요 변동 방향을 축에 맞춘다.

**개념·원리**
- **PCA**(주성분 분석): 분산이 가장 큰 방향을 첫 번째 축 $`z_1`$, 그와 직교하는 방향을 $`z_2`$로 하는 좌표로 데이터를 회전 (7장)
- 붓꽃의 꽃잎 길이·너비는 강한 양의 상관 → 데이터가 대각선 방향으로 분포 → PCA로 대각선을 $`z_1`$ 축에 정렬

**코드**

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())   # 스케일링 → 회전
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)

th1, th2 = tree_clf_pca.tree_.threshold[[0, 2]]          # -0.911, 0.758
```

- `StandardScaler`를 먼저 적용하는 이유: PCA는 분산 기반이므로 특성 스케일에 민감

**결과 해석**

![그림 5-8 PCA 후 붓꽃 트리](images/05/fig5-8_pca_iris.png)

| 항목 | 원본 좌표 | PCA 좌표 |
|---|---|---|
| 사용 특성 | 길이, 너비 | $`z_1`$만 (`tree_.feature` = `[0, -2, 0, -2, -2]`) |
| 결정 경계 | 세로선 + 가로선 | 세로선 2개 |
| 훈련 정확도 | 0.96 | 0.98 |

- $`z_1`$ 하나가 분산의 98.1% 설명 (추가 확인값)

**한계**
$`z_1`$은 원래 특성의 선형 결합이므로 "z₁ ≤ −0.911" 같은 규칙은 해석이 어렵다 → 화이트박스 장점 일부 상실.

**정리**
- 결정 트리는 데이터 방향에 민감하다
- PCA로 축을 정렬하면 트리가 단순해질 수 있으나 해석성은 떨어진다

---

## 10. 결정 트리의 분산 문제 (5.10)

**목적**
결정 트리가 작은 변화에도 크게 달라지는 고분산 모델임을 확인한다.

**개념·원리**
- **분산이 높은 모델**: 훈련 데이터나 하이퍼파라미터의 작은 변화에 학습 결과가 크게 달라지는 모델
- 사이킷런의 CART 구현은 노드마다 **특성을 무작위 순서로 검토**하고, 비용이 같은 분할이 있으면 먼저 찾은 것을 선택
- 붓꽃 루트에서 "길이 ≤ 2.45"와 "너비 ≤ 0.8"이 동일 비용(4-1) → `random_state`에 따라 루트 분할이 달라짐
- 상위 분할이 바뀌면 하위 구조와 결정 경계 전체가 바뀜

**코드**

```python
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)   # 42 → 40
tree_clf_tweaked.fit(X_iris, y_iris)

# 그림 5-9: 2-2와 같은 방식으로 결정 경계 표시
y_pred = tree_clf_tweaked.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)
...
th0, th1 = tree_clf_tweaked.tree_.threshold[[0, 2]]
plt.plot([0, 7.2], [th0, th0], "k-", linewidth=2)    # 깊이 0: 가로선
plt.plot([0, 7.2], [th1, th1], "k--", linewidth=2)   # 깊이 1: 가로선
```

시드에 따른 루트 분할 분포 (노트북 외 검증 코드):

```python
from collections import Counter
roots = Counter()
for rs in range(100):
    t = DecisionTreeClassifier(max_depth=2, random_state=rs).fit(X_iris, y_iris).tree_
    roots[(t.feature[0], round(t.threshold[0], 2))] += 1
roots   # (특성 1, 0.8) → 51회, (특성 0, 2.45) → 49회
```

**결과 해석**

![그림 5-9 시드만 바꾼 트리](images/05/fig5-9_tweaked_tree.png)

| 항목 | `random_state=42` | `random_state=40` |
|---|---|---|
| 루트 분할 | 길이 ≤ 2.45 | 너비 ≤ 0.80 |
| 깊이 1 분할 | 너비 ≤ 1.75 | 너비 ≤ 1.75 |
| 결정 경계 | 세로선 + 가로선 | 가로선 2개 |
| 훈련 정확도 | 0.96 | 0.96 |
| 입력 (길이 2, 너비 1.5) 예측 | setosa | versicolor |

- 시드 0~99 중 루트 분할: 너비 0.8 → 51회, 길이 2.45 → 49회
- 훈련 정확도는 같지만 훈련 샘플이 없는 영역에서 예측이 다름

**정리**
- 결정 트리는 고분산 모델이다
- 대응: 여러 트리의 예측을 평균(다수결)해 분산을 줄인다 → **랜덤 포레스트** (6장). 연습문제 8번에서 직접 구현

---

## 11. 부록: 트리 구조 접근 (`tree_`)

**목적**
학습된 트리의 내부 구조를 코드로 조회한다.

**개념·원리**
- `tree_` 속성(`sklearn.tree._tree.Tree`)에 **노드 번호를 인덱스로 하는 NumPy 배열**로 구조 저장
- 노드 번호: 루트 0부터 깊이 우선(전위) 순서

```
        0
       / \
      1   2
         / \
        3   4
```

| 속성 | 의미 | `tree_clf` 값 |
|---|---|---|
| `node_count` | 전체 노드 수 | 5 |
| `max_depth` | 최대 깊이 | 2 |
| `max_n_classes` | 클래스 수 | 3 |
| `n_features` | 특성 수 | 2 |
| `n_outputs` | 타깃 수 | 1 |
| `n_leaves` | 리프 수 | 3 |
| `impurity` | 노드별 불순도 | `[0.6667, 0, 0.5, 0.168, 0.0425]` |
| `children_left` | 왼쪽 자식 번호 (리프는 −1) | `[1, -1, 3, -1, -1]` |
| `children_right` | 오른쪽 자식 번호 (리프는 −1) | `[2, -1, 4, -1, -1]` |
| `feature` | 분할 특성 (리프는 −2) | `[0, -2, 1, -2, -2]` |
| `threshold` | 분할 임곗값 (리프는 −2) | `[2.45, -2, 1.75, -2, -2]` |
| `n_node_samples` | 노드별 샘플 수 | `[150, 50, 100, 54, 46]` |
| `value` | 노드별 클래스 분포 | 형태 (5, 1, 3) |

**코드: 리프 판별**

```python
tree = tree_clf.tree_

tree.children_left[0], tree.children_right[0]   # (1, 2)
tree.children_left[3], tree.children_right[3]   # (-1, -1)

is_leaf = (tree.children_left == tree.children_right)   # 리프는 두 자식 번호가 같음(-1)
np.arange(tree.node_count)[is_leaf]                     # array([1, 3, 4])
```

**코드: 노드 깊이 계산**

```python
def compute_depth(tree_clf):
    tree = tree_clf.tree_
    depth = np.zeros(tree.node_count)
    stack = [(0, 0)]                                    # (노드 번호, 깊이)
    while stack:
        node, node_depth = stack.pop()
        depth[node] = node_depth
        if tree.children_left[node] != tree.children_right[node]:   # 분할 노드
            stack.append((tree.children_left[node], node_depth + 1))
            stack.append((tree.children_right[node], node_depth + 1))
    return depth

depth = compute_depth(tree_clf)                          # array([0., 1., 1., 2., 2.])

tree_clf.tree_.feature[(depth == 1) & (~is_leaf)]        # array([1])     깊이 1 분할 노드의 특성
tree_clf.tree_.threshold[(depth == 1) & (~is_leaf)]      # array([1.75])  임곗값
```

**원리: 스택 기반 깊이 우선 탐색**

| 단계 | pop | 기록 | push | 스택 |
|---|---|---|---|---|
| 1 | (0, 0) | depth[0] = 0 | (1, 1), (2, 1) | [(1,1), (2,1)] |
| 2 | (2, 1) | depth[2] = 1 | (3, 2), (4, 2) | [(1,1), (3,2), (4,2)] |
| 3 | (4, 2) | depth[4] = 2 | — | [(1,1), (3,2)] |
| 4 | (3, 2) | depth[3] = 2 | — | [(1,1)] |
| 5 | (1, 1) | depth[1] = 1 | — | [] |

**코드: 클래스 분포**

```python
tree.value.squeeze(1)
# [[0.3333, 0.3333, 0.3333],
#  [1.    , 0.    , 0.    ],
#  [0.    , 0.5   , 0.5   ],
#  [0.    , 0.9074, 0.0926],
#  [0.    , 0.0217, 0.9783]]

np.all(tree.value.sum(axis=(1, 2)) == tree.n_node_samples)
# 노트북 기대값: True / scikit-learn 1.9.1 실행 결과: False
```

**버전 차이**
- 노트북은 `value`를 **클래스별 개수**로 가정, 최근 scikit-learn은 분류 트리에서 **클래스별 비율** 저장 → 행 합이 1이므로 비교 결과 `False`
- 개수 복원:

```python
counts = tree.value.squeeze(1) * tree.weighted_n_node_samples[:, None]
# [[50, 50, 50], [50, 0, 0], [0, 50, 50], [0, 49, 5], [0, 1, 45]]
np.all(counts.sum(axis=1).round() == tree.n_node_samples)   # True
```

**정리**
- 리프 판별: `children_left == children_right`
- 리프 위치의 `feature`, `threshold` 값(−2)은 무의미
- 최근 버전에서 `value`는 비율로 저장됨

---

## 12. 연습문제 (5.11)

### 12-1. 개념 문제 (1–6번)

| 번 | 질문 | 답 | 근거 |
|---|---|---|---|
| 1 | 샘플 100만 개로 제한 없이 학습한 트리의 깊이 | 약 20 | 리프 ≈ 샘플 수, 균형 이진 트리 깊이 ≈ $`\log_2 10^6 \approx 19.9`$. 완전한 균형이 아니므로 실제로는 약간 더 깊음 |
| 2 | 노드의 지니는 부모보다 항상 낮은가 | 일반적으로 낮으나 항상은 아님 | 아래 예시 |
| 3 | 과대적합 시 `max_depth` 조정 | 감소 | 모델 제약 = 규제 |
| 4 | 과소적합 시 특성 스케일링 | 효과 없음 | 결정 트리는 스케일과 무관 |
| 5 | 100만 개에 1시간 → 1,000만 개 | 약 11.7시간 | 5절 계산 |
| 6 | 특성 수 2배 | 훈련 시간 약 2배 | 훈련 복잡도가 $n$에 비례 |

**2번 예시**: 1차원 데이터가 A, B, A, A, A 순서로 정렬된 노드

| 노드 | 구성 | 지니 |
|---|---|---|
| 부모 | A 4, B 1 | $`1 - (4/5)^2 - (1/5)^2 = 0.32`$ |
| 왼쪽 자식 | A, B | $`1 - (1/2)^2 - (1/2)^2 = 0.5`$ (부모보다 높음) |
| 오른쪽 자식 | A, A, A | 0 |
| 가중 평균 $J$ | | $`\frac{2}{5} \times 0.5 + \frac{3}{5} \times 0 = 0.2 < 0.32`$ |

- CART는 자식 불순도의 **가중 평균**을 최소화 → 한 자식이 부모보다 불순해도 전체 비용이 줄면 선택

---

### 12-2. moons 데이터 튜닝 (7번)

**목적**
그리드 서치로 결정 트리 하이퍼파라미터를 찾아 테스트 정확도 85~87%를 달성한다.

**코드**

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# a. 데이터 생성
X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)

# b. 분할 (훈련 8,000 / 테스트 2,000)
X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons,
                                                    test_size=0.2,
                                                    random_state=42)

# c. 그리드 서치
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

# d. 테스트 평가
y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)   # 0.8595
```

**결과 해석**

| 항목 | 값 |
|---|---|
| 탐색 조합 수 | 98 × 6 × 3 = 1,764 (3-폴드 → 5,292회 학습) |
| 최적 조합 | `max_depth=6`, `max_leaf_nodes=17` |
| CV 정확도 | 0.854 |
| 테스트 정확도 | **0.8595** |

- `GridSearchCV`는 `refit=True`(기본값)로 최적 조합을 전체 훈련 세트에 재학습 → 바로 `predict` 가능

---

### 12-3. 숲 만들기 (8번)

**목적**
작은 부분 세트로 학습한 트리 여러 개의 다수결이 단일 트리보다 나은지 확인한다.

**코드: a. 크기 100의 부분 세트 1,000개 생성**

```python
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100

mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances,
                  random_state=42)                     # test_size = 7,900 → 훈련 쪽 100개
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))
```

**코드: b. 트리 1,000개 학습 및 개별 평가**

```python
from sklearn.base import clone

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]   # 미학습 복제본

accuracy_scores = []
for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train, y_mini_train)
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

np.mean(accuracy_scores)   # 0.8056605
```

**코드: c. 다수결 예측**

```python
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)   # (1000, 2000): 트리 × 테스트 샘플
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)       # 열(샘플)마다 최빈값
```

**코드: d. 평가**

```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))   # 0.873
```

- 최근 SciPy의 `mode`는 1차원 `(2000,)`을 반환 → `.reshape([-1])`은 구버전 호환용

**결과 해석**

| 모델 | 학습 데이터 | 테스트 정확도 |
|---|---|---|
| 개별 트리 (1,000개 평균) | 100개 | 0.806 |
| 튜닝한 단일 트리 (7번) | 8,000개 | 0.8595 |
| 트리 1,000개 다수결 | 각 100개 | **0.873** |

**정리**
- 약한 트리 여러 개의 다수결이 전체 데이터로 학습한 단일 트리보다 우수
- 10절의 고분산 문제를 평균으로 완화한 결과 → 랜덤 포레스트의 기본 원리 (6장)

---

## 13. 정리

### 13-1. 개념과 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| 학습 | `DecisionTreeClassifier(max_depth=...).fit(X, y)` | 스케일링 불필요 |
| 시각화 | `export_graphviz`, `export_text` | 노드 = 질문, samples, value, gini |
| 예측 | `predict`, `predict_proba` | 루트→리프 경로, 확률 = 리프 클래스 비율 |
| 결정 경계 | `meshgrid` → `predict` → `contourf` | 경계는 축에 수직 |
| 훈련 원리 | (CART) | 가중 불순도 $J$ 최소 분할을 탐욕적·재귀적으로 선택 |
| 불순도 | `criterion="gini"` / `"entropy"` | 결과 대부분 동일, 기본값 지니 |
| 규제 | `max_depth`, `min_samples_leaf`, `max_leaf_nodes` … | `max_*` ↓, `min_*` ↑ |
| 회귀 | `DecisionTreeRegressor` | 리프 = 평균, MSE 최소화, 계단 함수 |
| 축 민감성 대응 | `make_pipeline(StandardScaler(), PCA())` | 주요 방향을 축에 정렬 |
| 분산 대응 | 여러 트리 + `mode` | 다수결 → 랜덤 포레스트 |
| 구조 조회 | `tree_.feature`, `tree_.threshold`, `children_left/right` | 노드 번호로 인덱싱 |

### 13-2. 실험 결과

| 실험 | 비교 | 결과 |
|---|---|---|
| 붓꽃 깊이 2 트리 | — | 루트 길이 ≤ 2.45, 깊이 1 너비 ≤ 1.75, 훈련 정확도 0.96 |
| 루트 후보 분할 비용 | 길이 ≤ 2.45 vs 너비 ≤ 0.8 | 동일 0.333 |
| 지니 vs 엔트로피 (붓꽃) | — | 동일 트리 |
| moons 규제 | 없음 / `min_samples_leaf=5` | 테스트 0.898 / **0.920** |
| 회귀 규제 | 없음 / `min_samples_leaf=10` | 리프 200 / 15 |
| 회전 실험 | 원본 / 45° 회전 | 리프 2 / 6 |
| PCA 적용 (붓꽃) | 원본 / PCA | 훈련 정확도 0.96 / 0.98 |
| 시드 변경 (100회) | 길이 2.45 / 너비 0.8 | 루트 선택 49회 / 51회 |
| 연습 7 | 튜닝한 단일 트리 | 테스트 0.8595 |
| 연습 8 | 개별 트리 평균 / 다수결 | 0.806 / **0.873** |

### 13-3. 핵심 원칙

1. 결정 트리는 "특성 ≤ 임곗값" 질문으로 공간을 축 정렬 직사각형으로 나누며, 스케일링이 필요 없다.
2. CART는 자식 불순도(회귀는 MSE)의 가중 평균을 최소화하는 분할을 탐욕적으로 고른다. 전역 최적은 보장되지 않는다.
3. 예측은 $`O(\log_2 m)`$으로 빠르고, 훈련은 $`O(n \times m \log_2 m)`$이다.
4. 지니와 엔트로피는 대부분 같은 트리를 만든다. 기본값 지니로 충분하다.
5. 결정 트리는 비파라미터 모델이므로 규제(`max_*` ↓, `min_*` ↑) 없이는 과대적합된다. 효과는 테스트·교차 검증 점수로 판단한다.
6. 축에 수직인 분할만 가능하므로 데이터 방향에 민감하다 → PCA로 축 정렬을 고려한다.
7. 작은 변화에도 구조가 크게 바뀌는 고분산 모델이다 → 여러 트리의 평균(랜덤 포레스트, 6장)으로 해결한다.
