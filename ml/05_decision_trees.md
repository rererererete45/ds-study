# 5장. 결정 트리

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `05_decision_trees.ipynb`
>
> 실행 환경: 노트북 요구사항은 scikit-learn ≥ 1.6.1. 이 노트의 출력값과 그림은 **scikit-learn 1.9.1**로 직접 실행해 얻은 결과.

## 이 장을 한 문장으로

> **결정 트리 = "예/아니오 질문을 순서대로 던져서 답을 찾아가는 스무고개".**

스무고개를 떠올려 보자. "살아 있나요?" → 예 → "날 수 있나요?" → 아니오 → "다리가 4개인가요?" … 질문 몇 번이면 정답에 도달한다. 결정 트리도 똑같다. 다만 질문이 항상 **"(어떤 특성) ≤ (어떤 숫자)?"** 꼴이고, 어떤 질문을 어떤 순서로 할지를 **데이터를 보고 스스로 정한다**.

## 이 장에서 배우는 것

| 절 | 질문 | 한 줄 답 |
|---|---|---|
| 5.1 | 트리는 어떻게 생겼나? | 질문 상자(노드)가 가지를 치며 내려가는 모양 |
| 5.2 | 어떻게 예측하나? | 위에서부터 질문에 답하며 내려가 맨 아래 상자(리프)의 답을 출력 |
| 5.3 | 확률도 낼 수 있나? | 리프에 모인 훈련 샘플의 클래스 비율 |
| 5.4 | 질문은 어떻게 고르나? | 모든 후보를 다 해 보고 **가장 깔끔하게 나뉘는 것** (CART) |
| 5.5 | 얼마나 빠른가? | 예측은 아주 빠름, 훈련은 데이터 양에 비례해서 조금 더 |
| 5.6 | "깔끔함"은 뭘로 재나? | 지니 불순도 또는 엔트로피 — 결과는 거의 같음 |
| 5.7 | 너무 외우면 어떡하나? | 트리가 자라는 걸 제한 (규제) |
| 5.8 | 숫자도 맞힐 수 있나? | 리프마다 평균값을 출력 → 계단 모양 예측 |
| 5.9 | 약점 ① | 가로·세로로만 자를 수 있어서 비스듬한 경계에 약함 |
| 5.10 | 약점 ② | 조금만 바뀌어도 트리가 완전히 달라짐 → 6장 랜덤 포레스트로 해결 |

## 실습 데이터

| 데이터 | 쓰이는 절 | 내용 |
|---|---|---|
| 붓꽃(iris) | 5.1~5.3, 5.9, 5.10 | 꽃 150송이, 품종 3개(setosa·versicolor·virginica) 각 50송이. **꽃잎 길이·너비 2개만** 사용 |
| moons | 5.7, 연습문제 | 반달 2개가 맞물린 모양의 가짜 데이터 |
| 2차 함수 + 잡음 | 5.8 | `y = x² + 약간의 잡음`, 점 200개 |

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

> 🎯 **이 절의 목표**: 트리를 하나 만들어 보고, 그림으로 그려서 **각 상자에 적힌 숫자를 읽을 수 있게** 된다.

### 1-1. 트리 만들기

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)
```

한 줄씩 보면

| 줄 | 하는 일 | 왜? |
|---|---|---|
| `load_iris(as_frame=True)` | 붓꽃 데이터를 표(DataFrame) 형태로 불러옴 | 열 이름으로 특성을 고르기 편하려고 |
| `[["petal length (cm)", "petal width (cm)"]]` | 특성 4개 중 **꽃잎 길이·너비**만 선택 | 특성이 2개면 x축·y축 평면에 그림을 그릴 수 있음 |
| `.values` | 표 → NumPy 배열 | 나중에 `[[5, 1.5]]` 같은 리스트로 예측해도 경고가 안 뜸 |
| `y_iris` | 정답(품종 번호) | 0=setosa, 1=versicolor, 2=virginica |
| `max_depth=2` | 질문을 **최대 2단계**까지만 | 그림을 보기 좋게 일부러 작게 만듦 |
| `random_state=42` | 무작위성 고정 | 결정 트리도 훈련에 약간의 무작위성이 있음 (10절에서 이유가 나옴) |
| `.fit(X, y)` | 데이터를 보고 **어떤 질문을 할지** 학습 | |

> 💡 **스케일링(StandardScaler)을 하지 않았다는 점에 주목.** 4장의 선형 모델은 특성 크기를 맞춰 줘야 했다. 결정 트리는 "꽃잎 길이 ≤ 2.45?"처럼 **특성 하나를 숫자 하나와 비교**만 하므로, 단위가 cm든 mm든 결과가 같다. 전처리가 거의 필요 없다는 것이 큰 장점.

### 1-2. 그림으로 그리기

학습된 트리는 그냥 보면 알 수 없으니, 그림으로 그려 본다.

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
Source.from_file("my_iris_tree.dot")       # 노트북 화면에 바로 그림 표시

!dot -Tpng "my_iris_tree.dot" -o "my_iris_tree.png"   # PNG 파일로 저장
```

과정은 두 단계다.

1. `export_graphviz` → 트리 구조를 **`.dot`이라는 텍스트 파일**로 저장 (그림의 "설계도")
2. `Source.from_file` 또는 `dot` 명령 → 설계도를 **실제 그림**으로 변환

| 인자 | 의미 |
|---|---|
| `feature_names` | 그림에 `x[0]` 대신 "petal length (cm)"처럼 이름으로 표시 |
| `class_names` | 0, 1, 2 대신 setosa 같은 품종 이름으로 표시 |
| `rounded=True` | 상자 모서리를 둥글게 |
| `filled=True` | 상자를 다수 품종의 색으로 칠함. **한 품종만 있을수록 진하게** |

> 💡 `!dot ...` 앞의 `!`는 "파이썬 코드가 아니라 터미널 명령어를 실행하라"는 노트북 문법. Graphviz 프로그램이 설치돼 있어야 한다.
>
> 💡 Graphviz가 없으면 `sklearn.tree.plot_tree(tree_clf)`(matplotlib으로 그림)나 `sklearn.tree.export_text(tree_clf)`(글자로 출력)를 써도 된다.

### 1-3. 그림 읽는 법 (그림 5-1)

![그림 5-1 붓꽃 결정 트리](images/05/fig5-1_iris_tree.png)

**① 용어부터**

| 용어 | 뜻 | 그림에서 |
|---|---|---|
| **노드** | 상자 하나 | 상자 5개 |
| **루트 노드** | 맨 위 상자 (시작점, 깊이 0) | "petal length ≤ 2.45" 상자 |
| **리프 노드** | 아래로 더 갈라지지 않는 상자 = **최종 답이 나오는 곳** | 색칠된 상자 3개 |
| **깊이** | 루트에서 몇 단계 내려왔는지 | 루트 0, 그 아래 1, 맨 아래 2 |
| 왼쪽 화살표 | 질문의 답이 **True(예)** | |
| 오른쪽 화살표 | 질문의 답이 **False(아니오)** | |

**② 상자 안의 숫자**

루트 상자를 예로 들면

| 줄 | 값 | 뜻 |
|---|---|---|
| `petal length (cm) <= 2.45` | | 이 상자에서 던지는 **질문** (리프에는 없음) |
| `gini = 0.667` | | 이 상자가 얼마나 **섞여 있는지** (아래 ③에서 설명) |
| `samples = 150` | | 이 상자에 도착한 **훈련 샘플 수** |
| `value = [50, 50, 50]` | | 그중 **품종별 개수** [setosa, versicolor, virginica] |
| `class = setosa` | | 가장 많은 품종 (동점이면 앞 번호) |

**③ 지니 불순도(gini) — "얼마나 섞여 있나"**

> 🤔 **쉽게 말하면**: 상자에서 꽃 두 송이를 무작위로 (돌려놓으며) 뽑았을 때 **품종이 서로 다를 확률**.
> - 한 품종만 있으면 → 절대 다를 수 없음 → **0** (완벽하게 깔끔 = "순수하다")
> - 골고루 섞여 있으면 → 자주 다름 → 값이 큼

$$G = 1 - \sum_{k} p_k^2 \qquad (p_k = \text{그 상자에서 품종 } k \text{의 비율})$$

"1 − (같은 품종이 두 번 나올 확률들의 합)"이라고 읽으면 된다. 직접 계산해 보자.

**노드 3 (초록 상자, value = [0, 49, 5], samples = 54)**

| 단계 | 계산 |
|---|---|
| 1. 비율 구하기 | setosa 0/54 = 0, versicolor 49/54 ≈ 0.907, virginica 5/54 ≈ 0.093 |
| 2. 각각 제곱 | 0² = 0, 0.907² ≈ 0.823, 0.093² ≈ 0.009 |
| 3. 1에서 빼기 | 1 − (0 + 0.823 + 0.009) ≈ **0.168** ✔ 그림과 같음 |

나머지 상자도 같은 방법

| 상자 | value | 지니 | 해석 |
|---|---|---|---|
| 루트 | [50, 50, 50] | 1 − 3 × (1/3)² = **0.667** | 품종 3개가 완전히 반반반 → 3품종에서 가장 지저분한 상태 |
| 왼쪽 리프 | [50, 0, 0] | 1 − 1² = **0** | setosa만 → 완벽하게 순수 |
| 오른쪽 상자 | [0, 50, 50] | 1 − 0.5² − 0.5² = **0.5** | 두 품종이 반반 |
| 노드 3 | [0, 49, 5] | **0.168** | versicolor가 대부분, 조금 섞임 |
| 노드 4 | [0, 1, 45] | **0.043** | 거의 virginica만 |

> 📌 **트리는 내려갈수록 지니가 작아진다** = 질문을 할수록 상자가 깔끔해진다. 이것이 결정 트리가 하는 일의 전부다.

> 💡 사이킷런의 결정 트리는 항상 **가지가 정확히 2개**(예/아니오)인 이진 트리. 다른 알고리즘(ID3 등)은 가지가 3개 이상일 수도 있다.

---

## 2. 예측 (5.2)

> 🎯 **이 절의 목표**: 새 꽃이 들어왔을 때 트리가 답을 내는 과정과, 그것을 평면 위 **결정 경계**로 보는 법.

### 2-1. 손으로 따라가 보기

그림 5-1을 보며 꽃 3송이를 분류해 보자.

| 꽃 (길이, 너비) | 루트: 길이 ≤ 2.45? | 다음: 너비 ≤ 1.75? | 도착한 리프 | 예측 |
|---|---|---|---|---|
| (1.4, 0.2) | **예** → 왼쪽 | (질문 없음) | [50, 0, 0] | setosa |
| (5.0, 1.5) | 아니오 → 오른쪽 | **예** → 왼쪽 | [0, 49, 5] | versicolor |
| (5.5, 2.0) | 아니오 → 오른쪽 | 아니오 → 오른쪽 | [0, 1, 45] | virginica |

길어야 **질문 2번**이면 답이 나온다. 그래서 결정 트리는 예측이 아주 빠르다.

### 2-2. 결정 경계 그리기 (그림 5-2)

"평면 위 모든 점을 트리에 넣으면 어떤 품종으로 나올까?"를 색으로 칠한 그림.

![그림 5-2 붓꽃 결정 경계](images/05/fig5-2_iris_boundary.png)

**그림 읽기**
- 굵은 **실선(Depth=0)** = 루트 질문 "길이 = 2.45" → 왼쪽(노랑)은 전부 setosa
- **파선(Depth=1)** = 두 번째 질문 "너비 = 1.75" → 아래(파랑) versicolor, 위(초록) virginica
- **점선(Depth=2)** = `max_depth=3`이었다면 추가로 그었을 선 (실제 모델엔 없음, 참고용)
- 경계선이 **모두 가로 아니면 세로** → 결정 트리의 특징 (9절에서 약점이 됨)

**코드**

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])   # 노랑, 파랑, 초록
plt.figure(figsize=(8, 4))

lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)

for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
             style, label=f"Iris {name}")
```

핵심은 가운데 4줄. **"평면을 촘촘한 격자로 덮고 → 격자점마다 예측 → 색칠"** 하는 과정이다. 작은 예로 이해해 보자.

```python
xx, yy = np.meshgrid([1, 2, 3], [10, 20])
# xx = [[1, 2, 3],        yy = [[10, 10, 10],
#       [1, 2, 3]]              [20, 20, 20]]
# → 겹쳐 보면 격자점 6개: (1,10) (2,10) (3,10) (1,20) (2,20) (3,20)

np.c_[xx.ravel(), yy.ravel()]
# ravel(): 2차원을 1줄로 펼침 → [1,2,3,1,2,3] 와 [10,10,10,20,20,20]
# np.c_: 두 줄을 열로 붙임 → [[1,10], [2,10], [3,10], [1,20], [2,20], [3,20]]
#        = 모델에 넣을 수 있는 (6, 2) 모양
```

실제 코드는 이걸 100 × 100으로 한 것.

| 단계 | 코드 | 결과 모양 |
|---|---|---|
| ① 격자 만들기 | `np.meshgrid(x 100개, y 100개)` | 100×100 행렬 2개 |
| ② 1줄로 펴서 붙이기 | `np.c_[a.ravel(), b.ravel()]` | (10000, 2) |
| ③ 1만 개 점 모두 예측 | `tree_clf.predict(...)` | (10000,) |
| ④ 다시 100×100으로 접기 | `.reshape(lengths.shape)` | (100, 100) |
| ⑤ 같은 예측끼리 색칠 | `plt.contourf(...)` | 그림 |

나머지 부분

| 코드 | 의미 |
|---|---|
| `alpha=0.3` | 배경색을 30% 진하기로 (점이 잘 보이게) |
| `"yo"`, `"bs"`, `"g^"` | 색 + 모양: yellow circle, blue square, green 삼각형 |
| `X_iris[:, 0][y_iris == idx]` | "품종이 idx인 꽃들의 꽃잎 길이" (불리언 마스크로 골라내기) |

> 📌 **meshgrid → ravel → `np.c_` → predict → reshape → contourf** 는 2차원 결정 경계를 그리는 **표준 순서**. 이 장에서 계속 나오니 한 번 익혀 두면 편하다.

### 2-3. 분할선 그리기

그림 속 검은 선은 트리에서 **임곗값(threshold)** 을 꺼내 그린 것.

```python
tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_deeper.fit(X_iris, y_iris)
th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
# th0=2.45, th1=1.75, th2a=4.95, th2b=4.85

plt.plot([th0, th0], [0, 3], "k-", linewidth=2)       # x=2.45 세로선 (깊이 0)
plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)  # y=1.75 가로선 (깊이 1)
plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)   # x=4.95 점선 (깊이 2, 아래쪽)
plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)   # x=4.85 점선 (깊이 2, 위쪽)
```

- `tree_.threshold` = 노드 번호 순서대로 각 노드의 임곗값을 담은 배열
- `[[0, 2, 3, 6]]` = 그중 0, 2, 3, 6번 노드 값만 꺼내기
- `plt.plot([x1, x2], [y1, y2])` = 점 (x1, y1)에서 (x2, y2)까지 선 긋기

### 2-4. 화이트박스 모델

| 화이트박스 | 블랙박스 |
|---|---|
| 결정 트리 | 랜덤 포레스트, 신경망 |
| "꽃잎 길이가 2.45 이하라서 setosa입니다"처럼 **이유를 그대로 설명** 가능 | 성능은 좋지만 "왜 그렇게 판단했는지" 설명하기 어려움 |

> 💡 대출 심사·의료처럼 **판단 근거를 설명해야 하는 분야**에서 결정 트리가 사랑받는 이유.

---

## 3. 클래스 확률 추정 (5.3)

> 🎯 **이 절의 목표**: 트리가 "몇 % 확률로 이 품종"이라고 말하는 방법.

```python
tree_clf.predict_proba([[5, 1.5]]).round(3)
# array([[0.   , 0.907, 0.093]])
#          setosa  versicolor  virginica

tree_clf.predict([[5, 1.5]])
# array([1])  → versicolor
```

**계산 과정** — 아주 단순하다.

1. 꽃 (5, 1.5)이 도착한 리프를 찾는다 → 2-1절 표처럼 노드 3 (value = [0, 49, 5])
2. 그 리프의 **품종 비율**이 곧 확률 → 0/54, 49/54, 5/54 = **0, 0.907, 0.093**
3. `predict`는 가장 큰 확률의 품종 → versicolor

> 💡 `[[5, 1.5]]`처럼 대괄호가 **두 겹**인 이유: 사이킷런은 "샘플 여러 개 × 특성 여러 개" 모양(2차원)을 기대한다. 샘플 1개라도 `[ [샘플1] ]`로 감싸야 한다.

> ⚠️ **같은 리프에 도착하면 확률이 완전히 같다.** (5, 1.5)도, (6, 1.5)도, (5, 1.0)도 모두 노드 3 → 전부 90.7%. 경계에 아슬아슬하게 걸친 꽃이라도 확신도가 내려가지 않는다. 결정 트리의 확률은 이렇게 **거칠다**.

---

## 4. CART 훈련 알고리즘 (5.4)

> 🎯 **이 절의 목표**: 트리가 "길이 ≤ 2.45"라는 질문을 **어떻게 골랐는지** 이해하기.
>
> 노트북에는 이 절의 코드가 없다. 아래 코드는 원리를 확인하려고 **직접 작성**한 것.

### 4-1. 아이디어: 다 해 보고 가장 깔끔한 걸 고른다

> 🤔 **쉽게 말하면**: 꽃 150송이를 칼로 한 번 잘라 두 무더기로 나눠야 한다. 칼질은 "길이 몇 cm에서" 또는 "너비 몇 cm에서"만 가능하다. 가능한 칼질을 **전부 해 보고**, 나뉜 두 무더기가 **가장 깔끔한**(지니가 낮은) 칼질을 고른다. 그다음 각 무더기에서 또 같은 방법으로 자른다.

이 방법이 **CART** (Classification And Regression Tree). 정리하면

1. 특성 하나 *k* 와 기준값 *t* 를 골라 데이터를 둘로 나눈다
2. 가능한 모든 (*k*, *t*) 중 **비용 *J* 가 가장 작은** 것을 택한다
3. 나뉜 두 쪽에서 1~2를 **반복**한다
4. `max_depth`에 닿거나, 더 나눠도 깔끔해지지 않으면 멈춘다

### 4-2. 비용 함수: "두 무더기의 지니를 크기만큼 가중 평균"

$$J(k, t) = \frac{m_\text{왼쪽}}{m} G_\text{왼쪽} + \frac{m_\text{오른쪽}}{m} G_\text{오른쪽}$$

- *m* = 나누기 전 샘플 수, *m*\_왼쪽·*m*\_오른쪽 = 나눈 뒤 각 무더기 샘플 수
- 그냥 평균이 아니라 **샘플 수만큼 가중**한다. 꽃 2송이짜리 무더기가 깔끔한 것보다 100송이짜리 무더기가 깔끔한 게 훨씬 중요하니까

### 4-3. 루트에서 실제로 비교해 보기

붓꽃 데이터 루트에서 가능한 칼질은 **63가지**(길이·너비의 서로 다른 값 사이마다 하나씩). 그중 몇 개를 계산하면

| 칼질 | 왼쪽 무더기 | 왼쪽 지니 | 오른쪽 무더기 | 오른쪽 지니 | 비용 *J* |
|---|---|---|---|---|---|
| **길이 ≤ 2.45** | [50, 0, 0] | 0 | [0, 50, 50] | 0.5 | **0.333** ✅ |
| **너비 ≤ 0.8** | [50, 0, 0] | 0 | [0, 50, 50] | 0.5 | **0.333** ✅ |
| 너비 ≤ 1.75 | [50, 49, 5] | 0.545 | [0, 1, 45] | 0.043 | 0.391 |
| 길이 ≤ 4.75 | [50, 44, 1] | 0.508 | [0, 6, 49] | 0.194 | 0.393 |
| 길이 ≤ 5.05 | [50, 49, 9] | 0.573 | [0, 1, 41] | 0.046 | 0.425 |
| 너비 ≤ 1.35 | [50, 28, 0] | 0.460 | [0, 22, 50] | 0.424 | 0.443 |

- "길이 ≤ 2.45"의 비용 = 50/150 × 0 + 100/150 × 0.5 = **0.333** → 가장 작음 → 루트 질문으로 채택
- 2.45는 어디서 왔나? setosa의 최대 길이 1.9와 versicolor의 최소 길이 3.0의 **딱 중간** (1.9 + 3.0) / 2. 후보 기준값은 이렇게 **정렬된 값들 사이의 중간점**으로 만든다

> ⚠️ **동점이 있다!** "너비 ≤ 0.8"도 setosa를 완벽하게 떼어내서 비용이 똑같이 0.333이다 (0.8 = setosa 최대 너비 0.6과 versicolor 최소 너비 1.0의 중간). 둘 중 무엇이 뽑힐지는 `random_state`가 정한다 → **10절의 핵심 복선**.

### 4-4. 직접 구현해서 확인

```python
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=3) / len(y)   # 품종별 개수 → 비율
    return 1 - (p ** 2).sum()                  # 지니 공식 그대로

def best_split(X, y):
    m = len(y)
    best = (None, None, np.inf)                 # (특성 번호, 기준값, 비용) — 처음엔 비용 무한대
    for k in range(X.shape[1]):                 # 특성마다 (0=길이, 1=너비)
        values = np.unique(X[:, k])             # 중복 없이 정렬된 값들
        for t in (values[:-1] + values[1:]) / 2:   # 이웃한 값들의 중간점 = 후보 기준값
            left = X[:, k] <= t                 # 왼쪽으로 갈 샘플 (True/False 배열)
            J = left.sum() / m * gini(y[left]) + (~left).sum() / m * gini(y[~left])
            if J < best[2] - 1e-12:             # 더 작은 비용이면 갱신
                best = (k, t, J)
    return best

y = y_iris.values
best_split(X_iris, y)
# (0, 2.45, 0.3333)   → 특성 0(길이) ≤ 2.45  ✔ 사이킷런 루트와 같음

right = X_iris[:, 0] > 2.45                    # 오른쪽 무더기만 골라서
best_split(X_iris[right], y[right])
# (1, 1.75, 0.1103)   → 특성 1(너비) ≤ 1.75  ✔ 그림 5-1의 두 번째 질문과 같음
```

| 코드 | 쉬운 설명 |
|---|---|
| `np.bincount(y, minlength=3)` | 0, 1, 2가 각각 몇 개인지 셈 → `[50, 50, 50]` |
| `(values[:-1] + values[1:]) / 2` | `[1, 2, 4]`라면 `[1, 2]` + `[2, 4]` → `[1.5, 3.0]` = 이웃 값의 중간 |
| `~left` | True/False 뒤집기 → 오른쪽으로 갈 샘플 |
| `- 1e-12` | 컴퓨터 소수 계산 오차 때문에 "사실상 같은 값"을 더 좋다고 착각하지 않도록 |

**재귀적으로 반복**한다는 건, 이렇게 찾은 분할로 나눈 각 무더기에 `best_split`을 또 호출하는 것. 위 코드에서 오른쪽 무더기에 다시 적용한 게 바로 그 두 번째 단계다.

### 4-5. 탐욕 알고리즘이라는 한계

> 🤔 **쉽게 말하면**: 바둑에서 **지금 당장 가장 좋아 보이는 수**만 두고, 열 수 앞은 생각하지 않는 것.

- CART는 매 단계 **현재 가장 좋은** 분할만 고른다 (**탐욕적**, greedy)
- 그래서 "꽤 좋은" 트리는 만들지만 **세상에서 가장 좋은 트리**라는 보장은 없다
- 진짜 최적 트리를 찾으려면 모든 조합을 다 따져야 하는데, 이건 데이터가 조금만 커져도 시간이 폭발적으로 늘어나는 **NP-완전** 문제 → 현실적으로 불가능 → 그래서 탐욕적으로 타협

---

## 5. 계산 복잡도 (5.5)

> 🎯 **이 절의 목표**: 데이터가 커지면 결정 트리가 얼마나 느려지는지 감 잡기.

### 5-1. 예측: O(log₂ *m*) — 엄청 빠름

> 🤔 **쉽게 말하면**: "1부터 100만 사이 숫자 맞히기" 게임에서 "50만보다 큰가요?"처럼 **매번 절반씩 줄이면 20번** 만에 맞힌다. 균형 잡힌 결정 트리도 똑같다.

| 훈련 샘플 수 *m* | 트리 깊이 ≈ log₂ *m* | 예측 시 질문 수 |
|---|---|---|
| 1,000 | 약 10 | 약 10번 |
| 1,000,000 | 약 20 | 약 20번 |
| 1,000,000,000 | 약 30 | 약 30번 |

- 데이터가 1000배 늘어도 질문은 10번만 늘어난다
- 각 노드에서 **특성 하나만** 보므로 특성이 많아도 예측 속도는 그대로

### 5-2. 훈련: O(*n* × *m* log₂ *m*)

- 각 깊이에서 **모든 특성(*n*개)** × **모든 샘플(*m*개)** 을 비교해야 함 + 정렬 비용(log *m*)
- 대략 "특성 수에 비례, 샘플 수에 **비례보다 조금 더**" 늘어난다

> 💡 **연습문제 5번 미리 보기**: 100만 개로 훈련에 1시간 걸렸다면 1,000만 개는?
> 배율 *K* = (10*m* × log₂ 10*m*) / (*m* × log₂ *m*) = 10 × log₂(10⁷) / log₂(10⁶) = 10 × 23.25 / 19.93 ≈ **11.7시간**.
> 데이터 10배 → 시간은 10배보다 약간 더(11.7배).

---

## 6. 지니 불순도 또는 엔트로피? (5.6)

> 🎯 **이 절의 목표**: "섞인 정도"를 재는 또 다른 자, 엔트로피를 알고 지니와 비교하기.

### 6-1. 엔트로피란?

> 🤔 **쉽게 말하면**: "상자에서 하나 꺼냈을 때 결과를 맞히기가 **얼마나 어려운가**(=얼마나 놀라운가)".
> - 한 품종만 있으면 → 뭐가 나올지 뻔함 → **0**
> - 반반 섞여 있으면 → 예측하기 가장 어려움 → 값이 최대

$$H = -\sum_{k,\, p_k \ne 0} p_k \log_2 p_k$$

```python
DecisionTreeClassifier(criterion="entropy")   # 기본값은 criterion="gini"
```

**노드 3 [0, 49, 5]로 직접 계산**

| 단계 | 계산 |
|---|---|
| 비율 | 49/54 ≈ 0.907, 5/54 ≈ 0.093 (0인 setosa는 건너뜀) |
| 각 항 | 0.907 × log₂ 0.907 ≈ −0.128, 0.093 × log₂ 0.093 ≈ −0.318 |
| 합에 − 붙이기 | −(−0.128 − 0.318) ≈ **0.445** |

### 6-2. 지니와 엔트로피를 그래프로 비교 (추가 그림)

품종이 2개인 상자에서 "품종 1의 비율 *p*"를 0→1로 바꿔 가며 두 값을 그려 보면

![지니 vs 엔트로피](images/05/extra_gini_vs_entropy.png)

- 둘 다 **순수할 때(p=0, 1) 0**, **반반일 때(p=0.5) 최대**
- 엔트로피를 반으로 줄이면(점선) 지니와 거의 겹친다 → **모양이 거의 같은 자**
- 그래서 대부분 **같은 트리**가 나온다

**직접 확인** (노트북 외 추가 코드)

```python
tree_ent = DecisionTreeClassifier(max_depth=2, criterion="entropy", random_state=42)
tree_ent.fit(X_iris, y_iris)
(tree_ent.predict(X_iris) == tree_clf.predict(X_iris)).all()   # True → 똑같은 트리
```

### 6-3. 그럼 뭘 쓰나?

| | 지니 (기본값) | 엔트로피 |
|---|---|---|
| 속도 | **조금 빠름** (로그 계산이 없음) | 조금 느림 |
| 경향 | 가장 많은 클래스를 **한쪽 가지로 떼어내는** 경향 | 조금 더 **균형 잡힌** 트리 |
| 결론 | 대부분 차이 없음 → **기본값(지니) 그대로 써도 OK** | |

> 💡 사이킷런에는 `criterion="log_loss"`도 있는데 `"entropy"`와 같은 것이다.

---

## 7. 규제 하이퍼파라미터 (5.7)

> 🎯 **이 절의 목표**: 결정 트리가 왜 쉽게 **과대적합**되는지, 그리고 그걸 막는 손잡이들.

### 7-1. 왜 결정 트리는 잘 외우나?

> 🤔 **쉽게 말하면**: 시험 문제를 **이해하지 않고 통째로 외운 학생**. 연습 문제는 100점인데 새 문제는 틀린다.

제한 없는 결정 트리는 **모든 리프가 완벽하게 순수해질 때까지** 계속 질문을 만든다. 훈련 데이터의 이상한 점 하나를 위해서도 전용 칸을 만들어 버린다.

| 모델 종류 | 뜻 | 예 | 과대적합 |
|---|---|---|---|
| **파라미터 모델** | 학습할 숫자 개수가 **미리 정해짐** (예: 선형 회귀는 특성 수 + 1개) | 선형 회귀 | 구조가 고정돼서 덜 외움 |
| **비파라미터 모델** | 데이터를 보면서 **구조가 자유롭게 자람** | 결정 트리 | 제한 안 하면 다 외움 |

→ 그래서 트리가 **자라는 것을 제한**해야 한다 = **규제**.

### 7-2. 규제 손잡이들

| 하이퍼파라미터 | 뜻 | 비유 | 규제를 세게 하려면 |
|---|---|---|---|
| `max_depth` | 최대 깊이 (기본 무제한) | 질문은 최대 N번까지만 | ↓ 줄이기 |
| `max_leaf_nodes` | 리프 최대 개수 | 최종 반(班)은 최대 N개까지만 | ↓ 줄이기 |
| `max_features` | 각 분할에서 검토할 특성 수 | 칼질 후보를 일부 특성에서만 | ↓ 줄이기 |
| `min_samples_split` | 이 이상 샘플이 있어야 **나눌 수 있음** | N명 이상인 반만 쪼갤 수 있다 | ↑ 늘리기 |
| `min_samples_leaf` | 리프에 최소 이만큼 샘플이 있어야 함 | 반 하나에 최소 N명 | ↑ 늘리기 |
| `min_weight_fraction_leaf` | `min_samples_leaf`를 **전체 대비 비율**로 | 반 하나에 전체의 최소 N% | ↑ 늘리기 |

> 📌 외우는 법: **`max_`는 줄이고, `min_`은 늘리면** 규제가 세진다.

> 💡 **가지치기(pruning)**: 일단 끝까지 키운 다음, 별 도움 안 되는 가지를 잘라내는 방법도 있다. 교재는 χ² 검정("이 분할이 그냥 우연일 확률은?")으로 판단하는 방식을 소개. 사이킷런에서는 `ccp_alpha` 값을 키우면 가지치기를 한다.

### 7-3. 실험: moons 데이터 (그림 5-3)

```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

tree_clf1 = DecisionTreeClassifier(random_state=42)                      # 제한 없음
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)  # 리프당 최소 5개
tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
```

| 코드 | 뜻 |
|---|---|
| `make_moons(n_samples=150, noise=0.2)` | 반달 2개 모양 점 150개. `noise`는 점들이 얼마나 흩어지는지 (클수록 지저분) |
| `min_samples_leaf=5` | 샘플 1~4개짜리 작은 "전용 칸"을 못 만들게 막음 |

![그림 5-3 규제 비교](images/05/fig5-3_moons_regularization.png)

**그림 읽기**
- **왼쪽(제한 없음)**: 경계가 들쭉날쭉. 반대편 무리 속에 섞인 점 하나를 위해서도 **작은 섬**을 만들었다 → 훈련 데이터를 외움
- **오른쪽(min_samples_leaf=5)**: 경계가 단순하고 매끈 → 몇 개 틀리더라도 **큰 흐름**을 잡음

**그림을 그린 코드** — 2-2절의 결정 경계 그리기를 함수로 묶은 것

```python
def plot_decision_boundary(clf, X, y, axes, cmap):
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)    # 영역 색칠
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)  # 경계선만 회색으로
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

| 코드 | 뜻 |
|---|---|
| `axes=[x최소, x최대, y최소, y최대]` | 그림 범위 |
| `plt.contourf` vs `plt.contour` | f = fill(채우기) / f 없으면 **선만** |
| `linestyle="none"` | 점만 찍고 선으로 잇지 않음 |
| `plt.subplots(ncols=2)` | 그림 칸 2개를 옆으로 나란히 |
| `plt.sca(axes[0])` | "지금부터 첫 번째 칸에 그려라" (set current axes) |
| `sharey=True` | 두 칸이 y축 눈금 공유 |

### 7-4. 새 데이터로 성적표 확인

그림만 보고 판단하지 말고 **처음 보는 데이터**로 점수를 매겨 보자.

```python
X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2,
                                        random_state=43)   # 시드가 다름 = 새 데이터
tree_clf1.score(X_moons_test, y_moons_test)   # 0.898
tree_clf2.score(X_moons_test, y_moons_test)   # 0.92
```

| | 제한 없음 | `min_samples_leaf=5` |
|---|---|---|
| 깊이 / 리프 수 (직접 확인) | 8 / 19 | 6 / 13 |
| 훈련 데이터 점수 | **1.000** (연습 문제 만점) | 0.947 |
| 새 데이터 점수 | 0.898 | **0.920** (실전 승리) |

> 📌 **훈련 점수가 100%면 오히려 의심하자.** 규제로 훈련 점수를 조금 양보하면 새 데이터 점수(진짜 실력)가 오른다.

---

## 8. 회귀 (5.8)

> 🎯 **이 절의 목표**: 결정 트리로 **숫자**를 예측하기. 분류 트리와 거의 같고 "리프에 무엇을 적느냐"만 다르다.

### 8-1. 데이터와 모델

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

rng = np.random.default_rng(seed=42)
X_quad = rng.random((200, 1)) - 0.5    # -0.5 ~ 0.5 사이 숫자 200개 (특성 1개)
y_quad = X_quad ** 2 + 0.025 * rng.standard_normal((200, 1))   # y = x² + 작은 잡음

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_quad, y_quad)
```

| 코드 | 뜻 |
|---|---|
| `rng.random((200, 1))` | 0~1 사이 숫자 200개를 (200행, 1열)로. `- 0.5`로 −0.5~0.5로 이동 |
| `X_quad ** 2` | x² → **U자 모양** 곡선 |
| `0.025 * rng.standard_normal(...)` | 평균 0, 표준편차 0.025인 잡음 → 점들이 곡선 주변에 살짝 흩어짐 |
| `DecisionTreeRegressor` | 회귀용 결정 트리. 사용법은 분류기와 완전히 같음 |

### 8-2. 회귀 트리 읽기 (그림 5-4)

```python
export_graphviz(tree_reg, out_file="my_regression_tree.dot",
                feature_names=["x1"], rounded=True, filled=True)
Source.from_file("my_regression_tree.dot")
```

![그림 5-4 회귀 트리](images/05/fig5-4_regression_tree.png)

분류 트리와 다른 점은 두 가지뿐

| | 분류 트리 | 회귀 트리 |
|---|---|---|
| `value` | 품종별 **개수** | 그 상자에 모인 샘플들의 y **평균** |
| 깔끔함의 기준 | `gini` | `squared_error` (MSE, 평균제곱오차) |

**예측 따라가 보기**: *x*₁ = 0.2
1. 루트 "x1 ≤ 0.343?" → 예 → 왼쪽
2. "x1 ≤ −0.302?" → 아니오 → 오른쪽
3. 도착한 리프: samples = 133, **value = 0.038** → 예측값 **0.038**

```python
tree_reg.predict([[0.2]])   # array([0.0376])  ✔
```

> 💡 즉 "x가 −0.302 ~ 0.343 사이인 훈련 샘플 133개의 y 평균"을 답으로 내는 것. **실제 정답 0.2² = 0.04와 아주 가깝다.**

> 💡 이 데이터는 무작위로 생성돼서 교재 그림의 숫자와 다를 수 있다. 읽는 법만 같으면 된다.

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

- **`-2`는 "여기는 리프라서 기준값이 없음"** 이라는 표시 (사이킷런이 빈칸 대신 넣는 값)
- 깊이 3 트리도 위쪽 기준값(0.343, −0.302, 0.431)은 똑같고 **아래에 질문이 더 붙은 것**

![그림 5-5 회귀 트리 깊이 비교](images/05/fig5-5_regression_depth.png)

**그림 읽기**
- 파란 점 = 훈련 데이터, 빨간 선 = 트리의 예측
- 예측선이 **계단 모양** → 구간마다 그 구간 평균값 하나로 일정
- 깊이 2 → 계단 4개 / 깊이 3 → 계단 8개. 깊을수록 곡선을 촘촘히 따라감

> 🤔 **비유**: 택배 요금표. "1kg 이하 3,000원, 1~3kg 4,000원, 3~5kg 5,000원…" 무게 구간마다 요금이 하나씩 정해진 계단식 표. 회귀 트리는 데이터를 보고 **구간 경계와 구간별 값을 자동으로 정하는** 요금표다.

**그림 코드**

```python
def plot_regression_predictions(tree_reg, X, y, axes=[-0.5, 0.5, -0.05, 0.25]):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)   # x축 위 500개 점
    y_pred = tree_reg.predict(x1)                            # 그 점들의 예측
    plt.axis(axes)
    plt.xlabel("$x_1$")
    plt.plot(X, y, "b.")                                     # 훈련 데이터: 파란 점
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")   # 예측: 빨간 선

th0, th1a, th1b = tree_reg.tree_.threshold[[0, 1, 4]]   # 깊이 0 하나, 깊이 1 둘
th2s = tree_reg2.tree_.threshold[[2, 5, 9, 12]]         # 깊이 2 넷
```

> 💡 `.reshape(-1, 1)` = 숫자 500개 한 줄을 **(500행, 1열)** 로 세우기. `-1`은 "개수는 알아서 계산해"라는 뜻. 사이킷런은 특성이 1개여도 2차원 모양을 원한다.

### 8-4. 회귀에서 CART는 무엇을 최소화하나

분류에선 지니를 줄였다면, 회귀에선 **MSE**를 줄인다.

$$J(k, t) = \frac{m_\text{왼쪽}}{m} \text{MSE}_\text{왼쪽} + \frac{m_\text{오른쪽}}{m} \text{MSE}_\text{오른쪽}$$

- 상자의 MSE = 상자 안 각 샘플이 **상자 평균에서 얼마나 떨어져 있는지**를 제곱해 평균낸 것
- 즉 "나눈 뒤 각 무더기의 점들이 **자기 무더기 평균 근처에 오밀조밀 모이도록**" 자른다

### 8-5. 회귀 트리도 외운다 (그림 5-6)

```python
tree_reg1 = DecisionTreeRegressor(random_state=42)                       # 제한 없음
tree_reg2 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)  # 리프당 최소 10개
tree_reg1.fit(X_quad, y_quad)
tree_reg2.fit(X_quad, y_quad)
```

![그림 5-6 회귀 규제 비교](images/05/fig5-6_regression_regularization.png)

| | 제한 없음 (왼쪽) | `min_samples_leaf=10` (오른쪽) |
|---|---|---|
| 리프 수 (직접 확인) | **200** = 샘플 수와 같음 | 15 |
| 예측선 | 점 하나하나를 다 지나가며 **지그재그** | U자 곡선을 **매끈하게** 따라감 |

> ⚠️ 리프 200개 = 샘플마다 전용 칸 → 잡음까지 그대로 외운 상태. 회귀에서도 규제는 필수.

---

## 9. 축 방향에 대한 민감성 (5.9)

> 🎯 **이 절의 목표**: 결정 트리의 첫 번째 약점 — **비스듬한 경계를 잘 못 긋는다**.

> 🤔 **쉽게 말하면**: **가로·세로로만 자를 수 있는 가위**를 들고 대각선을 오려야 한다면? 계단처럼 조금씩 꺾어서 자를 수밖에 없다.

결정 트리의 질문은 항상 "특성 하나 ≤ 숫자"라서, 경계선이 항상 **축에 수직**(가로선 또는 세로선)이다.

### 9-1. 같은 데이터를 45° 돌려 보기 (그림 5-7)

```python
rng = np.random.default_rng(seed=42)
X_square = rng.random((100, 2)) - 0.5              # 정사각형 안에 점 100개
y_square = (X_square[:, 0] > 0).astype(np.int64)   # x₁ > 0 이면 클래스 1

angle = np.pi / 4  # 45도 (라디안)
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
X_rotated_square = X_square.dot(rotation_matrix)   # 모든 점을 45° 회전

tree_clf_square = DecisionTreeClassifier(random_state=42)
tree_clf_square.fit(X_square, y_square)
tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
tree_clf_rotated_square.fit(X_rotated_square, y_square)
```

| 코드 | 뜻 |
|---|---|
| `(X_square[:, 0] > 0)` | 첫 번째 좌표가 양수인가? → 정답은 **세로선 x₁ = 0 하나로 완벽히** 나뉨 |
| `.astype(np.int64)` | True/False → 1/0 |
| `rotation_matrix` | 점을 원점 중심으로 돌리는 **회전 행렬** (고등학교 수학의 회전 변환) |
| `X.dot(R)` | 행렬 곱으로 100개 점을 한 번에 회전 |

![그림 5-7 회전에 대한 민감성](images/05/fig5-7_rotation.png)

| | 원래 데이터 (왼쪽) | 45° 회전 (오른쪽) |
|---|---|---|
| 정답 경계 | 세로선 | 대각선 |
| 트리가 만든 경계 | **세로선 1개** | **계단** |
| 깊이 / 리프 수 (직접 확인) | 1 / 2 | 4 / 6 |

- 데이터의 본질은 똑같다 (돌렸을 뿐). 그런데 트리는 훨씬 복잡해졌다
- 계단 경계는 훈련 데이터엔 맞지만, **새 데이터가 계단 틈에 떨어지면 틀리기 쉽다** → 일반화가 나빠질 가능성

### 9-2. 해결책: PCA로 데이터를 "똑바로 세우기" (그림 5-8)

> 🤔 **쉽게 말하면**: 비스듬하게 놓인 종이를 가위질하기 전에 **종이를 돌려서 똑바로 놓는 것**.

**PCA**(주성분 분석, 7장에서 자세히)는 데이터가 **가장 길게 퍼진 방향을 새 x축**으로 삼도록 회전시킨다.

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pca_pipeline = make_pipeline(StandardScaler(), PCA())
X_iris_rotated = pca_pipeline.fit_transform(X_iris)
tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_pca.fit(X_iris_rotated, y_iris)
```

| 코드 | 뜻 |
|---|---|
| `StandardScaler()` | PCA 전에 단위 맞추기 (PCA는 "얼마나 퍼졌나"를 보는데, 단위가 다르면 왜곡됨) |
| `PCA()` | 가장 많이 퍼진 방향 = *z*₁, 그와 직각 방향 = *z*₂ 로 좌표 회전 |
| `make_pipeline(A, B)` | A → B 를 차례로 실행하는 묶음 (2장) |

붓꽃의 꽃잎 길이·너비는 **함께 커지는**(강한 상관) 관계라 점들이 대각선으로 늘어서 있다. PCA로 돌리면 그 대각선이 *z*₁ 축이 된다.

![그림 5-8 PCA 후 붓꽃 트리](images/05/fig5-8_pca_iris.png)

```python
th1, th2 = tree_clf_pca.tree_.threshold[[0, 2]]   # -0.911, 0.758
```

**결과 (직접 확인)**
- 두 질문 모두 ***z*₁ 하나만** 사용 (`tree_.feature` = `[0, -2, 0, -2, -2]`) → 경계가 **세로선 두 개**로 단순
- *z*₁ 하나가 데이터 분산의 **98.1%** 를 설명 → 사실상 1차원 문제였던 것
- 훈련 정확도: 원래 좌표 0.96 → PCA 좌표 **0.98**

> ⚠️ 대신 *z*₁ = "길이와 너비를 섞은 새 축"이라 **"z₁ ≤ −0.911이면 setosa"는 사람이 해석하기 어렵다.** 화이트박스 장점을 일부 내주는 셈.

---

## 10. 결정 트리의 분산 문제 (5.10)

> 🎯 **이 절의 목표**: 결정 트리의 두 번째 약점 — **조금만 바뀌어도 완전히 다른 트리**가 된다.

**"분산이 크다"** 는 말은, 훈련 데이터나 설정이 살짝만 달라져도 결과 모델이 **크게 흔들린다**는 뜻이다. 9절의 회전도 그 예. 이번엔 데이터를 **전혀 안 바꾸고** `random_state`만 바꿔 본다.

### 10-1. 시드 하나 바꿨을 뿐인데 (그림 5-9)

```python
tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)   # 42 → 40
tree_clf_tweaked.fit(X_iris, y_iris)
```

![그림 5-9 시드만 바꾼 트리](images/05/fig5-9_tweaked_tree.png)

그림 5-2와 비교하면 루트 경계가 **세로선(길이 2.45) → 가로선(너비 0.8)** 으로 바뀌었다.

```
random_state=42                         random_state=40
|--- 길이 <= 2.45  → setosa              |--- 너비 <= 0.80  → setosa      ← 첫 질문이 다름!
|--- 길이 >  2.45                        |--- 너비 >  0.80
|   |--- 너비 <= 1.75 → versicolor       |   |--- 너비 <= 1.75 → versicolor
|   |--- 너비 >  1.75 → virginica        |   |--- 너비 >  1.75 → virginica
```

**왜 이런 일이?** 4-3절의 **동점**을 떠올리자.
- "길이 ≤ 2.45"와 "너비 ≤ 0.8"은 비용이 똑같이 0.333
- 사이킷런은 칼질 후보를 찾을 때 **특성을 무작위 순서로** 살펴보고, 비용이 같으면 **먼저 찾은 쪽**을 택한다
- 그 무작위 순서를 정하는 게 `random_state` → 시드에 따라 둘 중 하나가 뽑힘

**직접 확인 — 시드 0~99로 100번 훈련**

| 루트 질문 | 뽑힌 횟수 |
|---|---|
| 너비 ≤ 0.80 | 51번 |
| 길이 ≤ 2.45 | 49번 |

거의 **동전 던지기**다.

> ⚠️ 두 트리 모두 훈련 정확도는 0.96으로 같다. 하지만 **훈련 데이터가 없던 영역**에서는 답이 다르다. 예를 들어 길이 2, 너비 1.5인 꽃은
> - `random_state=42` 트리: 길이 2 ≤ 2.45 → **setosa**
> - `random_state=40` 트리: 너비 1.5 > 0.8, ≤ 1.75 → **versicolor**
>
> **"점수는 같은데 모델은 다르다"** = 분산이 큰 모델의 전형적인 모습.

### 10-2. 해결책: 여러 트리에게 물어보고 다수결

> 🤔 **쉽게 말하면**: 한 사람의 의견은 기분에 따라 흔들리지만, **100명에게 물어 다수결**하면 안정적이다 (집단지성).

- 트리를 **여러 개** 조금씩 다르게 훈련시키고 **예측을 평균(다수결)** 내면 흔들림이 상쇄된다
- 이것이 **랜덤 포레스트** → 6장
- 연습문제 8번에서 직접 만들어 본다 (12절)

---

## 11. 부록: `tree_`로 트리 구조 직접 다루기

> 🎯 그림 말고 **코드로** 트리 내부를 들여다보는 방법. 2-3절, 8-3절에서 쓴 `tree_.threshold`의 정체.

훈련된 트리의 모든 정보는 `tree_` 속성에 **NumPy 배열**로 들어 있다. 배열의 **i번째 칸 = i번 노드의 정보**. 노드 번호는 루트 0부터 **왼쪽 가지를 끝까지 먼저** 내려가는 순서(깊이 우선)로 매겨진다.

```
노드 번호 (그림 5-1 기준)
        0 (루트)
       / \
      1   2
         / \
        3   4
```

### 11-1. 트리 전체 정보

```python
tree = tree_clf.tree_

tree.node_count     # 5   노드 개수
tree.max_depth      # 2   깊이
tree.max_n_classes  # 3   클래스 개수
tree.n_features     # 2   특성 개수
tree.n_outputs      # 1   예측하는 타깃 개수 (다중 출력이면 2 이상)
tree.n_leaves       # 3   리프 개수
tree.impurity       # array([0.6667, 0.    , 0.5   , 0.168 , 0.0425])  노드별 지니
```

### 11-2. 누가 누구의 자식인가

```python
tree.children_left[0], tree.children_right[0]   # (1, 2)    → 0번의 자식은 1번(왼쪽), 2번(오른쪽)
tree.children_left[3], tree.children_right[3]   # (-1, -1)  → 3번은 자식 없음 = 리프

is_leaf = (tree.children_left == tree.children_right)
np.arange(tree.node_count)[is_leaf]             # array([1, 3, 4])  리프 번호들
```

- 리프는 왼쪽·오른쪽 자식이 둘 다 `-1`로 **같다** → 이 성질로 리프를 골라냄
- `np.arange(5)` = `[0, 1, 2, 3, 4]` → 그중 리프인 것만 남김

### 11-3. 각 노드의 질문

```python
tree.feature     # array([ 0, -2,  1, -2, -2])           어떤 특성으로 물었나 (0=길이, 1=너비)
tree.threshold   # array([ 2.45, -2.  ,  1.75, -2.  , -2.  ])  기준값
```

→ 0번: "특성 0(길이) ≤ 2.45?", 2번: "특성 1(너비) ≤ 1.75?". 리프 자리의 `-2`는 **의미 없는 빈칸 표시**.

### 11-4. 각 노드에 모인 샘플

```python
tree.n_node_samples   # array([150,  50, 100,  54,  46])   그림의 samples

tree.value
# array([[[0.3333, 0.3333, 0.3333]],
#        [[1.    , 0.    , 0.    ]],
#        [[0.    , 0.5   , 0.5   ]],
#        [[0.    , 0.9074, 0.0926]],
#        [[0.    , 0.0217, 0.9783]]])     모양 (5, 1, 3) = (노드, 타깃, 클래스)

np.all(tree.value.sum(axis=(1, 2)) == tree.n_node_samples)
# 노트북이 기대한 결과: True  /  scikit-learn 1.9.1 실제 결과: False
```

> ⚠️ **버전 차이로 노트북과 결과가 다르다.** 노트북은 `value`에 **개수**([50, 50, 50])가 들어 있다고 가정했지만, 최근 scikit-learn은 **비율**([0.333, 0.333, 0.333])을 저장한다. 비율의 합은 1이니 150과 같을 리 없어 `False`. 개수가 필요하면 샘플 수를 곱해 되돌리면 된다.
>
> ```python
> counts = tree.value.squeeze(1) * tree.weighted_n_node_samples[:, None]
> # [[50, 50, 50], [50, 0, 0], [0, 50, 50], [0, 49, 5], [0, 1, 45]]  ← 그림과 같은 개수
> np.all(counts.sum(axis=1).round() == tree.n_node_samples)   # True
> ```
>
> - `.squeeze(1)` = 크기 1짜리 가운데 차원 없애기 → (5, 3)
> - `[:, None]` = (5,)를 (5, 1)로 세워서 행마다 곱할 수 있게

### 11-5. 노드별 깊이 계산하기

```python
def compute_depth(tree_clf):
    tree = tree_clf.tree_
    depth = np.zeros(tree.node_count)       # 노드별 깊이를 담을 칸 (일단 0)
    stack = [(0, 0)]                        # 할 일 목록: (노드 번호, 깊이). 루트부터 시작
    while stack:                            # 할 일이 남아 있는 동안
        node, node_depth = stack.pop()      # 하나 꺼내서
        depth[node] = node_depth            # 깊이 기록
        if tree.children_left[node] != tree.children_right[node]:   # 리프가 아니면
            stack.append((tree.children_left[node], node_depth + 1))    # 자식들을
            stack.append((tree.children_right[node], node_depth + 1))   # 깊이+1로 목록에 추가
    return depth

depth = compute_depth(tree_clf)
depth   # array([0., 1., 1., 2., 2.])
```

**따라가 보기**

| 단계 | 꺼낸 것 | 기록 | 목록에 추가 | 남은 목록 |
|---|---|---|---|---|
| 1 | (0, 0) | depth[0]=0 | (1,1), (2,1) | [(1,1), (2,1)] |
| 2 | (2, 1) | depth[2]=1 | (3,2), (4,2) | [(1,1), (3,2), (4,2)] |
| 3 | (4, 2) | depth[4]=2 | 리프라 없음 | [(1,1), (3,2)] |
| 4 | (3, 2) | depth[3]=2 | 리프라 없음 | [(1,1)] |
| 5 | (1, 1) | depth[1]=1 | 리프라 없음 | [] → 끝 |

```python
tree_clf.tree_.feature[(depth == 1) & (~is_leaf)]     # array([1])     깊이 1의 질문 노드가 쓰는 특성
tree_clf.tree_.threshold[(depth == 1) & (~is_leaf)]   # array([1.75])  그 기준값
```

- `(depth == 1) & (~is_leaf)` = "깊이가 1이면서 리프가 아닌 노드" → 2번 노드
- 그림 5-2의 "Depth=1" 가로선(너비 1.75)을 코드로 꺼낸 것

---

## 12. 연습문제 풀이

### 1~6번 (개념 문제)

**1. 샘플 100만 개로 제한 없이 훈련한 트리의 깊이는?**
제한이 없으면 대략 샘플마다 리프가 하나 → 리프 100만 개. 질문 하나마다 절반씩 나누므로 깊이 ≈ log₂(1,000,000) ≈ **20** (완벽한 균형은 아니라 실제론 조금 더 깊음).

**2. 자식 노드의 지니는 항상 부모보다 낮을까?**
**대체로 낮지만 항상은 아니다.** 예를 들어 한 줄로 A, B, A, A, A가 놓인 노드(A 4개, B 1개)를 생각해 보자.

| 노드 | 구성 | 지니 |
|---|---|---|
| 부모 | A 4, B 1 | 1 − (4/5)² − (1/5)² = **0.32** |
| 왼쪽 자식 | A, B | 1 − 0.5² − 0.5² = **0.5** ← 부모보다 높다! |
| 오른쪽 자식 | A, A, A | **0** |
| 가중 평균 | | 2/5 × 0.5 + 3/5 × 0 = **0.2** < 0.32 |

CART가 줄이는 건 **두 자식의 가중 평균**이다. 한쪽이 나빠져도 다른 쪽이 그보다 더 좋아지면 그 분할을 택한다.

**3. 과대적합이면 `max_depth`를?** → **줄인다.** 질문 수를 제한 = 규제.

**4. 과소적합인데 특성 스케일링을 하면?** → **소용없다.** 트리는 스케일과 무관하니까 시간 낭비.

**5. 100만 개 훈련에 1시간이면 1,000만 개는?** → 5-2절 계산대로 **약 11.7시간.**

**6. 특성 수가 2배가 되면?** → 훈련 시간이 *n*에 비례하므로 **약 2배.**

### 7번 — moons 데이터로 최적의 트리 찾기

> 목표: 그리드 서치로 좋은 하이퍼파라미터를 찾아 테스트 정확도 85~87% 달성

**a. 데이터 만들기**

```python
from sklearn.datasets import make_moons

X_moons, y_moons = make_moons(n_samples=10000, noise=0.4, random_state=42)
```

- 5.7절(150개, noise 0.2)보다 **훨씬 많고 훨씬 지저분**하다 → 완벽 분리가 불가능, 규제가 중요해짐

**b. 훈련/테스트 나누기**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_moons, y_moons,
                                                    test_size=0.2,
                                                    random_state=42)
```

- 20%를 테스트로 → 훈련 8,000 / 테스트 2,000

**c. 그리드 서치**

```python
from sklearn.model_selection import GridSearchCV

params = {
    'max_leaf_nodes': list(range(2, 100)),   # 리프 2~99개
    'max_depth': [1, 2, 3, 4, 5, 6],
    'min_samples_split': [2, 3, 4]
}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                              params, cv=3)
grid_search_cv.fit(X_train, y_train)

grid_search_cv.best_estimator_
# DecisionTreeClassifier(max_depth=6, max_leaf_nodes=17, random_state=42)
```

- 조합 수 = 98 × 6 × 3 = **1,764개**, 각각 3폴드 교차 검증 → 트리를 5,292번 훈련 (트리는 빨라서 수십 초면 끝)
- 최적: **깊이 6, 리프 17개**

**d. 테스트 세트로 평가**

```python
from sklearn.metrics import accuracy_score

y_pred = grid_search_cv.predict(X_test)
accuracy_score(y_test, y_pred)   # 0.8595
```

- `GridSearchCV`는 최적 조합을 찾으면 **전체 훈련 세트로 다시 학습**해 둔다(`refit=True` 기본) → 바로 `predict` 가능
- 테스트 정확도 **85.95%** → 목표 달성

### 8번 — 숲을 직접 키워 보기 (랜덤 포레스트 맛보기)

> 아이디어: **작은 데이터로 훈련한 약한 트리 1,000개**를 만들어 다수결을 내면?

**a. 100개짜리 미니 훈련 세트 1,000개 만들기**

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

- `ShuffleSplit` = "매번 새로 섞어서 훈련/테스트로 나누기"를 `n_splits`번 반복
- 트릭: 테스트 크기를 `8000 − 100 = 7900`으로 주면 **훈련 쪽이 정확히 100개**
- 여기선 훈련 쪽 인덱스만 쓰고 `mini_test_index`는 버린다

**b. 트리 1,000개 각각 훈련하고 평가**

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

- `clone` = 7번에서 찾은 설정(깊이 6, 리프 17)을 복사한 **아직 학습 안 된** 트리
- `zip(forest, mini_sets)` = 트리 1개와 미니 세트 1개를 짝지어 반복
- 100개로만 배워서 하나하나는 약하다 → 평균 **80.6%**

**c. 다수결 예측 만들기**

```python
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)   # (1000, 2000) 빈 표
for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)               # 트리마다 한 줄씩 채우기

from scipy.stats import mode
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)       # 세로로 최빈값
```

표로 그려 보면

```
            테스트 샘플 →
            샘플1  샘플2  샘플3  …  샘플2000
트리1         1      0      1
트리2         1      0      0
 ⋮            ⋮      ⋮      ⋮
트리1000      1      1      1
─────────────────────────────
최빈값        1      0      1    ← mode(axis=0): 세로(트리 방향)로 다수결
```

| 코드 | 뜻 |
|---|---|
| `np.empty(..., dtype=np.uint8)` | 값은 안 채우고 공간만 확보. 0/1만 담으니 가장 작은 정수형 |
| `mode(..., axis=0)` | 열마다(=샘플마다) 가장 많이 나온 값 + 그 득표수 |

**d. 평가**

```python
accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))   # 0.873
```

| 모델 | 테스트 정확도 |
|---|---|
| 작은 트리 1개 (평균) | 0.806 |
| 7번의 튜닝된 트리 1개 (8,000개로 학습) | 0.8595 |
| **작은 트리 1,000개 다수결** | **0.873** 🎉 |

> 📌 **하나하나는 약해도 다수결로 모으면, 데이터를 다 본 최고의 트리 하나보다 낫다.** 10절의 "분산이 크다"는 약점이 평균으로 상쇄된 것. 축하합니다, 이게 바로 **랜덤 포레스트**의 기본 아이디어다 → 6장.
>
> 💡 `.reshape([-1])` = 1줄로 펴기. 예전 SciPy의 `mode`는 2차원으로 돌려줘서 필요했지만, 최근 버전은 이미 1차원 `(2000,)`이라 있어도 그만이다.

---

## 13. 전체 흐름 요약

```
붓꽃 꽃잎 길이·너비                                 ← 데이터 (스케일링 필요 없음!)
   ↓
DecisionTreeClassifier(max_depth=2).fit             ← 5.1 트리 만들기
export_graphviz → 그림 / 지니로 "섞인 정도" 읽기
   ↓
루트부터 예/아니오로 내려가 리프의 답               ← 5.2 예측
리프의 클래스 비율 = 확률                           ← 5.3
   ↓
모든 칼질 후보 중 가중 지니가 최소인 것 (CART)      ← 5.4 질문 고르는 법 (탐욕적)
예측 O(log m) 빠름 / 훈련 O(n·m log m)              ← 5.5
지니 ≈ 엔트로피, 기본값 지니면 충분                 ← 5.6
   ↓
그냥 두면 다 외움 → max_↓, min_↑ 로 규제            ← 5.7
리프에 평균값 → 계단 모양 회귀, MSE 최소화          ← 5.8
   ↓
약점 ① 가로·세로로만 자름 → 회전에 약함 → PCA      ← 5.9
약점 ② 작은 변화(시드, 데이터)에 크게 흔들림       ← 5.10 → 여러 트리 다수결 = 랜덤 포레스트 (6장)
```

### 결정 트리 장단점 한눈에

| 👍 장점 | 👎 단점 |
|---|---|
| 이유를 설명할 수 있음 (화이트박스) | 훈련 데이터를 쉽게 외움 → **규제 필수** |
| 스케일링 등 전처리가 거의 필요 없음 | 경계가 가로·세로뿐 → **비스듬한 경계에 약함** |
| 분류·회귀 모두 가능 | **분산이 큼** (조금만 바뀌어도 다른 트리) |
| 예측이 매우 빠름 | 확률이 거침 (같은 리프면 같은 확률) |

### 분류 트리 vs 회귀 트리

| | `DecisionTreeClassifier` | `DecisionTreeRegressor` |
|---|---|---|
| 리프가 내는 답 | 가장 많은 클래스 (+ 비율 = 확률) | 샘플 y값의 **평균** |
| "깔끔함" 기준 | 지니 / 엔트로피 | MSE |
| 예측 모양 | 직사각형 영역들 | 계단 |

### 주요 결과 모음

| 실험 | 결과 |
|---|---|
| 붓꽃 깊이 2 트리 | 길이 ≤ 2.45 → 너비 ≤ 1.75, 훈련 정확도 0.96 |
| 꽃 (5, 1.5)의 확률 | [0, 0.907, 0.093] → versicolor |
| moons: 제한 없음 / `min_samples_leaf=5` | 새 데이터 0.898 / **0.920** |
| 회귀 리프 수: 제한 없음 / `min_samples_leaf=10` | 200 / 15 |
| 정사각형 데이터 리프 수: 원본 / 45° 회전 | 2 / 6 |
| 붓꽃 PCA 후 트리 | *z*₁만 사용, 훈련 정확도 0.98 |
| 시드 100개 중 루트 질문 | 길이 2.45 → 49번, 너비 0.8 → 51번 |
| 연습 7: 튜닝한 트리 | 깊이 6·리프 17, 테스트 0.8595 |
| 연습 8: 작은 트리 평균 / 다수결 | 0.806 / **0.873** |

### 자주 쓴 코드 패턴

| 패턴 | 언제 쓰나 | 위치 |
|---|---|---|
| `export_graphviz(..., filled=True)` + `Source.from_file` | 트리를 그림으로 | 1-2, 8-2 |
| `export_text(...)` | 트리를 글자로 빠르게 확인 | 1-2 |
| `meshgrid` → `ravel` → `np.c_` → `predict` → `reshape` → `contourf` | 2차원 결정 경계 그리기 | 2-2, 7-3, 9 |
| `model.predict([[a, b]])` | 샘플 1개도 2차원으로 감싸기 | 3 |
| `tree_.threshold[[i, j]]` | 원하는 노드의 기준값 꺼내기 | 2-3, 8-3, 9-2 |
| `children_left == children_right` | 리프 노드 찾기 | 11-2 |
| `X.reshape(-1, 1)` | 특성 1개짜리를 2차원으로 | 8-3 |
| `make_pipeline(StandardScaler(), PCA())` | 데이터 축 맞추기 | 9-2 |
| `ShuffleSplit(test_size=전체-100)` | 작은 무작위 부분집합 여러 개 | 12 (8번) |
| `scipy.stats.mode(Y, axis=0)` | 여러 모델 다수결 | 12 (8번) |
