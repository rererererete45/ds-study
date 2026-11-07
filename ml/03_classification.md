# 3장. 분류

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `03_classification.ipynb`
>
> 범주를 예측하는 분류 문제를 다룬다. 분류기를 학습하는 것보다 **성능을 올바르게 측정하는 방법**(오차 행렬, 정밀도·재현율, 임곗값, PR·ROC 곡선)에 중점을 두고, 이후 문제 형태를 이진 → 다중 → 다중 레이블 → 다중 출력으로 확장한다.

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
| 1. 데이터 준비 | 3.1 | MNIST 원본 | 훈련 세트, 테스트 세트 |
| 2. 이진 분류 | 3.2 | 훈련 세트, 이진 레이블 | 학습된 이진 분류기 |
| 3. 성능 측정 | 3.3 | 이진 분류기 | 오차 행렬, 정밀도·재현율·F1, PR·ROC 곡선, 임곗값 |
| 4. 다중 분류 | 3.4 | 10개 클래스 레이블 | 다중 분류기 (OvR / OvO) |
| 5. 오류 분석 | 3.5 | 다중 분류기의 예측 | 오류 패턴, 개선 방향 |
| 6. 다중 레이블 분류 | 3.6 | 이진 레이블 여러 개 | 다중 레이블 분류기 |
| 7. 다중 출력 분류 | 3.7 | 다중 클래스 레이블 여러 개 | 다중 출력 분류기 (노이즈 제거) |

**분류 문제의 형태** — 이 장 전체를 관통하는 구분

| | 레이블 1개 | 레이블 여러 개 |
|---|---|---|
| **레이블당 값 2개** | 이진 분류 (3.2–3.3) | 다중 레이블 분류 (3.6) |
| **레이블당 값 3개 이상** | 다중 분류 (3.4–3.5) | 다중 출력 분류 (3.7) |

## 데이터: MNIST

고등학생과 미국 인구조사국 직원이 손으로 쓴 숫자 이미지. 샘플 70,000개, 특성 784개.

| 항목 | 형태 | 설명 | 주의 |
|---|---|---|---|
| `X` (`mnist.data`) | `(70000, 784)` | 28×28 이미지를 1차원으로 펼친 픽셀 값 | 0(흰색) ~ 255(검정) |
| `y` (`mnist.target`) | `(70000,)` | 정답 숫자 | **문자열** `'0'`~`'9'` |
| 훈련 세트 | 앞 60,000개 | | 이미 섞여 있음 |
| 테스트 세트 | 뒤 10,000개 | | |

## 목차

1. [MNIST (3.1)](#1-mnist-31)
   - [1-1. 데이터 로드와 구조 확인](#1-1-데이터-로드와-구조-확인)
   - [1-2. 이미지 시각화](#1-2-이미지-시각화)
   - [1-3. 훈련/테스트 분할](#1-3-훈련테스트-분할)
2. [이진 분류기 훈련 (3.2)](#2-이진-분류기-훈련-32)
   - [2-1. 이진 레이블 생성](#2-1-이진-레이블-생성)
   - [2-2. SGD 분류기 학습](#2-2-sgd-분류기-학습)
3. [성능 측정 (3.3)](#3-성능-측정-33)
   - [3-1. 교차 검증으로 정확도 측정](#3-1-교차-검증으로-정확도-측정)
   - [3-2. 기준 모델과 비교](#3-2-기준-모델과-비교)
   - [3-3. 오차 행렬](#3-3-오차-행렬)
   - [3-4. 정밀도·재현율·F1 점수](#3-4-정밀도재현율f1-점수)
   - [3-5. 결정 점수와 임곗값](#3-5-결정-점수와-임곗값)
   - [3-6. PR 곡선과 임곗값 선택](#3-6-pr-곡선과-임곗값-선택)
   - [3-7. ROC 곡선과 AUC](#3-7-roc-곡선과-auc)
   - [3-8. 모델 비교: 랜덤 포레스트](#3-8-모델-비교-랜덤-포레스트)
4. [다중 분류 (3.4)](#4-다중-분류-34)
   - [4-1. 다중 분류 전략](#4-1-다중-분류-전략)
   - [4-2. SVC: 자동 OvO](#4-2-svc-자동-ovo)
   - [4-3. 전략 강제 지정](#4-3-전략-강제-지정)
   - [4-4. SGD 분류기: 자동 OvR](#4-4-sgd-분류기-자동-ovr)
   - [4-5. 성능 평가와 스케일링](#4-5-성능-평가와-스케일링)
5. [오류 분석 (3.5)](#5-오류-분석-35)
   - [5-1. 오차 행렬 시각화](#5-1-오차-행렬-시각화)
   - [5-2. 혼동 클래스 샘플 관찰](#5-2-혼동-클래스-샘플-관찰)
   - [5-3. 원인 분석과 개선 방향](#5-3-원인-분석과-개선-방향)
6. [다중 레이블 분류 (3.6)](#6-다중-레이블-분류-36)
   - [6-1. KNN 다중 레이블 분류](#6-1-knn-다중-레이블-분류)
   - [6-2. 다중 레이블 평가](#6-2-다중-레이블-평가)
   - [6-3. 분류기 체인](#6-3-분류기-체인)
7. [다중 출력 분류 (3.7)](#7-다중-출력-분류-37)
   - [7-1. 노이즈 제거 데이터 구성](#7-1-노이즈-제거-데이터-구성)
   - [7-2. 학습과 예측](#7-2-학습과-예측)
8. [연습문제](#8-연습문제)
9. [정리](#9-정리)

---

## 1. MNIST (3.1)

### 1-1. 데이터 로드와 구조 확인

**목적**
분류 실습에 사용할 데이터를 불러오고, 특성과 레이블의 형태를 확인한다.

**개념·원리**
- MNIST: 손글씨 숫자 이미지 70,000개로 이루어진 표준 분류 데이터셋. 새 분류 알고리즘의 성능을 확인하는 기준으로 널리 쓰인다.
- 이미지 1개 = 28×28 = 784픽셀. **픽셀 1개가 특성 1개**이므로 샘플 1개는 784차원 특성 벡터다.

**코드**

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)   # NumPy 배열로 로드
mnist.keys()

X, y = mnist.data, mnist.target
X.shape
y.shape
y
```

| 코드 | 의미 |
|---|---|
| `fetch_openml('mnist_784')` | OpenML 저장소에서 데이터셋을 이름으로 내려받음. 이후 로컬 캐시(`~/scikit_learn_data`) 사용 |
| `as_frame=False` | DataFrame 대신 NumPy 배열로 반환 (이미지 데이터에 적합) |
| `mnist.data` / `mnist.target` | 반환 객체(`Bunch`)에서 특성 행렬과 레이블 벡터를 꺼냄 |

**결과 해석**

```
dict_keys(['data', 'target', 'frame', 'categories', 'feature_names',
           'target_names', 'DESCR', 'details', 'url'])
(70000, 784)
(70000,)
array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)
```

| 항목 | 값 | 해석 |
|---|---|---|
| `X.shape` | (70000, 784) | 샘플 70,000개 × 특성 784개 |
| `y.shape` | (70000,) | 샘플마다 레이블 1개 |
| `y.dtype` | object | 레이블이 **문자열** → 이후 비교는 `y == '5'` 형태 |

**정리**
- 이 장에서는 `data`(특성)와 `target`(레이블)만 사용한다
- 레이블이 문자열이라는 점을 이후 모든 비교 연산에서 유의한다

---

### 1-2. 이미지 시각화

**목적**
784차원 벡터를 이미지로 복원해 데이터를 눈으로 확인한다.

**코드**

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)   # (784,) → (28, 28)
    plt.imshow(image, cmap="binary")     # 0 = 흰색, 255 = 검정
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()

y[0]   # '5'
```

```python
# 그림 3-2: 처음 100개 샘플을 10×10 격자로 출력
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)        # 10행 10열 중 (idx+1)번째 칸
    plot_digit(image_data)
plt.subplots_adjust(wspace=0, hspace=0) # 칸 사이 여백 제거
plt.show()
```

**결과 해석**
- 첫 샘플(`X[0]`)은 숫자 5. 이후 `some_digit`이라는 이름으로 장 전체에서 예측 확인용으로 쓰인다.
- 같은 숫자라도 필체 차이가 크다. 이 변동성이 분류의 난이도를 결정한다.

**정리**
- 모델 입력은 1차원 벡터, 사람이 확인할 때는 `reshape(28, 28)`로 복원

---

### 1-3. 훈련/테스트 분할

**목적**
최종 평가용 테스트 세트를 분리한다.

**개념·원리**
- 테스트 세트는 최종 평가 전까지 사용하지 않는다 (2장의 데이터 스누핑 방지 원칙).
- MNIST는 배포 시점에 이미 섞여 있고 분할 기준(앞 60,000 / 뒤 10,000)이 정해져 있다.

**코드**

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

**정리**
- 이미 섞여 있으므로 `train_test_split` 없이 슬라이싱으로 분할
- 섞여 있다는 점은 교차 검증에서 중요하다. 정렬된 데이터를 그대로 나누면 특정 폴드에 일부 클래스가 빠질 수 있다

---

## 2. 이진 분류기 훈련 (3.2)

### 2-1. 이진 레이블 생성

**목적**
10개 클래스 문제를 2개 클래스 문제로 단순화한다.

**개념·원리**
- **이진 분류**(binary classification): 두 클래스 중 하나를 예측하는 문제.
- 관심 대상을 **양성**(positive), 나머지를 **음성**(negative)이라 한다. 여기서는 숫자 5 = 양성, 5가 아닌 숫자 = 음성.

**코드**

```python
y_train_5 = (y_train == '5')   # True: 5, False: 5 아님
y_test_5 = (y_test == '5')
```

**정리**
- 원소별 비교 → 불리언 배열 → 이진 레이블

---

### 2-2. SGD 분류기 학습

**목적**
이진 분류기를 학습하고 예측해 본다.

**개념·원리**

`SGDClassifier`: **확률적 경사 하강법**(Stochastic Gradient Descent)으로 학습하는 선형 분류기.

결정 함수와 예측 규칙:

```math
h(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b, \qquad
\hat{y} = \begin{cases} 1 & h(\mathbf{x}) > 0 \\ 0 & \text{otherwise} \end{cases}
```

- $`\mathbf{w}`$: 픽셀별 가중치 (784개), $`b`$: 절편
- 학습: 훈련 샘플을 하나씩 무작위로 골라 손실의 기울기 방향으로 $`\mathbf{w}, b`$를 갱신
- 기본 손실 함수 `hinge` → 선형 SVM과 같은 모델
- 샘플 단위로 갱신하므로 대규모 데이터와 온라인 학습에 적합

**코드**

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)   # 무작위성 고정 → 재현 가능
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])              # 입력은 2차원: 샘플 1개도 리스트로 감쌈
```

**결과 해석**

```
array([ True])
```

`some_digit`(5)을 양성으로 올바르게 예측했다.

**정리**
- `predict()`의 입력은 `(샘플 수, 특성 수)` 2차원 → 단일 샘플은 `[x]`
- 선형 모델 = 픽셀 위치마다 고정된 가중치. 이 성질이 5절 오류 분석의 원인으로 다시 등장한다

---

## 3. 성능 측정 (3.3)

**절의 논리 흐름**

```
정확도 측정 (3-1) → 기준 모델과 비교하니 의미 없음 (3-2)
  → 오류를 유형별로 분해: 오차 행렬 (3-3)
  → 유형별 비율로 요약: 정밀도·재현율·F1 (3-4)
  → 두 지표는 임곗값에 따라 반대로 움직임 (3-5)
  → 임곗값 전체를 곡선으로: PR 곡선 (3-6), ROC 곡선·AUC (3-7)
  → 이 지표들로 모델 비교 (3-8)
```

### 3-1. 교차 검증으로 정확도 측정

**목적**
테스트 세트를 쓰지 않고 분류기의 정확도를 추정한다.

**개념·원리**

정확도(accuracy):

```math
\text{정확도} = \frac{\text{올바르게 예측한 샘플 수}}{\text{전체 샘플 수}}
```

**계층적 k-폴드 교차 검증** (stratified k-fold)
- 훈련 세트를 k개 폴드로 나누고, 각 폴드를 한 번씩 검증용으로 사용 (2장 5-2)
- 각 폴드의 **클래스 비율을 전체 비율과 같게** 유지 → 폴드 간 분포 차이로 인한 평가 편향 방지

**코드: `cross_val_score`**

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

**코드: 교차 검증 직접 구현**

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3)  # 데이터가 섞여 있지 않다면 shuffle=True 추가

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)                 # 학습되지 않은 복제본
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)     # True = 1로 합산 → 정답 수
    print(n_correct / len(y_pred))
```

| 코드 | 의미 |
|---|---|
| `StratifiedKFold(n_splits=3)` | 클래스 비율을 유지하는 3-폴드 분할기 |
| `.split(X, y)` | 회차마다 (훈련 인덱스, 검증 인덱스) 생성. 비율 계산에 `y` 필요 |
| `clone(sgd_clf)` | 하이퍼파라미터는 같고 학습 상태는 없는 복제본. 회차 간 학습 결과가 섞이지 않게 함 |

**결과 해석**

```
array([0.95035, 0.96035, 0.9604 ])   # cross_val_score
0.95035                              # 직접 구현
0.96035
0.9604
```

- 두 방법의 결과가 같다 → `cross_val_score`의 내부 동작이 위 반복문과 동일
- 정확도 95% 이상. 성능이 좋아 보이지만 3-2에서 검증이 필요하다

---

### 3-2. 기준 모델과 비교

**목적**
정확도 95%가 실제로 의미 있는 성능인지 확인한다.

**개념·원리**
- **기준 모델**(baseline): 입력을 보지 않는 단순한 규칙의 모델. 비교 기준선으로 사용한다.
- `DummyClassifier`의 기본 전략 `"prior"`: 가장 빈도가 높은 클래스만 예측

**코드**

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))   # 양성 예측이 하나라도 있는가

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

**결과 해석**

```
False
array([0.90965, 0.90965, 0.90965])
```

| 모델 | 예측 방식 | 정확도 |
|---|---|---|
| 더미 분류기 | 항상 "5 아님" | 약 0.910 |
| SGD 분류기 | 학습된 선형 모델 | 약 0.957 |

- 모든 샘플을 음성으로 예측해도 정확도 91%. 음성(5가 아님)이 전체의 약 91%이기 때문
- SGD의 정확도는 기준 모델보다 약 4%p 높을 뿐이다

**정리**
- **불균형 데이터**(클래스 비율이 크게 차이 나는 데이터)에서 정확도는 성능 지표로 부적절하다
- 오류를 유형별로 나눠 보는 지표가 필요하다 → 오차 행렬

---

### 3-3. 오차 행렬

**목적**
분류기의 오류를 유형별로 분해한다.

**개념·원리**

**오차 행렬**(confusion matrix): 실제 클래스(행)와 예측 클래스(열)의 조합별 샘플 수를 센 표.

| | 예측: 음성 | 예측: 양성 |
|---|---|---|
| **실제: 음성** | TN (진짜 음성) | FP (거짓 양성, 1종 오류) |
| **실제: 양성** | FN (거짓 음성, 2종 오류) | TP (진짜 양성) |

- 명명 규칙: 뒷글자(P/N) = **예측값**, 앞글자(T/F) = **예측이 맞았는지**
- 대각선(TN, TP) = 정답, 비대각선(FP, FN) = 오류

**교차 검증 예측** — 오차 행렬에 필요한 샘플별 예측을 테스트 세트 없이 얻는 방법

| 함수 | 반환값 |
|---|---|
| `cross_val_score` | 폴드별 **점수** (k개) |
| `cross_val_predict` | 샘플별 **예측값** (n개). 각 예측은 그 샘플을 학습에 사용하지 않은 모델이 생성 |

**코드**

```python
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

cm = confusion_matrix(y_train_5, y_train_pred)    # (실제, 예측) 순서
cm

y_train_perfect_predictions = y_train_5            # 완벽한 분류기를 가정
confusion_matrix(y_train_5, y_train_perfect_predictions)
```

**결과 해석**

```
array([[53892,   687],
       [ 1891,  3530]])

array([[54579,     0],      # 완벽한 분류기
       [    0,  5421]])
```

| | 예측: 5 아님 | 예측: 5 |
|---|---|---|
| **실제: 5 아님** | TN = 53,892 `cm[0,0]` | FP = 687 `cm[0,1]` |
| **실제: 5** | FN = 1,891 `cm[1,0]` | TP = 3,530 `cm[1,1]` |

- 양성으로 예측한 4,217개(FP+TP) 중 687개가 오류
- 실제 양성 5,421개(FN+TP) 중 1,891개를 놓침
- 완벽한 분류기는 비대각선이 모두 0

**정리**
- 오차 행렬 분석의 초점 = **비대각선 원소의 크기와 위치**

---

### 3-4. 정밀도·재현율·F1 점수

**목적**
오차 행렬을 해석 가능한 비율 지표로 요약한다.

**개념·원리**

**정밀도**(precision) — 양성으로 예측한 것 중 실제 양성의 비율

```math
\text{정밀도} = \frac{TP}{TP + FP}
```

**재현율**(recall, 민감도, TPR) — 실제 양성 중 양성으로 예측한 비율

```math
\text{재현율} = \frac{TP}{TP + FN}
```

**F1 점수** — 정밀도와 재현율의 조화평균

```math
F_1 = \frac{2}{\frac{1}{\text{정밀도}} + \frac{1}{\text{재현율}}}
    = 2 \times \frac{\text{정밀도} \times \text{재현율}}{\text{정밀도} + \text{재현율}}
    = \frac{TP}{TP + \frac{FN + FP}{2}}
```

| 지표 | 답하는 질문 | 낮추는 오류 |
|---|---|---|
| 정밀도 | 양성 판정을 얼마나 믿을 수 있는가 | FP |
| 재현율 | 실제 양성을 얼마나 빠짐없이 찾는가 | FN |
| F1 | 두 지표가 모두 높은가 | FP, FN |

- 조화평균은 작은 값 쪽으로 치우친다. 예: 정밀도 1.0, 재현율 0.1 → 산술평균 0.55, F1 ≈ 0.18
- 따라서 F1이 높으려면 두 지표가 모두 높아야 한다

**코드**

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred)
cm[1, 1] / (cm[0, 1] + cm[1, 1])                     # TP / (FP + TP)

recall_score(y_train_5, y_train_pred)
cm[1, 1] / (cm[1, 0] + cm[1, 1])                     # TP / (FN + TP)

f1_score(y_train_5, y_train_pred)
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)    # TP / (TP + (FN+FP)/2)
```

**결과 해석**

| 지표 | 계산 | 값 | 의미 |
|---|---|---|---|
| 정밀도 | 3530 / (687 + 3530) | 0.837 | "5"라는 판정의 83.7%만 실제 5 |
| 재현율 | 3530 / (1891 + 3530) | 0.651 | 실제 5의 65.1%만 찾아냄 |
| F1 | 3530 / (3530 + 1289) | 0.733 | |

정확도 95%와 비교하면 훨씬 낮다. 이것이 이 분류기의 실제 수준이다.

**지표 선택 기준**

| 상황 | 중시할 지표 | 예 |
|---|---|---|
| FP 비용이 큼 | 정밀도 | 어린이용 영상 필터 (안전하다고 판정한 영상은 반드시 안전해야 함) |
| FN 비용이 큼 | 재현율 | 도난 감지, 질병 선별 검사 (실제 양성을 놓치면 안 됨) |
| 두 비용이 비슷함 | F1 | |

**정리**
- 정밀도 = FP에 대한 지표, 재현율 = FN에 대한 지표
- 둘 중 무엇을 중시할지는 오류 비용에 따라 문제마다 다르다

---

### 3-5. 결정 점수와 임곗값

**목적**
분류기 내부의 판정 과정을 이해하고, 정밀도와 재현율을 조절하는 수단을 찾는다.

**개념·원리**
- **결정 점수**(decision score): 분류기가 샘플마다 계산하는 실수값 $`h(\mathbf{x})`$. 클수록 양성에 가깝다.
- **결정 임곗값**(threshold): 결정 점수가 이 값을 넘으면 양성으로 판정. `SGDClassifier` 기본값은 0.

```math
\hat{y} = \begin{cases} 1 & h(\mathbf{x}) > \text{임곗값} \\ 0 & \text{otherwise} \end{cases}
```

| 임곗값 | 양성 판정 수 | FP | FN | 정밀도 | 재현율 |
|---|---|---|---|---|---|
| 올림 ↑ | 감소 | 감소 | 증가 | 대체로 ↑ | ↓ |
| 내림 ↓ | 증가 | 증가 | 감소 | 대체로 ↓ | ↑ |

→ **정밀도/재현율 트레이드오프**: 한쪽을 올리면 다른 쪽이 내려간다. 모델을 다시 학습하지 않고 임곗값만으로 균형을 조정할 수 있다.

**코드**

```python
y_scores = sgd_clf.decision_function([some_digit])   # 결정 점수만 반환
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
```

**결과 해석**

```
array([2164.22030239])
array([ True])     # 임곗값 0
array([False])     # 임곗값 3000
```

- `predict()`는 내부적으로 `decision_function() > 0`과 같다
- 임곗값 3000에서는 점수 2164인 실제 5가 음성으로 바뀐다 → FN 발생 → 재현율 감소

**정리**
- `decision_function()`으로 점수를 직접 받으면 임곗값을 자유롭게 정할 수 있다

---

### 3-6. PR 곡선과 임곗값 선택

**목적**
모든 임곗값에서의 정밀도·재현율을 계산하고, 목표에 맞는 임곗값을 고른다.

**개념·원리**
- `cross_val_predict(..., method="decision_function")`: 교차 검증으로 샘플별 **결정 점수**를 얻는다
- `precision_recall_curve`: 가능한 모든 임곗값에 대한 정밀도·재현율을 계산. `thresholds`는 **오름차순**
- **PR 곡선**: x축 재현율, y축 정밀도. 오른쪽 위에 가까울수록 좋은 분류기

**코드: 점수와 곡선 데이터**

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

> ⚠️ `precisions`, `recalls`의 길이는 `len(thresholds) + 1`이다(끝에 정밀도 1, 재현율 0인 점이 추가됨). 임곗값과 짝지어 그릴 때는 `[:-1]`로 마지막 원소를 제외한다.

**코드: 임곗값에 따른 정밀도·재현율 (그림 3-5)**

```python
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

idx = (thresholds >= threshold).argmax()   # 임곗값 3000 이상인 첫 인덱스
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc="center right")
plt.show()
```

**코드: PR 곡선 (그림 3-6)**

```python
plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
plt.show()
```

> **관용구 `(조건 배열).argmax()`**: 불리언 배열에서 `argmax()`는 최댓값(True = 1)이 처음 나타나는 위치를 반환한다 → **조건을 처음 만족하는 인덱스**.

**코드: 정밀도 90% 임곗값 찾기**

```python
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision

y_train_pred_90 = (y_scores >= threshold_for_90_precision)   # 새 임곗값으로 판정

precision_score(y_train_5, y_train_pred_90)
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision
```

**결과 해석**

그림 3-5, 3-6:
- 재현율은 임곗값이 커질수록 단조 감소
- 정밀도는 대체로 증가하지만 단조 증가는 아니다. 임곗값을 올렸을 때 TP 하나가 빠지면 일시적으로 떨어질 수 있다
- PR 곡선에서 재현율 약 80% 부근부터 정밀도가 급락한다 → 그 직전이 일반적인 절충점

정밀도 90% 임곗값:

| 항목 | 값 |
|---|---|
| 임곗값 | 3370.02 |
| 정밀도 | 약 0.900 |
| 재현율 | 0.480 |

- 임곗값만 바꿔 정밀도 90% 분류기를 얻었지만 재현율은 48%로 떨어졌다

**정리**
- 목표 정밀도(또는 재현율)를 제시할 때는 **반대 지표의 값도 함께 명시**해야 의미가 있다
- 임곗값 선택 절차: 점수 계산 → `precision_recall_curve` → 조건을 만족하는 첫 인덱스 → 해당 임곗값

---

### 3-7. ROC 곡선과 AUC

**목적**
임곗값 전체에 걸친 분류기 성능을 다른 관점(FPR 대 TPR)으로 평가하고, 숫자 하나로 요약한다.

**개념·원리**

**ROC 곡선**(Receiver Operating Characteristic): 임곗값을 바꿔 가며 FPR(x축)에 대한 TPR(y축)을 그린 곡선

```math
TPR = \frac{TP}{TP + FN} \;(= \text{재현율}), \qquad
FPR = \frac{FP}{FP + TN} \;(= 1 - \text{특이도})
```

**AUC**(Area Under the Curve): ROC 곡선 아래 면적. 1 = 완벽, 0.5 = 무작위.

| 곡선 | x축 | y축 | 이상적 위치 | 무작위 분류기 |
|---|---|---|---|---|
| PR | 재현율 | 정밀도 | 오른쪽 위 | |
| ROC | FPR | TPR | 왼쪽 위 | 대각선 |

**코드**

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# thresholds는 내림차순 → 정밀도 90% 임곗값의 위치를 <= 로 찾음
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

# 그림 3-7
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)
plt.show()

roc_auc_score(y_train_5, y_scores)
```

> ⚠️ `roc_curve`의 `thresholds`는 **내림차순**이다(`precision_recall_curve`는 오름차순). 위치를 찾는 조건의 방향이 반대가 된다.

**결과 해석**

```
0.9604938554008616
```

- TPR을 높이면 FPR도 함께 올라간다 → ROC 곡선에서도 같은 트레이드오프
- AUC 0.96은 높아 보이지만, 음성이 많아 FPR의 분모(FP + TN)가 크기 때문이다

**PR 곡선과 ROC 곡선의 선택 기준**

| 조건 | 사용할 곡선 | 이유 |
|---|---|---|
| 양성 클래스가 드묾 | PR | TN이 커서 FPR이 작게 나옴 → ROC가 낙관적 |
| FP를 FN보다 중시 | PR | 정밀도가 FP를 직접 반영 |
| 그 외 | ROC | |

**정리**
- 이번 문제(양성 약 10%)에는 PR 곡선이 적절하다. PR 곡선으로 보면 개선 여지가 크다
- AUC는 모델 비교용 단일 지표로 쓴다

---

### 3-8. 모델 비교: 랜덤 포레스트

**목적**
앞에서 정의한 지표로 다른 모델과 성능을 비교한다.

**개념·원리**
- `RandomForestClassifier`에는 `decision_function()`이 없다. 대신 `predict_proba()`로 **클래스별 추정 확률**을 반환한다.
- 양성 확률을 결정 점수 대신 사용하면 같은 방법(PR 곡선, AUC)으로 비교할 수 있다.
- 랜덤 포레스트: 결정 트리 여러 개의 예측을 모으는 앙상블 모델 (6장)

**코드: 클래스 확률**

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_probas_forest[:2]
```

```
array([[0.11, 0.89],     # [음성 확률, 양성 확률]
       [0.99, 0.01]])
```

**코드: 추정 확률의 신뢰성 확인**

```python
# 양성 확률이 0.5~0.6인 샘플 중 실제 양성의 비율
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")
```

```
94.0%
```

**코드: PR 곡선 비교와 지표 계산**

```python
y_scores_forest = y_probas_forest[:, 1]          # 양성 확률을 점수로 사용
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

# 그림 3-8
plt.figure(figsize=(6, 5))
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")
plt.show()

y_train_pred_forest = y_probas_forest[:, 1] >= 0.5   # 임곗값 0.5 적용
f1_score(y_train_5, y_train_pred_forest)
roc_auc_score(y_train_5, y_scores_forest)
precision_score(y_train_5, y_train_pred_forest)
recall_score(y_train_5, y_train_pred_forest)
```

- 이미 구한 확률에 임곗값 0.5를 적용하면 `cross_val_predict`를 다시 실행하지 않고도 같은 예측을 얻는다

**결과 해석**

추정 확률:
- 양성 확률 0.5~0.6으로 추정된 샘플 중 실제 양성은 94%. 확률이 정확하다면 약 55%여야 한다
- `predict_proba`의 출력은 **보정(calibration)되지 않은 추정치**다. 확률값 자체를 의사결정에 쓰려면 보정이 필요하다

성능 비교:

| 지표 | SGD | 랜덤 포레스트 |
|---|---|---|
| 정밀도 | 0.837 | **0.990** |
| 재현율 | 0.651 | **0.873** |
| F1 | 0.733 | **0.927** |
| ROC AUC | 0.960 | **0.998** |

- 랜덤 포레스트의 PR 곡선이 오른쪽 위 모서리에 훨씬 가깝고, 모든 지표에서 우수하다

**정리**

| 모델 유형 | 점수를 얻는 방법 |
|---|---|
| `decision_function` 제공 (SGD, SVC 등) | `cross_val_predict(..., method="decision_function")` |
| `predict_proba`만 제공 (랜덤 포레스트 등) | `cross_val_predict(..., method="predict_proba")[:, 1]` |

---

## 4. 다중 분류 (3.4)

### 4-1. 다중 분류 전략

**목적**
이진 분류기로 셋 이상의 클래스를 구분하는 방법을 정리한다.

**개념·원리**
- **다중 분류**(multiclass classification): 셋 이상의 클래스 중 하나를 예측
- 다중 분류를 기본 지원하는 모델: 로지스틱 회귀, 랜덤 포레스트, KNN 등
- 이진 전용 모델(SGD 분류기, SVM 등)은 아래 두 전략으로 확장한다

| 전략 | 구성 | 예측 방법 | 분류기 수 (N 클래스) | 적합한 경우 |
|---|---|---|---|---|
| **OvR** (One-versus-Rest) | 클래스마다 "해당 클래스 vs 나머지" | 결정 점수가 가장 높은 클래스 | N | 대부분의 알고리즘 |
| **OvO** (One-versus-One) | 클래스 쌍마다 "A vs B" | 가장 많은 대결에서 이긴 클래스 | N(N−1)/2 | 훈련 세트 크기에 따라 비용이 급증하는 알고리즘 (SVM) |

- MNIST(N = 10): OvR 10개, OvO 45개
- OvO의 각 분류기는 두 클래스의 샘플만 사용하므로 개별 학습 비용이 작다
- 사이킷런은 이진 분류기에 다중 클래스 레이블이 주어지면 알고리즘에 맞게 OvR 또는 OvO를 **자동 선택**한다

**정리**
- OvR: 분류기 수 적음, 각 분류기는 전체 데이터 사용
- OvO: 분류기 수 많음, 각 분류기는 일부 데이터만 사용

---

### 4-2. SVC: 자동 OvO

**목적**
SVC로 다중 분류를 수행하고, 결정 점수에서 예측 클래스를 얻는 방법을 확인한다.

**코드: 학습과 예측**

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])   # y_train: 10개 클래스 (y_train_5 아님)

svm_clf.predict([some_digit])
```

- SVC는 훈련 세트가 커지면 학습 시간이 급격히 늘어나므로 2,000개만 사용

**코드: 결정 점수와 예측 클래스**

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

class_id = some_digit_scores.argmax()   # 최고 점수의 인덱스
class_id
svm_clf.classes_                        # 학습된 클래스 목록 (정렬됨)
svm_clf.classes_[class_id]              # 인덱스 → 클래스 레이블
```

**코드: OvO 원점수 45개**

```python
svm_clf.decision_function_shape = "ovo"
some_digit_scores_ovo = svm_clf.decision_function([some_digit])
some_digit_scores_ovo.round(2)
```

**결과 해석**

```
array(['5'], dtype=object)

array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,  4.82]])
#        '0'    '1'    '2'    '3'    '4'    '5'    '6'    '7'    '8'    '9'
5
array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
'5'

array([[ 0.11, -0.21, -0.97,  0.51, -1.01,  0.19,  0.09, -0.31, -0.04,
        -0.45, -1.28,  0.25, -1.01, -0.13, -0.32, -0.9 , -0.36, -0.93,
         0.79, -1.  ,  0.45,  0.24, -0.24,  0.25,  1.54, -0.77,  1.11,
         1.13,  1.04,  1.2 , -1.42, -0.53, -0.45, -0.99, -0.95,  1.21,
         1.  ,  1.  ,  1.08, -0.02, -0.67, -0.14, -0.3 , -0.13,  0.25]])
```

| 항목 | 해석 |
|---|---|
| 클래스별 점수 10개 | **OvO 승수 + 분류기 신뢰도에 기반한 작은 보정값**. '5'(9.3)는 9번의 대결에서 모두 승리 |
| 차순위 '3'(8.3) | 이 샘플이 3과 유사한 형태 |
| `argmax()` → 5 | **인덱스**. 클래스가 '0'~'9' 순서라 레이블과 우연히 일치 |
| `decision_function_shape` | **출력 형태만** 결정. `"ovr"`(기본) = 45개 점수를 클래스별 10개로 집계, `"ovo"` = 쌍별 점수 45개(0vs1, 0vs2, …, 8vs9) 그대로. **학습은 항상 OvO** |

**정리**
- 예측 클래스는 `classes_[점수.argmax()]`로 얻는다 (인덱스 ≠ 레이블일 수 있음)

---

### 4-3. 전략 강제 지정

**목적**
자동 선택 대신 OvR 또는 OvO를 직접 지정한다.

**코드**

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)
```

**결과 해석**

```
array(['5'], dtype='<U1')
10
```

- 분류기를 `OneVsRestClassifier`로 감싸면 OvR이 적용되어 이진 분류기 10개(`estimators_`)가 생성된다

**정리**

| 클래스 | 강제하는 전략 |
|---|---|
| `OneVsRestClassifier(모델)` | OvR |
| `OneVsOneClassifier(모델)` | OvO |

---

### 4-4. SGD 분류기: 자동 OvR

**목적**
SGD 분류기로 다중 분류를 수행하고 예측 근거를 확인한다.

**코드**

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

sgd_clf.decision_function([some_digit]).round()
```

**결과 해석**

```
array(['3'], dtype='<U1')

array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
        -16148.,  -4604., -12051.]])
#        '0'      '1'      '2'      '3'      '4'      '5'   ...
```

- SGD는 OvR을 사용 → 클래스별 점수 10개 중 최댓값 선택
- 양수 점수는 '3'(1,824)뿐이고 정답 '5'(−1,386)는 2위 → **오답**
- 3과 5의 혼동이 관찰된다 → 5절에서 원인 분석

---

### 4-5. 성능 평가와 스케일링

**목적**
다중 분류 성능을 측정하고, 입력 스케일링의 효과를 확인한다.

**개념·원리**

표준화(2장 4-4):

```math
z = \frac{x - \mu}{\sigma}
```

- 경사 하강법 기반 모델은 특성 스케일에 민감하다. 스케일이 다르면 비용 함수의 등고선이 길쭉해져 최적화가 느려지고 불안정해진다 (4장).

**코드**

```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))   # 실수형으로 변환 후 표준화
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```

**결과 해석**

```
array([0.87365, 0.85835, 0.8689 ])   # 원본
array([0.8983, 0.891 , 0.9018])      # 표준화
```

| 입력 | 평균 정확도 |
|---|---|
| 원본 | 약 0.867 |
| 표준화 | 약 0.897 |

- 10개 클래스 무작위 추측(10%) 대비 높은 수준
- 표준화만으로 약 3%p 향상

**정리**

| 분류기 | 다중 분류 방식 | 분류기 수 | `some_digit` 예측 |
|---|---|---|---|
| `SVC` | 자동 OvO | 45 | '5' (정답) |
| `OneVsRestClassifier(SVC())` | 강제 OvR | 10 | '5' (정답) |
| `SGDClassifier` | 자동 OvR | 10 | '3' (오답) |

- 모델을 바꾸기 전에 전처리(스케일링)부터 점검한다

---

## 5. 오류 분석 (3.5)

**분석 절차**

```
오차 행렬(개수) → 행 기준 정규화(비율) → 정답 제거(오류만)
  → 혼동 클래스 쌍의 샘플 관찰 → 원인 추론 → 개선 방향
```

### 5-1. 오차 행렬 시각화

**목적**
다중 클래스 오차 행렬을 시각화해 오류가 집중된 클래스를 찾는다.

**개념·원리**
- 다중 클래스 오차 행렬은 N×N (행: 실제, 열: 예측). 원소가 많으므로 색으로 시각화한다.
- **정규화**: 원소를 합계로 나눠 비율로 변환. 기준에 따라 해석이 달라진다.

| `normalize=` | 나누는 기준 | 읽는 법 | 관점 |
|---|---|---|---|
| `"true"` | 행(실제 클래스) 합 | 실제 X 중 몇 %를 Y로 예측했나 | 재현율 |
| `"pred"` | 열(예측 클래스) 합 | Y로 예측한 것 중 몇 %가 실제 X였나 | 정밀도 |
| `"all"` | 전체 합 | 전체 샘플 대비 비율 | |

**코드**

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# (a) 개수
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()

# (b) 행 기준 정규화
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()

# (c) 오류만: 정답 샘플의 가중치를 0으로
sample_weight = (y_train_pred != y_train)
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show()
```

| 코드 | 의미 |
|---|---|
| `ConfusionMatrixDisplay.from_predictions(실제, 예측)` | 오차 행렬 계산과 시각화를 한 번에 |
| `values_format=".0%"` | 칸의 값을 정수 퍼센트로 표시 |
| `sample_weight=(y_train_pred != y_train)` | 정답 = 0, 오답 = 1 → 대각선이 0이 되고 오류 분포만 남음 |

**결과 해석**

| 그림 | 관찰 |
|---|---|
| (a) 개수 | 대각선이 진함 → 대체로 정확. '5' 칸이 상대적으로 옅으나, 성능이 낮아서인지 샘플이 적어서인지 구분 불가 |
| (b) 행 정규화 | 샘플 수 영향 제거. 실제 5의 약 82%만 정답, **약 10%가 8로** 오분류 |
| (c) 오류만 | **'8' 열이 밝음** → 여러 클래스가 8로 오분류되는 경향 |

> ⚠️ 정규화 기준에 따라 같은 오류의 크기가 다르게 보인다. 행 기준에서 큰 "5 → 3" 오류가 열 기준(`"pred"`)에서는 다르게 보일 수 있으므로 해석할 때 기준을 명시한다.

**정리**
- 개수 → 비율 → 오류만 순서로 보면 오류가 집중된 위치가 드러난다
- 발견: '8'로의 오분류, 3과 5의 상호 혼동

---

### 5-2. 혼동 클래스 샘플 관찰

**목적**
혼동이 잦은 클래스 쌍(3, 5)의 실제 이미지를 보고 오류의 성격을 파악한다.

**코드: 네 그룹으로 분리**

```python
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]   # 실제 3, 예측 3
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]   # 실제 3, 예측 5
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]   # 실제 5, 예측 3
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]   # 실제 5, 예측 5
```

| 코드 | 의미 |
|---|---|
| `X_train[불리언 마스크]` | 마스크가 True인 행만 선택 (불리언 인덱싱) |
| `(조건1) & (조건2)` | 원소별 AND. NumPy 배열에는 `and` 대신 `&`, 연산자 우선순위 때문에 각 조건을 괄호로 감쌈 |

**코드: 2×2 블록 시각화 (그림 3-11)**

```python
size = 5      # 블록당 5×5 = 25개 이미지
pad = 0.2     # 블록 간격
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)    # 블록 내 열 + 블록 오프셋
        y = idx // size + label_row * (size + pad)   # 블록 내 행 + 블록 오프셋
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))      # 좌표 영역에 직접 배치
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()
```

| 코드 | 의미 |
|---|---|
| `idx % size`, `idx // size` | 블록 내 열·행 위치 (0~4) |
| `label_col * (size + pad)` | 블록 단위 오프셋. (0,0) = 왼쪽 아래, (1,1) = 오른쪽 위 |
| `imshow(..., extent=(왼, 오, 아래, 위))` | 이미지를 지정 좌표 영역에 배치. 서브플롯 100개보다 효율적 |

**결과 해석**

배치:

| | 예측 3 | 예측 5 |
|---|---|---|
| **실제 3** (위) | `X_aa` 정답 | `X_ab` 오답 |
| **실제 5** (아래) | `X_ba` 오답 | `X_bb` 정답 |

- 오분류된 샘플 일부는 사람도 판별하기 어렵지만, 대부분은 사람에게는 명확하다
- 3과 5의 형태적 차이는 **위쪽 가로획과 아래쪽 호를 잇는 짧은 세로획의 위치**뿐이다

---

### 5-3. 원인 분석과 개선 방향

**목적**
관찰 결과로부터 오류의 원인을 추론하고 개선책을 정한다.

**개념·원리**
- `SGDClassifier`는 선형 모델로, 픽셀 위치마다 고정된 가중치를 곱해 합산한다 (2-2)
- 숫자가 조금만 이동·회전해도 같은 획이 다른 픽셀 위치에 놓여 점수가 크게 바뀐다
- 결론: 이 모델은 **이동과 회전에 민감**하다 → 3/5처럼 세부 위치만 다른 클래스를 혼동

**개선 방향**

| 방향 | 방법 | 대상 문제 |
|---|---|---|
| 전처리 | 이미지를 중앙 정렬하고 회전 보정 | 이동·회전 민감성 |
| 데이터 증강 | 이동·회전한 이미지를 훈련 세트에 추가 (연습문제 2) | 이동·회전 민감성 |
| 특성 공학 | 닫힌 고리 수 등 형태 특성 추가 (8: 2개, 6: 1개, 5: 0개) | '8'로의 오분류 |
| 모델 변경 | 비선형 모델 사용 (예: 3-8의 랜덤 포레스트) | 선형 모델의 표현력 한계 |

**정리**
- 오류 분석 = **정량(오차 행렬) → 정성(샘플 관찰) → 원인 → 개선책**
- 이번 결론: 선형 모델의 위치 민감성 → 3/5 혼동, '8'로의 오분류 → 전처리·데이터 증강으로 개선

---

## 6. 다중 레이블 분류 (3.6)

### 6-1. KNN 다중 레이블 분류

**목적**
샘플 하나에 여러 개의 이진 레이블을 동시에 예측한다.

**개념·원리**
- **다중 레이블 분류**(multilabel classification): 샘플마다 **이진 레이블 여러 개**를 출력
- 예: 사진 한 장에서 인물 A·B·C의 등장 여부 → `[1, 0, 1]`
- `KNeighborsClassifier`: 가장 가까운 k개 훈련 샘플의 레이블로 예측. 다중 레이블을 기본 지원

**코드**

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')                  # 레이블 1: 7 이상인가
y_train_odd = (y_train.astype('int8') % 2 == 1)   # 레이블 2: 홀수인가
y_multilabel = np.c_[y_train_large, y_train_odd]  # (60000, 2)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])
```

| 코드 | 의미 |
|---|---|
| `y_train >= '7'` | 문자열 비교. 한 자리 숫자이므로 사전순 = 크기순 |
| `.astype('int8') % 2 == 1` | 정수 변환 후 홀수 판별. 0~9는 `int8` 범위로 충분 |
| `np.c_[a, b]` | 1차원 배열들을 열 방향으로 결합 → 레이블 행렬 |

**결과 해석**

```
array([[False,  True]])
```

- 5는 7 미만(False), 홀수(True) → 두 레이블 모두 정답

---

### 6-2. 다중 레이블 평가

**목적**
여러 레이블에 걸친 성능을 하나의 지표로 요약한다.

**개념·원리**
- 레이블별로 이진 분류 지표를 계산한 뒤 평균한다.

| `average=` | 계산 | 사용 시점 |
|---|---|---|
| `"macro"` | 레이블별 F1의 단순 평균 | 모든 레이블이 동등하게 중요 |
| `"weighted"` | 레이블별 **지지도**(양성 샘플 수)로 가중 평균 | 레이블 빈도 차이를 반영 |

**코드**

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
f1_score(y_multilabel, y_train_knn_pred, average="weighted")
```

**결과 해석**

```
0.9764102655606048   # macro
```

- macro F1 0.976
- 두 레이블의 양성 비율이 비슷해 `"weighted"`와의 차이는 미미하다

---

### 6-3. 분류기 체인

**목적**
다중 레이블을 지원하지 않는 모델을 다중 레이블로 확장하고, 레이블 간 의존성을 활용한다.

**개념·원리**
- 단순한 방법: 레이블마다 독립 모델 → **레이블 간 의존성**을 활용하지 못함
- **`ClassifierChain`**: 레이블마다 모델을 두되, 사슬 순서대로 앞 모델들의 예측을 다음 모델의 입력 특성에 추가

```
모델 1:  X                 → ŷ₁ (7 이상인가)
모델 2:  [X, ŷ₁]           → ŷ₂ (홀수인가)
  ⋮
모델 k:  [X, ŷ₁, …, ŷₖ₋₁]  → ŷₖ
```

- `cv` 지정 시: 학습 단계에서 앞 모델의 출력으로 **실제 레이블 대신 교차 검증 예측**을 사용한다. 예측 단계에서는 실제 레이블을 알 수 없으므로 학습과 예측의 입력 조건을 일치시키는 것이다.

**코드**

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
chain_clf.predict([some_digit])
```

**결과 해석**

```
array([[0., 1.]])
```

- 출력은 실수형이지만 `[False, True]`와 같은 의미 → 정답

**정리**

| 방법 | 레이블 간 의존성 반영 | 적용 대상 |
|---|---|---|
| 다중 레이블 지원 모델 (KNN 등) | 모델에 따라 다름 | 지원 모델 |
| 레이블별 독립 모델 | ✗ | 모든 이진 분류기 |
| `ClassifierChain` | ✓ (앞 레이블 → 뒤 레이블) | 모든 이진 분류기 |

---

## 7. 다중 출력 분류 (3.7)

### 7-1. 노이즈 제거 데이터 구성

**목적**
다중 출력 분류 문제를 구성한다: 노이즈가 섞인 이미지 → 깨끗한 이미지.

**개념·원리**
- **다중 출력 분류**(multioutput-multiclass classification): 레이블이 여러 개이고, 각 레이블이 셋 이상의 값을 가질 수 있는 분류. 다중 레이블 분류의 일반화.

| | 레이블 수 | 레이블당 값 |
|---|---|---|
| 다중 레이블 | 여러 개 | 2개 |
| 다중 출력 | 여러 개 | 3개 이상 |

**코드**

```python
rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0, 100, (len(X_train), 784))   # [0, 100) 정수 노이즈
X_train_mod = X_train + noise_train                        # 입력: 노이즈 이미지
noise_test = rng.integers(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train                                      # 타깃: 원본 이미지
y_test_mod = X_test

# 그림 3-12: 입력(왼쪽) / 타깃(오른쪽)
plt.subplot(121); plot_digit(X_test_mod[0])
plt.subplot(122); plot_digit(y_test_mod[0])
plt.show()
```

**결과 해석**
- 타깃의 형태 `(60000, 784)` → **레이블 784개**(픽셀당 1개), 각 레이블의 값 **0~255**
- 레이블 여러 개 × 레이블당 값 여러 개 → 다중 출력 분류

---

### 7-2. 학습과 예측

**목적**
KNN으로 노이즈를 제거한다.

**개념·원리**
- KNN은 입력과 가장 가까운 훈련 샘플 k개(기본 k = 5)를 찾고, **출력 레이블(픽셀)마다 이웃들의 타깃 값 중 다수결**로 예측한다.
- 이웃들의 타깃(원본 이미지)에는 무작위 노이즈가 없으므로 다수결 결과에서 노이즈가 제거된다.

**코드**

```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)           # (60000, 784) → (60000, 784)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
```

**결과 해석**
- 노이즈가 제거된 숫자 이미지가 출력된다
- 픽셀 값 예측은 회귀에 가까운 성격도 있다. 분류와 회귀의 경계가 항상 명확하지는 않다

**정리**

| 항목 | 내용 |
|---|---|
| 입력 | 노이즈 이미지 `(n, 784)` |
| 출력 | 원본 이미지 `(n, 784)`, 각 값 0~255 |
| 모델 | `KNeighborsClassifier` (다중 출력 기본 지원) |
| 원리 | 픽셀별 이웃 다수결 |

---

## 8. 연습문제

| 번호 | 과제 | 핵심 개념 | 결과 |
|---|---|---|---|
| 1 | MNIST 정확도 97% 이상 | KNN, 그리드 서치 | 테스트 정확도 0.9714 |
| 2 | 데이터 증강 | 이미지 이동, 훈련 세트 확장 | 0.9763 (오류율 −17%) |
| 3 | 타이타닉 생존자 예측 | 전처리 파이프라인, 모델 비교 | SVC CV 정확도 0.825 |
| 4 | 스팸 분류기 | 텍스트 → 단어 벡터 변환기, 희소 행렬 | 정밀도 96.9%, 재현율 97.9% |

### 8-1. MNIST 정확도 97% 이상

**목적**: 테스트 세트 정확도 97% 이상의 분류기를 만든다.

```python
# 기준 성능: 기본 하이퍼파라미터 KNN
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
baseline_accuracy = knn_clf.score(X_test, y_test)    # 0.9688

# 그리드 서치 (탐색 속도를 위해 10,000개만 사용)
from sklearn.model_selection import GridSearchCV

param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5, 6]}]

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5)
grid_search.fit(X_train[:10_000], y_train[:10_000])
grid_search.best_params_     # {'n_neighbors': 4, 'weights': 'distance'}
grid_search.best_score_      # 0.9442

# 최적 하이퍼파라미터로 전체 훈련 세트 재학습
grid_search.best_estimator_.fit(X_train, y_train)
tuned_accuracy = grid_search.score(X_test, y_test)   # 0.9714
```

| 하이퍼파라미터 | 의미 |
|---|---|
| `n_neighbors` | 참고할 이웃 수 k |
| `weights="uniform"` | 이웃의 투표 가중치가 모두 같음 |
| `weights="distance"` | 가까운 이웃일수록 큰 가중치 (거리의 역수) |

**결과 해석**

| 단계 | 정확도 |
|---|---|
| 기본 KNN (테스트) | 0.9688 |
| 그리드 서치 최적 (CV, 10,000개) | 0.9442 |
| 최적 모델 전체 재학습 (테스트) | **0.9714** |

- CV 점수가 낮은 것은 훈련 샘플을 10,000개만 썼기 때문. 전체 데이터로 재학습하면 목표 달성
- 최적 조합: k = 4, 거리 가중

---

### 8-2. 데이터 증강

**목적**: 이미지를 상하좌우로 1픽셀씩 이동한 복사본을 추가해 훈련 세트를 늘리고 성능 변화를 확인한다 (5-3의 개선 방향 검증).

```python
from scipy.ndimage import shift

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")   # 빈 곳은 0으로 채움
    return shifted_image.reshape([-1])

# 원본 + 4방향 이동 → 훈련 세트 5배 (300,000개)
X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

# 섞기: 이동 방향별로 묶여 있지 않게
rng = np.random.default_rng(seed=42)
shuffle_idx = rng.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)   # 8-1의 최적 하이퍼파라미터
knn_clf.fit(X_train_augmented, y_train_augmented)
augmented_accuracy = knn_clf.score(X_test, y_test)           # 0.9763

error_rate_change = (1 - augmented_accuracy) / (1 - tuned_accuracy) - 1
print(f"error_rate_change = {error_rate_change:.0%}")         # -17%
```

| 코드 | 의미 |
|---|---|
| `shift(image, [dy, dx], cval=0, mode="constant")` | 행 방향 dy, 열 방향 dx만큼 이동. 비어 있는 영역은 `cval`(0)로 채움 |
| `reshape([-1])` | 2차원 이미지를 다시 1차원으로 |
| `**grid_search.best_params_` | 딕셔너리를 키워드 인수로 풀어 전달 |

**결과 해석**

```math
\text{오류율 변화} = \frac{1 - 0.9763}{1 - 0.9714} - 1 = \frac{0.0237}{0.0286} - 1 \approx -17\%
```

- 정확도는 0.5%p 상승이지만, 오류율 기준으로는 **17% 감소**
- 정확도가 이미 높을 때는 오류율의 상대 변화로 개선 폭을 보는 것이 적절하다

---

### 8-3. 타이타닉 생존자 예측

**목적**: 승객 특성으로 생존 여부(`Survived`)를 예측한다. 2장의 전처리 파이프라인을 분류 문제에 적용한다.

**데이터**

| 컬럼 | 설명 | 결측 |
|---|---|---|
| `Survived` | **타깃** (0 = 사망, 1 = 생존) | |
| `Pclass` | 객실 등급 (1, 2, 3) | |
| `Name`, `Sex`, `Age` | 이름, 성별, 나이 | `Age` 177개 (약 19%) |
| `SibSp` | 함께 탄 형제자매·배우자 수 | |
| `Parch` | 함께 탄 부모·자녀 수 | |
| `Ticket` | 티켓 번호 | |
| `Fare` | 요금 (파운드) | |
| `Cabin` | 객실 번호 | 약 77% |
| `Embarked` | 승선 항구 (C, Q, S) | 2개 |

**코드: 로드와 탐색**

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets", filter="data")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]

train_data, test_data = load_titanic_data()          # 테스트 세트에는 레이블 없음 (Kaggle 제출용)
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

train_data.info()                     # 891행, Age·Cabin·Embarked 결측
train_data.describe()                 # 생존율 평균 0.384
train_data["Survived"].value_counts() # 0: 549, 1: 342
```

- `Cabin`(결측 77%), `Name`, `Ticket`(수치 변환이 어려움)은 제외
- 생존 비율 약 38% → 클래스 불균형이 심하지 않아 정확도를 지표로 사용

**코드: 전처리 파이프라인**

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),      # 결측 → 중앙값
        ("scaler", StandardScaler())
    ])

cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),               # 문자열 → 정수 (결측은 유지)
        ("imputer", SimpleImputer(strategy="most_frequent")),# 결측 → 최빈값
        ("cat_encoder", OneHotEncoder(sparse_output=False)),
    ])

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]
```

| 단계 | 이유 |
|---|---|
| `OrdinalEncoder` → `SimpleImputer` → `OneHotEncoder` | 범주를 먼저 정수로 바꾼 뒤 결측을 최빈값으로 채우고 원-핫 인코딩 |

**코드: 모델 비교**

```python
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
X_test = preprocess_pipeline.transform(test_data)    # 테스트 세트는 transform만
y_pred = forest_clf.predict(X_test)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()                                 # 0.8138

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()                                    # 0.8249

# 폴드별 점수 분포 비교
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], tick_labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
```

**결과 해석**

| 모델 | 10-폴드 CV 평균 정확도 |
|---|---|
| 랜덤 포레스트 | 0.814 |
| **SVC** | **0.825** |

- 상자 그림: 랜덤 포레스트는 한 폴드에서 매우 높은 점수를 냈지만 평균이 낮고 분산이 크다 → **SVC가 일반화 가능성이 더 높다**
- 평균뿐 아니라 **폴드별 분포**를 함께 봐야 한다

**코드: 특성 공학 아이디어**

```python
train_data["AgeBucket"] = train_data["Age"] // 15 * 15          # 15세 단위 구간
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]   # 동반 가족 수
train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean()
```

| `AgeBucket` | 생존율 | | `RelativesOnboard` | 생존율 |
|---|---|---|---|---|
| 0 | 0.577 | | 0 (혼자) | 0.304 |
| 15 | 0.363 | | 1 | 0.553 |
| 30 | 0.423 | | 2 | 0.578 |
| 45 | 0.404 | | 3 | 0.724 |
| 60 | 0.240 | | 4 이상 | 0.0 ~ 0.33 |

- 연령대별, 동반 가족 수별 생존율 차이가 크다 → 수치 특성을 범주로 바꾸거나 합산 특성을 만들면 성능 개선 가능
- 그 밖의 개선 방향: 이름에서 호칭(Mr, Mrs, Miss 등) 추출, 더 많은 모델 비교와 하이퍼파라미터 튜닝

---

### 8-4. 스팸 분류기

**목적**: 이메일 원문을 단어 벡터로 변환하는 전처리 파이프라인을 만들고 스팸 여부를 분류한다.

**데이터**: Apache SpamAssassin 공개 데이터 — 정상 메일(ham) 2,500개, 스팸 500개

**코드: 로드와 구조 탐색**

```python
import email
import email.policy

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(filepath) for filepath in ham_filenames]    # 2500
spam_emails = [load_email(filepath) for filepath in spam_filenames]  # 500

def get_email_structure(email):          # 메일의 MIME 구조를 재귀적으로 문자열화
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email)
                               for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()
```

| 구조 | 정상 메일 | 스팸 |
|---|---|---|
| `text/plain` | 2,408 | 218 |
| `text/html` | — | 183 |
| PGP 서명 포함 | 66 | 0 |

- 스팸은 HTML 비중이 높고 PGP 서명이 없다 → 메일 구조 자체가 유용한 정보

**코드: 분할**

```python
X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
```

**코드: 텍스트 추출**

```python
import re
from html import unescape

def html_to_plain_text(html):
    text = re.sub(r'<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)   # <head> 제거
    text = re.sub(r'<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)  # 링크 → HYPERLINK
    text = re.sub(r'<.*?>', '', text, flags=re.M | re.S)                        # 나머지 태그 제거
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)                   # 빈 줄 정리
    return unescape(text)                                                       # &gt; 등 복원

def email_to_text(email):               # 형식과 무관하게 본문 텍스트 반환
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:                          # 인코딩 오류 대비
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)
```

**코드: 메일 → 단어 빈도 변환기**

```python
import nltk
import urlextract
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

stemmer = nltk.PorterStemmer()            # Computations, Computing → comput
url_extractor = urlextract.URLExtract()

class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True,
                 remove_punctuation=True, replace_urls=True,
                 replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)   # 긴 URL부터 치환
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)
```

| 처리 | 목적 |
|---|---|
| 소문자화 | "Free"와 "free"를 같은 단어로 |
| URL → `URL`, 숫자 → `NUMBER` | 개별 값이 아닌 "URL/숫자가 있다"는 사실만 특성으로 |
| 구두점 제거 | 단어 단위 분리 |
| 어간 추출(stemming) | 활용형을 같은 단어로 통합 |
| 생성자 인수 | 각 처리를 켜고 끌 수 있어 하이퍼파라미터로 튜닝 가능 |

**코드: 단어 빈도 → 희소 벡터 변환기**

```python
from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)          # 한 메일의 기여를 최대 10으로 제한
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1                  # 0번은 어휘 밖 단어용
                            for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))   # 어휘에 없으면 0번 열
                data.append(count)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(X), self.vocabulary_size + 1))
```

| 요소 | 의미 |
|---|---|
| `fit` | 훈련 데이터에서 가장 빈번한 단어 `vocabulary_size`개로 **어휘 사전** 생성 |
| `transform` | 각 메일을 길이 `vocabulary_size + 1`의 단어 빈도 벡터로 변환 |
| 0번 열 | 어휘에 없는 단어 수의 합 |
| `csr_matrix` | 대부분 0인 행렬을 0이 아닌 값만 저장하는 **희소 행렬**로 표현 → 메모리 절약 |

어휘 10개로 메일 3개를 변환한 예:

```
array([[ 6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [99, 11,  9,  8,  3,  1,  3,  1,  3,  2,  3],
       [67,  0,  1,  2,  3,  4,  1,  2,  0,  1,  0]])
vocabulary_: {'the': 1, 'of': 2, 'and': 3, 'to': 4, 'url': 5, ...}
```

→ 두 번째 메일: 어휘 밖 단어 99개, "the" 11회, "of" 9회, …

**코드: 학습과 평가**

```python
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])
X_train_transformed = preprocess_pipeline.fit_transform(X_train)

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3)
score.mean()                                          # 0.985

X_test_transformed = preprocess_pipeline.transform(X_test)
log_clf = LogisticRegression(max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)
y_pred = log_clf.predict(X_test_transformed)

print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")
```

**결과 해석**

| 평가 | 값 |
|---|---|
| CV 정확도 (훈련) | 0.985 |
| 테스트 정밀도 | 96.88% |
| 테스트 재현율 | 97.89% |

- 스팸 비율이 약 17%로 불균형하므로 정확도 외에 정밀도·재현율을 함께 보고한다 (3-2의 원칙)
- 이 데이터는 "easy" 세트다. 어려운 세트에서는 여러 모델 비교와 튜닝이 필요하다

---

## 9. 정리

### 9-1. 워크플로와 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| 데이터 준비 | `fetch_openml('mnist_784')`, 슬라이싱 분할 | 레이블 자료형 확인, 테스트 세트 격리 |
| 이진 분류 | `SGDClassifier().fit(X, y_train_5)` | 문제를 단순화해 시작 |
| 기준 비교 | `DummyClassifier`, `cross_val_score` | 불균형 데이터에서 정확도를 믿지 않는다 |
| 오류 분해 | `cross_val_predict`, `confusion_matrix` | 테스트 세트 없이 샘플별 예측 생성 |
| 지표 요약 | `precision_score`, `recall_score`, `f1_score` | 오류 비용에 따라 지표 선택 |
| 임곗값 선택 | `decision_function`, `precision_recall_curve` | 재학습 없이 임곗값으로 균형 조정 |
| 모델 비교 | `roc_auc_score`, PR 곡선 겹쳐 그리기 | 양성이 드물면 PR 곡선 |
| 다중 분류 | `SVC`, `OneVsRestClassifier`, `StandardScaler` | OvR/OvO 자동 선택, 스케일링 먼저 |
| 오류 분석 | `ConfusionMatrixDisplay(normalize, sample_weight)` | 정량 → 정성 → 원인 → 개선책 |
| 다중 레이블 | `KNeighborsClassifier`, `ClassifierChain` | 레이블 간 의존성 활용 |
| 다중 출력 | `KNeighborsClassifier` (이미지 → 이미지) | 레이블당 값이 여러 개 |

### 9-2. 성능 변화

**5 감지 (이진 분류)**

| 모델 / 설정 | 정확도 | 정밀도 | 재현율 | F1 | AUC |
|---|---|---|---|---|---|
| 더미 분류기 | 0.910 | — | 0 | — | — |
| SGD | 0.957 | 0.837 | 0.651 | 0.733 | 0.960 |
| SGD, 임곗값 3,370 | | 0.900 | 0.480 | | |
| **랜덤 포레스트** | | **0.990** | **0.873** | **0.927** | **0.998** |

**0~9 분류 (다중 분류)**

| 모델 / 설정 | 정확도 |
|---|---|
| SGD (CV) | 0.867 |
| SGD + 표준화 (CV) | 0.897 |
| KNN 기본 (테스트) | 0.9688 |
| KNN 튜닝 (테스트) | 0.9714 |
| **KNN 튜닝 + 데이터 증강 (테스트)** | **0.9763** |

### 9-3. 핵심 원칙

1. 클래스가 불균형하면 정확도 대신 오차 행렬과 정밀도·재현율로 평가한다.
2. 성능은 항상 기준 모델(더미 분류기)과 비교해 해석한다.
3. `cross_val_predict`로 테스트 세트를 쓰지 않고 샘플별 예측·점수·확률을 얻는다.
4. 정밀도와 재현율은 트레이드오프 관계다. 목표 지표를 제시할 때는 반대 지표의 값도 함께 명시한다.
5. 임곗값은 모델 재학습 없이 조정할 수 있으며, PR 곡선에서 목표에 맞게 고른다.
6. 양성이 드물거나 FP가 중요하면 ROC 곡선보다 PR 곡선을 본다.
7. `argmax()`는 인덱스를 반환한다. 클래스 이름은 `classes_[인덱스]`로 얻는다.
8. 오류 분석은 정규화 기준을 명시하고, 수치 확인 후 실제 샘플을 관찰해 원인을 찾는다.
9. 정확도가 이미 높을 때는 오류율의 상대 변화로 개선 폭을 평가한다.
