# 3장. 분류

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `03_classification.ipynb`
>
> **장의 목적**: 범주를 예측하는 **분류** 문제에서 ① 분류기를 학습하고, ② 그 성능을 올바르게 측정하고, ③ 오류를 분석해 개선 방향을 찾고, ④ 문제 형태를 이진 → 다중 → 다중 레이블 → 다중 출력으로 확장하는 방법을 다룬다.
>
> **실습 데이터: MNIST**
>
> | 항목 | 내용 |
> |---|---|
> | `X` (`mnist.data`) | `(70000, 784)` — 28×28 흑백 이미지를 1차원으로 펼친 것. 값 0(흰색)~255(검정) |
> | `y` (`mnist.target`) | `(70000,)` — 정답 레이블. 문자열 `'0'`~`'9'` |
> | 분할 | 앞 60,000개 훈련 / 뒤 10,000개 테스트 (사전에 섞여 있음) |

## 목차

- [0. 장 구성](#0-장-구성)
- [3.1 MNIST](#31-mnist)
- [3.2 이진 분류기 훈련](#32-이진-분류기-훈련)
- [3.3 성능 측정](#33-성능-측정)
  - [3.3.1 교차 검증을 사용한 정확도 측정](#331-교차-검증을-사용한-정확도-측정)
  - [3.3.2 오차 행렬](#332-오차-행렬)
  - [3.3.3 정밀도와 재현율](#333-정밀도와-재현율)
  - [3.3.4 정밀도/재현율 트레이드오프](#334-정밀도재현율-트레이드오프)
  - [3.3.5 ROC 곡선](#335-roc-곡선)
  - [3.3.6 모델 비교: 랜덤 포레스트](#336-모델-비교-랜덤-포레스트)
- [3.4 다중 분류](#34-다중-분류)
- [3.5 오류 분석](#35-오류-분석)
- [3.6 다중 레이블 분류](#36-다중-레이블-분류)
- [3.7 다중 출력 분류](#37-다중-출력-분류)
- [3.8 장 요약](#38-장-요약)

> **읽는 법**: 각 항목은 같은 순서로 정리했다.
> **개념**(무엇인가) → **원리**(왜·어떻게, 수식) → **코드** → **결과**(노트북 실제 출력) → **해석**(결과가 의미하는 것)

---

## 0. 장 구성

### 0-1. 절 사이의 관계

```
[3.1 데이터]  MNIST 로드·분할
     │
[3.2 모델]    이진 분류기 (5 / 5 아님) ── SGDClassifier
     │
[3.3 평가]    정확도의 한계 ─→ 오차 행렬 ─→ 정밀도·재현율·F1
     │                              │
     │                        임곗값 조정 ─→ PR 곡선 / ROC 곡선·AUC ─→ 모델 비교
     │
[3.4 확장①]  다중 분류 (0~9) ── OvR / OvO 전략, 스케일링
     │
[3.5 개선]    오류 분석 ── 정규화 오차 행렬, 오분류 샘플 관찰 ─→ 개선 방향
     │
[3.6 확장②]  다중 레이블 (여러 이진 레이블) ── KNN, ClassifierChain
     │
[3.7 확장③]  다중 출력 (여러 다중 클래스 레이블) ── 노이즈 제거
```

### 0-2. 절별 핵심 질문

| 절 | 핵심 질문 | 핵심 개념 | 주요 API |
|---|---|---|---|
| 3.1 | 데이터는 어떤 구조인가 | 특성 벡터, 훈련/테스트 분할 | `fetch_openml` |
| 3.2 | 이진 분류기는 어떻게 만드는가 | 이진 분류, 선형 분류기 | `SGDClassifier` |
| 3.3 | 분류기를 어떻게 평가하는가 | 정확도, 오차 행렬, 정밀도, 재현율, F1, 임곗값, PR·ROC, AUC | `cross_val_score`, `cross_val_predict`, `confusion_matrix`, `precision_recall_curve`, `roc_curve` |
| 3.4 | 클래스가 여러 개면 어떻게 하는가 | OvR, OvO | `SVC`, `OneVsRestClassifier`, `StandardScaler` |
| 3.5 | 무엇을 틀리고 왜 틀리는가 | 정규화 오차 행렬, 오류 패턴 | `ConfusionMatrixDisplay` |
| 3.6 | 레이블이 여러 개면 어떻게 하는가 | 다중 레이블, 레이블 간 의존성 | `KNeighborsClassifier`, `ClassifierChain` |
| 3.7 | 레이블마다 값이 여러 개면 어떻게 하는가 | 다중 출력 | `KNeighborsClassifier` |

### 0-3. 분류 문제의 네 가지 형태

이 장 전체를 관통하는 분류 체계. **레이블의 개수**와 **레이블 하나가 가질 수 있는 값의 개수**로 구분한다.

| | 레이블 1개 | 레이블 여러 개 |
|---|---|---|
| **값 2개** | 이진 분류 (3.2–3.3) | 다중 레이블 분류 (3.6) |
| **값 3개 이상** | 다중 분류 (3.4–3.5) | 다중 출력 분류 (3.7) |

---

## 3.1 MNIST

**학습 목표**
- MNIST 데이터의 구조(샘플 수, 특성 수, 레이블 형태)를 파악한다.
- 1차원 특성 벡터와 2차원 이미지의 관계를 이해한다.
- 훈련/테스트 세트를 분할한다.

### 3.1.1 데이터 로드

**개념**
- MNIST: 손글씨 숫자 이미지 70,000개로 구성된 표준 분류 데이터셋.
- 각 이미지는 28×28 = 784개 픽셀로 이루어지며, **픽셀 하나가 특성 하나**다. 즉 샘플 하나는 784차원 특성 벡터다.

**코드**

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
mnist.keys()

X, y = mnist.data, mnist.target
X.shape
y.shape
y
```

| 코드 | 설명 |
|---|---|
| `fetch_openml('mnist_784')` | OpenML 저장소에서 데이터셋을 이름으로 내려받는다. 이후에는 로컬 캐시(`~/scikit_learn_data`)를 사용 |
| `as_frame=False` | 반환 형식을 pandas DataFrame이 아닌 NumPy 배열로 지정 |
| `mnist.data`, `mnist.target` | 반환값(`Bunch`)에서 특성 행렬과 레이블 벡터를 꺼낸다 |

**결과**

```
dict_keys(['data', 'target', 'frame', 'categories', 'feature_names',
           'target_names', 'DESCR', 'details', 'url'])
(70000, 784)
(70000,)
array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)
```

**해석**
- `X`는 70,000(샘플) × 784(특성) 행렬, `y`는 길이 70,000의 레이블 벡터다.
- 레이블의 자료형이 `object`(문자열)이다. 따라서 이후 비교 연산은 `y == '5'`처럼 **문자열**로 해야 한다.

### 3.1.2 이미지 시각화

**개념**
- 모델은 784차원 벡터를 입력으로 받지만, 사람이 확인하려면 28×28 행렬로 복원해야 한다.

**코드**

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)   # (784,) → (28, 28)
    plt.imshow(image, cmap="binary")     # 0=흰색, 255=검정
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()

y[0]
```

```python
# 그림 3-2: 처음 100개 샘플을 10×10 격자로 출력
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)
    plot_digit(image_data)
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
```

**결과**

```
'5'
```

**해석**
- 첫 샘플은 숫자 5이며, 이후 `some_digit`이라는 이름으로 이 장 전체에서 예측 확인용 샘플로 쓰인다.
- 같은 숫자라도 필체 차이가 커서, 이 변동성이 분류의 난이도를 결정한다.

### 3.1.3 훈련/테스트 분할

**개념**
- 테스트 세트는 최종 평가 전까지 사용하지 않는다 (2장의 데이터 스누핑 방지 원칙).

**코드**

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

**해석**
- MNIST는 배포 시점에 이미 섞여 있고 분할 기준(앞 60,000 / 뒤 10,000)이 정해져 있으므로 슬라이싱만으로 분할한다.
- 섞여 있다는 사실은 교차 검증에서 중요하다. 정렬된 데이터를 그대로 나누면 특정 폴드에 일부 클래스가 빠질 수 있다.

### 3.1 정리

| 항목 | 내용 |
|---|---|
| 특성 | 픽셀 784개 (0~255) |
| 레이블 | 문자열 `'0'`~`'9'` (10개 클래스) |
| 훈련 / 테스트 | 60,000 / 10,000 |

---

## 3.2 이진 분류기 훈련

**학습 목표**
- 다중 클래스 레이블을 이진 레이블로 변환한다.
- `SGDClassifier`로 이진 분류기를 학습하고 예측한다.

### 3.2.1 이진 레이블 생성

**개념**
- **이진 분류**: 두 클래스 중 하나를 예측하는 문제. 관심 대상을 **양성**(positive), 나머지를 **음성**(negative)이라 한다.
- 여기서는 "5"를 양성, "5가 아닌 숫자"를 음성으로 정의한다.

**코드**

```python
y_train_5 = (y_train == '5')   # True: 5, False: 5 아님
y_test_5 = (y_test == '5')
```

**해석**
- 원소별 비교로 불리언 배열이 생성된다. 10개 클래스 문제가 2개 클래스 문제로 단순화된다.

### 3.2.2 SGD 분류기 학습

**개념**
- `SGDClassifier`: **확률적 경사 하강법**(Stochastic Gradient Descent)으로 학습하는 선형 분류기.

**원리**
- 결정 함수: $h(\mathbf{x}) = \mathbf{w}^\top\mathbf{x} + b$ — 784개 픽셀의 가중합에 절편을 더한 값
- 예측: $\hat{y} = 1 \;\text{if}\; h(\mathbf{x}) > 0 \;\text{else}\; 0$
- 학습: 샘플을 하나씩 무작위로 골라 손실의 기울기 방향으로 $\mathbf{w}, b$를 갱신 (기본 손실 `hinge` → 선형 SVM과 동일한 모델)
- 장점: 샘플 단위로 갱신하므로 대규모 데이터와 온라인 학습에 적합하다.

**코드**

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])
```

| 코드 | 설명 |
|---|---|
| `random_state=42` | 샘플 추출 순서의 무작위성을 고정 → 재현 가능 |
| `predict([some_digit])` | 입력은 `(샘플 수, 특성 수)` 2차원이어야 하므로 샘플 1개도 리스트로 감싼다 |

**결과**

```
array([ True])
```

**해석**
- `some_digit`(5)을 양성으로 올바르게 예측했다.
- 이 모델은 **선형**이므로 픽셀 위치마다 고정된 가중치를 가진다. 이 성질이 3.5 오류 분석의 핵심 원인으로 다시 등장한다.

### 3.2 정리

| 단계 | 코드 |
|---|---|
| 이진 레이블 | `y_train_5 = (y_train == '5')` |
| 학습 | `SGDClassifier(random_state=42).fit(X_train, y_train_5)` |
| 예측 | `sgd_clf.predict([x])` |

---

## 3.3 성능 측정

**학습 목표**
- 불균형 데이터에서 정확도가 부적절한 이유를 설명한다.
- 오차 행렬, 정밀도, 재현율, F1 점수를 계산하고 해석한다.
- 결정 임곗값과 정밀도/재현율 트레이드오프의 관계를 이해한다.
- PR 곡선과 ROC 곡선을 그리고 용도를 구분한다.
- 지표를 이용해 두 모델을 비교한다.

**절의 논리 구조**

```
정확도로 측정 (3.3.1) → 기준 모델과 비교하니 의미 없음
   → 오류를 유형별로 분해: 오차 행렬 (3.3.2)
   → 유형별 비율로 요약: 정밀도·재현율·F1 (3.3.3)
   → 두 지표는 임곗값에 따라 반대로 움직임: 트레이드오프 (3.3.4)
   → 임곗값 전체를 한 그래프로: PR 곡선 / ROC 곡선·AUC (3.3.5)
   → 이 지표들로 모델 비교 (3.3.6)
```

### 3.3.1 교차 검증을 사용한 정확도 측정

**개념**
- **정확도**(accuracy) = 올바르게 예측한 샘플 수 / 전체 샘플 수
- **k-폴드 교차 검증**: 훈련 세트를 k개 폴드로 나누고, 각 폴드를 한 번씩 평가용으로, 나머지를 학습용으로 사용해 k개의 점수를 얻는다.
- **계층적(stratified) k-폴드**: 각 폴드의 클래스 비율을 전체 비율과 같게 유지한다.

**코드 ① — `cross_val_score`**

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

**결과**

```
array([0.95035, 0.96035, 0.9604 ])
```

**코드 ② — 교차 검증 직접 구현**

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3)  # 데이터가 섞여 있지 않으면 shuffle=True

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
```

| 코드 | 설명 |
|---|---|
| `StratifiedKFold(n_splits=3)` | 클래스 비율을 유지하는 3-폴드 분할기 |
| `skfolds.split(X, y)` | 회차마다 (학습 인덱스, 평가 인덱스)를 생성. 비율 계산을 위해 `y`가 필요 |
| `clone(sgd_clf)` | 하이퍼파라미터는 같고 학습 상태는 없는 복제본. 회차 간 학습 결과가 섞이지 않게 한다 |
| `sum(y_pred == y_test_fold)` | 불리언 합 = 정답 개수 |

**결과**

```
0.95035
0.96035
0.9604
```

→ `cross_val_score`와 동일. `cross_val_score`의 내부 동작이 위 반복문과 같다는 것을 확인할 수 있다.

**코드 ③ — 기준 모델(더미 분류기)과 비교**

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
```

| 코드 | 설명 |
|---|---|
| `DummyClassifier()` | 기본 전략 `"prior"` — 입력을 무시하고 가장 빈도가 높은 클래스만 예측하는 기준 모델 |
| `any(...)` | 양성 예측이 하나라도 있는지 확인 |

**결과**

```
False
array([0.90965, 0.90965, 0.90965])
```

**해석**
- 더미 분류기는 모든 샘플을 음성으로 예측하는데도 정확도가 **약 91%**다. 음성(5가 아님)이 전체의 약 91%를 차지하기 때문이다.
- 따라서 SGD의 95%는 기준 모델 대비 약 4%p 높은 수준에 불과하다.
- **결론: 클래스 분포가 불균형한 데이터에서는 정확도가 분류기의 성능을 제대로 나타내지 못한다.** 오류를 유형별로 나눠 보는 지표가 필요하다.

### 3.3.2 오차 행렬

**개념**
- **오차 행렬**(confusion matrix): 실제 클래스(행)와 예측 클래스(열)의 조합별 샘플 수를 센 표.

**원리** — 이진 분류의 오차 행렬

| | 예측: 음성 | 예측: 양성 |
|---|---|---|
| **실제: 음성** | TN (진짜 음성) | FP (거짓 양성, 1종 오류) |
| **실제: 양성** | FN (거짓 음성, 2종 오류) | TP (진짜 양성) |

- 명명 규칙: 뒷글자(P/N)는 **예측값**, 앞글자(T/F)는 **예측이 맞았는지 여부**
- 대각선(TN, TP)이 정답, 비대각선(FP, FN)이 오류

**코드 ① — 교차 검증 예측 생성**

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

| 함수 | 반환값 |
|---|---|
| `cross_val_score` | 폴드별 **점수** (k개) |
| `cross_val_predict` | 샘플별 **예측값** (n개) — 각 예측은 그 샘플을 학습에 사용하지 않은 모델이 생성 |

→ 테스트 세트를 사용하지 않고도 훈련 세트 전체에 대한 "보지 않은 데이터 예측"을 얻는다.

**코드 ② — 오차 행렬 계산**

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)   # (실제, 예측) 순서
cm
```

**결과**

```
array([[53892,   687],
       [ 1891,  3530]])
```

| | 예측: 5 아님 | 예측: 5 |
|---|---|---|
| **실제: 5 아님** | TN = 53,892 `cm[0,0]` | FP = 687 `cm[0,1]` |
| **실제: 5** | FN = 1,891 `cm[1,0]` | TP = 3,530 `cm[1,1]` |

**코드 ③ — 완벽한 분류기의 오차 행렬**

```python
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)
```

```
array([[54579,     0],
       [    0,  5421]])
```

**해석**
- SGD는 5를 4,217번 양성으로 예측했고(FP+TP), 그중 687번은 오류다.
- 실제 5 5,421개 중 1,891개를 놓쳤다(FN).
- 완벽한 분류기는 비대각선이 모두 0이다. 오차 행렬 분석의 초점은 **비대각선 원소의 크기와 위치**다.

### 3.3.3 정밀도와 재현율

**개념**

| 지표 | 정의 | 답하는 질문 |
|---|---|---|
| **정밀도** (precision) | 양성으로 예측한 것 중 실제 양성의 비율 | 양성이라고 판정했을 때 얼마나 믿을 수 있는가 |
| **재현율** (recall, 민감도, TPR) | 실제 양성 중 양성으로 예측한 비율 | 실제 양성을 얼마나 빠짐없이 찾는가 |
| **F1 점수** | 정밀도와 재현율의 조화평균 | 두 지표를 하나로 요약하면 어느 수준인가 |

**원리**

$$\text{정밀도} = \frac{TP}{TP + FP} \qquad \text{재현율} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2}{\frac{1}{\text{정밀도}} + \frac{1}{\text{재현율}}} = 2 \times \frac{\text{정밀도} \times \text{재현율}}{\text{정밀도} + \text{재현율}} = \frac{TP}{TP + \frac{FN + FP}{2}}$$

- 정밀도는 FP에, 재현율은 FN에 의해 낮아진다.
- 조화평균은 작은 값 쪽으로 치우친다. 예: 정밀도 1.0, 재현율 0.1이면 산술평균 0.55, F1 ≈ 0.18. 따라서 F1이 높으려면 두 지표가 모두 높아야 한다.

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

**결과**

| 지표 | 계산 | 값 |
|---|---|---|
| 정밀도 | 3530 / (687 + 3530) | 0.8371 |
| 재현율 | 3530 / (1891 + 3530) | 0.6512 |
| F1 | 3530 / (3530 + (1891 + 687)/2) | 0.7325 |

**해석**
- 5라고 판정한 것의 83.7%만 실제 5이고, 실제 5의 65.1%만 찾아낸다.
- 정확도 95%와 비교하면 훨씬 낮은 수치로, 이것이 이 분류기의 실제 수준이다.

**지표 선택 기준**

| 상황 | 중시할 지표 | 이유 |
|---|---|---|
| FP 비용이 큼 (예: 어린이용 콘텐츠 필터) | 정밀도 | 안전하다고 판정한 것은 반드시 안전해야 함 |
| FN 비용이 큼 (예: 질병 선별 검사, 도난 감지) | 재현율 | 실제 양성을 놓치면 안 됨 |
| 두 비용이 비슷함 | F1 | 균형 |

### 3.3.4 정밀도/재현율 트레이드오프

**개념**
- **결정 점수**(decision score): 분류기가 샘플마다 계산하는 실수값. 클수록 양성일 가능성이 높다고 판단한 것.
- **결정 임곗값**(threshold): 결정 점수가 이 값을 넘으면 양성으로 판정하는 기준. `SGDClassifier`의 기본값은 0.

**원리**

$$\hat{y} = \begin{cases} 1 & h(\mathbf{x}) > \text{임곗값} \\ 0 & \text{otherwise} \end{cases}$$

| 임곗값 변화 | 양성 판정 수 | FP | FN | 정밀도 | 재현율 |
|---|---|---|---|---|---|
| 올림 ↑ | 감소 | 감소 | 증가 | 대체로 ↑ | ↓ |
| 내림 ↓ | 증가 | 증가 | 감소 | 대체로 ↓ | ↑ |

→ 모델을 다시 학습하지 않고도 **임곗값만으로 정밀도와 재현율의 균형을 조정**할 수 있다.

**코드 ① — 단일 샘플의 결정 점수와 임곗값**

```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

threshold = 3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
```

**결과**

```
array([2164.22030239])
array([ True])      # 임곗값 0
array([False])      # 임곗값 3000
```

**해석**
- `predict()`는 내부적으로 `decision_function() > 0`과 같다.
- 임곗값을 3000으로 올리면 점수 2164인 실제 5가 음성으로 바뀐다(FN 발생 → 재현율 감소).

**코드 ② — 전체 샘플의 결정 점수와 PR 곡선 데이터**

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

| 코드 | 설명 |
|---|---|
| `method="decision_function"` | 예측값 대신 결정 점수를 교차 검증으로 얻는다 |
| `precision_recall_curve` | 가능한 모든 임곗값에 대한 정밀도·재현율을 계산. `thresholds`는 **오름차순** |

> ⚠️ `precisions`, `recalls`의 길이는 `len(thresholds) + 1`이다(마지막에 정밀도 1, 재현율 0인 점이 추가됨). 임곗값과 짝지어 그릴 때는 `[:-1]`로 마지막 원소를 제외한다.

**코드 ③ — 임곗값에 따른 정밀도·재현율 (그림 3-5)**

```python
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")

idx = (thresholds >= threshold).argmax()   # 임곗값 3000 이상인 첫 인덱스
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid(); plt.xlabel("Threshold"); plt.legend(loc="center right")
plt.show()
```

> **관용구 `(조건 배열).argmax()`**: 불리언 배열에서 `argmax()`는 최댓값(True=1)이 처음 나타나는 위치를 반환한다. 따라서 **조건을 처음 만족하는 인덱스**를 구하는 데 쓴다.

**코드 ④ — PR 곡선 (그림 3-6)**

```python
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
```

**해석 (그림 3-5, 3-6)**
- 재현율은 임곗값이 커질수록 단조 감소한다.
- 정밀도는 대체로 증가하지만 단조 증가는 아니다. 임곗값 상승으로 TP 하나가 빠지면 정밀도가 일시적으로 떨어질 수 있다.
- PR 곡선에서 재현율 약 80% 부근부터 정밀도가 급격히 하락한다. 그 직전이 일반적인 절충점이다.

**코드 ⑤ — 목표 정밀도(90%)에 해당하는 임곗값 찾기**

```python
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

precision_score(y_train_5, y_train_pred_90)
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision
```

**결과**

| 항목 | 값 |
|---|---|
| 정밀도 90% 임곗값 | 3370.02 |
| 정밀도 | 약 0.900 |
| 재현율 | 0.4800 |

**해석**
- 임곗값만 조정해 정밀도 90% 분류기를 얻었으나 재현율은 48%로 하락했다.
- 따라서 정밀도(또는 재현율) 목표를 제시할 때는 **반대 지표의 값을 함께 명시**해야 의미가 있다.

### 3.3.5 ROC 곡선

**개념**
- **ROC 곡선**(Receiver Operating Characteristic): 임곗값을 바꿔 가며 **FPR**(x축)에 대한 **TPR**(y축)을 그린 곡선.
- **AUC**(Area Under the Curve): ROC 곡선 아래 면적. 1이면 완벽한 분류기, 0.5면 무작위 분류기.

**원리**

$$TPR = \frac{TP}{TP + FN} \;(= \text{재현율}) \qquad FPR = \frac{FP}{FP + TN} \;(= 1 - \text{특이도})$$

| 곡선 | x축 | y축 | 이상적인 위치 |
|---|---|---|---|
| PR 곡선 | 재현율 | 정밀도 | 오른쪽 위 |
| ROC 곡선 | FPR | TPR | 왼쪽 위 |

**코드 ① — ROC 곡선 계산과 시각화 (그림 3-7)**

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid(); plt.axis([0, 1, 0, 1]); plt.legend(loc="lower right")
plt.show()
```

> ⚠️ `roc_curve`의 `thresholds`는 **내림차순**이다(`precision_recall_curve`는 오름차순). 그래서 정밀도 90% 임곗값의 위치를 찾을 때 `<=` 조건을 쓴다.

**코드 ② — AUC**

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
```

**결과**

```
0.9604938554008616
```

**해석**
- 점선 대각선은 무작위 분류기의 ROC 곡선이다. 좋은 분류기일수록 곡선이 왼쪽 위로 붙는다.
- TPR(재현율)을 높이면 FPR도 함께 올라간다. ROC 곡선에서도 같은 트레이드오프가 나타난다.
- AUC 0.96은 높아 보이지만, 이는 음성이 많아 FPR의 분모(FP+TN)가 크기 때문이다.

**PR 곡선과 ROC 곡선의 선택 기준**

| 조건 | 사용할 곡선 | 이유 |
|---|---|---|
| 양성 클래스가 드묾 | PR 곡선 | TN이 커서 FPR이 작게 나오므로 ROC가 낙관적으로 보인다 |
| FP를 FN보다 중시 | PR 곡선 | 정밀도가 FP를 직접 반영 |
| 그 외 | ROC 곡선 | 클래스 비율 변화에 덜 민감한 일반적 비교 |

→ 이번 문제(양성 약 10%)에서는 PR 곡선이 더 적절하며, PR 곡선상으로는 개선 여지가 크다.

### 3.3.6 모델 비교: 랜덤 포레스트

**개념**
- `RandomForestClassifier`는 `decision_function()`을 제공하지 않는다. 대신 `predict_proba()`로 **클래스별 추정 확률**을 반환하며, 양성 확률을 결정 점수 대신 사용한다.

**코드 ① — 클래스 확률 얻기**

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_probas_forest[:2]
```

**결과**

```
array([[0.11, 0.89],
       [0.99, 0.01]])
```

- 열 순서: `[음성 확률, 양성 확률]`. 첫 샘플(5)은 양성 확률 0.89, 두 번째 샘플(0)은 0.01.

**코드 ② — 추정 확률의 신뢰성 확인**

```python
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")
```

**결과**

```
94.0%
```

**해석**
- 양성 확률 0.5~0.6으로 추정된 샘플 중 실제 양성 비율은 94%다. 확률이 정확하다면 약 55%여야 한다.
- 즉 `predict_proba`의 출력은 **보정(calibration)되지 않은 추정치**이며, 확률 값 자체를 의사결정에 쓰려면 별도의 보정이 필요하다.

**코드 ③ — PR 곡선 비교 (그림 3-8)**

```python
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
```

**코드 ④ — 지표 비교**

```python
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5   # 임곗값 0.5 적용
f1_score(y_train_5, y_train_pred_forest)
roc_auc_score(y_train_5, y_scores_forest)
precision_score(y_train_5, y_train_pred_forest)
recall_score(y_train_5, y_train_pred_forest)
```

- 이미 구한 확률에 임곗값 0.5를 적용하면 `cross_val_predict`를 다시 실행하지 않고 동일한 예측을 얻는다.

**결과**

| 지표 | SGD | 랜덤 포레스트 |
|---|---|---|
| 정밀도 | 0.837 | **0.990** |
| 재현율 | 0.651 | **0.873** |
| F1 | 0.733 | **0.927** |
| ROC AUC | 0.960 | **0.998** |

**해석**
- 랜덤 포레스트의 PR 곡선이 오른쪽 위 모서리에 훨씬 가깝고, 모든 지표에서 SGD보다 우수하다.

### 3.3 정리

| 도구 | 입력 | 산출물 | 용도 |
|---|---|---|---|
| `cross_val_score(..., scoring=...)` | 모델, X, y | 폴드별 점수 | 단일 지표 평가 |
| `cross_val_predict(...)` | 모델, X, y | 샘플별 예측 | 오차 행렬·지표 계산 |
| `cross_val_predict(..., method="decision_function")` | 〃 | 샘플별 결정 점수 | 임곗값 분석 |
| `cross_val_predict(..., method="predict_proba")` | 〃 | 샘플별 클래스 확률 | 점수가 없는 모델의 임곗값 분석 |
| `confusion_matrix` | 실제, 예측 | TN·FP·FN·TP | 오류 유형 분해 |
| `precision_score` / `recall_score` / `f1_score` | 실제, 예측 | 지표값 | 요약 평가 |
| `precision_recall_curve` | 실제, 점수 | 정밀도·재현율·임곗값 배열 | PR 곡선, 임곗값 선택 |
| `roc_curve` / `roc_auc_score` | 실제, 점수 | FPR·TPR·임곗값 / AUC | ROC 곡선, 모델 비교 |

---

## 3.4 다중 분류

**학습 목표**
- 이진 분류기를 다중 분류로 확장하는 OvR, OvO 전략을 구분한다.
- 사이킷런의 자동 전략 선택과 수동 지정 방법을 익힌다.
- 다중 분류 결정 점수에서 예측 클래스를 얻는 방법을 이해한다.
- 입력 스케일링이 성능에 미치는 영향을 확인한다.

### 3.4.1 다중 분류 전략

**개념**
- **다중 분류**(multiclass classification): 셋 이상의 클래스 중 하나를 예측하는 문제.
- 로지스틱 회귀, 랜덤 포레스트, KNN 등은 다중 분류를 **기본 지원**한다.
- SGD 분류기, SVM 등 **이진 전용** 분류기는 아래 두 전략으로 확장한다.

**원리**

| 전략 | 구성 | 예측 방법 | 분류기 수 (N 클래스) | 적합한 경우 |
|---|---|---|---|---|
| **OvR** (One-versus-Rest) | 클래스마다 "해당 클래스 vs 나머지" 분류기 | 결정 점수가 가장 높은 클래스 | N | 대부분의 알고리즘 |
| **OvO** (One-versus-One) | 모든 클래스 쌍마다 "A vs B" 분류기 | 가장 많은 대결에서 이긴 클래스 | N(N−1)/2 | 훈련 세트 크기에 따라 비용이 크게 증가하는 알고리즘 (예: SVM) |

- MNIST(N=10): OvR 10개, OvO 45개.
- OvO의 각 분류기는 해당 두 클래스의 샘플만으로 학습하므로, 개별 학습 비용이 작다.
- 사이킷런은 이진 분류기에 다중 클래스 레이블이 주어지면 알고리즘에 따라 **OvR 또는 OvO를 자동 선택**한다.

### 3.4.2 SVC: 자동 OvO

**코드 ① — 학습과 예측**

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])   # 레이블: y_train (10개 클래스)

svm_clf.predict([some_digit])
```

- SVC는 훈련 세트 크기에 따라 학습 시간이 급격히 늘어나므로 2,000개만 사용한다.

**결과**

```
array(['5'], dtype=object)
```

**코드 ② — 결정 점수와 예측 클래스**

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)

class_id = some_digit_scores.argmax()
class_id
svm_clf.classes_
svm_clf.classes_[class_id]
```

**결과**

```
array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,  4.82]])
#        '0'    '1'    '2'    '3'    '4'    '5'    '6'    '7'    '8'    '9'
5
array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype=object)
'5'
```

**해석**
- 클래스별 점수 10개 중 '5'가 9.3으로 최대다. 이 점수는 **OvO 대결의 승수 + 분류기 신뢰도에 기반한 작은 보정값**이다. '5'는 9번의 대결에서 모두 이겼다.
- 차순위는 '3'(8.3)이다. 이 샘플이 3과 유사한 형태임을 시사한다.
- `argmax()`는 **인덱스**를 반환한다. 클래스 이름은 반드시 `classes_[인덱스]`로 얻어야 한다. (이 예에서는 클래스가 '0'~'9' 순서라 인덱스와 레이블이 우연히 일치)

**코드 ③ — OvO 원점수 45개 확인**

```python
svm_clf.decision_function_shape = "ovo"
some_digit_scores_ovo = svm_clf.decision_function([some_digit])
some_digit_scores_ovo.round(2)
```

**결과**

```
array([[ 0.11, -0.21, -0.97,  0.51, -1.01,  0.19,  0.09, -0.31, -0.04,
        -0.45, -1.28,  0.25, -1.01, -0.13, -0.32, -0.9 , -0.36, -0.93,
         0.79, -1.  ,  0.45,  0.24, -0.24,  0.25,  1.54, -0.77,  1.11,
         1.13,  1.04,  1.2 , -1.42, -0.53, -0.45, -0.99, -0.95,  1.21,
         1.  ,  1.  ,  1.08, -0.02, -0.67, -0.14, -0.3 , -0.13,  0.25]])
```

**해석**
- `decision_function_shape`는 **점수 출력 형태만** 결정한다. 기본값 `"ovr"`은 45개 점수를 클래스별 10개로 집계하고, `"ovo"`는 쌍별 점수 45개(0vs1, 0vs2, …, 8vs9)를 그대로 반환한다.
- **학습은 이 설정과 무관하게 항상 OvO**로 이루어진다.

### 3.4.3 전략 강제 지정: `OneVsRestClassifier`

**코드**

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)
```

**결과**

```
array(['5'], dtype='<U1')
10
```

**해석**
- 분류기를 `OneVsRestClassifier`로 감싸면 OvR이 적용되어 이진 분류기 10개(`estimators_`)가 생성된다.
- OvO를 강제하려면 `OneVsOneClassifier`를 사용한다.

### 3.4.4 SGD 분류기: 자동 OvR

**코드**

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

sgd_clf.decision_function([some_digit]).round()
```

**결과**

```
array(['3'], dtype='<U1')
array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
        -16148.,  -4604., -12051.]])
#        '0'      '1'      '2'      '3'      '4'      '5'   ...
```

**해석**
- SGD는 OvR을 사용하며, 10개 클래스별 점수 중 최댓값을 택한다.
- 양수 점수는 '3'(1824)뿐이고 정답 '5'(−1386)는 2위다. **3과 5의 혼동**이 관찰되며, 3.5에서 원인을 분석한다.

### 3.4.5 성능 평가와 스케일링

**코드**

```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
```

| 코드 | 설명 |
|---|---|
| `StandardScaler` | 특성(픽셀)마다 평균 0, 표준편차 1로 변환: $z = (x - \mu) / \sigma$ |
| `.astype("float64")` | 정수형 픽셀을 실수형으로 변환해 표준화 결과를 정확히 저장 |

**결과**

| 입력 | 폴드별 정확도 | 평균 |
|---|---|---|
| 원본 | 0.874, 0.858, 0.869 | 약 0.867 |
| 표준화 | 0.898, 0.891, 0.902 | 약 0.897 |

**해석**
- 10개 클래스 무작위 추측(10%) 대비 높은 수준이다.
- 표준화만으로 정확도가 약 3%p 향상되었다. 경사 하강법 기반 모델은 **특성 스케일에 민감**하므로, 모델 변경 이전에 전처리를 먼저 점검해야 한다.

### 3.4 정리

| 분류기 | 다중 분류 방식 | 분류기 수 | `some_digit` 예측 |
|---|---|---|---|
| `SVC` | 자동 OvO | 45 | '5' (정답) |
| `OneVsRestClassifier(SVC())` | 강제 OvR | 10 | '5' (정답) |
| `SGDClassifier` | 자동 OvR | 10 | '3' (오답) |

---

## 3.5 오류 분석

**학습 목표**
- 다중 클래스 오차 행렬을 시각화하고 정규화 방식에 따른 해석 차이를 이해한다.
- 오분류 패턴을 찾아 원인을 추론한다.
- 분석 결과로부터 개선 방향을 도출한다.

**분석 절차**

```
① 오차 행렬(개수) → ② 행 기준 정규화(비율) → ③ 정답 제거(오류만)
   → ④ 혼동 클래스 쌍의 샘플 직접 관찰 → ⑤ 원인 추론 → ⑥ 개선 방향
```

### 3.5.1 오차 행렬 시각화

**개념**
- 다중 클래스 오차 행렬은 N×N 행렬이다(행: 실제 클래스, 열: 예측 클래스). 원소가 많으므로 색으로 시각화한다.
- **정규화**: 원소를 합계로 나눠 비율로 바꾼다. 기준에 따라 해석이 달라진다.

| `normalize=` | 나누는 기준 | 읽는 법 |
|---|---|---|
| `"true"` | 행(실제 클래스) 합 | "실제 X 중 몇 %를 Y로 예측했나" — 재현율 관점 |
| `"pred"` | 열(예측 클래스) 합 | "Y로 예측한 것 중 몇 %가 실제 X였나" — 정밀도 관점 |
| `"all"` | 전체 합 | 전체 샘플 대비 비율 |

**코드 ① — 원본 오차 행렬**

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

- `ConfusionMatrixDisplay.from_predictions(실제, 예측)`: 오차 행렬 계산과 시각화를 한 번에 수행한다.

**해석**: 대각선이 진해 대체로 정확하다. '5'의 대각선 칸이 상대적으로 옅은데, 이것이 **5의 분류 성능이 낮아서인지, 5의 샘플 수가 적어서인지** 개수만으로는 구분할 수 없다.

**코드 ② — 행 기준 정규화**

```python
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()
```

**해석**: 샘플 수의 영향이 제거된다. 실제 5 중 약 82%만 올바르게 분류되었고, **약 10%가 8로** 분류되었다.

**코드 ③ — 오류만 표시**

```python
sample_weight = (y_train_pred != y_train)
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show()
```

| 코드 | 설명 |
|---|---|
| `sample_weight=(y_train_pred != y_train)` | 정답 샘플의 가중치를 0, 오답 샘플을 1로 지정 → 대각선이 0이 되고 오류 분포만 남는다 |

**해석**: **'8' 열이 전반적으로 밝다.** 여러 클래스의 샘플이 8로 오분류되는 경향이 있다.

**코드 ①~③ 결과 요약**

| 단계 | 관찰 |
|---|---|
| ① 개수 | 전반적으로 정확. '5' 성능이 다소 낮아 보이나 원인 불명확 |
| ② 행 정규화 | 실제 5의 약 10%가 8로 오분류 |
| ③ 오류만 | 여러 클래스가 '8'로 오분류되는 경향 (8 열 집중) |

> ⚠️ 정규화 기준에 따라 같은 오류의 크기가 달리 보인다. 행 기준에서 큰 "5 → 3" 오류가 열 기준(`"pred"`)에서는 다르게 보일 수 있으므로, 해석할 때 기준을 명시해야 한다.

### 3.5.2 혼동 클래스 쌍의 샘플 관찰 (3 vs 5)

**코드 ① — 네 그룹으로 분리**

```python
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]   # 실제 3, 예측 3
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]   # 실제 3, 예측 5
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]   # 실제 5, 예측 3
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]   # 실제 5, 예측 5
```

| 코드 | 설명 |
|---|---|
| `X_train[불리언 마스크]` | 마스크가 True인 행만 선택 (불리언 인덱싱) |
| `(조건1) & (조건2)` | 원소별 AND. NumPy 배열에는 `and`가 아닌 `&`를 쓰며, 연산자 우선순위 때문에 각 조건을 괄호로 감싼다 |

**코드 ② — 2×2 블록으로 시각화 (그림 3-11)**

```python
size = 5      # 블록당 5×5 = 25개 이미지
pad = 0.2     # 블록 간격
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)
        y = idx // size + label_row * (size + pad)
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.show()
```

| 코드 | 설명 |
|---|---|
| `idx % size`, `idx // size` | 블록 내 열·행 위치 (0~4) |
| `label_col * (size + pad)` | 블록 오프셋. (0,0)=왼쪽 아래 … (1,1)=오른쪽 위 |
| `imshow(..., extent=(왼, 오, 아래, 위))` | 이미지를 지정 좌표 영역에 배치. 서브플롯 100개를 만드는 것보다 효율적 |

**배치 결과**

| | 예측 3 | 예측 5 |
|---|---|---|
| **실제 3** (위) | `X_aa` 정답 | `X_ab` 오답 |
| **실제 5** (아래) | `X_ba` 오답 | `X_bb` 정답 |

### 3.5.3 원인 추론

**관찰**
- 오분류된 샘플 일부는 사람도 판별하기 어렵지만, 대부분은 사람에게는 명확하다.
- 3과 5의 형태적 차이는 **위쪽 가로획과 아래쪽 호를 잇는 짧은 세로획의 위치**뿐이다.

**원인**
- `SGDClassifier`는 선형 모델로, 픽셀 위치마다 **고정된 가중치**를 곱해 합산한다(3.2.2).
- 따라서 숫자가 약간 이동하거나 회전하면, 같은 획이 다른 픽셀 위치에 놓여 점수가 크게 달라진다. 즉 이 모델은 **이동과 회전에 민감**하다.

### 3.5.4 개선 방향

| 방향 | 방법 | 대상 문제 |
|---|---|---|
| 전처리 | 이미지를 중앙 정렬하고 기울기를 보정 | 이동·회전 민감성 |
| 데이터 증강 | 이동·회전시킨 이미지를 훈련 세트에 추가 (연습문제 2) | 이동·회전 민감성 |
| 특성 공학 | 닫힌 고리 수 등 형태 특성 추가 (8: 2개, 6: 1개, 5: 0개) | '8'로의 오분류 |
| 모델 변경 | 비선형 모델 사용 (예: 3.3.6의 랜덤 포레스트) | 선형 모델의 표현력 한계 |

### 3.5 정리

- 오류 분석은 **정량(오차 행렬) → 정성(샘플 관찰) → 원인 → 개선책** 순서로 진행한다.
- 이번 분석의 결론: 선형 모델의 위치 민감성 때문에 3/5 혼동과 '8'로의 오분류가 발생하며, 전처리·데이터 증강으로 개선할 수 있다.

---

## 3.6 다중 레이블 분류

**학습 목표**
- 다중 레이블 분류의 정의를 이해한다.
- 다중 레이블을 지원하는 모델로 학습·평가한다.
- 지원하지 않는 모델에 `ClassifierChain`을 적용하고 그 원리를 이해한다.

### 3.6.1 정의

**개념**
- **다중 레이블 분류**(multilabel classification): 샘플 하나에 대해 **여러 개의 이진 레이블**을 동시에 출력하는 분류.
- 예: 사진 한 장에서 인물 A·B·C의 등장 여부를 각각 판단 → 출력 `[1, 0, 1]`.

**원리**
- 출력이 레이블 수만큼의 이진 벡터다. 레이블 사이에는 상관관계가 있을 수 있다.

### 3.6.2 KNN으로 다중 레이블 분류

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

| 코드 | 설명 |
|---|---|
| `y_train >= '7'` | 문자열 비교. 한 자리 숫자이므로 사전순과 크기순이 같다 |
| `.astype('int8') % 2 == 1` | 정수로 변환 후 홀수 판별. 0~9는 `int8` 범위(−128~127)로 충분 |
| `np.c_[a, b]` | 1차원 배열들을 열 방향으로 결합 → 레이블 행렬 |
| `KNeighborsClassifier` | 다중 레이블을 기본 지원하는 모델. 가장 가까운 k개 훈련 샘플의 레이블로 예측 |

**결과**

```
array([[False,  True]])
```

**해석**: 5는 7 미만(False)이고 홀수(True)이므로 두 레이블 모두 정답이다.

### 3.6.3 평가

**개념**
- 레이블별로 이진 분류 지표를 계산한 뒤 평균한다.

| `average=` | 계산 방식 | 사용 시점 |
|---|---|---|
| `"macro"` | 레이블별 F1의 단순 평균 | 모든 레이블이 동등하게 중요할 때 |
| `"weighted"` | 레이블별 **지지도**(양성 샘플 수)로 가중 평균 | 레이블별 빈도 차이를 반영할 때 |

**코드**

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
f1_score(y_multilabel, y_train_knn_pred, average="weighted")
```

**결과**

```
0.9764102655606048   # macro
```

**해석**: macro F1 0.976. 두 레이블의 양성 비율이 비슷해 `"weighted"`와의 차이는 미미하다.

### 3.6.4 `ClassifierChain`

**개념**
- 다중 레이블을 지원하지 않는 모델(예: SVC)을 다중 레이블로 확장하는 방법.
- 단순한 방법은 레이블마다 독립 모델을 두는 것이지만, 이 경우 **레이블 간 의존성**을 활용하지 못한다.

**원리**
- 레이블마다 모델을 하나씩 두되, **사슬 순서대로 앞 모델들의 예측을 다음 모델의 입력 특성에 추가**한다.

```
모델 1:  X                → ŷ₁ (7 이상인가)
모델 2:  [X, ŷ₁]          → ŷ₂ (홀수인가)
모델 k:  [X, ŷ₁, …, ŷₖ₋₁] → ŷₖ
```

- `cv` 지정 시: 학습 단계에서 앞 모델의 출력으로 **실제 레이블 대신 교차 검증 예측**을 사용한다. 예측 단계에서는 실제 레이블을 알 수 없으므로, 학습과 예측의 입력 조건을 일치시키는 것이다.

**코드**

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
chain_clf.predict([some_digit])
```

**결과**

```
array([[0., 1.]])
```

**해석**: 출력은 실수형이지만 `[False, True]`와 같은 의미이며, 정답이다.

### 3.6 정리

| 방법 | 레이블 간 의존성 반영 | 적용 대상 |
|---|---|---|
| 다중 레이블 기본 지원 모델 (KNN 등) | 모델에 따라 다름 | 지원 모델 |
| 레이블별 독립 모델 | ✗ | 모든 이진 분류기 |
| `ClassifierChain` | ✓ (앞 레이블 → 뒤 레이블) | 모든 이진 분류기 |

---

## 3.7 다중 출력 분류

**학습 목표**
- 다중 출력 분류의 정의와 다중 레이블 분류와의 관계를 이해한다.
- 이미지 노이즈 제거를 다중 출력 분류로 구현한다.

### 3.7.1 정의

**개념**
- **다중 출력 분류**(multioutput-multiclass classification): **레이블이 여러 개**이고, **각 레이블이 셋 이상의 값**을 가질 수 있는 분류. 다중 레이블 분류의 일반화다.

| | 레이블 수 | 레이블당 값 |
|---|---|---|
| 다중 레이블 | 여러 개 | 2개 |
| 다중 출력 | 여러 개 | 여러 개 |

### 3.7.2 데이터 구성: 노이즈 추가

**코드**

```python
rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise_train
noise_test = rng.integers(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test

# 그림 3-12: 입력(노이즈) / 타깃(원본)
plt.subplot(121); plot_digit(X_test_mod[0])
plt.subplot(122); plot_digit(y_test_mod[0])
plt.show()
```

| 코드 | 설명 |
|---|---|
| `rng.integers(0, 100, shape)` | [0, 100) 범위의 정수 노이즈를 이미지와 같은 크기로 생성 |
| `X_train_mod = X_train + noise_train` | 입력: 노이즈가 섞인 이미지 |
| `y_train_mod = X_train` | **타깃: 원본 이미지 자체** |
| `subplot(121)` | 1행 2열 중 1번째 칸 |

**해석**
- 타깃의 형태는 `(60000, 784)`다. 즉 **레이블 784개**(픽셀당 하나), 각 레이블의 값은 **0~255**다.
- 따라서 이 문제는 다중 출력 분류에 해당한다.

### 3.7.3 학습과 예측

**코드**

```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)          # (60000, 784) → (60000, 784)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
```

**원리**
- KNN은 입력과 가장 가까운 훈련 샘플 k개(기본 k=5)를 찾고, **각 출력 레이블(픽셀)마다 이웃들의 타깃 값 중 최빈값**을 예측값으로 쓴다.
- 이웃들의 원본 이미지에는 무작위 노이즈가 없으므로, 다수결 결과에서 노이즈가 제거된다.

**결과**: 노이즈가 제거된 숫자 이미지가 출력된다.

**해석**
- 픽셀 값 예측은 회귀에 가까운 성격도 있다. 이처럼 분류와 회귀의 경계가 항상 명확한 것은 아니다.

### 3.7 정리

| 항목 | 내용 |
|---|---|
| 입력 | 노이즈 이미지 `(n, 784)` |
| 출력 | 원본 이미지 `(n, 784)`, 각 값 0~255 |
| 모델 | `KNeighborsClassifier` (다중 출력 기본 지원) |
| 원리 | 픽셀별 이웃 다수결 |

---

## 3.8 장 요약

### 3.8.1 전체 흐름

```
fetch_openml('mnist_784') → 앞 60,000 / 뒤 10,000 분할                  [3.1]
   ↓ y_train == '5'
SGDClassifier().fit(X_train, y_train_5)                                 [3.2]
   ↓ cross_val_predict (예측 / decision_function / predict_proba)
confusion_matrix → precision · recall · F1                              [3.3]
precision_recall_curve / roc_curve → 임곗값 선택, 모델 비교
   ↓ y_train (10개 클래스)
SVC(OvO) / OneVsRestClassifier / SGD(OvR) + StandardScaler              [3.4]
   ↓
ConfusionMatrixDisplay (normalize, sample_weight) → 샘플 관찰 → 원인 → 개선  [3.5]
   ↓
KNeighborsClassifier / ClassifierChain                                  [3.6]
KNeighborsClassifier (노이즈 → 원본)                                    [3.7]
```

### 3.8.2 핵심 개념

| 개념 | 정의 | 절 |
|---|---|---|
| 정확도 | 정답 수 / 전체 수. 불균형 데이터에서 부적절 | 3.3.1 |
| 오차 행렬 | (실제, 예측) 조합별 샘플 수. TN·FP·FN·TP | 3.3.2 |
| 정밀도 | TP / (TP+FP) | 3.3.3 |
| 재현율 (TPR) | TP / (TP+FN) | 3.3.3 |
| F1 | 정밀도와 재현율의 조화평균 | 3.3.3 |
| 결정 임곗값 | 결정 점수가 이 값을 넘으면 양성. 정밀도↔재현율 조절 | 3.3.4 |
| FPR | FP / (FP+TN) | 3.3.5 |
| ROC / AUC | FPR 대 TPR 곡선 / 곡선 아래 면적 | 3.3.5 |
| OvR / OvO | 이진 분류기의 다중 분류 확장 전략 (N개 / N(N−1)/2개) | 3.4.1 |
| 다중 레이블 | 이진 레이블 여러 개 | 3.6 |
| 다중 출력 | 다중 클래스 레이블 여러 개 | 3.7 |

### 3.8.3 주요 결과

| 모델 / 설정 | 지표 | 값 |
|---|---|---|
| 더미 분류기 (5 감지) | 정확도 | 0.910 |
| SGD (5 감지) | 정확도 | 0.950 ~ 0.960 |
| SGD (5 감지) | 정밀도 / 재현율 / F1 | 0.837 / 0.651 / 0.733 |
| SGD, 임곗값 3,370 | 정밀도 / 재현율 | 0.900 / 0.480 |
| SGD (5 감지) | ROC AUC | 0.960 |
| 랜덤 포레스트 (5 감지) | 정밀도 / 재현율 / F1 / AUC | 0.990 / 0.873 / 0.927 / 0.998 |
| SGD (10 클래스) | 정확도 | 약 0.867 |
| SGD (10 클래스) + 표준화 | 정확도 | 약 0.897 |
| KNN (다중 레이블) | F1 (macro) | 0.976 |

### 3.8.4 코드 패턴

| 패턴 | 용도 | 절 |
|---|---|---|
| `model.predict([x])` | 단일 샘플도 2차원으로 입력 | 3.2 |
| `clone(model)` | 학습되지 않은 복제본 생성 | 3.3.1 |
| `cross_val_predict(..., method=...)` | 교차 검증 예측 / 점수 / 확률 | 3.3 |
| `(조건 배열).argmax()` | 조건을 처음 만족하는 인덱스 | 3.3.4, 3.3.5 |
| `scores >= threshold` | 재학습 없이 임곗값 변경 | 3.3.4 |
| `model.classes_[idx]` | 인덱스 → 클래스 레이블 | 3.4.2 |
| `sample_weight=(pred != y)` | 오차 행렬에서 오류만 표시 | 3.5.1 |
| `X[(cond1) & (cond2)]` | 불리언 마스크 필터링 | 3.5.2 |
| `np.c_[a, b]` | 1차원 배열을 열로 결합 | 3.6.2 |

### 3.8.5 확인 문제

<details>
<summary>Q1. 불균형 데이터에서 정확도가 부적절한 이유는?</summary>

다수 클래스만 예측해도 정확도가 높게 나오기 때문이다. MNIST에서 항상 "5 아님"을 예측하는 더미 분류기의 정확도는 91%지만 재현율은 0이다.
</details>

<details>
<summary>Q2. 정밀도와 재현율의 정의와, 각각을 낮추는 오류 유형은?</summary>

정밀도 = TP/(TP+FP), FP가 낮춘다. 재현율 = TP/(TP+FN), FN이 낮춘다.
</details>

<details>
<summary>Q3. F1에 조화평균을 쓰는 이유는?</summary>

조화평균은 작은 값 쪽으로 치우치므로, 두 지표가 모두 높아야 F1이 높다. 한 지표만 높은 분류기를 걸러 낸다.
</details>

<details>
<summary>Q4. 결정 임곗값을 올리면 정밀도와 재현율은 어떻게 변하는가?</summary>

양성 판정이 줄어 FP는 감소, FN은 증가한다. 따라서 정밀도는 대체로 오르고 재현율은 내려간다.
</details>

<details>
<summary>Q5. PR 곡선을 ROC 곡선보다 우선해야 하는 경우는?</summary>

양성 클래스가 드물거나 FP를 FN보다 중시할 때. 음성이 많으면 FPR의 분모(FP+TN)가 커서 ROC 곡선이 낙관적으로 보인다.
</details>

<details>
<summary>Q6. 클래스가 N개일 때 OvR과 OvO의 분류기 수와, SVC가 OvO를 쓰는 이유는?</summary>

OvR은 N개, OvO는 N(N−1)/2개. SVC는 훈련 세트 크기에 따라 학습 비용이 크게 증가하므로, 두 클래스의 샘플만 쓰는 작은 분류기를 여러 개 학습하는 OvO가 유리하다.
</details>

<details>
<summary>Q7. SGD 분류기가 3과 5를 혼동하는 원인과 개선책은?</summary>

선형 모델이 픽셀 위치별 고정 가중치를 사용해 이동·회전에 민감하기 때문이다. 이미지 중앙 정렬·회전 보정, 데이터 증강으로 개선할 수 있다.
</details>

<details>
<summary>Q8. ClassifierChain이 레이블별 독립 모델보다 나은 점과 cv 옵션의 역할은?</summary>

앞 레이블의 예측을 다음 모델의 입력으로 넘겨 레이블 간 의존성을 활용한다. cv를 지정하면 학습 시 실제 레이블 대신 교차 검증 예측을 넘겨, 예측 단계와 같은 조건에서 학습한다.
</details>

<details>
<summary>Q9. 다중 레이블 분류와 다중 출력 분류의 차이는?</summary>

둘 다 레이블이 여러 개다. 다중 레이블은 레이블당 값이 2개(이진), 다중 출력은 레이블당 값이 3개 이상이다.
</details>
