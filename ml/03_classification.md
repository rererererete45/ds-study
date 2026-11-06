# 3장. 분류

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `03_classification.ipynb`
>
> 2장이 숫자를 맞히는 **회귀**였다면 3장은 범주를 맞히는 **분류**. 분류기를 만드는 것보다 **어떻게 평가하느냐**(오차 행렬, 정밀도·재현율, PR·ROC 곡선)에 가장 많은 분량을 쓴다. 이후 이진 → 다중 → 다중 레이블 → 다중 출력으로 문제를 넓혀 간다.
>
> 실습 데이터: MNIST 손글씨 숫자 (70,000장, 28×28 흑백)
>
> | 항목 | 내용 |
> |---|---|
> | `X` (`mnist.data`) | `(70000, 784)` — 이미지 1장 = 픽셀 784개, 값 0(흰색)~255(검정) |
> | `y` (`mnist.target`) | `(70000,)` — 정답 숫자. **문자열** `'0'`~`'9'` |
> | 분할 | 앞 60,000장 훈련 / 뒤 10,000장 테스트 (이미 섞여 있음) |

## 목차

1. [MNIST (3.1)](#1-mnist-31)
2. [이진 분류기 훈련 (3.2)](#2-이진-분류기-훈련-32)
3. [성능 측정 (3.3)](#3-성능-측정-33)
4. [다중 분류 (3.4)](#4-다중-분류-34)
5. [오류 분석 (3.5)](#5-오류-분석-35)
6. [다중 레이블 분류 (3.6)](#6-다중-레이블-분류-36)
7. [다중 출력 분류 (3.7)](#7-다중-출력-분류-37)
8. [전체 흐름 요약](#8-전체-흐름-요약)

---

## 1. MNIST (3.1)

손글씨 숫자 70,000장. 분류 문제의 "Hello World".

### 1-1. 데이터 불러오기

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
mnist.keys()
# dict_keys(['data', 'target', 'frame', 'categories', 'feature_names',
#            'target_names', 'DESCR', 'details', 'url'])

X, y = mnist.data, mnist.target
X.shape   # (70000, 784)
y.shape   # (70000,)
y         # array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)
```

| 줄 | 의미 |
|---|---|
| `fetch_openml(이름)` | OpenML.org에서 데이터셋을 이름으로 받음. 첫 실행 때만 다운로드, 이후 `~/scikit_learn_data` 캐시 사용 |
| `as_frame=False` | DataFrame 대신 **NumPy 배열**로 받음 (이미지는 표로 다룰 이유가 없음) |
| `mnist.data` / `mnist.target` | 반환값은 딕셔너리처럼 쓰는 `Bunch`. 이 장에선 이 둘만 사용 |

> ⚠️ `y`는 숫자가 아니라 **문자열**. 이후 `y_train == '5'`처럼 따옴표로 비교해야 한다.

### 1-2. 이미지 그려 보기

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)   # 1차원 784개 → 2차원 28×28
    plt.imshow(image, cmap="binary")     # binary: 0=흰색, 255=검정
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()

y[0]   # '5'
```

- 모델은 이미지를 **784개 숫자가 한 줄로 늘어선 것**으로 본다. 사람이 보려면 `reshape(28, 28)`로 다시 접어야 함
- `some_digit`(숫자 5)은 이 장 내내 **테스트용 샘플**로 등장

```python
# 그림 3-2: 처음 100장을 10×10 격자로
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)        # 10행 10열 중 (idx+1)번째 칸
    plot_digit(image_data)
plt.subplots_adjust(wspace=0, hspace=0) # 칸 사이 여백 제거
plt.show()
```

### 1-3. 훈련/테스트 분할

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

- MNIST는 **이미 섞여 있고 분할도 정해져 있음** → `train_test_split` 없이 슬라이싱
- 섞여 있다는 점이 교차 검증에서 중요. 정렬돼 있으면 어떤 폴드엔 특정 숫자가 아예 없을 수 있음

---

## 2. 이진 분류기 훈련 (3.2)

문제를 단순화 → **"5인가, 아닌가"**만 구분.

```python
y_train_5 = (y_train == '5')   # 5면 True, 나머지 False
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

sgd_clf.predict([some_digit])  # array([ True])  → 맞춤
```

| 줄 | 의미 |
|---|---|
| `y_train == '5'` | 비교 연산 → **불리언 배열**. 10개 클래스 문제가 이진 문제로 바뀜 |
| `SGDClassifier` | 확률적 경사 하강법. 샘플을 무작위로 하나씩 보며 가중치를 갱신 → 대용량·온라인 학습에 유리 |
| `random_state=42` | SGD는 무작위성이 있으므로 재현을 위해 고정 |
| `predict([some_digit])` | `predict`는 **2차원 입력**을 기대 → 샘플 1개도 리스트로 감쌈 |

> 💡 기본 손실이 `hinge` → **선형 SVM**과 같은 모델. 픽셀 784개에 가중치를 곱해 더하는 **선형 결합**으로 판단한다. 이 성질이 5절 오류 분석에서 다시 등장.

---

## 3. 성능 측정 (3.3)

### 3-1. 교차 검증으로 정확도 측정

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.95035, 0.96035, 0.9604 ])
```

- 3등분 → 2개로 학습, 1개로 평가 × 3회
- 95% 이상으로 좋아 보이지만… (3-2에서 반전)

#### 교차 검증 직접 구현

`cross_val_score`가 내부에서 하는 일:

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3)  # 데이터가 안 섞여 있으면 shuffle=True 추가

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
# 0.95035
# 0.96035
# 0.9604
```

| 부분 | 의미 |
|---|---|
| `StratifiedKFold` | 폴드마다 **클래스 비율을 전체와 같게** 유지 (5가 약 10%면 모든 폴드에 약 10%) |
| `.split(X, y)` | 회차마다 (학습 인덱스, 평가 인덱스)를 내놓는 제너레이터. 비율을 맞추려면 `y` 필요 |
| `clone(sgd_clf)` | **하이퍼파라미터만 같고 학습 안 된** 복제본. 회차마다 새로 시작해야 이전 학습이 섞이지 않음 |
| `sum(y_pred == y_test_fold)` | True=1로 더해져 맞힌 개수가 됨 |

→ 결과가 `cross_val_score`와 똑같음.

### 3-2. 정확도의 함정: 더미 분류기

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))   # False

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.90965, 0.90965, 0.90965])
```

- `DummyClassifier` 기본 전략(`"prior"`) → 입력을 보지 않고 **가장 많은 클래스만 예측** = 항상 False
- `any(...)`가 False → 60,000개 예측 중 True가 **하나도 없음**
- 그런데 정확도 **91%**. 5가 아닌 이미지가 약 91%이기 때문

> ⚠️ **불균형 데이터에서 정확도는 좋은 지표가 아니다.** SGD의 95%는 "아무것도 안 하는 모델"보다 4%p 나을 뿐.

### 3-3. 오차 행렬

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

- `cross_val_score` → **점수**를, `cross_val_predict` → **예측값 자체**를 반환
- 각 샘플의 예측 = **그 샘플을 학습에 쓰지 않은 모델**이 낸 값 → 테스트 세트를 건드리지 않고 "처음 보는 데이터에 대한 예측"을 얻음

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)   # (정답, 예측) 순서
cm
# array([[53892,   687],
#        [ 1891,  3530]])
```

행 = **실제**, 열 = **예측** (False가 0번, True가 1번)

| | 예측 False | 예측 True |
|---|---|---|
| **실제 False** | TN = 53,892 `cm[0,0]` | FP = 687 `cm[0,1]` |
| **실제 True** | FN = 1,891 `cm[1,0]` | TP = 3,530 `cm[1,1]` |

- **TN** 5 아닌 걸 아니라고 맞힘 / **FP** 5 아닌 걸 5라고 함 (1종 오류)
- **FN** 5를 놓침 (2종 오류) / **TP** 5를 5라고 맞힘

```python
y_train_perfect_predictions = y_train_5   # 완벽한 모델이라고 가정
confusion_matrix(y_train_5, y_train_perfect_predictions)
# array([[54579,     0],
#        [    0,  5421]])
```

완벽한 분류기는 **대각선에만 값**이 있다.

### 3-4. 정밀도·재현율·F1

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision_score(y_train_5, y_train_pred)   # 0.8370879772350012  == 3530 / (687 + 3530)
cm[1, 1] / (cm[0, 1] + cm[1, 1])            # 직접: TP / (FP + TP)

recall_score(y_train_5, y_train_pred)      # 0.6511713705958311  == 3530 / (1891 + 3530)
cm[1, 1] / (cm[1, 0] + cm[1, 1])            # 직접: TP / (FN + TP)

f1_score(y_train_5, y_train_pred)          # 0.7325171197343847
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)   # 직접: TP / (TP + (FN+FP)/2)
```

| 지표 | 공식 | 의미 | 결과 |
|---|---|---|---|
| **정밀도** (precision) | TP / (TP+FP) | "5라고 한 것 중 진짜 5의 비율" | 83.7% |
| **재현율** (recall, TPR) | TP / (TP+FN) | "진짜 5 중 찾아낸 비율" | 65.1% |
| **F1** | 2 / (1/P + 1/R) | 정밀도·재현율의 **조화평균** | 73.3% |

- 조화평균은 **작은 값에 끌려감** → 둘 다 높아야 F1이 높다
- F1 공식을 정리하면 `TP / (TP + (FN+FP)/2)` → 오차 행렬에서 바로 계산 가능

### 3-5. 정밀도/재현율 트레이드오프

#### ① 결정 점수와 임곗값

```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores                                     # array([2164.22030239])

threshold = 0
y_some_digit_pred = (y_scores > threshold)   # array([ True])
y_scores > 0                                 # predict()와 동일

threshold = 3000
y_some_digit_pred = (y_scores > threshold)   # array([False])  → 5를 놓침
```

- `predict()` = ① **결정 점수**(가중치·입력 내적 + 절편) 계산 → ② **점수 > 0이면 True**
- `decision_function()` = ②를 건너뛰고 **점수만** 반환 → 임곗값을 직접 정할 수 있음
- 임곗값 3000으로 올리면 점수 2164인 5를 놓침(FN) → **임곗값↑ = 재현율↓**

#### ② 모든 임곗값에서의 정밀도·재현율

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

- `method="decision_function"` → 예측값 대신 **결정 점수** 60,000개를 받음
- `precision_recall_curve` → 가능한 모든 임곗값의 정밀도·재현율을 한 번에

> ⚠️ `precisions`·`recalls`는 `thresholds`보다 **1개 더 길다**(맨 끝에 정밀도=1, 재현율=0인 점이 붙음). 그래서 그래프에서 `[:-1]`로 자름.

```python
# 그림 3-5: 임곗값에 따른 정밀도·재현율
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")   # 3000에 세로선

idx = (thresholds >= threshold).argmax()   # 3000 이상인 첫 인덱스
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid(); plt.xlabel("Threshold"); plt.legend(loc="center right")
plt.show()
```

> 💡 **`(조건 배열).argmax()`** → 처음으로 True가 나오는 인덱스. True=1이 최댓값이고 동률이면 첫 위치를 반환하기 때문. 이 장에서 계속 쓰는 관용구.

- 재현율은 임곗값을 올릴수록 **매끄럽게** 감소
- 정밀도는 대체로 오르지만 **울퉁불퉁** — 임곗값을 올렸는데 TP가 하나 빠지면 잠깐 떨어질 수 있음

```python
# 그림 3-6: PR 곡선 (x=재현율, y=정밀도)
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
```

→ 재현율 80% 근처부터 정밀도가 급락. 그 직전이 좋은 절충점.

#### ③ 목표 정밀도에 맞는 임곗값 찾기

```python
idx_for_90_precision = (precisions >= 0.90).argmax()   # 정밀도 90%를 처음 넘는 위치
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision                  # 3370.0194991439557

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

precision_score(y_train_5, y_train_pred_90)  # 약 0.90
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision                       # 0.4799852425751706
```

- 모델 재학습 없이 **임곗값만 바꿔** 정밀도 90% 분류기 완성
- 대가: **재현율 48%** — 5의 절반 이상을 놓침

> 💡 "정밀도 99% 달성하자"는 말을 들으면 **"재현율은 얼마에서?"**라고 되물을 것.
>
> | 상황 | 중요한 쪽 |
> |---|---|
> | 아이용 안전 동영상 필터 (나쁜 영상은 절대 통과 X) | 정밀도 |
> | 좀도둑 감지 CCTV (오경보 OK, 도둑은 놓치면 안 됨) | 재현율 |

### 3-6. ROC 곡선

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

- **FPR** (거짓 양성 비율) = FP / (FP + TN) — 음성 중 양성으로 잘못 판정한 비율
- **TPR** = 재현율

> ⚠️ `roc_curve`의 `thresholds`는 **내림차순** (`precision_recall_curve`는 오름차순).

```python
# 그림 3-7: ROC 곡선 + 정밀도 90% 지점 표시
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

- 내림차순이므로 `<=`로 첫 True를 찾음 (PR 곡선 때 `>=`와 방향 반대)
- **대각선** = 무작위 분류기. 좋은 분류기일수록 **왼쪽 위 구석**에 붙음

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)   # 0.9604938554008616
```

- **AUC** (곡선 아래 면적): 1 = 완벽, 0.5 = 무작위

> 💡 **PR vs ROC**: 양성이 드물거나 FP가 더 신경 쓰이면 **PR 곡선**. 이번처럼 음성이 압도적으로 많으면 TN이 커서 FPR이 작게 나오고 ROC가 실제보다 좋아 보인다. AUC 0.96은 훌륭해 보이지만 PR 곡선을 보면 개선 여지가 많음.

### 3-7. 랜덤 포레스트와 비교

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_probas_forest[:2]
# array([[0.11, 0.89],
#        [0.99, 0.01]])
```

- 랜덤 포레스트엔 `decision_function()`이 **없음** → `predict_proba()`로 **클래스별 확률**
- 각 행 = `[False 확률, True 확률]`. 첫 이미지(5)는 89% 확률로 5, 둘째(0)는 1%

```python
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")
# 94.0%
```

- 양성 확률이 50~60%라고 나온 샘플 중 실제 양성 비율 → 확률이 정확하다면 ~55%여야 하는데 **94%**
- → 이 확률은 **추정치**일 뿐 보정(calibration)된 확률이 아님

```python
y_scores_forest = y_probas_forest[:, 1]   # 양성 확률을 '점수'로 사용
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

# 그림 3-8: PR 곡선 비교
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
```

→ 랜덤 포레스트 곡선이 오른쪽 위 구석에 훨씬 가까움.

```python
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5   # 확률 50% 이상 → 양성
f1_score(y_train_5, y_train_pred_forest)          # 0.9274509803921569
roc_auc_score(y_train_5, y_scores_forest)         # 0.9983436731328145
precision_score(y_train_5, y_train_pred_forest)   # 0.9897468089558485
recall_score(y_train_5, y_train_pred_forest)      # 0.8725327430363402
```

- `cross_val_predict`를 다시 돌리면 몇 분 걸림 → 이미 구한 확률에 **임곗값 0.5만 적용**하면 같은 예측을 즉시 얻음

| | SGD | 랜덤 포레스트 |
|---|---|---|
| 정밀도 | 0.837 | **0.990** |
| 재현율 | 0.651 | **0.873** |
| F1 | 0.733 | **0.927** |
| ROC AUC | 0.960 | **0.998** |

---

## 4. 다중 분류 (3.4)

0~9 열 개 클래스를 모두 구분.

- 로지스틱 회귀, 랜덤 포레스트 등 → **원래 다중 분류 지원**
- SGD, SVM 등 이진 전용 → 두 전략으로 확장

| 전략 | 방법 | 분류기 수 (10 클래스) |
|---|---|---|
| **OvR** (One-versus-Rest) | "5 vs 나머지"를 클래스마다 → 점수 최고 클래스 선택 | 10개 |
| **OvO** (One-versus-One) | "0 vs 1", "0 vs 2", … 모든 쌍 → 가장 많이 이긴 클래스 | N(N−1)/2 = **45개** |

사이킷런은 이진 분류기에 다중 클래스 타깃이 들어오면 **자동으로 OvR 또는 OvO**를 적용.

### 4-1. SVC — 자동으로 OvO

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])   # y_train_5가 아니라 y_train!

svm_clf.predict([some_digit])                 # array(['5'], dtype=object)  → 정답
```

- 타깃이 `y_train` = **'0'~'9' 10개 클래스**
- SVC는 데이터가 커지면 매우 느려짐 → **2,000개만** 사용. 쌍마다 작은 데이터만 쓰는 OvO가 유리

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)
# array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,  4.82]])
#          '0'    '1'    '2'    '3'    '4'    '5'    '6'    '7'    '8'    '9'

class_id = some_digit_scores.argmax()   # 5
svm_clf.classes_                        # array(['0', '1', ..., '9'], dtype=object)
svm_clf.classes_[class_id]              # '5'
```

- 클래스마다 점수 1개씩 10개. '5'가 9.3으로 최고
- 이 점수 ≈ **이긴 대결 수 + 신뢰도 기반 작은 보정값**. '5'는 9번 대결 전승

> ⚠️ `argmax()`는 **인덱스**를 반환. 여기선 클래스가 '0'~'9' 순서라 인덱스 5 = 레이블 '5'로 우연히 일치. 일반적으로는 반드시 `classes_[인덱스]`로 변환.

```python
svm_clf.decision_function_shape = "ovo"
some_digit_scores_ovo = svm_clf.decision_function([some_digit])
some_digit_scores_ovo.round(2)
# 45개 값: [[ 0.11, -0.21, -0.97,  0.51, ... ]]   (0vs1, 0vs2, ..., 8vs9)
```

- 기본값 `"ovr"`은 45개 점수를 클래스별로 **모아서 보여 줄 뿐**. **학습은 항상 OvO**
- `"ovo"`로 바꾸면 45개 쌍별 점수를 그대로 볼 수 있음

### 4-2. OvR 강제하기

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])   # array(['5'], dtype='<U1')
len(ovr_clf.estimators_)        # 10
```

- 모델을 `OneVsRestClassifier`로 감싸면 "0 vs 나머지" … 이진 분류기 **10개**
- 학습된 모델 목록은 `estimators_`에 (OvO를 강제하려면 `OneVsOneClassifier`)

### 4-3. SGD 다중 분류 (OvR)

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])   # array(['3'], dtype='<U1')  ← 틀림!

sgd_clf.decision_function([some_digit]).round()
# array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
#         -16148.,  -4604., -12051.]])
#           '0'      '1'      '2'      '3'      '4'      '5'   ...
```

- SGD는 내부적으로 OvR → 이진 분류기 10개가 각자 점수
- 양수는 **'3'(1824)**뿐, 정답 **'5'(−1386)**가 2위 → **3과 5를 헷갈림** (5절에서 분석)

### 4-4. 평가와 스케일링

```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([0.87365, 0.85835, 0.8689 ])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# array([0.8983, 0.891 , 0.9018])
```

- 10 클래스 무작위 = 10% → 87%면 나쁘지 않음
- `StandardScaler` → 픽셀마다 평균 0, 표준편차 1
- `.astype("float64")` → 실수형으로 바꿔 변환 결과의 소수점을 정확히 담음
- **경사 하강법 계열은 스케일에 민감** → 스케일링만으로 87% → **90%**

---

## 5. 오류 분석 (3.5)

좋은 후보 모델을 찾았다고 가정하고, **어떤 오류를 내는지** 분석해 개선 방향을 찾는 단계.

### 5-1. 오차 행렬 세 가지 버전

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

# (a) 원본 개수
plt.rc('font', size=9)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()

# (b) 행 기준 정규화
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()

# (c) 오류만 남기기
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show()
```

| 버전 | 코드 포인트 | 읽을 것 |
|---|---|---|
| (a) 원본 | `from_predictions(정답, 예측)` → 계산 + 색칠 그림까지 한 번에 | 대각선이 진함 → 대체로 잘 맞힘. '5' 칸이 약간 옅음 — 5를 못 맞혀서인지, 5 샘플이 적어서인지 **구분 불가** |
| (b) 정규화 | `normalize="true"` → **행(실제 클래스) 합 = 1**. `"pred"`는 열 기준, `"all"`은 전체 기준. `values_format=".0%"` → 정수 퍼센트 표시 | 샘플 수 차이가 사라짐. 실제 5 중 약 82%만 맞히고 **약 10%를 8로** 분류 |
| (c) 오류만 | `sample_weight = (pred != y)` → 맞힌 샘플 가중치 0 → **대각선이 0**이 되고 오류만 남음 | **'8' 열이 유독 밝음** → 여러 숫자가 8로 잘못 분류됨 |

> ⚠️ 행 기준이냐 열 기준이냐에 따라 해석이 달라진다. 행 기준에서 큰 "5→3" 오류가 열 기준(`"pred"`)에선 다르게 보일 수 있음.

### 5-2. 3과 5 혼동 사례 직접 보기

```python
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]   # 실제 3, 예측 3 (정답)
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]   # 실제 3, 예측 5 (오류)
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]   # 실제 5, 예측 3 (오류)
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]   # 실제 5, 예측 5 (정답)
```

- 두 조건을 묶은 **불리언 마스크**로 이미지를 골라냄

> ⚠️ `and`가 아니라 **`&`**, 각 조건은 **괄호로 감싸기** (연산자 우선순위 때문).

```python
# 그림 3-11: 네 그룹을 2×2 블록으로 (블록당 5×5 = 25장)
size = 5
pad = 0.2
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
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")   # 세로 구분선
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")   # 가로 구분선
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.show()
```

| 줄 | 의미 |
|---|---|
| `idx % size` / `idx // size` | 블록 안 열 번호(0~4) / 행 번호(0~4) |
| `label_col * (size + pad)` | 블록 단위 오프셋 (0이면 왼쪽·아래, 1이면 오른쪽·위) |
| `imshow(..., extent=(왼, 오, 아래, 위))` | 이미지를 **원하는 좌표 칸에 직접 배치**. subplot 100개보다 가벼움 |

**해석**
- 잘못 분류된 것 중 상당수는 사람에게는 명백함
- 3과 5의 차이 = **위 가로선과 아래 호를 잇는 짧은 세로선의 위치**뿐
- SGD는 **선형 모델** → 픽셀마다 고정 가중치뿐이라 이미지가 조금 이동·회전해도 쉽게 헷갈림

**개선 방향**
- 이미지 **중앙 정렬·회전 보정** 전처리
- 이동·회전시킨 이미지로 **데이터 증강** (연습문제 2번)
- 8 오분류를 줄일 특성 추가 (예: 닫힌 고리 개수)

---

## 6. 다중 레이블 분류 (3.6)

샘플 하나에 **여러 개의 True/False 레이블**을 동시에 출력. 예: 사진 한 장에서 앨리스·밥·찰리가 각각 있는지.

### 6-1. KNN으로 레이블 2개 학습

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')                  # 레이블1: 7, 8, 9인가?
y_train_odd = (y_train.astype('int8') % 2 == 1)   # 레이블2: 홀수인가?
y_multilabel = np.c_[y_train_large, y_train_odd]  # (60000, 2)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])   # array([[False,  True]])  → 5는 '크지 않고 홀수'
```

| 줄 | 의미 |
|---|---|
| `y_train >= '7'` | **문자열 비교**지만 한 글자 숫자라 사전순 = 크기순 |
| `.astype('int8')` | `%` 연산을 위해 정수로. 0~9는 `int8`(−128~127)로 충분 |
| `np.c_[a, b]` | 1차원 배열 두 개를 **열 방향으로 결합** → 행마다 `[큰 수?, 홀수?]` |
| `KNeighborsClassifier` | 다중 레이블을 **기본 지원** → 2열 타깃을 그대로 학습 |

### 6-2. 평가

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")      # 0.9764102655606048
f1_score(y_multilabel, y_train_knn_pred, average="weighted")
```

- 레이블별 F1을 따로 구해 평균
- `"macro"` → 모든 레이블 **동일 비중**
- `"weighted"` → 레이블별 **지지도(양성 샘플 수)로 가중**. 이 데이터는 두 레이블 균형이 비슷해 차이가 거의 없음

### 6-3. ClassifierChain — 다중 레이블 미지원 모델용

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
chain_clf.predict([some_digit])   # array([[0., 1.]])
```

- SVC는 다중 레이블 미지원. 레이블마다 독립 모델을 두면 **레이블 간 관계를 못 씀**
- `ClassifierChain` → 모델을 **사슬처럼 연결**
  - 모델 1: `X` → "큰 수인가?"
  - 모델 2: `X` + **모델 1의 예측** → "홀수인가?"
- `cv=3` → 학습 때 앞 모델 출력으로 **정답 대신 교차 검증 예측**을 넘김. 실제 예측 땐 정답을 모르니 학습도 같은 조건으로 맞춰 안정성↑
- 결과가 `[0., 1.]` **실수형**이지만 의미는 `[False, True]`

---

## 7. 다중 출력 분류 (3.7)

**다중 레이블의 일반화** — 레이블이 여러 개이고, **각 레이블이 값을 여러 개** 가질 수 있음. 예제: **이미지 노이즈 제거기**.

### 7-1. 노이즈 섞인 데이터 만들기

```python
rng = np.random.default_rng(seed=42)
noise_train = rng.integers(0, 100, (len(X_train), 784))   # 0~99 정수, (60000, 784)
X_train_mod = X_train + noise_train                        # 입력: 원본 + 노이즈
noise_test = rng.integers(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train                                      # 타깃: 깨끗한 원본
y_test_mod = X_test

# 그림 3-12: 왼쪽 노이즈 입력 / 오른쪽 깨끗한 타깃
plt.subplot(121); plot_digit(X_test_mod[0])
plt.subplot(122); plot_digit(y_test_mod[0])
plt.show()
```

- `rng.integers(low, high, size)` → `low` 이상 `high` **미만** 정수
- **타깃이 이미지 자체(784개 값)** → 레이블 784개(픽셀당 1개) × 각 값 0~255 = **다중 출력 분류**
- `subplot(121)` = 1행 2열 중 1번째 (축약 표기)

### 7-2. 학습과 복원

```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)           # (60000, 784) → (60000, 784)
clean_digit = knn_clf.predict([X_test_mod[0]])
plot_digit(clean_digit)
plt.show()
```

- KNN이 노이즈 이미지와 가장 가까운 훈련 샘플 **5개**(`n_neighbors=5` 기본)를 찾고, **픽셀마다 이웃들의 값을 다수결**
- 결과: 노이즈가 사라진 깨끗한 숫자

> 💡 픽셀값 예측은 회귀에 더 가까워 보이기도 한다. 분류와 회귀의 경계가 늘 명확하진 않음.

---

## 8. 전체 흐름 요약

```
fetch_openml('mnist_784')                       ← 3.1 데이터 수집, 앞 60k / 뒤 10k 분할
   ↓ y_train == '5'
SGDClassifier().fit(X_train, y_train_5)         ← 3.2 이진 분류
   ↓ cross_val_predict (예측 / decision_function / predict_proba)
confusion_matrix → 정밀도·재현율·F1             ← 3.3 정확도 대신 쓸 지표
precision_recall_curve / roc_curve → 임곗값 선택
   ↓ y_train (10 클래스)
SVC(OvO) / OneVsRestClassifier / SGD(OvR)       ← 3.4 다중 분류 + StandardScaler
   ↓
ConfusionMatrixDisplay (normalize, sample_weight) ← 3.5 오류 패턴 찾기 → 3 vs 5, 8 열
   ↓
KNN / ClassifierChain                            ← 3.6 다중 레이블
KNN (노이즈 → 원본)                              ← 3.7 다중 출력
```

### 분류 유형

| 유형 | 출력 형태 | 예시 | 사용 모델·도구 |
|---|---|---|---|
| 이진 분류 | 레이블 1개, 값 2개 | 5인가? | `SGDClassifier`, `RandomForestClassifier` |
| 다중 분류 | 레이블 1개, 값 N개 | 0~9 중 무엇? | `SVC`(OvO), `OneVsRestClassifier`, SGD(OvR) |
| 다중 레이블 | 레이블 여러 개, 각 값 2개 | 큰 수? 홀수? | `KNeighborsClassifier`, `ClassifierChain` |
| 다중 출력 | 레이블 여러 개, 각 값 여러 개 | 픽셀 784개 복원 | `KNeighborsClassifier` |

### 주요 결과

| 모델 / 설정 | 지표 | 값 |
|---|---|---|
| 더미 (항상 False) | 정확도 (CV) | 0.910 |
| SGD (5 감지) | 정확도 (CV) | 0.950 ~ 0.960 |
| SGD (5 감지) | 정밀도 / 재현율 / F1 | 0.837 / 0.651 / 0.733 |
| SGD, 임곗값 3,370 | 정밀도 / 재현율 | 0.90 / 0.480 |
| SGD (5 감지) | ROC AUC | 0.960 |
| **랜덤 포레스트 (5 감지)** | 정밀도 / 재현율 / F1 / AUC | **0.990 / 0.873 / 0.927 / 0.998** |
| SGD 10 클래스 | 정확도 (CV) | 0.86 ~ 0.87 |
| SGD 10 클래스 + 스케일링 | 정확도 (CV) | 0.89 ~ 0.90 |
| KNN 다중 레이블 | F1 (macro) | 0.976 |

### 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `model.predict([x])` | 샘플 1개도 2차원으로 감싸 입력 | 전체 |
| `cross_val_predict(..., method="decision_function")` | 교차 검증 **점수** | 3-5 |
| `cross_val_predict(..., method="predict_proba")` | 교차 검증 **확률** | 3-7 |
| `(조건 배열).argmax()` | 조건을 처음 만족하는 인덱스 | 3-5, 3-6 |
| `scores >= 임곗값` | 재학습 없이 임곗값 조정 | 3-5 |
| `model.classes_[idx]` | 인덱스 → 클래스 이름 | 4-1 |
| `sample_weight=(pred != y)` | 오차 행렬에서 오류만 남기기 | 5-1 |
| `X[(cond1) & (cond2)]` | 불리언 마스크 필터링 | 5-2 |
| `np.c_[a, b]` | 1차원 배열을 열로 합쳐 다중 레이블 | 6-1 |
