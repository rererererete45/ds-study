# 3장. 분류

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `03_classification.ipynb`
>
> 2장에서는 집값 같은 **숫자**를 맞혔어요(회귀). 3장에서는 "이 사진은 어떤 숫자인가?"처럼 **정해진 보기 중 하나를 고르는 문제**(분류)를 다룹니다.
>
> 분류기를 만드는 코드는 사실 몇 줄이면 끝나요. 이 장에서 진짜 시간을 많이 쓰는 부분은 **"이 분류기가 정말 잘하는 걸까?"를 따지는 방법**이에요. 정확도 하나만 보면 왜 속을 수 있는지, 대신 무엇을 봐야 하는지를 차근차근 배웁니다.
>
> **실습 데이터: MNIST 손글씨 숫자**
>
> | 항목 | 내용 |
> |---|---|
> | `X` (`mnist.data`) | `(70000, 784)` — 이미지 70,000장. 한 장은 28×28 = **픽셀 784개**, 각 값은 0(흰색)~255(검정) |
> | `y` (`mnist.target`) | `(70000,)` — 정답 숫자. 숫자가 아니라 **문자열** `'0'`~`'9'` |
> | 분할 | 앞 60,000장은 훈련용, 뒤 10,000장은 테스트용 (이미 잘 섞여 있음) |

## 목차

1. [MNIST (3.1)](#1-mnist-31)
2. [이진 분류기 훈련 (3.2)](#2-이진-분류기-훈련-32)
3. [성능 측정 (3.3)](#3-성능-측정-33)
4. [다중 분류 (3.4)](#4-다중-분류-34)
5. [오류 분석 (3.5)](#5-오류-분석-35)
6. [다중 레이블 분류 (3.6)](#6-다중-레이블-분류-36)
7. [다중 출력 분류 (3.7)](#7-다중-출력-분류-37)
8. [전체 흐름 요약](#8-전체-흐름-요약)
9. [스스로 확인해 보기](#9-스스로-확인해-보기)

### 이 장에서 처음 나오는 용어

먼저 훑어보고, 본문에서 막히면 다시 돌아와서 확인하세요.

| 용어 | 쉬운 뜻 |
|---|---|
| **분류**(classification) | 정해진 보기(클래스) 중 하나를 고르는 문제. 예: 스팸/정상, 0~9 |
| **클래스**(class) | 고를 수 있는 보기 하나하나. MNIST는 '0'~'9' 열 개 |
| **이진 분류** | 보기가 딱 두 개인 분류. 예: "5다 / 5가 아니다" |
| **양성 / 음성** | 이진 분류에서 우리가 찾고 싶은 쪽이 양성(5), 나머지가 음성(5 아님) |
| **결정 점수** | 모델이 "이건 양성 같다"고 느끼는 정도를 숫자로 나타낸 것. 클수록 양성 쪽 |
| **임곗값**(threshold) | 결정 점수가 몇 점 이상이면 양성으로 볼지 정하는 합격선 |
| **교차 검증** | 훈련 데이터를 여러 조각으로 나눠 "한 조각은 시험, 나머지는 공부"를 돌아가며 반복하는 평가 방법 |

---

## 1. MNIST (3.1)

> **이 절에서 하는 일**: 데이터를 불러와서 생김새를 확인하고, 훈련용과 테스트용으로 나눕니다.

MNIST는 고등학생과 미국 인구조사국 직원들이 손으로 쓴 숫자 이미지 70,000장이에요. 너무 유명해서 머신러닝 분야의 "Hello World"라고 불립니다. 새 분류 알고리즘이 나오면 일단 MNIST로 시험해 보는 경우가 많아요.

### 1-1. 데이터 불러오기

```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
```

- `fetch_openml`은 공개 데이터 저장소(OpenML.org)에서 **이름만으로 데이터를 받아 오는** 함수예요. 처음 실행할 때만 내려받고, 다음부터는 컴퓨터에 저장된 사본(`~/scikit_learn_data`)을 씁니다.
- `as_frame=False`는 "판다스 표(DataFrame) 말고 **넘파이 배열**로 주세요"라는 뜻이에요. 이미지는 표 형태로 볼 일이 없어서 배열이 더 편합니다.

받아 온 `mnist`는 딕셔너리처럼 쓸 수 있는 객체예요. 안에 뭐가 들었는지 열어 볼게요.

```python
mnist.keys()
# dict_keys(['data', 'target', 'frame', 'categories', 'feature_names',
#            'target_names', 'DESCR', 'details', 'url'])
```

여러 항목이 있지만 이 장에서는 **`data`(문제)와 `target`(정답)** 두 개만 씁니다.

```python
X, y = mnist.data, mnist.target
X.shape   # (70000, 784)
y.shape   # (70000,)
y         # array(['5', '0', '4', ..., '4', '5', '6'], dtype=object)
```

- `X`는 70,000행 × 784열짜리 표라고 생각하면 돼요. **한 행이 이미지 한 장**이고, 784개 열은 각 픽셀의 진하기예요.
- 왜 784일까요? 이미지가 가로 28칸 × 세로 28칸이라서 28 × 28 = 784입니다. 2차원 그림을 **한 줄로 길게 펼쳐 놓은** 거예요.

> ⚠️ **자주 하는 실수**: `y`의 값은 숫자 5가 아니라 **문자열 `'5'`**예요. 그래서 뒤에서 `y_train == 5`라고 쓰면 전부 False가 나와요. 반드시 `y_train == '5'`처럼 따옴표를 붙여야 합니다.

### 1-2. 이미지 그려 보기

숫자 784개만 봐서는 무슨 그림인지 알 수 없어요. 다시 28×28로 접어서 그려 봅시다.

```python
import matplotlib.pyplot as plt

def plot_digit(image_data):
    image = image_data.reshape(28, 28)   # 한 줄(784) → 28행 28열로 다시 접기
    plt.imshow(image, cmap="binary")     # 흑백으로 그리기 (0=흰색, 255=검정)
    plt.axis("off")                       # 눈금은 필요 없으니 숨기기

some_digit = X[0]
plot_digit(some_digit)
plt.show()

y[0]   # '5'
```

- `reshape(28, 28)`은 784개를 **28개씩 끊어서 28줄로** 다시 배치하는 함수예요. 펼쳐 놓은 종이를 다시 접는 느낌입니다.
- 첫 번째 이미지는 숫자 5예요. 이 이미지를 `some_digit`이라는 이름으로 저장해 두는데, **이 장 내내 "모델이 이걸 맞히나?" 시험해 보는 샘플**로 계속 등장합니다.

여러 장을 한꺼번에 보고 싶으면 이렇게 할 수 있어요.

```python
# 그림 3-2: 처음 100장을 10×10 격자로
plt.figure(figsize=(9, 9))
for idx, image_data in enumerate(X[:100]):
    plt.subplot(10, 10, idx + 1)        # 10행 10열 격자 중 (idx+1)번째 칸에
    plot_digit(image_data)               # 이미지 하나 그리기
plt.subplots_adjust(wspace=0, hspace=0) # 칸 사이 여백 없애기
plt.show()
```

그려 보면 같은 5라도 사람마다 쓰는 모양이 꽤 다르다는 걸 알 수 있어요. 이 다양성이 분류를 어렵게 만드는 이유예요.

### 1-3. 훈련 세트와 테스트 세트 나누기

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

2장에서는 `train_test_split`으로 무작위로 나눴는데, MNIST는 그럴 필요가 없어요. 데이터를 만든 사람들이 **이미 잘 섞어 두었고, 앞 60,000장은 훈련용 / 뒤 10,000장은 테스트용**이라고 정해 두었거든요. 그래서 앞뒤로 잘라 쓰기만 하면 됩니다.

> 💡 "이미 섞여 있다"는 점이 왜 중요할까요? 만약 데이터가 0, 0, 0, …, 1, 1, 1, … 순서로 정렬돼 있다면, 앞에서부터 잘랐을 때 어떤 조각에는 9가 하나도 없을 수 있어요. 그러면 공정한 평가가 안 되겠죠. 뒤에 나오는 교차 검증에서 특히 중요한 이야기입니다.

---

## 2. 이진 분류기 훈련 (3.2)

> **이 절에서 하는 일**: 처음부터 0~9를 다 구분하면 어려우니까, 일단 **"이 숫자가 5인가, 아닌가?"**만 맞히는 간단한 분류기를 만들어 봅니다.

### 2-1. 정답을 True/False로 바꾸기

```python
y_train_5 = (y_train == '5')   # 5면 True, 나머지 숫자는 전부 False
y_test_5 = (y_test == '5')
```

`y_train == '5'`는 6만 개 정답 하나하나에 "이거 '5'야?"라고 물어보는 거예요. 그 결과 `[True, False, False, False, …]` 같은 **참/거짓 배열**이 만들어집니다. 이걸 새 정답으로 쓰면 10개 중 하나를 고르는 문제가 **둘 중 하나를 고르는 문제**(이진 분류)로 바뀌어요.

### 2-2. 분류기 학습시키기

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
```

**SGDClassifier**는 "확률적 경사 하강법(Stochastic Gradient Descent) 분류기"의 줄임말이에요. 이름은 어렵지만 동작은 이렇게 이해하면 됩니다.

- 훈련 데이터에서 **샘플을 무작위로 하나씩 꺼내서**, 틀렸으면 틀린 만큼 조금씩 고쳐 나가요.
- 한 번에 하나씩 보기 때문에 데이터가 아주 많아도 감당할 수 있고, 새 데이터가 계속 들어오는 상황(온라인 학습)에도 잘 맞아요.
- "무작위로 꺼낸다"는 과정이 있어서 실행할 때마다 결과가 조금씩 달라질 수 있어요. `random_state=42`로 **무작위 순서를 고정**하면 매번 같은 결과가 나옵니다. (42는 아무 숫자나 써도 되는데, 관습적으로 많이 써요.)

> 💡 SGDClassifier는 기본적으로 **선형 모델**이에요. 픽셀 784개에 각각 "이 픽셀이 진하면 5일 가능성이 이만큼 올라간다/내려간다"는 가중치를 매기고, 그걸 다 더해서 판단합니다. 이 성질 때문에 5절에서 약점이 드러나니 기억해 두세요.

### 2-3. 시험해 보기

```python
sgd_clf.predict([some_digit])   # array([ True])
```

`some_digit`(숫자 5)을 넣었더니 True, 즉 "5가 맞다"고 했어요. 첫 시험은 통과입니다!

> ⚠️ **자주 하는 실수**: `predict(some_digit)`처럼 쓰면 에러가 나요. `predict`는 **여러 샘플을 한꺼번에** 받도록 만들어져 있어서 입력이 2차원이어야 하거든요. 샘플이 하나뿐이어도 `[some_digit]`처럼 대괄호로 한 번 감싸서 "샘플 1개짜리 묶음"으로 만들어 줘야 합니다.

---

## 3. 성능 측정 (3.3)

> **이 절에서 하는 일**: 이 장의 핵심이에요. "정확도 95%"가 왜 믿을 수 없는 숫자인지 확인하고, 대신 쓸 수 있는 평가 도구를 하나씩 배웁니다.
>
> **흐름**: 정확도의 함정 → 오차 행렬 → 정밀도·재현율 → 둘 사이의 줄다리기 → ROC 곡선 → 다른 모델과 비교

### 3-1. 교차 검증으로 정확도 재기

```python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.95035, 0.96035, 0.9604 ])
```

**교차 검증**을 그림으로 보면 이래요. 훈련 데이터를 3조각(A, B, C)으로 나눈 다음,

| 회차 | 공부(학습)에 쓰는 조각 | 시험(평가)에 쓰는 조각 |
|---|---|---|
| 1회차 | B, C | A |
| 2회차 | A, C | B |
| 3회차 | A, B | C |

이렇게 돌아가며 3번 시험을 봐서 점수 3개를 받아요. `cv=3`이 바로 "3조각으로 나눠라"는 뜻입니다.

결과는 95~96%예요. 100개 중 95개 이상을 맞혔으니 아주 잘하는 것 같죠? 그런데…

#### (참고) 교차 검증을 직접 만들어 보면

`cross_val_score`가 속에서 하는 일을 직접 코드로 써 보면 이해가 확실해져요.

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3)  # 데이터가 안 섞여 있으면 shuffle=True 추가

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)                 # ① 백지 상태의 모델 복사본 만들기
    X_train_folds = X_train[train_index]       # ② 이번 회차 공부용 데이터
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]          # ③ 이번 회차 시험용 데이터
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)   # ④ 공부시키고
    y_pred = clone_clf.predict(X_test_fold)        # ⑤ 시험 보게 하고
    n_correct = sum(y_pred == y_test_fold)         # ⑥ 맞힌 개수 세고
    print(n_correct / len(y_pred))                 # ⑦ 정답률 출력
# 0.95035
# 0.96035
# 0.9604
```

- **`StratifiedKFold`**: 조각을 나눌 때 **5와 5 아닌 것의 비율을 모든 조각에서 똑같이** 맞춰 줘요. 전체에서 5가 약 10%라면 A, B, C 각 조각에도 5가 약 10%씩 들어가요. (*stratified* = "층을 나눈", 2장의 계층 샘플링과 같은 아이디어예요.)
- **`clone(sgd_clf)`**: 설정은 똑같고 **아직 아무것도 배우지 않은** 새 모델을 만들어요. 회차마다 새 모델로 시작해야 이전 회차에 배운 게 섞이지 않습니다.
- **`sum(y_pred == y_test_fold)`**: 예측과 정답이 같은지 비교하면 True/False 배열이 나오는데, 파이썬에서 True는 1, False는 0으로 계산돼요. 그래서 더하면 **맞힌 개수**가 됩니다.

출력이 위의 `cross_val_score`와 똑같죠? 같은 일을 하고 있다는 뜻이에요.

### 3-2. 정확도의 함정: "무조건 아니라고 하는" 모델

여기서 장난을 하나 쳐 볼게요. **사진을 보지도 않고 무조건 "5 아니야!"라고만 답하는 엉터리 모델**을 만들어서 같은 시험을 보게 합니다.

```python
from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train_5)
print(any(dummy_clf.predict(X_train)))   # False

cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# array([0.90965, 0.90965, 0.90965])
```

- `DummyClassifier`는 입력은 무시하고 **가장 흔한 답만 내는** 비교용 가짜 모델이에요. 여기서 가장 흔한 답은 "5 아님(False)"이에요.
- `any(...)`는 "True가 하나라도 있니?"라고 묻는 함수예요. False가 나왔으니 6만 개 예측이 **전부 "5 아님"**이라는 뜻입니다.

그런데 정확도가 **91%**나 나왔어요! 이유는 간단해요. **원래 사진의 약 91%가 5가 아니니까**, 무조건 "아니야"라고만 해도 91%는 맞는 거예요.

> 🎯 **비유**: 1년 중 비 오는 날이 10%뿐인 동네에서 "매일 비 안 옴"이라고만 예보하는 기상캐스터도 정확도가 90%예요. 그런데 이 사람은 비 오는 날을 **단 한 번도** 맞히지 못했죠. 정확도만 보면 이런 사람도 훌륭해 보입니다.

그러니까 SGD의 95%는 "아무 생각 없는 모델(91%)"보다 고작 4%p 나은 수준이에요.

> ⚠️ **핵심 교훈**: 한쪽 답이 압도적으로 많은 데이터(**불균형 데이터**)에서는 정확도가 좋은 성적표가 아니에요. 질병 진단, 사기 거래 탐지, 불량품 검출처럼 실무에서 중요한 문제는 대부분 불균형 데이터라서 이 점이 정말 중요합니다.

그럼 무엇을 봐야 할까요? 답은 **"어떤 종류의 실수를 했는지"를 나눠서 보는 것**이에요.

### 3-3. 오차 행렬: 실수를 종류별로 세기

먼저 6만 개 각각에 대한 예측값이 필요해요.

```python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

`cross_val_score`와 이름이 비슷하지만 돌려주는 게 달라요.

| 함수 | 돌려주는 것 |
|---|---|
| `cross_val_score` | 회차별 **점수** 3개 (예: 0.95, 0.96, 0.96) |
| `cross_val_predict` | 샘플마다 **예측값** 60,000개 (예: True, False, …) |

여기서 중요한 점이 있어요. 각 샘플의 예측값은 **그 샘플을 공부할 때 보지 않았던 모델**이 낸 값이에요. 예를 들어 A 조각에 있는 사진은 B, C로 공부한 모델이 예측합니다. 그래서 테스트 세트를 건드리지 않고도 "처음 보는 문제를 얼마나 잘 푸는지"를 확인할 수 있어요.

이제 정답과 예측을 맞대어 표로 정리합니다. 이 표를 **오차 행렬**(confusion matrix, 혼동 행렬)이라고 해요.

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_train_5, y_train_pred)   # (정답, 예측) 순서로 넣기
cm
# array([[53892,   687],
#        [ 1891,  3530]])
```

숫자만 보면 헷갈리니까 이름표를 붙여 볼게요. **행(가로줄)은 실제 정답, 열(세로줄)은 모델의 예측**이에요.

| | 모델: "5 아니야" | 모델: "5야" |
|---|---|---|
| **실제로 5 아님** | ✅ **TN** 53,892개 — 아닌 걸 아니라고 맞힘 | ❌ **FP** 687개 — 아닌데 5라고 함 (헛경보) |
| **실제로 5임** | ❌ **FN** 1,891개 — 5인데 놓침 | ✅ **TP** 3,530개 — 5를 5라고 맞힘 |

이름이 외우기 어렵다면 이렇게 쪼개 보세요.

- 뒷글자 **P / N** = 모델이 "양성(Positive, 5야)" / "음성(Negative, 아니야)" 중 무엇이라고 **말했는지**
- 앞글자 **T / F** = 그 말이 **맞았는지**(True) / **틀렸는지**(False)

예를 들어 **FP**는 "5라고 했는데(P) 틀렸다(F)", 즉 헛경보예요. **FN**은 "아니라고 했는데(N) 틀렸다(F)", 즉 놓친 거예요.

> 🎯 **비유**: 코로나 검사로 생각해 보세요. FP는 건강한데 양성이 나온 경우(괜히 격리됨), FN은 감염됐는데 음성이 나온 경우(모르고 돌아다님)예요. 둘 다 실수지만 결과가 전혀 다르죠. 오차 행렬은 이 두 실수를 **따로 세어 준다**는 점이 정확도와의 결정적인 차이예요.

만약 완벽한 모델이라면 어떨까요?

```python
y_train_perfect_predictions = y_train_5   # 예측이 정답과 100% 같다고 가정
confusion_matrix(y_train_5, y_train_perfect_predictions)
# array([[54579,     0],
#        [    0,  5421]])
```

실수 칸(FP, FN)이 모두 0이고 **왼쪽 위 → 오른쪽 아래 대각선에만 숫자**가 있어요. 오차 행렬을 볼 때는 "대각선 밖에 숫자가 얼마나 있나"를 보면 됩니다.

### 3-4. 정밀도와 재현율: 오차 행렬을 숫자 하나로

오차 행렬은 정보가 풍부하지만 4개 숫자를 매번 보기는 번거로워요. 그래서 두 가지 질문으로 요약합니다.

#### 정밀도(precision): "5라고 한 것 중 진짜 5는 몇 %?"

```python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred)   # 0.8370879772350012
cm[1, 1] / (cm[0, 1] + cm[1, 1])            # 직접 계산: TP / (FP + TP) = 3530 / (687 + 3530)
```

모델이 "5야!"라고 외친 게 총 4,217번(687 + 3,530)인데, 그중 진짜 5는 3,530개였어요. **정밀도 83.7%**, 즉 "5라고 할 때 믿을 만한 정도"예요.

#### 재현율(recall): "진짜 5 중에서 몇 %를 찾아냈나?"

```python
recall_score(y_train_5, y_train_pred)      # 0.6511713705958311
cm[1, 1] / (cm[1, 0] + cm[1, 1])            # 직접 계산: TP / (FN + TP) = 3530 / (1891 + 3530)
```

진짜 5는 총 5,421개(1,891 + 3,530)인데, 그중 3,530개만 찾아냈어요. **재현율 65.1%**, 즉 "5를 빠뜨리지 않는 정도"예요. 나머지 35%는 놓쳤다는 뜻이죠.

> 🎯 **비유: 그물로 물고기 잡기**
> - **정밀도** = 건져 올린 것 중 물고기 비율 (쓰레기를 덜 건질수록 ↑)
> - **재현율** = 바다에 있던 물고기 중 건져 올린 비율 (물고기를 덜 놓칠수록 ↑)
>
> 그물을 아주 촘촘하게 넓게 치면 물고기를 거의 다 잡지만(재현율↑) 쓰레기도 잔뜩 딸려 와요(정밀도↓). 확실한 물고기만 골라 잡으면 쓰레기는 없지만(정밀도↑) 많이 놓치죠(재현율↓).

#### F1 점수: 둘을 합친 점수 하나

```python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)          # 0.7325171197343847
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)   # 직접 계산: TP / (TP + (FN+FP)/2)
```

두 모델을 비교할 때 숫자가 두 개면 불편하니까, 하나로 합친 게 **F1 점수**예요. 그냥 평균이 아니라 **조화평균**을 써요.

$$F_1 = \frac{2}{\frac{1}{정밀도} + \frac{1}{재현율}}$$

조화평균은 **작은 쪽에 끌려가는** 성질이 있어요. 예를 들어 정밀도 100%, 재현율 10%면 그냥 평균은 55%지만 F1은 약 18%밖에 안 돼요. 그래서 F1이 높으려면 **둘 다 높아야** 합니다. 한쪽만 잘하는 "편식 모델"을 걸러 주는 거예요.

| 지표 | 계산 | 한마디로 | 결과 |
|---|---|---|---|
| **정밀도** | TP / (TP+FP) | "5라고 할 때 믿어도 되나?" | 83.7% |
| **재현율** | TP / (TP+FN) | "5를 빠뜨리지 않나?" | 65.1% |
| **F1** | 두 값의 조화평균 | "둘 다 괜찮나?" | 73.3% |

정확도 95%에 비하면 훨씬 냉정한 성적표죠. 이게 이 모델의 진짜 실력이에요.

### 3-5. 정밀도와 재현율의 줄다리기

> **이 소절의 핵심**: 정밀도와 재현율은 한쪽을 올리면 다른 쪽이 내려가는 **시소** 관계예요. 그리고 그 시소를 움직이는 손잡이가 **임곗값**입니다.

#### ① 모델은 속으로 "점수"를 매긴다

지금까지는 모델이 바로 True/False를 내놓는 것처럼 보였지만, 사실 속에서는 두 단계를 거쳐요.

1. 사진마다 **결정 점수**를 계산한다 ("이건 5 같은 정도가 2164점이야")
2. 점수가 **임곗값(기본 0)보다 크면 True**라고 답한다

`decision_function()`을 쓰면 2단계를 건너뛰고 **1단계의 점수를 직접** 볼 수 있어요.

```python
y_scores = sgd_clf.decision_function([some_digit])
y_scores                                     # array([2164.22030239])

threshold = 0
y_some_digit_pred = (y_scores > threshold)   # array([ True])
```

점수 2164가 0보다 크니까 True예요. `predict()`가 속에서 하는 일과 똑같습니다.

이제 **합격선을 3000으로 올려** 볼까요?

```python
threshold = 3000
y_some_digit_pred = (y_scores > threshold)   # array([False])
```

같은 사진인데 이번엔 "5 아님"이 됐어요. 2164점은 3000점 합격선을 못 넘었으니까요. 진짜 5를 하나 **놓쳤으니** 재현율이 떨어진 거예요.

> 🎯 **비유: 시험 합격선**
> - 합격선을 **높이면** → 합격자는 확실한 실력자뿐(정밀도↑), 대신 아깝게 떨어지는 사람이 늘어남(재현율↓)
> - 합격선을 **낮추면** → 실력자를 거의 다 붙임(재현율↑), 대신 실력 없는 사람도 섞여 붙음(정밀도↓)

#### ② 모든 합격선을 한꺼번에 시험해 보기

그럼 합격선을 어디에 두는 게 좋을까요? 일단 모든 사진의 점수부터 구합니다.

```python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
```

`method="decision_function"`을 붙이면 `cross_val_predict`가 True/False 대신 **점수**를 돌려줘요. 6만 개 사진 각각의 점수가 담긴 배열이 됩니다.

```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
```

`precision_recall_curve`는 "합격선을 이 점수로 두면 정밀도·재현율은 얼마?"를 **가능한 모든 합격선에 대해** 한 번에 계산해 줘요. 세 배열이 같은 순서로 짝지어져 있어서, `thresholds[i]`를 합격선으로 쓰면 정밀도는 `precisions[i]`, 재현율은 `recalls[i]`예요.

> ⚠️ **주의**: `precisions`와 `recalls`는 `thresholds`보다 **1개 더 길어요.** 맨 끝에 "정밀도 1, 재현율 0"인 점이 하나 더 붙어 있어서예요. 그래서 그래프를 그릴 때 `[:-1]`로 마지막 하나를 잘라 길이를 맞춥니다.

그래프로 그려 보면 시소가 눈에 보여요.

```python
# 그림 3-5: 합격선에 따른 정밀도·재현율
plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)  # 파란 점선
plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)         # 초록 실선
plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold")  # 3000 위치에 세로선

idx = (thresholds >= threshold).argmax()   # 합격선 3000에 해당하는 위치 찾기
plt.plot(thresholds[idx], precisions[idx], "bo")   # 그 지점에 동그라미
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid(); plt.xlabel("Threshold"); plt.legend(loc="center right")
plt.show()
```

> 💡 **`(조건).argmax()`는 "조건을 처음 만족하는 위치 찾기"예요.** 이 장에서 계속 나오니 꼭 알아 두세요.
> `thresholds >= 3000`은 `[False, False, …, True, True, …]` 같은 배열이 돼요. `argmax()`는 가장 큰 값의 위치를 찾는데, True(=1)가 가장 크고, 같은 값이 여러 개면 **첫 번째** 위치를 알려 줘요. 결국 "처음으로 3000 이상이 되는 위치"를 얻게 됩니다.

그래프를 보면,
- 오른쪽으로 갈수록(합격선↑) **재현율(초록)은 꾸준히 내려가고**
- **정밀도(파랑)는 대체로 올라가지만 중간중간 울퉁불퉁**해요. 합격선을 올렸을 때 진짜 5가 하나 떨어지면 정밀도가 잠깐 내려갈 수도 있거든요.

x축을 재현율, y축을 정밀도로 바꿔서 그리면 **PR 곡선**이 돼요.

```python
# 그림 3-6: PR 곡선
plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall curve")
```

재현율 80% 근처부터 정밀도가 뚝 떨어져요. 보통 **그 직전**이 좋은 타협점입니다.

#### ③ "정밀도 90%"가 목표라면?

상사가 "정밀도 90%짜리로 만들어 주세요"라고 했다고 해 볼게요. 모델을 새로 학습할 필요 없이 **합격선만 찾아서 바꾸면** 돼요.

```python
idx_for_90_precision = (precisions >= 0.90).argmax()   # 정밀도가 처음 90%를 넘는 위치
threshold_for_90_precision = thresholds[idx_for_90_precision]
threshold_for_90_precision                  # 3370.0194991439557  → 합격선은 약 3370점

y_train_pred_90 = (y_scores >= threshold_for_90_precision)   # 새 합격선으로 다시 판정

precision_score(y_train_5, y_train_pred_90)  # 약 0.90  → 목표 달성!
recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
recall_at_90_precision                       # 0.4799852425751706
```

정밀도는 90%가 됐는데 **재현율이 48%**로 떨어졌어요. 진짜 5의 절반 이상을 놓친다는 뜻이에요. 시소의 한쪽을 올리니 반대쪽이 내려간 거죠.

> 💡 그래서 누가 "정밀도 99%를 달성했어요!"라고 하면 **"재현율은 얼마인가요?"**라고 되물어야 해요. 합격선을 엄청 높이면 정밀도는 얼마든지 올릴 수 있거든요.

**어느 쪽을 중요하게 볼지는 문제마다 달라요.**

| 상황 | 더 중요한 것 | 이유 |
|---|---|---|
| 어린이용 영상 필터 | **정밀도** | 좋은 영상을 좀 막더라도, "안전하다"고 통과시킨 건 반드시 안전해야 함 |
| 마트 좀도둑 감지 CCTV | **재현율** | 경보가 좀 잘못 울려도 괜찮지만, 도둑은 놓치면 안 됨 |
| 암 검진 1차 검사 | **재현율** | 놓치면 치명적. 헛경보는 정밀 검사로 다시 거르면 됨 |

### 3-6. ROC 곡선: 또 하나의 성적 그래프

PR 곡선과 비슷한 도구로 **ROC 곡선**이 있어요. 축만 다릅니다.

| | x축 | y축 |
|---|---|---|
| PR 곡선 | 재현율 | 정밀도 |
| **ROC 곡선** | **FPR** (헛경보 비율) | **TPR** (= 재현율) |

새로 나온 **FPR**(False Positive Rate)은 "5가 아닌 사진 중에서 실수로 5라고 한 비율"이에요. 계산은 FP / (FP + TN)이고, **낮을수록 좋아요.**

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
```

> ⚠️ **주의**: `roc_curve`가 돌려주는 `thresholds`는 **큰 점수부터 작은 점수 순서(내림차순)**예요. 앞의 `precision_recall_curve`는 반대로 작은 것부터(오름차순)였어요. 그래서 아래 코드에서 위치를 찾을 때 `>=` 대신 `<=`를 씁니다.

```python
# 그림 3-7: ROC 곡선에 "정밀도 90% 지점"도 표시
idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")  # 대각선
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid(); plt.axis([0, 1, 0, 1]); plt.legend(loc="lower right")
plt.show()
```

**ROC 곡선 읽는 법**
- **점선 대각선**은 동전 던지기로 찍는 모델이에요. 이 선 근처면 아무 쓸모 없는 모델입니다.
- 좋은 모델일수록 곡선이 **왼쪽 위 모서리**(헛경보 0%, 다 찾아냄 100%)에 바짝 붙어요.

곡선이 얼마나 왼쪽 위에 붙었는지를 숫자 하나로 나타낸 게 **AUC**(Area Under the Curve, 곡선 아래 넓이)예요.

```python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)   # 0.9604938554008616
```

- AUC **1.0** = 완벽한 모델
- AUC **0.5** = 동전 던지기 수준

0.96이면 꽤 좋아 보이죠? 그런데 여기서도 함정이 있어요.

> 💡 **PR 곡선과 ROC 곡선, 언제 뭘 쓸까?**
> **찾는 대상(양성)이 드물 때는 PR 곡선**을 보세요. 이번처럼 "5 아님"이 압도적으로 많으면 TN이 엄청 커서, FP가 꽤 있어도 FPR(= FP / (FP + TN))이 작게 나와요. 그래서 ROC 곡선은 실제보다 좋아 보이게 됩니다. 실제로 AUC는 0.96으로 훌륭해 보이지만, PR 곡선을 보면 아직 개선할 여지가 많아요.

### 3-7. 랜덤 포레스트와 비교해 보기

SGD의 실력을 알았으니, 다른 모델인 **랜덤 포레스트**와 비교해 봐요. (랜덤 포레스트는 6장에서 자세히 배워요. 지금은 "결정 트리를 여러 개 모아 투표하는 강한 모델" 정도로 알면 됩니다.)

```python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")
y_probas_forest[:2]
# array([[0.11, 0.89],
#        [0.99, 0.01]])
```

랜덤 포레스트에는 `decision_function()`이 **없어요.** 대신 `predict_proba()`로 **확률**을 알려 줘요.

- 각 행은 `[5가 아닐 확률, 5일 확률]`이에요.
- 첫 번째 사진(숫자 5): "89% 확률로 5" → 잘 맞힘
- 두 번째 사진(숫자 0): "1% 확률로 5" → 역시 잘 맞힘

그런데 이 확률을 그대로 믿어도 될까요? 확인해 봅시다.

```python
idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")
# 94.0%
```

이 코드는 **"5일 확률 50~60%"라고 나온 사진들만 모아서, 그중 진짜 5가 몇 %인지** 세어 봐요. 확률이 정확하다면 55% 정도가 나와야겠죠. 그런데 실제로는 **94%**예요. 모델이 실제보다 자신 없게 말하고 있는 거예요.

> 💡 그러니 `predict_proba`의 값은 "대략 이쪽일 가능성이 높다" 정도의 **추정치**로 보세요. 확률 자체를 정확하게 써야 한다면 보정(calibration) 작업이 따로 필요해요.

확률을 점수처럼 써서 PR 곡선을 그려 SGD와 비교합니다.

```python
y_scores_forest = y_probas_forest[:, 1]   # "5일 확률"만 꺼내서 점수로 사용
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
    y_train_5, y_scores_forest)

# 그림 3-8: 두 모델의 PR 곡선 겹쳐 그리기
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2, label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")
```

랜덤 포레스트의 곡선이 **오른쪽 위 구석에 훨씬 가깝게** 그려져요. 정밀도와 재현율을 둘 다 높게 유지할 수 있다는 뜻이에요.

점수로도 확인해 볼게요.

```python
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5   # 확률 50% 이상이면 "5야"
f1_score(y_train_5, y_train_pred_forest)          # 0.9274509803921569
roc_auc_score(y_train_5, y_scores_forest)         # 0.9983436731328145
precision_score(y_train_5, y_train_pred_forest)   # 0.9897468089558485
recall_score(y_train_5, y_train_pred_forest)      # 0.8725327430363402
```

> 💡 `cross_val_predict(forest_clf, …)`로 예측을 다시 구하면 몇 분이 걸려요. 이미 확률을 구해 뒀으니 **"50% 이상이면 True"라는 합격선만 적용**하면 같은 결과를 바로 얻을 수 있어요.

| | SGD | 랜덤 포레스트 |
|---|---|---|
| 정밀도 | 0.837 | **0.990** |
| 재현율 | 0.651 | **0.873** |
| F1 | 0.733 | **0.927** |
| ROC AUC | 0.960 | **0.998** |

모든 지표에서 랜덤 포레스트가 확실히 앞서요.

---

## 4. 다중 분류 (3.4)

> **이 절에서 하는 일**: 이제 "5인가 아닌가"가 아니라 **0~9 중 무엇인가**를 맞힙니다. 그리고 원래 둘 중 하나만 고를 줄 아는 모델로 10개 중 하나를 고르게 하는 요령을 배워요.

### 4-1. 두 가지 요령: OvR과 OvO

어떤 모델(로지스틱 회귀, 랜덤 포레스트 등)은 처음부터 여러 클래스를 다룰 수 있어요. 하지만 SGD나 SVM처럼 **두 개 중 하나만 고를 줄 아는 모델**도 있어요. 이런 모델에게 10개 중 하나를 고르게 하는 방법이 두 가지 있습니다.

| 요령 | 방법 | 필요한 모델 수 (10개 클래스) |
|---|---|---|
| **OvR** (One-versus-Rest, 일대다) | "0이냐 나머지냐", "1이냐 나머지냐", … 숫자마다 감별사를 한 명씩 두고, **가장 자신 있게 손드는** 감별사의 숫자를 고름 | **10개** |
| **OvO** (One-versus-One, 일대일) | "0 vs 1", "0 vs 2", … "8 vs 9"처럼 **모든 짝끼리 1:1 대결**을 시키고, **가장 많이 이긴** 숫자를 고름 | 10×9÷2 = **45개** |

> 🎯 **비유**: OvR은 "숫자별 오디션 심사위원 10명이 각자 점수를 매기는 방식", OvO는 "10개 팀이 리그전으로 모든 팀과 한 번씩 붙어서 승수로 순위를 정하는 방식"이에요.

OvO는 모델 수가 많지만, 각 모델은 **두 숫자의 사진만** 가지고 공부하면 돼서 하나하나는 가벼워요. 데이터가 많아질수록 급격히 느려지는 모델(SVM 등)에는 OvO가 유리합니다.

다행히 **사이킷런이 알아서 골라 줘요.** 두 개 중 하나만 고르는 모델에 10개 클래스 정답을 주면, 자동으로 OvR이나 OvO를 적용합니다.

### 4-2. SVC로 해 보기 (자동으로 OvO)

```python
from sklearn.svm import SVC

svm_clf = SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])   # y_train_5가 아니라 y_train!

svm_clf.predict([some_digit])                 # array(['5'], dtype=object)  → 정답!
```

- 이번엔 정답으로 `y_train_5`(True/False)가 아니라 **`y_train`('0'~'9')**을 줬어요.
- SVC는 데이터가 많아지면 아주 느려져서 **2,000장만** 써서 학습했어요.
- SVC는 OvO 방식이 유리한 모델이라 사이킷런이 자동으로 **45개 모델**을 만들어 학습했어요.

어떤 근거로 '5'를 골랐는지 점수를 볼게요.

```python
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores.round(2)
# array([[ 3.79,  0.73,  6.06,  8.3 , -0.29,  9.3 ,  1.75,  2.77,  7.21,  4.82]])
#          '0'    '1'    '2'    '3'    '4'    '5'    '6'    '7'    '8'    '9'
```

숫자마다 점수가 하나씩, 총 10개 나왔어요. '5'가 **9.3**으로 가장 높아요. 이 점수는 대략 **"1:1 대결에서 이긴 횟수 + 얼마나 확실하게 이겼는지에 따른 작은 보너스"**예요. '5'는 다른 9개 숫자와의 대결에서 **9전 9승**을 한 셈입니다. '3'(8.3)이 2등인 걸 보면, 이 사진이 3과 약간 헷갈리는 모양이라는 것도 짐작할 수 있어요.

가장 높은 점수의 숫자를 꺼내는 방법은 이래요.

```python
class_id = some_digit_scores.argmax()   # 5  → 가장 높은 점수의 "위치"
svm_clf.classes_                        # array(['0', '1', ..., '9'], dtype=object)
svm_clf.classes_[class_id]              # '5'  → 그 위치의 "클래스 이름"
```

> ⚠️ **자주 하는 실수**: `argmax()`가 알려 주는 건 **몇 번째 칸인지(위치)**이지, **클래스 이름**이 아니에요. 여기서는 클래스가 '0', '1', …, '9' 순서라서 5번째 칸이 우연히 '5'와 같을 뿐이에요. 클래스가 '고양이', '강아지', '새'라면 `argmax()`는 1을 돌려줄 뿐이니, 반드시 `classes_[위치]`로 이름을 꺼내야 합니다.

45개 대결 결과를 하나하나 보고 싶다면 이렇게 해요.

```python
svm_clf.decision_function_shape = "ovo"
some_digit_scores_ovo = svm_clf.decision_function([some_digit])
some_digit_scores_ovo.round(2)
# 45개 값: [[ 0.11, -0.21, -0.97,  0.51, ... ]]   (0vs1, 0vs2, ..., 8vs9 순서)
```

> 💡 헷갈리기 쉬운 부분: `decision_function_shape`의 기본값은 `"ovr"`이지만, 이건 **점수를 어떤 모양으로 보여 줄지**만 정하는 설정이에요. 실제 학습은 이 값과 관계없이 **항상 OvO**로 해요. `"ovr"`이면 45개 대결 결과를 숫자별로 모아서 10개로 보여 주고, `"ovo"`면 45개를 그대로 보여 줍니다.

### 4-3. OvR로 바꾸고 싶다면

사이킷런이 자동으로 고르는 방식 대신 원하는 방식을 쓰고 싶으면, 모델을 **포장지로 한 번 감싸면** 돼요.

```python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

ovr_clf.predict([some_digit])   # array(['5'], dtype='<U1')
len(ovr_clf.estimators_)        # 10
```

- `OneVsRestClassifier(SVC(...))`는 "SVC를 OvR 방식으로 써 줘"라는 뜻이에요.
- 학습된 감별사 목록이 `estimators_`에 들어 있고, 개수를 세어 보면 **10개**예요. (OvO를 강제하고 싶다면 `OneVsOneClassifier`를 쓰면 됩니다.)

### 4-4. SGD로 해 보기 (자동으로 OvR)

```python
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])   # array(['3'], dtype='<U1')  ← 앗, 틀렸어요!
```

SGD는 `some_digit`을 **'3'으로 잘못** 맞혔어요. 점수를 보면 이유가 보입니다.

```python
sgd_clf.decision_function([some_digit]).round()
# array([[-31893., -34420.,  -9531.,   1824., -22320.,  -1386., -26189.,
#         -16148.,  -4604., -12051.]])
#           '0'      '1'      '2'      '3'      '4'      '5'   ...
```

- SGD는 OvR 방식이라 숫자별 감별사 10명이 각자 점수를 매겼어요.
- 양수 점수는 **'3' 감별사(1824점)**뿐이에요. 정답인 **'5' 감별사는 −1386점**으로 2등이었어요.
- 즉 모델이 **3과 5 사이에서 헷갈리고 있다**는 걸 알 수 있어요. 이 문제를 5절에서 자세히 파헤칩니다.

### 4-5. 평가해 보기, 그리고 스케일링의 힘

```python
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([0.87365, 0.85835, 0.8689 ])
```

10개 중 하나를 고르는 문제라 아무렇게나 찍으면 10%예요. 87%면 나쁘지 않아요. 그런데 아주 간단한 방법 하나로 더 올릴 수 있어요.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
# array([0.8983, 0.891 , 0.9018])
```

- `StandardScaler`는 2장에서 배운 **표준화**예요. 각 픽셀 값을 "평균 0, 표준편차 1"이 되도록 바꿔 줘요.
- `.astype("float64")`는 정수 픽셀 값을 소수점이 있는 실수로 바꾸는 거예요. 표준화 결과는 소수(예: −0.42)라서 실수형으로 담아야 정확해요.

이것만으로 정확도가 87%에서 **약 90%**로 올랐어요. SGD처럼 **경사 하강법으로 학습하는 모델은 입력 값의 크기(스케일)에 아주 민감**하기 때문이에요. 모델을 바꾸기 전에 전처리부터 점검해 보는 습관이 중요하다는 걸 보여 주는 예입니다.

---

## 5. 오류 분석 (3.5)

> **이 절에서 하는 일**: 성적표(점수)를 받았으니 이제 **오답 노트**를 만들 차례예요. 모델이 어떤 문제를 자주 틀리는지 찾아내면, 무엇을 고쳐야 할지 보입니다.

### 5-1. 오차 행렬을 그림으로 보기

10개 클래스면 오차 행렬이 10×10 = 100칸이라 숫자로는 읽기 힘들어요. 색깔로 칠해서 봅시다.

```python
from sklearn.metrics import ConfusionMatrixDisplay

y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

plt.rc('font', size=9)   # 칸이 많으니 글씨를 작게
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()
```

`ConfusionMatrixDisplay.from_predictions(정답, 예측)`은 오차 행렬 계산부터 **색칠한 그림**까지 한 번에 해 줘요. 3.3에서와 마찬가지로 **가로줄 = 실제 숫자, 세로줄 = 모델이 고른 숫자**예요.

대각선(정답 칸)이 진하게 나오니까 대체로 잘 맞히고 있어요. 다만 '5' 줄의 대각선 칸이 다른 숫자보다 조금 옅어요. 그런데 이건 두 가지로 해석될 수 있어요.

1. 모델이 5를 잘 못 맞혀서
2. 애초에 5 사진이 다른 숫자보다 적어서

개수로 칠하면 이 둘을 구분할 수 없어요. 그래서 **비율로** 바꿔 봅니다.

```python
plt.rc('font', size=10)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        normalize="true", values_format=".0%")
plt.show()
```

- `normalize="true"`: 각 **가로줄(실제 숫자)의 합이 100%**가 되도록 나눠요. "실제 5 사진 중 몇 %를 무엇으로 골랐나"를 보여 주니까, 사진 개수 차이와 상관없이 공정하게 비교할 수 있어요.
  - 참고로 `"pred"`로 하면 세로줄 기준, `"all"`로 하면 전체 기준이에요.
- `values_format=".0%"`: 칸 안의 숫자를 "82%"처럼 소수점 없는 퍼센트로 보여 줘요.

이제 보면 **실제 5 중 약 82%만 맞혔고, 약 10%는 8로** 잘못 골랐다는 게 드러나요.

마지막으로 **정답은 빼고 오답만** 남겨서 봅시다.

```python
sample_weight = (y_train_pred != y_train)   # 틀린 것만 True(=1), 맞힌 건 False(=0)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred,
                                        sample_weight=sample_weight,
                                        normalize="true", values_format=".0%")
plt.show()
```

여기서 쓴 요령이 재미있어요. `sample_weight`는 원래 "이 샘플을 얼마나 중요하게 셀지" 정하는 가중치예요. **맞힌 샘플의 가중치를 0으로** 주면 아예 세지 않으니까, 대각선(정답)이 전부 0이 되고 **오답만 남은 오차 행렬**이 만들어집니다.

이렇게 보면 **'8' 세로줄이 유독 밝게** 빛나요. 여러 숫자가 **8로 잘못 분류**되는 경우가 많다는 뜻이에요. 오답 노트의 첫 번째 발견이죠.

| 그림 | 설정 | 알 수 있는 것 |
|---|---|---|
| ① 개수 | 기본 | 대체로 잘 맞힘. '5'가 약간 약해 보이지만 원인은 불분명 |
| ② 비율 | `normalize="true"` | 실제 5의 약 10%를 8로 착각 |
| ③ 오답만 | `sample_weight=(오답)` | 여러 숫자가 '8'로 잘못 분류됨 |

> ⚠️ 비율을 **가로줄 기준**으로 냈는지 **세로줄 기준**으로 냈는지에 따라 해석이 달라져요. 가로줄 기준에서 "5→3 오류"가 커 보여도, 세로줄 기준(`"pred"`)으로 보면 다른 모습일 수 있어요. 그림을 볼 때 무엇을 기준으로 나눴는지 꼭 확인하세요.

### 5-2. 3과 5, 직접 눈으로 비교하기

4.4에서 SGD가 3과 5를 헷갈렸죠. 실제로 어떤 사진들을 헷갈리는지 직접 보는 게 가장 확실해요. 사진을 네 무리로 나눕니다.

```python
cl_a, cl_b = '3', '5'
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]   # 실제 3 → 3이라고 함 (정답)
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]   # 실제 3 → 5라고 함 (오답)
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]   # 실제 5 → 3이라고 함 (오답)
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]   # 실제 5 → 5라고 함 (정답)
```

`X_train[조건]`은 조건이 True인 사진만 골라내요(**불리언 인덱싱**). 예를 들어 `X_ab`의 조건은 "정답이 3이면서 **그리고** 예측이 5인 사진"이에요.

> ⚠️ **자주 하는 실수**: 넘파이 배열에서 "그리고"는 `and`가 아니라 **`&`**를 써야 해요. 그리고 각 조건을 **괄호로 꼭 감싸야** 해요. `y_train == cl_a & y_train_pred == cl_a`처럼 괄호를 빼면, 파이썬이 `&`를 먼저 계산해서 엉뚱한 결과가 나옵니다.

네 무리를 2×2로 배치해서 그립니다. (그림 그리는 코드라 원리만 이해하면 돼요.)

```python
# 그림 3-11: 네 무리를 2×2 블록으로 (블록마다 5×5 = 25장)
size = 5     # 블록 한 변에 사진 5장
pad = 0.2    # 블록 사이 간격
plt.figure(figsize=(size, size))
for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)),
                                       (X_aa, (0, 1)), (X_ab, (1, 1))]:
    for idx, image_data in enumerate(images[:size*size]):
        x = idx % size + label_col * (size + pad)    # 가로 위치
        y = idx // size + label_row * (size + pad)   # 세로 위치
        plt.imshow(image_data.reshape(28, 28), cmap="binary",
                   extent=(x, x + 1, y, y + 1))      # (x, y)에 1×1 크기로 붙이기
plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], "k:")   # 세로 구분선
plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], "k:")   # 가로 구분선
plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.show()
```

| 부분 | 하는 일 |
|---|---|
| `idx % size`, `idx // size` | 사진 번호를 블록 안의 (열, 행) 위치로 바꿈. 예: 7번 사진 → 7%5=2열, 7//5=1행 |
| `label_col * (size + pad)` | 블록 단위로 옆(또는 위)으로 밀어냄. 0이면 왼쪽(아래) 블록, 1이면 오른쪽(위) 블록 |
| `imshow(..., extent=(왼, 오, 아래, 위))` | 사진을 **정해진 좌표 칸에 딱 붙여** 그림. 작은 그래프 100개를 만드는 것보다 가벼움 |

**그림을 보면 알게 되는 것**

- 틀린 사진 중 일부는 사람이 봐도 헷갈리지만, **대부분은 사람 눈에는 너무 쉬운 문제**예요.
- 3과 5는 사실 거의 같은 모양이에요. 차이는 **위쪽 가로선과 아래쪽 둥근 부분을 잇는 짧은 세로선이 왼쪽에 있느냐 오른쪽에 있느냐**뿐이에요.
- 그런데 SGD는 **선형 모델**이에요. "이 자리 픽셀이 진하면 +, 저 자리 픽셀이 진하면 −" 하고 **자리별 고정 점수**를 더할 뿐이라서, 글씨가 조금만 옆으로 밀리거나 기울어져도 다른 자리 픽셀이 칠해져 쉽게 헷갈려요.

**그래서 어떻게 고치면 될까?**

1. **사진을 가운데로 정렬하고 기울기를 바로잡는** 전처리를 한다.
2. 사진을 조금씩 옮기거나 돌린 **복사본을 만들어 훈련 데이터를 늘린다**(데이터 증강). 다양한 위치와 기울기를 미리 보여 주는 거예요. (연습문제 2번에서 실제로 해 보면 오류가 확 줄어요.)
3. '8' 오답을 줄이기 위해, 예를 들어 **닫힌 동그라미 개수**(8은 2개, 6은 1개, 5는 0개) 같은 특징을 새로 만들어 준다.

---

## 6. 다중 레이블 분류 (3.6)

> **이 절에서 하는 일**: 지금까지는 사진 하나에 정답이 **하나**였어요. 이번엔 사진 하나에 **여러 개의 예/아니요 질문**을 동시에 답하는 분류를 배웁니다.

> 🎯 **비유**: 인스타그램 해시태그를 생각해 보세요. 사진 한 장에 #바다 #여름 #친구처럼 태그가 여러 개 붙을 수 있죠. 단체 사진에서 "앨리스 있음? 밥 있음? 찰리 있음?"을 한꺼번에 답하는 얼굴 인식도 같은 경우예요. 이런 분류를 **다중 레이블 분류**라고 해요.

### 6-1. 질문 두 개를 동시에 답하는 모델

MNIST로 연습할 수 있게 질문 두 개를 만들어 볼게요.

- 질문 1: **"7 이상의 큰 숫자인가?"** (7, 8, 9면 예)
- 질문 2: **"홀수인가?"** (1, 3, 5, 7, 9면 예)

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= '7')                  # 질문 1의 정답
y_train_odd = (y_train.astype('int8') % 2 == 1)   # 질문 2의 정답
y_multilabel = np.c_[y_train_large, y_train_odd]  # 두 정답을 나란히 붙이기 → (60000, 2)

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
```

| 줄 | 쉬운 설명 |
|---|---|
| `y_train >= '7'` | 문자열끼리 비교하지만, 한 글자 숫자라서 '7' ≤ '8' ≤ '9' 순서가 숫자 크기와 같아 문제없어요 |
| `.astype('int8')` | 홀짝을 보려면 나머지 계산(`% 2`)이 필요한데, 문자열은 계산이 안 되니 정수로 바꿔요. 0~9는 작은 정수형(`int8`)이면 충분해요 |
| `% 2 == 1` | 2로 나눈 나머지가 1이면 홀수 |
| `np.c_[a, b]` | 1차원 배열 두 개를 **옆으로 나란히 붙여** 2열짜리 표로 만들어요. 각 행이 `[큰 숫자?, 홀수?]`가 돼요 |

**KNN**(K-최근접 이웃)은 "새 사진과 **가장 닮은 훈련 사진 몇 장**을 찾아서, 그 사진들의 답을 따라가는" 단순한 모델이에요. 원래부터 여러 질문을 한꺼번에 답할 수 있어서, 2열짜리 정답을 그대로 줘도 학습이 됩니다.

```python
knn_clf.predict([some_digit])   # array([[False,  True]])
```

5는 "7 이상? → 아니요(False)", "홀수? → 예(True)". 두 질문 모두 맞혔어요!

### 6-2. 어떻게 채점할까?

```python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")      # 0.9764102655606048
f1_score(y_multilabel, y_train_knn_pred, average="weighted")
```

질문마다 F1 점수를 따로 구한 다음 평균을 내요. 평균 내는 방법이 두 가지예요.

- `average="macro"`: 모든 질문을 **똑같이 중요하게** 보고 그냥 평균 → 0.976
- `average="weighted"`: "예"라는 답이 **많은 질문에 더 무게**를 줘서 평균. 예를 들어 사진에 앨리스가 밥보다 훨씬 자주 나온다면, 앨리스 질문의 점수를 더 중요하게 세는 식이에요.

이 데이터는 두 질문의 "예" 비율이 비슷해서 두 방법의 결과 차이가 거의 없어요.

### 6-3. 여러 질문을 못 하는 모델이라면: ClassifierChain

SVC 같은 모델은 한 번에 질문 하나만 답할 수 있어요. 가장 쉬운 해결책은 질문마다 모델을 하나씩 따로 두는 거지만, 그러면 **질문끼리의 관계**를 활용하지 못해요. 예를 들어 "7 이상이면 7, 8, 9 중 하나니까 홀수일 가능성이 2/3"라는 힌트를 버리는 셈이죠.

그래서 **모델을 줄줄이 사슬처럼 연결**하는 방법이 있어요.

```python
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])
chain_clf.predict([some_digit])   # array([[0., 1.]])
```

**동작 순서**
1. 1번 모델: 사진을 보고 → "7 이상인가?" 답하기
2. 2번 모델: 사진 **+ 1번 모델의 답**을 함께 보고 → "홀수인가?" 답하기

앞 모델의 답이 뒤 모델에게 **힌트로 전달**되는 거예요.

- `cv=3`: 학습할 때 1번 모델의 답 자리에 **정답 대신 교차 검증 예측값**을 넣어 줘요. 실제로 쓸 때는 정답을 모르고 1번 모델의 (틀릴 수도 있는) 예측을 쓰게 되니까, 학습 때도 같은 조건으로 연습시켜서 더 튼튼하게 만드는 거예요.
- 결과가 `[0., 1.]`처럼 소수로 나오지만 뜻은 `[False, True]`와 같아요. 이번에도 정답이에요.

---

## 7. 다중 출력 분류 (3.7)

> **이 절에서 하는 일**: 다중 레이블을 한 단계 더 넓혀서, **질문이 여러 개이고 각 질문의 답도 예/아니요가 아니라 여러 값 중 하나**인 분류를 배웁니다. 예제는 **사진의 잡티(노이즈)를 지워 주는 모델**이에요.

### 7-1. 일부러 지저분한 사진 만들기

```python
rng = np.random.default_rng(seed=42)                       # 재현 가능한 난수 생성기
noise_train = rng.integers(0, 100, (len(X_train), 784))    # 0~99 사이 무작위 잡티
X_train_mod = X_train + noise_train                         # 문제: 원본 + 잡티
noise_test = rng.integers(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
y_train_mod = X_train                                       # 정답: 깨끗한 원본
y_test_mod = X_test

# 그림 3-12: 왼쪽은 잡티 낀 문제, 오른쪽은 깨끗한 정답
plt.subplot(121); plot_digit(X_test_mod[0])
plt.subplot(122); plot_digit(y_test_mod[0])
plt.show()
```

- `rng.integers(0, 100, 모양)`은 0 이상 **100 미만**(즉 0~99)의 정수를 원하는 모양으로 만들어요. 이걸 원래 픽셀에 더하면 사진 전체에 지글지글한 잡티가 생겨요.
- `subplot(121)`은 "1행 2열짜리 칸 중 1번째"라는 뜻의 줄임 표기예요.

여기서 가장 중요한 부분은 **정답(`y_train_mod`)이 숫자 라벨이 아니라 깨끗한 사진 자체**라는 거예요.

- 맞혀야 할 것: 픽셀 **784개** 각각의 값 → 질문이 784개
- 각 질문의 답: 0~255 중 하나 → 답이 예/아니요가 아니라 256가지

질문이 여러 개(다중 레이블)이면서 각 답도 여러 값 중 하나(다중 클래스)라서 **다중 출력 분류**라고 불러요.

### 7-2. 잡티 지우기

```python
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)           # 잡티 사진 → 깨끗한 사진을 학습
clean_digit = knn_clf.predict([X_test_mod[0]])  # 처음 보는 잡티 사진을 넣어 보기
plot_digit(clean_digit)
plt.show()
```

KNN은 이렇게 일해요.
1. 잡티 사진과 **가장 닮은 훈련 사진 5장**을 찾아요. (`n_neighbors=5`가 기본값)
2. 그 5장의 **깨끗한 정답 사진**을 꺼내서, **픽셀마다 다수결**로 값을 정해요.

결과는 잡티가 말끔히 사라진 깨끗한 숫자예요. 비슷한 사진들의 깨끗한 버전을 모아서 다수결하니까, 무작위 잡티는 투표에서 지고 진짜 글씨 모양만 남는 거죠.

> 💡 "픽셀 값(0~255)을 맞히는 건 분류보다 회귀에 가깝지 않나?" 하는 생각이 들 수 있어요. 맞아요. 책에서도 이 예시처럼 **분류와 회귀의 경계가 늘 딱 잘라지는 건 아니라고** 이야기합니다.

---

## 8. 전체 흐름 요약

```
fetch_openml('mnist_784')                       ← 3.1 데이터 받기, 앞 6만 / 뒤 1만으로 나누기
   ↓ y_train == '5'  (10개 → 2개 문제로 단순화)
SGDClassifier().fit(X_train, y_train_5)         ← 3.2 "5인가?" 이진 분류기
   ↓ cross_val_predict (예측값 / 점수 / 확률)
confusion_matrix → 정밀도·재현율·F1             ← 3.3 정확도의 함정을 피하는 성적표
precision_recall_curve / roc_curve → 합격선(임곗값) 고르기
   ↓ y_train (다시 10개 클래스로)
SVC(OvO) / OneVsRestClassifier / SGD(OvR)       ← 3.4 다중 분류 + 표준화로 성능↑
   ↓
ConfusionMatrixDisplay (비율, 오답만)            ← 3.5 오답 노트 → 3 vs 5, '8' 쏠림 발견
   ↓
KNN / ClassifierChain                            ← 3.6 질문 여러 개 동시에 (다중 레이블)
KNN (잡티 사진 → 깨끗한 사진)                    ← 3.7 답이 여러 값인 질문 여러 개 (다중 출력)
```

### 분류의 네 가지 종류

| 종류 | 질문 수 × 답의 가짓수 | 예시 | 사용한 도구 |
|---|---|---|---|
| 이진 분류 | 질문 1개 × 답 2가지 | "5인가?" | `SGDClassifier`, `RandomForestClassifier` |
| 다중 분류 | 질문 1개 × 답 여러 가지 | "0~9 중 무엇?" | `SVC`(OvO), `OneVsRestClassifier`, SGD(OvR) |
| 다중 레이블 | 질문 여러 개 × 답 2가지 | "큰 숫자? 홀수?" | `KNeighborsClassifier`, `ClassifierChain` |
| 다중 출력 | 질문 여러 개 × 답 여러 가지 | "픽셀 784개 값은?" | `KNeighborsClassifier` |

### 주요 결과

| 모델 / 설정 | 지표 | 값 |
|---|---|---|
| 더미 (무조건 "5 아님") | 정확도 | 0.910 |
| SGD ("5인가?") | 정확도 | 0.950 ~ 0.960 |
| SGD ("5인가?") | 정밀도 / 재현율 / F1 | 0.837 / 0.651 / 0.733 |
| SGD, 합격선 3,370 | 정밀도 / 재현율 | 0.90 / 0.480 |
| SGD ("5인가?") | ROC AUC | 0.960 |
| **랜덤 포레스트 ("5인가?")** | 정밀도 / 재현율 / F1 / AUC | **0.990 / 0.873 / 0.927 / 0.998** |
| SGD (0~9) | 정확도 | 0.86 ~ 0.87 |
| SGD (0~9) + 표준화 | 정확도 | 0.89 ~ 0.90 |
| KNN (다중 레이블) | F1 (macro) | 0.976 |

### 자주 쓴 코드 패턴

| 패턴 | 언제 쓰나 | 나온 곳 |
|---|---|---|
| `model.predict([x])` | 샘플 1개도 대괄호로 감싸서 넣기 | 전체 |
| `cross_val_predict(..., method="decision_function")` | 교차 검증으로 **점수** 얻기 | 3-5 |
| `cross_val_predict(..., method="predict_proba")` | 교차 검증으로 **확률** 얻기 | 3-7 |
| `(조건 배열).argmax()` | 조건을 **처음** 만족하는 위치 찾기 | 3-5, 3-6 |
| `scores >= 합격선` | 다시 학습하지 않고 합격선만 바꾸기 | 3-5 |
| `model.classes_[위치]` | 위치 번호를 클래스 이름으로 바꾸기 | 4-2 |
| `sample_weight=(pred != y)` | 오차 행렬에서 오답만 남기기 | 5-1 |
| `X[(조건1) & (조건2)]` | 두 조건을 모두 만족하는 샘플 골라내기 | 5-2 |
| `np.c_[a, b]` | 1차원 배열 두 개를 나란히 붙이기 | 6-1 |

---

## 9. 스스로 확인해 보기

답을 먼저 떠올려 본 뒤 펼쳐서 확인하세요.

<details>
<summary><b>Q1.</b> 정확도 91%인 모델이 쓸모없을 수도 있는 이유는?</summary>

데이터가 불균형하면 가장 흔한 답만 계속 말해도 정확도가 높게 나오기 때문이에요. MNIST에서 "무조건 5 아님"이라고만 해도 91%가 나왔어요. 이 모델의 재현율은 0%예요.
</details>

<details>
<summary><b>Q2.</b> 정밀도와 재현율을 한 문장씩으로 설명하면?</summary>

- 정밀도: "5라고 한 것 중 진짜 5의 비율" → 양성이라고 할 때 믿을 만한가
- 재현율: "진짜 5 중 찾아낸 비율" → 양성을 빠뜨리지 않는가
</details>

<details>
<summary><b>Q3.</b> 임곗값(합격선)을 올리면 정밀도와 재현율은 각각 어떻게 될까?</summary>

정밀도는 대체로 올라가고 재현율은 내려가요. 확실한 것만 양성으로 판정하니 헛경보는 줄지만 놓치는 양성이 늘어나요.
</details>

<details>
<summary><b>Q4.</b> 양성이 아주 드문 문제에서 PR 곡선과 ROC 곡선 중 무엇을 봐야 할까?</summary>

PR 곡선이에요. 음성이 압도적으로 많으면 TN이 커서 FPR이 작게 나오고, ROC 곡선과 AUC가 실제보다 좋아 보여요.
</details>

<details>
<summary><b>Q5.</b> 클래스가 10개일 때 OvR과 OvO는 각각 모델을 몇 개 만들까?</summary>

OvR은 10개, OvO는 10×9÷2 = 45개예요.
</details>

<details>
<summary><b>Q6.</b> SGD가 3과 5를 자주 헷갈리는 이유와 해결책은?</summary>

SGD는 픽셀 자리마다 고정된 가중치를 더하는 선형 모델이라, 글씨가 조금 이동하거나 기울면 쉽게 헷갈려요. 이미지를 가운데 정렬·기울기 보정하거나, 이동·회전한 사진으로 데이터를 늘리면(데이터 증강) 나아져요.
</details>

<details>
<summary><b>Q7.</b> 다중 레이블 분류와 다중 출력 분류의 차이는?</summary>

둘 다 질문이 여러 개예요. 다중 레이블은 각 질문의 답이 예/아니요 두 가지이고, 다중 출력은 각 질문의 답이 여러 값 중 하나예요(예: 픽셀 값 0~255).
</details>
