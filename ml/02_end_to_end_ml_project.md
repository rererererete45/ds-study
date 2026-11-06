# 2장. 머신러닝 프로젝트 처음부터 끝까지

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `02_end_to_end_machine_learning_project.ipynb`
>
> 이 장은 **머신러닝 프로젝트 하나를 처음부터 끝까지 직접 해 보는 장**이에요.
> 알고리즘을 깊게 파고들기보다는 "실무에서는 어떤 순서로 일이 진행되는가"를 몸으로 익히는 게 목표입니다.
> 3장부터 배우는 모든 내용은 이 장에서 만든 뼈대 위에 하나씩 얹히게 돼요.

---

## 🗺️ 먼저 큰 그림부터

요리에 비유하면 이런 순서예요.

| 단계 | 요리로 치면 | 이 장에서 하는 일 | 절 |
|---|---|---|---|
| 1. 문제 정의 | 무슨 요리를 할지 정하기 | "구역별 집값을 맞히자" | 2.1–2.2 |
| 2. 데이터 수집 | 장보기 | 집값 데이터 다운로드, **시험용 데이터 따로 빼 두기** | 2.3 |
| 3. 탐색 | 재료 살펴보기 | 그래프 그려 보고 어떤 특성이 집값과 관련 있는지 확인 | 2.4 |
| 4. 전처리 | 재료 손질 | 빈칸 채우기, 글자를 숫자로, 크기 맞추기 | 2.5 |
| 5. 모델 훈련 | 여러 레시피로 시험 조리 | 선형 회귀·결정 트리·랜덤 포레스트 비교 | 2.6 |
| 6. 튜닝 | 간 맞추기 | 가장 좋은 설정값 찾기 | 2.7 |
| 7. 최종 평가 | 손님에게 내기 전 마지막 시식 | 따로 빼 둔 시험용 데이터로 단 한 번 채점 | 2.7 |
| 8. 배포 | 가게 메뉴에 올리기 | 모델 저장하고 운영 | 2.8 |

---

## 📖 이 장에서 처음 나오는 용어

읽다가 막히면 여기로 돌아오세요.

| 용어 | 쉬운 뜻 |
|---|---|
| **샘플** (sample, 인스턴스) | 데이터 한 줄. 여기서는 "구역 하나" |
| **특성** (feature) | 샘플을 설명하는 값 하나. 여기서는 소득, 방 개수, 위도 같은 열 |
| **레이블 / 타깃** (label, target) | 맞히고 싶은 정답. 여기서는 집값 |
| **훈련 세트** (train set) | 모델이 공부하는 데이터 (문제집) |
| **테스트 세트** (test set) | 마지막에 실력을 재는 데이터 (수능). 공부할 때 절대 보면 안 됨 |
| **모델** (model) | 특성을 넣으면 예측값을 내놓는 함수 |
| **훈련** (fit, 학습) | 모델이 훈련 세트를 보고 규칙을 배우는 과정 |
| **하이퍼파라미터** | 사람이 훈련 **전에** 정해 주는 설정값 (예: 트리 개수) |
| **과대적합** (overfitting) | 문제집 답을 통째로 외워서, 새 문제는 못 푸는 상태 |
| **과소적합** (underfitting) | 공부가 너무 부족해서 문제집조차 못 푸는 상태 |
| **파이프라인** (pipeline) | 여러 처리 단계를 순서대로 묶어 버튼 하나로 실행하게 만든 것 |

---

## 📦 실습 데이터: 캘리포니아 주택 가격

1990년 미국 캘리포니아 인구조사 데이터예요. 한 줄이 **구역(district) 하나**이고(보통 인구 600~3,000명 규모), 총 **20,640개** 구역이 있습니다.

| 컬럼 | 뜻 | 메모 |
|---|---|---|
| `longitude`, `latitude` | 경도, 위도 | 구역의 위치 |
| `housing_median_age` | 주택 연식 중간값 | 52년에서 잘려 있음(상한) |
| `total_rooms` | 구역 전체의 **방** 개수 | 집 한 채가 아니라 구역 합계! |
| `total_bedrooms` | 구역 전체의 **침실** 개수 | ⚠️ 207개 빈칸(결측치) |
| `population` | 구역 인구 | |
| `households` | 구역의 가구 수 | |
| `median_income` | 소득 중간값 | 달러가 아님. 대략 **1 = $10,000** |
| `median_house_value` | **집값 중간값 ← 우리가 맞힐 정답** | $500,000에서 잘려 있음(상한) |
| `ocean_proximity` | 바다와의 거리 | 숫자가 아닌 **글자** (5종류) |

---

## 목차

1. [실제 데이터로 작업하기 · 큰 그림 보기 (2.1–2.2)](#1-실제-데이터로-작업하기--큰-그림-보기-2122)
2. [데이터 가져오기 (2.3)](#2-데이터-가져오기-23)
3. [탐색과 시각화 (2.4)](#3-탐색과-시각화-24)
4. [데이터 준비 (2.5)](#4-데이터-준비-25)
5. [모델 선택과 훈련 (2.6)](#5-모델-선택과-훈련-26)
6. [모델 미세 튜닝 (2.7)](#6-모델-미세-튜닝-27)
7. [론칭·모니터링·유지 보수 (2.8)](#7-론칭모니터링유지-보수-28)
8. [연습문제 (2.9)](#8-연습문제-29)
9. [전체 흐름 요약](#9-전체-흐름-요약)

---

## 1. 실제 데이터로 작업하기 · 큰 그림 보기 (2.1–2.2)

### 1-1. 상황 설정

여러분은 "머신러닝 주택 회사"에 막 입사했어요. 팀장님이 이렇게 말합니다.

> "구역 정보를 주면 그 구역의 **집값 중간값**을 예측하는 모델을 만들어 주세요. 이 예측값은 '이 구역에 투자할까?'를 판단하는 다른 시스템에 들어갑니다. 지금은 전문가들이 손으로 추정하는데, 시간도 오래 걸리고 20% 넘게 틀릴 때도 있어요."

여기서 중요한 포인트는 **"이 모델이 어디에 쓰이는가"를 먼저 물어봤다**는 거예요. 용도를 알아야 얼마나 정확해야 하는지, 어떤 형태로 결과를 줘야 하는지 정할 수 있어요.

### 1-2. 문제의 종류 정하기

코드를 짜기 전에, 이 문제가 머신러닝의 어떤 종류인지 분류해 봅니다.

| 질문 | 이 문제의 답 | 그래서? |
|---|---|---|
| 정답(집값)이 데이터에 있나요? | 있어요 | **지도 학습** — 정답을 보며 배우는 방식 |
| 맞히려는 게 숫자인가요, 종류인가요? | 숫자(가격) | **회귀** — 숫자를 예측. 종류를 맞히면 "분류" |
| 데이터가 계속 실시간으로 들어오나요? | 아니요, 한 번에 다 있어요 | **배치 학습** — 한꺼번에 학습 |

### 1-3. 성적표 정하기: RMSE

모델이 잘하는지 못하는지 판단하려면 **채점 기준**이 필요해요. 회귀에서 가장 많이 쓰는 건 **RMSE**(평균 제곱근 오차)입니다.

계산은 이렇게 해요.
1. 각 구역마다 (정답 − 예측) 을 구한다 → 오차
2. 오차를 **제곱**한다 → 음수를 없애고, 큰 오차를 더 크게 만든다
3. 전부 **평균**낸다
4. **제곱근**을 씌운다 → 단위를 다시 "달러"로 되돌린다

예를 들어 RMSE가 50,000이면 "평균적으로 $50,000 정도 틀린다"고 읽으면 돼요.

| 지표 | 특징 | 언제 쓰나 |
|---|---|---|
| **RMSE** | 큰 실수를 특히 싫어함 (제곱하니까) | 기본값 |
| **MAE** (평균 절대 오차) | 오차를 그냥 절댓값으로 평균 | 이상한 값(이상치)이 많을 때 |

> 💡 **가정 확인하기**: 혹시 투자 시스템이 정확한 가격이 아니라 "싸다/보통/비싸다"만 필요하다면? 그러면 회귀가 아니라 분류 문제가 돼요. 시작 전에 꼭 확인해야 할 부분이에요.

### 1-4. 환경 확인 코드

```python
import sys
assert sys.version_info >= (3, 10)     # 파이썬이 3.10 이상인지 확인

from packaging.version import Version
import sklearn
assert Version(sklearn.__version__) >= Version("1.6.1")   # 사이킷런이 1.6.1 이상인지 확인
```

- `assert 조건`은 "이 조건이 맞아야 해. 아니면 여기서 멈춰!"라는 뜻이에요.
- 왜 확인하냐면, 뒤에서 쓰는 함수 중 일부(`root_mean_squared_error` 등)가 옛날 버전엔 없기 때문이에요. 나중에 이상한 에러가 나는 것보다 처음에 바로 알려주는 게 친절하죠.
- `Version()`으로 감싸는 이유: 문자열로 비교하면 `"1.10" < "1.6"`이 돼 버려요(글자 순서로 비교하니까). `Version`은 숫자로 제대로 비교해 줍니다.

---

## 2. 데이터 가져오기 (2.3)

### 2-1. 데이터 다운로드 함수

```python
from pathlib import Path          # 파일 경로를 다루는 도구
import pandas as pd               # 표 형태 데이터를 다루는 도구
import tarfile                    # .tgz 압축 파일을 푸는 도구
import urllib.request             # 인터넷에서 파일을 받는 도구

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")        # 압축 파일을 저장할 위치
    if not tarball_path.is_file():                     # 아직 안 받았으면
        Path("datasets").mkdir(parents=True, exist_ok=True)   # datasets 폴더 만들기
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)  # 다운로드
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")   # 압축 풀기
    return pd.read_csv(Path("datasets/housing/housing.csv"))   # CSV를 표로 읽어서 돌려주기

housing_full = load_housing_data()
```

**왜 굳이 함수로 만들까요?**
데이터가 나중에 업데이트되면 함수만 다시 실행하면 되니까요. 그리고 `if not ... is_file()` 덕분에 이미 받은 파일이 있으면 다시 받지 않아요. 노트북을 열 번 실행해도 다운로드는 한 번만 일어납니다.

몇 가지 세부사항:
- `mkdir(parents=True, exist_ok=True)` → "필요하면 상위 폴더까지 만들고, 이미 있어도 에러 내지 마"
- `with ... as ...:` → 파일을 열고, 블록이 끝나면 **자동으로 닫아** 줘요. 닫는 걸 깜빡할 일이 없어요.
- `filter="data"` → 압축 파일 안에 `../../중요파일` 같은 위험한 경로가 숨어 있어도 막아 주는 안전장치예요.

### 2-2. 데이터 처음 들여다보기

```python
housing_full.head()          # 맨 위 5줄 보기
housing_full.info()          # 각 열의 이름, 개수, 자료형 보기
housing_full["ocean_proximity"].value_counts()   # 글자 열의 종류별 개수
housing_full.describe()      # 숫자 열의 평균, 최솟값, 최댓값 등
```

`info()` 결과에서 꼭 봐야 할 두 줄:

```
RangeIndex: 20640 entries, 0 to 20639             ← 전체 20,640줄
 4   total_bedrooms      20433 non-null  float64   ← 20,433개만 값이 있음 → 207개가 비어 있다!
 9   ocean_proximity     20640 non-null  object    ← object = 글자. 이것만 숫자가 아님
```

- **비어 있는 칸(결측치)**: 대부분의 머신러닝 모델은 빈칸이 있으면 동작하지 않아요. 나중에 채워 줘야 해요.
- **글자 열**: 모델은 숫자만 이해해요. 나중에 숫자로 바꿔 줘야 해요.

`describe()`가 보여주는 값:

| 이름 | 뜻 |
|---|---|
| count | 값이 있는 개수 |
| mean | 평균 |
| std | 표준편차 (값들이 평균에서 얼마나 퍼져 있나) |
| min / max | 최솟값 / 최댓값 |
| 25% / 50% / 75% | 작은 순으로 줄 세웠을 때 25%, 50%(=중간값), 75% 위치의 값 |

### 2-3. 히스토그램으로 분포 보기

```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)            # 그래프 글자 크기 기본값 설정 (보기 좋게)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12, 8))   # 숫자 열마다 히스토그램, 막대 50개
plt.show()
```

**히스토그램**은 "어떤 값이 몇 개나 있나"를 막대로 보여주는 그래프예요. 여기서 발견할 수 있는 것:

1. **소득은 달러 단위가 아니다** — 0.5~15 사이의 값이에요. 데이터를 만든 사람이 미리 스케일을 바꿔 둔 거예요. 이런 건 데이터 출처에 꼭 물어봐야 해요.
2. **집값과 주택 연식은 끝이 잘려 있다** — 히스토그램 오른쪽 끝에 막대가 툭 솟아 있어요. $500,000 넘는 집은 전부 $500,000으로 적혀 있다는 뜻이에요. 그러면 모델은 "$500,000보다 비싼 집"이 있다는 걸 배울 수 없어요.
3. **오른쪽으로 꼬리가 긴 분포가 많다** — 대부분 작은 값인데 가끔 아주 큰 값이 있는 모양이에요. 나중에 로그를 씌워서 고르게 펴 줄 거예요.

### 2-4. 테스트 세트 먼저 떼어 두기 ⭐

> **가장 중요한 원칙**: 데이터를 자세히 들여다보기 **전에** 시험용 데이터(테스트 세트)부터 따로 떼어 서랍에 넣어 두세요.

**왜 그래야 할까요?** 수능 문제를 미리 훑어보고 공부하면, 실제 실력보다 점수가 잘 나오겠죠. 데이터도 똑같아요. 테스트 세트를 보고 나면 무의식적으로 거기에 맞는 모델을 고르게 돼요. 이걸 **데이터 스누핑 편향**이라고 불러요.

보통 전체의 **20%**를 테스트 세트로 떼어 둡니다. 떼는 방법이 여러 가지인데, 하나씩 발전시켜 볼게요.

#### ① 가장 단순한 방법: 섞어서 자르기

```python
import numpy as np

def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))        # 0~20639 번호를 무작위로 섞기
    test_set_size = int(len(data) * test_ratio)          # 테스트 세트 크기 = 20640 × 0.2 = 4128
    test_indices = shuffled_indices[:test_set_size]      # 섞인 번호 앞쪽 4128개 → 테스트
    train_indices = shuffled_indices[test_set_size:]     # 나머지 → 훈련
    return data.iloc[train_indices], data.iloc[test_indices]   # 번호로 행 뽑기

rng = np.random.default_rng(seed=42)   # 난수 생성기. seed를 고정하면 매번 같은 결과
train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
len(train_set)   # 16512
len(test_set)    # 4128
```

카드를 섞어서 위에서 20%를 떼는 것과 같아요. `seed=42`는 "섞는 방식을 고정해 줘"라는 뜻이라 다시 실행해도 같은 결과가 나와요. (42에 특별한 의미는 없어요. 관례예요.)

**문제점**: 나중에 데이터가 추가되면 카드 수가 바뀌니까 섞는 결과도 완전히 달라져요. 그러면 전에 시험 문제였던 데이터가 이번엔 공부 데이터로 들어갈 수 있어요. 수능 문제가 문제집에 섞여 들어가는 셈이죠.

#### ② 데이터가 추가돼도 안전한 방법: ID 해시

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    # ID를 0 ~ 약 43억 사이의 숫자(해시)로 바꾸고, 그게 하위 20% 안이면 테스트 세트
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))   # 각 ID마다 True/False
    return data.loc[~in_test_set], data.loc[in_test_set]   # ~ 는 True/False 뒤집기
```

**해시**는 어떤 값을 넣으면 뒤죽박죽인 숫자를 돌려주는 함수예요. 핵심은 **같은 값을 넣으면 언제나 같은 숫자가 나온다**는 거예요. 그래서 "ID 123번은 항상 테스트 세트"처럼 고정돼요. 데이터가 늘어나도 기존 샘플의 자리는 바뀌지 않아요.

```python
housing_with_id = housing_full.reset_index()      # 행 번호(0,1,2...)를 'index'라는 열로 추가
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
```

행 번호를 ID로 쓰는 방법이에요. 그런데 이러면 새 데이터를 꼭 맨 끝에 붙여야 하고, 중간 행을 지우면 안 돼요(번호가 밀리니까).

```python
housing_with_id["id"] = (housing_full["longitude"] * 1000
                         + housing_full["latitude"])   # 위치로 ID 만들기
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
```

그래서 **절대 안 바뀌는 값**인 위치(경도·위도)로 ID를 만들었어요. 구역 위치는 데이터가 추가돼도 그대로니까요.

#### ③ 실무에서 쓰는 방법: `train_test_split`

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)
```

①을 사이킷런이 한 줄로 해 주는 함수예요. `random_state`는 ①의 `seed`와 같은 역할이에요.

#### ④ 그런데 무작위로 뽑으면 운이 나쁠 수도 있어요

예를 하나 들어볼게요. 어떤 나라 인구의 51.6%가 여성이라고 해요. 1,000명을 무작위로 뽑아 설문하면 표본도 51.6% 근처가 나올까요?

```python
from scipy.stats import binom      # 이항분포: "동전 n번 던져서 앞면이 k번 나올 확률"

sample_size = 1000
ratio_female = 0.516
proba_too_small = binom(sample_size, ratio_female).cdf(490 - 1)   # 여성이 489명 이하일 확률
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)   # 여성이 541명 이상일 확률
print(proba_too_small + proba_too_large)   # 약 0.107
```

결과는 약 **10.7%**예요. 열 번 중 한 번꼴로 여성 비율이 49% 미만이거나 54% 초과인, **치우친 표본**이 뽑힌다는 뜻이에요.

같은 걸 직접 10만 번 실험해서 확인할 수도 있어요.

```python
rng = np.random.default_rng(seed=42)
samples = (rng.random((100_000, sample_size)) < ratio_female).sum(axis=1)
#  ① 0~1 사이 난수를 10만 × 1000개 만들고
#  ② 0.516보다 작으면 "여성"(True)으로 치고
#  ③ 줄마다 True 개수를 세면 → 설문 10만 번의 여성 수
((samples < 490) | (samples > 540)).mean()   # 치우친 경우의 비율 → 역시 약 0.107
```

**해결책 = 계층 샘플링**: 전체를 그룹(계층)으로 나누고, **각 그룹에서 비율대로** 뽑는 방법이에요. 남녀 비율이 51.6 : 48.4라면 표본도 정확히 516명 : 484명을 뽑는 식이죠.

#### ⑤ 소득을 기준으로 그룹 나누기

전문가가 "집값을 예측할 때 **소득**이 제일 중요하다"고 알려줬다고 해요. 그러면 테스트 세트의 소득 분포가 전체와 똑같아야 공정한 시험이 되겠죠. 소득은 연속된 숫자라서, 먼저 5개 구간으로 나눠 줍니다.

```python
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],   # 구간 경계
                                    labels=[1, 2, 3, 4, 5])                 # 구간 이름
```

| 소득 값 | 구간 |
|---|---|
| 0 ~ 1.5 | 1 |
| 1.5 ~ 3.0 | 2 |
| 3.0 ~ 4.5 | 3 |
| 4.5 ~ 6.0 | 4 |
| 6.0 이상 (`np.inf` = 무한대) | 5 |

```python
cat_counts = housing_full["income_cat"].value_counts().sort_index()   # 구간별 개수, 1~5 순서로
cat_counts.plot.bar(rot=0, grid=True)                                 # 막대그래프 (rot=0: 글자 안 기울이기)
```

> 💡 구간을 너무 잘게 나누면 구간마다 데이터가 몇 개 없어서 비율을 맞추기 어려워요. 그래서 5개 정도로 적당히 나눠요.

#### ⑥ 계층 샘플링 실행

```python
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],   # ← 이 기준으로 비율 맞춰서!
    random_state=42)
```

`stratify=` 한 줄만 추가하면 끝이에요. "income_cat 비율을 유지하면서 나눠 줘"라는 뜻이에요.

참고로, 나누는 방법을 여러 개(예: 10개) 만들고 싶을 때는 이렇게 해요. 나중에 교차 검증에 쓰는 방식이에요.

```python
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)   # 서로 다른 분할 10개
strat_splits = []
for train_index, test_index in splitter.split(housing_full,
                                              housing_full["income_cat"]):   # 두 번째 인자가 비율 기준
    strat_train_set_n = housing_full.iloc[train_index]   # 받은 번호로 행 뽑기
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]   # 그중 첫 번째 사용
```

#### ⑦ 정말 효과가 있었는지 비교

```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)   # 구간별 비율

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing_full),       # 전체 데이터
    "Stratified %": income_cat_proportions(strat_test_set),  # 계층 샘플링 테스트 세트
    "Random %": income_cat_proportions(test_set),            # 무작위 테스트 세트
}).sort_index()
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)   # 전체 대비 얼마나 어긋났나
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)
```

| 소득 구간 | 전체 % | 계층 % | 무작위 % | 계층 오차 % | 무작위 오차 % |
|---|---|---|---|---|---|
| 1 | 3.98 | 4.00 | 4.24 | 0.36 | **6.45** |
| 2 | 31.88 | 31.88 | 30.74 | −0.02 | −3.59 |
| 3 | 35.06 | 35.05 | 34.52 | −0.01 | −1.53 |
| 4 | 17.63 | 17.64 | 18.41 | 0.03 | 4.42 |
| 5 | 11.44 | 11.43 | 12.09 | −0.08 | 5.63 |

계층 샘플링은 거의 완벽하게 비율이 맞고(최대 0.36% 차이), 무작위는 최대 6.45%나 어긋났어요. 특히 **원래 개수가 적은 그룹**(1구간, 3.98%)일수록 무작위로 뽑으면 크게 흔들려요.

#### ⑧ 다 쓴 임시 열 지우기

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)   # axis=1: 열을 지움, inplace=True: 원본을 직접 수정
```

`income_cat`은 나누는 데만 쓰려고 만든 열이라 이제 지워요. 변수 이름을 `set`이 아니라 `set_`로 쓴 건, `set`이 파이썬에 이미 있는 이름이라 덮어쓰지 않으려는 거예요.

---

## 3. 탐색과 시각화 (2.4)

이제 데이터와 친해질 시간이에요. **훈련 세트만** 사용합니다.

```python
housing = strat_train_set.copy()   # 복사본을 만들어서 마음껏 실험 (원본은 보호)
```

### 3-1. 지도처럼 그려 보기

데이터에 위도·경도가 있으니까 점으로 찍으면 캘리포니아 지도가 나와요.

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)              # 그냥 찍기
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)   # 투명하게 찍기
```

첫 번째 그림은 점이 너무 빽빽해서 캘리포니아 모양만 보여요. `alpha=0.2`로 점을 반투명하게 만들면, **점이 많이 겹친 곳이 진하게** 보여서 사람이 많이 사는 곳(샌프란시스코, LA, 샌디에이고 등)이 드러나요.

정보를 더 담아 볼게요.

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",   # 점 크기 = 인구
             c="median_house_value", cmap="jet", colorbar=True,   # 점 색깔 = 집값
             legend=True, sharex=False, figsize=(10, 7))
```

| 옵션 | 하는 일 |
|---|---|
| `s=population / 100` | 인구가 많을수록 큰 원. 100으로 나누지 않으면 원이 너무 커서 화면을 덮어요 |
| `c="median_house_value"` | 집값에 따라 색을 칠해요 |
| `cmap="jet"` | 파란색(싸다) → 빨간색(비싸다) 색상표 |
| `colorbar=True` | 오른쪽에 색깔 범례 막대 표시 |
| `sharex=False` | pandas 그래프의 x축 글자 버그를 피하는 옵션 (신경 안 써도 돼요) |

**발견**: 바닷가와 대도시 근처가 빨갛다 = 비싸다. 그러면 "**대도시 중심에서 얼마나 가까운가**"가 좋은 특성이 될 것 같아요. 이 아이디어를 나중에 `ClusterSimilarity`라는 도구로 구현합니다.

### 3-2. 어떤 특성이 집값과 관련 있을까? (상관관계)

```python
corr_matrix = housing.corr(numeric_only=True)    # 모든 숫자 열끼리 상관계수 계산
corr_matrix["median_house_value"].sort_values(ascending=False)   # 집값 기준으로 큰 순 정렬
```

```
median_income         0.688380    ← 소득이 높을수록 집값도 높다 (강한 관계)
total_rooms           0.137455
housing_median_age    0.102175
households            0.071426
total_bedrooms        0.054635
population           -0.020153    ← 거의 관계 없음
longitude            -0.050859
latitude             -0.139584    ← 북쪽으로 갈수록 살짝 싸진다
```

**상관계수**는 −1 ~ 1 사이의 값이에요.

| 값 | 뜻 |
|---|---|
| 1에 가까움 | 하나가 오르면 다른 것도 오른다 |
| 0에 가까움 | 별 관계 없다 |
| −1에 가까움 | 하나가 오르면 다른 건 내린다 |

> ⚠️ 상관계수는 **직선 관계**만 잡아요. "x가 0 근처일 때만 y가 크다" 같은 곡선 관계는 상관계수가 0으로 나올 수 있어요. 그래서 그래프로도 꼭 확인해야 해요.

그래프로 확인:

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))   # 4개 열의 모든 짝을 산점도로 (4×4)
```

대각선 칸은 자기 자신과의 산점도라 그냥 직선이 돼서 의미가 없어요. 그래서 그 자리엔 히스토그램을 대신 그려 줘요.

가장 관련 있어 보이는 소득만 크게 봅니다.

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
```

- 소득이 오르면 집값도 오르는 경향이 확실히 보여요.
- 그런데 **$500,000 위치에 수평선**이 있어요. 아까 본 "상한으로 잘린" 흔적이에요. $450,000, $350,000 근처에도 희미한 선이 있어요.
- 모델이 이런 이상한 선까지 따라 배우지 않도록, 이런 구역들을 빼는 것도 고려할 수 있어요.

### 3-3. 특성을 조합해서 새로 만들어 보기

`total_rooms`는 **구역 전체**의 방 개수예요. 인구가 많은 큰 구역이면 당연히 많겠죠. 이건 "집이 넓은가"를 말해 주지 않아요. 진짜 궁금한 건 **집 한 채당** 방 개수예요.

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]      # 집당 방 수
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]   # 방 중 침실 비율
housing["people_per_house"] = housing["population"] / housing["households"]      # 집당 인원

corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

```
median_income         0.688380
rooms_per_house       0.143663   ← total_rooms(0.137)보다 조금 더 관련 있음
...
people_per_house     -0.038224
bedrooms_ratio       -0.256397   ← total_bedrooms(0.055)보다 훨씬 관련 있음!
```

`bedrooms_ratio`가 −0.26이라는 건, **방 중에 침실 비율이 낮을수록 비싸다**는 뜻이에요. 생각해 보면 당연하죠. 비싼 집일수록 침실 말고도 거실, 서재, 드레스룸 같은 방이 많으니까요.

> 💡 **교훈**: 원래 있는 열을 그대로 쓰는 것보다, 머리를 써서 **의미 있는 비율**을 만드는 게 더 강력할 수 있어요. 이걸 **특성 공학**(feature engineering)이라고 해요.

---

## 4. 데이터 준비 (2.5)

이제 모델이 먹을 수 있게 재료를 손질할 차례예요. 이 장에서 가장 길고 중요한 부분이에요.

손질해야 할 것 3가지:
1. **빈칸**이 있다 (`total_bedrooms` 207개) → 채워야 함
2. **글자 열**이 있다 (`ocean_proximity`) → 숫자로 바꿔야 함
3. **숫자 크기가 제각각**이다 (방 수는 수만, 소득은 0~15) → 크기를 맞춰야 함

그리고 최종 목표는 이 손질 과정을 **자동화**하는 거예요. 새 데이터가 들어올 때마다 손으로 하면 실수하기 쉬우니까요.

### 4-0. 입력과 정답 분리하기

```python
housing = strat_train_set.drop("median_house_value", axis=1)   # 입력(X): 집값 열을 뺀 나머지
housing_labels = strat_train_set["median_house_value"].copy()  # 정답(y): 집값 열만
```

모델에게는 **문제(X)**와 **정답(y)**을 따로 줘야 해요. `drop()`은 원본을 건드리지 않고 새 표를 돌려줘요. (3-3에서 만든 비율 열들도 여기서 사라져요. 나중에 파이프라인이 자동으로 다시 만들어 줄 거예요.)

### 4-1. 빈칸 채우기

먼저 어떤 행이 비어 있는지 확인:

```python
null_rows_idx = housing.isnull().any(axis=1)   # 빈칸이 하나라도 있는 행이면 True
housing.loc[null_rows_idx].head()              # 그런 행만 골라서 보기
```

빈칸을 처리하는 방법은 3가지예요.

```python
# 방법 1: 빈칸 있는 행을 통째로 버리기
housing_option1 = housing.copy()
housing_option1.dropna(subset=["total_bedrooms"], inplace=True)

# 방법 2: 빈칸 있는 열을 통째로 버리기
housing_option2 = housing.copy()
housing_option2.drop("total_bedrooms", axis=1, inplace=True)

# 방법 3: 빈칸을 그 열의 중간값으로 채우기
housing_option3 = housing.copy()
median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"] = housing_option3["total_bedrooms"].fillna(median)
```

| 방법 | 좋은 점 | 아쉬운 점 |
|---|---|---|
| 1. 행 버리기 | 간단 | 데이터가 줄어요 |
| 2. 열 버리기 | 간단 | 쓸모 있는 정보까지 버려요 |
| 3. **채우기** ✅ | 데이터를 살려요 | 채운 값이 진짜 값은 아니에요 |

보통 **방법 3**을 써요. 그런데 직접 하면 귀찮은 점이 있어요. 훈련 세트에서 구한 중간값을 **어딘가 기억해 뒀다가** 테스트 세트나 새 데이터에도 **같은 값**으로 채워야 하거든요. (테스트 세트의 중간값으로 채우면 반칙이에요!)

이걸 대신 해 주는 도구가 `SimpleImputer`예요.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")              # "중간값으로 채우는 도구" 만들기
housing_num = housing.select_dtypes(include=[np.number])   # 숫자 열만 고르기 (글자는 중간값이 없으니까)
imputer.fit(housing_num)        # 각 열의 중간값을 계산해서 기억 ← "배우기"
imputer.statistics_             # 기억한 중간값들 확인
housing_num.median().values     # 직접 계산한 것과 똑같아요

X = imputer.transform(housing_num)   # 기억한 중간값으로 빈칸 채우기 ← "적용하기"
```

**`fit`과 `transform`의 차이** (이 장 전체에서 가장 중요한 개념!)

| 메서드 | 하는 일 | 비유 |
|---|---|---|
| `fit(데이터)` | 데이터를 보고 필요한 값을 **배워서 기억** | 레시피 익히기 |
| `transform(데이터)` | 기억한 값으로 데이터를 **변환** | 익힌 레시피대로 요리하기 |
| `fit_transform(데이터)` | 둘을 한 번에 | |

> ⚠️ 훈련 세트로만 `fit`하고, 테스트 세트에는 `transform`만 해요. 테스트 세트로 `fit`하면 시험 문제를 미리 보는 거예요.

결과는 표(DataFrame)가 아니라 숫자 배열(NumPy)로 나와서 열 이름이 사라져요. 다시 표로 감싸 줍니다.

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns,   # 원래 열 이름 붙이기
                          index=housing_num.index)           # 원래 행 번호 붙이기
```

#### 📌 사이킷런 도구들의 공통 규칙

사이킷런의 모든 도구는 같은 규칙을 따라서, 하나만 익히면 나머지도 쉽게 써요.

| 종류 | 가진 메서드 | 예 |
|---|---|---|
| **추정기** (뭔가를 배우는 것) | `fit()` | 모든 도구 |
| **변환기** (데이터를 바꾸는 것) | `transform()`, `fit_transform()` | `SimpleImputer`, `StandardScaler` |
| **예측기** (예측하는 것) | `predict()`, `score()` | `LinearRegression` |

- 사람이 정하는 설정값(하이퍼파라미터)은 만들 때 넣어요 → `SimpleImputer(strategy="median")`
- 배워서 알게 된 값은 **이름 끝에 밑줄 `_`**이 붙어요 → `statistics_`, `mean_`

#### 이상치 찾기 (참고)

```python
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)   # 1이면 정상, -1이면 이상치

# 이상치를 빼고 싶다면 (이 장에서는 실제로 하지 않음):
#housing = housing.iloc[outlier_pred == 1]
#housing_labels = housing_labels.iloc[outlier_pred == 1]
```

`IsolationForest`는 "다른 데이터와 쉽게 떼어지는 외톨이"를 이상치로 판단해요. 만약 빼려면 **입력(X)과 정답(y)을 같이** 빼야 해요. 한쪽만 빼면 줄이 어긋나요.

### 4-2. 글자를 숫자로 바꾸기

```python
housing_cat = housing[["ocean_proximity"]]   # 대괄호 두 겹 → 표(2차원) 형태 유지
```

사이킷런 도구들은 입력이 **표 모양(2차원)**이어야 해요. `housing["열"]`처럼 대괄호 한 겹이면 한 줄짜리(1차원)가 돼서 에러가 나요.

#### 방법 1: 번호 붙이기 (OrdinalEncoder)

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]          # [[3.], [0.], [1.], [1.], [4.], [1.], [0.], [3.]]
ordinal_encoder.categories_      # 어떤 글자가 몇 번이 됐는지
# ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
#      0          1         2          3            4
```

간단하지만 문제가 있어요. 모델은 숫자를 보고 **"0과 1은 가깝고, 0과 4는 멀다"**고 생각해요. 그런데 실제로는 0(바다 1시간 이내)과 4(바다 근처)가 오히려 비슷하잖아요. 엉뚱한 거리 관계를 배우게 되는 거예요.

→ "나쁨 < 보통 < 좋음"처럼 **순서가 있는 경우**에만 써요.

#### 방법 2: 원-핫 인코딩 (OneHotEncoder) ✅

종류마다 **열을 하나씩** 만들고, 해당하는 칸에만 1을 넣는 방법이에요.

| 원래 값 | <1H OCEAN | INLAND | ISLAND | NEAR BAY | NEAR OCEAN |
|---|---|---|---|---|---|
| INLAND | 0 | **1** | 0 | 0 | 0 |
| NEAR BAY | 0 | 0 | 0 | **1** | 0 |

이러면 모든 종류 사이의 거리가 똑같아서 엉뚱한 순서가 생기지 않아요.

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot               # <16512x5 sparse matrix ...>  ← "희소 행렬"
housing_cat_1hot.toarray()     # 일반 배열로 펼쳐 보기
```

**희소 행렬**이 뭘까요? 위 표를 보면 대부분이 0이에요. 0을 전부 저장하는 건 낭비라서 **"1이 어디 있는지"만** 기록하는 압축 형식이에요. 종류가 수천 개일 때 메모리를 크게 아껴 줘요.

```python
cat_encoder = OneHotEncoder(sparse_output=False)   # 처음부터 일반 배열로 받고 싶으면
```

#### ⚠️ 왜 `pd.get_dummies()`가 아니라 `OneHotEncoder`를 쓸까?

pandas에도 원-핫 인코딩 함수 `get_dummies()`가 있어요. 그런데 중요한 차이가 있어요.

```python
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)          # 열이 2개만 생김 (INLAND, NEAR BAY)
cat_encoder.transform(df_test)   # 훈련 때 본 5개 열 그대로 나옴
```

`get_dummies()`는 **지금 보이는 종류만** 열로 만들어요. 훈련할 땐 5열이었는데 예측할 땐 2열이면, 모델은 입력 모양이 달라서 동작하지 않아요. `OneHotEncoder`는 **훈련 때 본 종류를 기억**하니까 항상 같은 모양을 내 줘요.

처음 보는 종류가 들어오면?

```python
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
cat_encoder.handle_unknown = "ignore"     # 처음 보는 값은 무시해 줘
cat_encoder.transform(df_test_unknown)
# [[0., 0., 0., 0., 0.],   ← 처음 본 "<2H OCEAN"은 전부 0
#  [0., 0., 1., 0., 0.]]
```

기본 설정에선 에러가 나요. `"ignore"`로 바꾸면 모든 칸을 0으로 처리해 줘요.

```python
cat_encoder.get_feature_names_out()   # 만들어진 열 이름 확인
# ['ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', ...]
```

### 4-3. 숫자 크기 맞추기 (스케일링)

방 개수는 6 ~ 39,320인데 소득은 0.5 ~ 15예요. 대부분의 모델은 **숫자가 큰 특성을 더 중요하게** 착각해요. 그래서 크기를 비슷하게 맞춰 줘야 해요.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))     # 모든 값을 -1 ~ 1 사이로
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

std_scaler = StandardScaler()                             # 평균 0, 표준편차 1로
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

| 방법 | 어떻게 바꾸나 | 장단점 |
|---|---|---|
| **MinMaxScaler** | 가장 작은 값 → −1, 가장 큰 값 → 1 | 이상치 하나가 엄청 크면 나머지가 다 한쪽으로 몰려요 |
| **StandardScaler** ✅ | (값 − 평균) ÷ 표준편차 | 범위는 정해지지 않지만 이상치에 덜 휘둘려요 |

> 예: 평균 100, 표준편차 20인 열에서 140은 → (140 − 100) ÷ 20 = **2** → "평균보다 표준편차 2개만큼 크다"

#### 꼬리가 긴 분포 → 로그를 씌워 펴 주기

```python
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)                 # 원래 인구 분포
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)   # 로그 씌운 분포
```

인구 분포는 대부분 작은 값에 몰려 있고 오른쪽으로 길게 늘어져 있어요. 로그를 씌우면 큰 값은 많이 줄고 작은 값은 덜 줄어서, **종 모양**에 가까워져요. 스케일러는 크기만 바꾸지 모양은 못 바꾸니까, **로그 → 스케일링** 순서로 해요.

#### 순위(백분위)로 바꾸기

```python
percentiles = [np.percentile(housing["median_income"], p) for p in range(1, 100)]   # 1%, 2%, ..., 99% 지점의 값
flattened_median_income = pd.cut(housing["median_income"],
                                 bins=[-np.inf] + percentiles + [np.inf],   # 그 지점들을 경계로
                                 labels=range(1, 100 + 1))                  # 1~100등급
```

"이 구역의 소득은 상위 몇 %인가"로 바꾸는 방법이에요. 각 등급에 정확히 1%씩 들어가니까 분포가 평평해져요.

#### 봉우리가 여러 개인 분포 → "특정 값과 얼마나 가까운가"

```python
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

주택 연식 분포는 35년 근처에 봉우리가 있어요. 그래서 "**35년과 얼마나 비슷한가**"라는 새 특성을 만들어요.

- 연식이 정확히 35년이면 → 1
- 35년에서 멀어질수록 → 0에 가까워짐
- `gamma`가 클수록 → 조금만 멀어져도 빨리 0이 됨 (더 까다로움)

이렇게 하면 단순한 모델도 "35년 근처 집은 특별하다" 같은 패턴을 잡을 수 있어요.

#### 정답(집값)도 변환했다면 → 예측 후 되돌리기

```python
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())   # 집값을 표준화

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)     # 표준화된 집값으로 학습
some_new_data = housing[["median_income"]].iloc[:5]      # 새 데이터라고 칠게요

scaled_predictions = model.predict(some_new_data)        # 예측값도 표준화된 숫자로 나옴 (예: 0.3)
predictions = target_scaler.inverse_transform(scaled_predictions)   # 다시 달러로 되돌리기
```

정답을 바꿔서 학습했으면 예측값도 바뀐 단위로 나와요. `inverse_transform`으로 원래 단위로 돌려놔야 해요. 이걸 깜빡하기 쉬우니까, 자동으로 해 주는 도구가 있어요.

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())   # 정답 변환·복원을 알아서
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)    # 바로 달러 단위로 나옴
```

### 4-4. 나만의 변환기 만들기

사이킷런에 없는 변환이 필요할 때는 직접 만들 수 있어요.

#### 간단한 경우: 함수 하나로 끝나는 변환 (`FunctionTransformer`)

배울 게 없는 변환(로그 씌우기, 나누기 등)은 함수를 감싸기만 하면 돼요.

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)   # 로그 / 되돌리기는 exp
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))   # 함수에 줄 추가 인자
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41    # 샌프란시스코 좌표
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])   # 샌프란시스코와 얼마나 가까운가

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])   # 첫째 열 ÷ 둘째 열
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))   # [[0.5], [0.75]]
```

> 💡 `X[:, [0]]`처럼 대괄호 안에 **리스트**로 쓰면 결과가 세로 표 모양(2차원)으로 유지돼요. `X[:, 0]`이라고 쓰면 한 줄(1차원)이 돼 버려요.

#### 배워야 하는 경우: 클래스로 만들기

평균처럼 **데이터를 보고 배워야 하는** 변환은 클래스로 만들어요. 연습 삼아 `StandardScaler`를 직접 만들어 봅니다.

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):   # 설정값을 받아서
        self.with_mean = with_mean        # 같은 이름으로 저장만 해요

    def fit(self, X, y=None):             # y는 안 쓰지만 자리는 있어야 해요
        X = check_array(X)                # 제대로 된 숫자 배열인지 검사
        self.mean_ = X.mean(axis=0)       # 열마다 평균을 배워서 기억
        self.scale_ = X.std(axis=0)       # 열마다 표준편차를 배워서 기억
        self.n_features_in_ = X.shape[1]  # 열 개수도 기억
        return self                       # 꼭 자기 자신을 돌려주기!

    def transform(self, X):
        check_is_fitted(self)             # fit을 먼저 했는지 확인 (밑줄 붙은 속성이 있나 봄)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]   # 열 개수가 훈련 때와 같은지
        if self.with_mean:
            X = X - self.mean_            # 평균 빼기
        return X / self.scale_            # 표준편차로 나누기
```

규칙이 많아 보이지만 이유를 알면 쉬워요.

| 규칙 | 왜? |
|---|---|
| `BaseEstimator`를 물려받기 | 설정값을 조회·변경하는 기능(`get_params`, `set_params`)이 자동으로 생겨요. 나중에 자동 튜닝할 때 꼭 필요해요 |
| `TransformerMixin`을 물려받기 | `fit_transform()`을 따로 안 만들어도 공짜로 생겨요 |
| `__init__`에서는 **저장만** | 사이킷런이 "이 도구의 설정값이 뭐지?"를 `__init__`의 인자 이름으로 알아내요 |
| `fit(X, y=None)` | `y`를 안 써도 자리가 있어야 파이프라인에서 문제없이 돌아가요 |
| 배운 값은 `이름_` | "이건 배워서 생긴 값이야"라는 표시. `check_is_fitted`가 이걸 보고 학습 여부를 판단해요 |
| `return self` | `scaler.fit(X).transform(X)`처럼 이어서 쓸 수 있어요 |

#### ⭐ 이 장의 핵심 변환기: `ClusterSimilarity`

3-1에서 "대도시와 얼마나 가까운가가 중요하겠다"고 했죠. 그런데 어디가 대도시인지 일일이 좌표를 적기는 힘들어요. 그래서 **데이터가 몰린 곳을 자동으로 찾아서**, 각 구역이 그곳들과 얼마나 가까운지를 특성으로 만들어요.

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters      # 중심을 몇 개 찾을지
        self.gamma = gamma                # 가까움을 얼마나 까다롭게 볼지
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)   # 데이터가 몰린 중심 10곳 찾기
        return self

    def transform(self, X):
        # 각 구역이 10개 중심 각각과 얼마나 가까운지 (0~1)
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]   # 결과 열 이름
```

**KMeans**는 비슷한 점끼리 묶어서 그룹(군집)을 만들고, 각 그룹의 **중심점**을 찾아주는 알고리즘이에요(8장에서 자세히 배워요).

| 단계 | 하는 일 |
|---|---|
| `fit` | 위도·경도로 KMeans를 돌려 구역들이 몰린 중심 10곳을 찾아 기억 |
| `transform` | 각 구역이 중심 10곳 각각과 얼마나 가까운지 계산 → 10개 열 |
| `get_feature_names_out` | 나중에 "이 열이 뭐였지?" 할 때 이름을 알려줌 |

```python
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
similarities[:3].round(2)
# [[0.  , 0.14, 0.  , 0.  , 0.  , 0.08, 0.  , 0.99, 0.  , 0.6 ],   ← 첫 구역: 7번 중심 바로 옆(0.99)
#  [0.63, 0.  , 0.99, 0.  , 0.  , 0.  , 0.04, 0.  , 0.11, 0.  ], ...]  ← 둘째 구역: 2번 중심 바로 옆
```

### 4-5. 전처리를 한 번에: 파이프라인

지금까지 빈칸 채우기 → 숫자 변환 → 스케일링을 따로따로 했어요. 이걸 매번 손으로 순서대로 하면 실수하기 쉬워요. **파이프라인**은 이 단계들을 컨베이어 벨트처럼 연결해 줘요.

#### 기본 파이프라인

```python
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),   # 1단계: 빈칸을 중간값으로
    ("standardize", StandardScaler()),              # 2단계: 표준화
])
```

`("이름", 도구)` 쌍을 순서대로 적으면 돼요. 이름을 짓기 귀찮으면:

```python
from sklearn.pipeline import make_pipeline

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
# 이름이 자동으로 "simpleimputer", "standardscaler"가 돼요
```

**어떻게 동작하나요?** `num_pipeline.fit_transform(데이터)`을 호출하면:
1. 1단계가 데이터를 `fit_transform` → 결과를
2. 2단계에 넘겨서 `fit_transform` → 최종 결과

```python
housing_num_prepared = num_pipeline.fit_transform(housing_num)   # 두 단계가 한 번에!
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)
```

파이프라인 안을 들여다보는 방법:

```python
from sklearn import set_config
set_config(display='diagram')
num_pipeline                              # 주피터에서 상자 그림으로 예쁘게 보여줌

num_pipeline.steps                        # 단계 목록
num_pipeline[1]                           # 두 번째 단계 (리스트처럼 꺼내기)
num_pipeline[:-1]                         # 마지막 단계 빼고 나머지
num_pipeline.named_steps["simpleimputer"] # 이름으로 꺼내기
num_pipeline.set_params(simpleimputer__strategy="median")   # 안쪽 설정값 바꾸기
```

> 💡 `simpleimputer__strategy`처럼 **`단계이름__설정이름`** (밑줄 두 개)으로 파이프라인 안쪽 설정을 바꿀 수 있어요. 나중에 자동 튜닝할 때 이 문법을 그대로 써요. 꼭 기억해 두세요!

#### 열마다 다른 처리: `ColumnTransformer`

숫자 열과 글자 열은 손질 방법이 달라요. `ColumnTransformer`는 "이 열들엔 이 처리, 저 열들엔 저 처리"를 한 번에 해 줘요.

```python
from sklearn.compose import ColumnTransformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),         # 글자 열 빈칸은 가장 많은 값으로
    OneHotEncoder(handle_unknown="ignore"))          # 그리고 원-핫 인코딩

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),   # 숫자 열 → 숫자 파이프라인
    ("cat", cat_pipeline, cat_attribs),   # 글자 열 → 글자 파이프라인
])
```

각 처리 결과는 **옆으로 나란히 붙여서** 하나의 큰 표가 돼요.

열 이름을 일일이 적기 귀찮으면, 자료형으로 자동 선택할 수도 있어요.

```python
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),   # 숫자 열 전부
    (cat_pipeline, make_column_selector(dtype_include=object)),      # 글자 열 전부
)
housing_prepared = preprocessing.fit_transform(housing)
```

#### ⭐⭐ 최종 전처리 파이프라인

지금까지 배운 걸 전부 합친 최종 버전이에요. 이 코드 하나가 이 장의 하이라이트예요.

```python
# ── 재료 1: 비율 계산 ──
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]          # 첫째 열 ÷ 둘째 열

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]                      # 결과 열 이름은 "ratio"

def ratio_pipeline():                     # 함수로 만든 이유는 아래 설명 참고
    return make_pipeline(
        SimpleImputer(strategy="median"),                                  # ① 빈칸 채우고
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),   # ② 나누고
        StandardScaler())                                                  # ③ 표준화

# ── 재료 2: 로그 변환 ──
log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),                                # ① 빈칸 채우고
    FunctionTransformer(np.log, feature_names_out="one-to-one"),     # ② 로그 (열 이름은 그대로)
    StandardScaler())                                                # ③ 표준화

# ── 재료 3: 대도시 근접도 ──
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

# ── 재료 4: 나머지 숫자 열 기본 처리 ──
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

# ── 전부 조립 ──
preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),      # 침실 비율
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),   # 집당 방 수
        ("people_per_house", ratio_pipeline(), ["population", "households"]),   # 집당 인원
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),                 # 꼬리 긴 5개
        ("geo", cluster_simil, ["latitude", "longitude"]),                      # 대도시 근접도
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),      # 글자 열
    ],
    remainder=default_num_pipeline)   # 위에서 안 다룬 열(housing_median_age)은 기본 처리
```

**몇 가지 궁금증 풀이**

**Q. `ratio_pipeline`은 왜 그냥 변수가 아니라 함수예요?**
비율 특성을 3개 만드는데, 파이프라인 **하나**를 세 군데서 같이 쓰면 서로 배운 내용이 덮어써져서 꼬여요. 함수로 만들면 `ratio_pipeline()`을 부를 때마다 **새 파이프라인**이 만들어져서 안전해요.

**Q. 왜 나누기 전에 빈칸부터 채워요?**
`total_bedrooms`에 빈칸이 있어요. 빈칸 ÷ 숫자 = 빈칸이 돼 버리니까, 먼저 채우고 나눠야 해요.

**Q. `ratio_name`이나 `feature_names_out`은 왜 필요해요?**
`FunctionTransformer`는 함수가 뭘 하는지 몰라서 결과 열 이름을 지을 수 없어요. 그래서 이름을 알려주는 거예요. 이걸 해 두면 나중에 "어떤 특성이 중요했나" 볼 때 열 이름으로 확인할 수 있어요.

**Q. `total_rooms`가 여러 군데 들어가도 돼요?**
네, 같은 열을 여러 처리에 써도 괜찮아요. 각 처리가 복사본을 받아요.

**Q. `remainder`는 뭐예요?**
위 목록에 없는 열을 어떻게 할지 정하는 거예요. 기본값은 `"drop"`(버림)이고, 여기선 기본 숫자 처리를 지정했어요.

어떤 특성이 몇 개 나오는지 정리하면:

| 이름 | 입력 | 나오는 열 | 의미 |
|---|---|---|---|
| bedrooms | 침실, 방 | 1 | 방 중 침실 비율 |
| rooms_per_house | 방, 가구 | 1 | 집 한 채당 방 수 |
| people_per_house | 인구, 가구 | 1 | 집 한 채당 인원 |
| log | 5개 열 | 5 | 로그로 편 값 |
| geo | 위도, 경도 | 10 | 대도시 중심 10곳과의 근접도 |
| cat | ocean_proximity | 5 | 원-핫 |
| remainder | housing_median_age | 1 | 표준화만 |
| **합계** | | **24** | |

```python
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape          # (16512, 24) ← 16,512개 구역 × 24개 특성
preprocessing.get_feature_names_out()
# ['bedrooms__ratio', 'rooms_per_house__ratio', 'people_per_house__ratio',
#  'log__total_bedrooms', ..., 'log__median_income',
#  'geo__Cluster 0 similarity', ..., 'geo__Cluster 9 similarity',
#  'cat__ocean_proximity_<1H OCEAN', ..., 'cat__ocean_proximity_NEAR OCEAN',
#  'remainder__housing_median_age']
```

이제 **원본 표를 넣으면 버튼 하나로 모델용 데이터가 나와요.** 새 데이터가 와도 `preprocessing.transform(새데이터)` 한 줄이면 끝이에요.

---

## 5. 모델 선택과 훈련 (2.6)

재료 손질이 끝났으니 드디어 요리(모델 훈련)예요! 여러 모델을 시험해 보고 가장 나은 걸 고릅니다.

### 5-1. 첫 번째 시도: 선형 회귀

**선형 회귀**는 "집값 = a×소득 + b×방수 + … + c" 같은 **직선 공식**을 찾는 가장 기본적인 모델이에요.

```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())   # 전처리 + 모델을 하나로 연결!
lin_reg.fit(housing, housing_labels)                         # 원본 표를 그대로 넣으면 됨
```

전처리 파이프라인 뒤에 모델을 붙였어요. 이제 `fit`에 **원본 데이터를 그대로** 넣으면 알아서 전처리하고 학습까지 해요.

예측해 보고 정답과 비교:

```python
housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2)    # 예측값 5개 (100원 단위로 반올림)
housing_labels.iloc[:5].values       # 실제 정답 5개

error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))
# -46.3%, -23.0%, 33.4%, -4.9%, -8.5%    ← 첫 번째는 46%나 틀렸어요
```

전체 성적(RMSE)을 계산하면:

```python
from sklearn.metrics import root_mean_squared_error

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
lin_rmse      # 68972.89 ← 평균 약 $69,000 틀림
```

집값 대부분이 $120,000 ~ $265,000 사이인데 $69,000이나 틀린다면 꽤 나쁜 성적이에요. 이건 **과소적합**이에요. 모델이 너무 단순해서 패턴을 제대로 못 잡은 거예요. (직선 공식으로는 복잡한 집값을 설명하기 어려워요.)

### 5-2. 두 번째 시도: 결정 트리

**결정 트리**는 "소득이 5보다 큰가? → 바닷가인가? → …"처럼 **스무고개 질문**을 이어 가며 예측하는 모델이에요. 복잡한 패턴도 잘 잡아요.

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
tree_rmse     # 0.0  ← ???
```

오차가 0? 완벽한 모델일까요? **아니에요!** 훈련 데이터를 통째로 **외워 버린** 거예요. 문제집 답지를 외워서 문제집은 100점이지만, 처음 보는 문제는 못 푸는 상태죠. 이게 **과대적합**이에요.

그런데 테스트 세트는 마지막까지 쓰면 안 되니까, 새 문제에서 실력이 어떤지 확인할 다른 방법이 필요해요.

### 5-3. 교차 검증: 훈련 세트 안에서 모의고사 보기

**교차 검증**은 훈련 세트를 여러 조각으로 나눠서 모의고사를 여러 번 보는 방법이에요.

```
훈련 세트를 10조각으로 나눔: [1][2][3][4][5][6][7][8][9][10]

1회차: [2]~[10]으로 공부 → [1]로 시험
2회차: [1],[3]~[10]으로 공부 → [2]로 시험
...
10회차: [1]~[9]로 공부 → [10]으로 시험

→ 점수 10개를 얻어요
```

모든 조각이 한 번씩 시험 문제가 되고, 시험 문제는 공부할 때 안 봤으니 **공정한 점수**가 나와요.

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)   # 10조각
pd.Series(tree_rmses).describe()
# mean 66573.73  ← 평균 $66,574 틀림 (0이 아니었어요!)
# std   1103.40  ← 10번의 점수가 이 정도로 들쭉날쭉
```

> 💡 **앞에 왜 마이너스(`-`)를 붙이나요?** 사이킷런은 "점수는 클수록 좋다"는 규칙을 써요. 그런데 RMSE는 작을수록 좋잖아요. 그래서 RMSE에 마이너스를 붙인 `neg_root_mean_squared_error`를 돌려줘요. 우리가 다시 마이너스를 붙여서 원래 RMSE로 되돌리는 거예요.

결정 트리는 훈련 세트에선 0점(오차), 모의고사에선 66,574점. 엄청난 차이죠? 심하게 외운 거예요. 선형 회귀보다 아주 조금 나을 뿐이에요.

```python
lin_rmses = -cross_val_score(lin_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_rmses).describe()      # mean 70003.40
```

### 5-4. 세 번째 시도: 랜덤 포레스트

**랜덤 포레스트**는 결정 트리를 **여러 개** 만들어서 **평균**내는 모델이에요. 한 사람의 의견보다 여러 사람 의견의 평균이 더 정확한 것과 같은 원리예요(6장에서 자세히 배워요). 각 트리는 조금씩 다른 데이터로 공부해서 서로 다른 실수를 하는데, 평균내면 실수가 상쇄돼요.

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
pd.Series(forest_rmses).describe()   # mean 47038.09  ← 훨씬 좋아졌어요!
```

훈련 세트 점수도 확인해 볼게요.

```python
forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)
forest_rmse = root_mean_squared_error(housing_labels, housing_predictions)
forest_rmse   # 17551.21
```

훈련 세트에선 17,551, 모의고사에선 47,038. 차이가 여전히 커요. 아직 **과대적합**이 있다는 뜻이에요. 그래도 셋 중엔 가장 좋아요.

### 5-5. 성적 비교

| 모델 | 훈련 세트 RMSE | 모의고사(교차 검증) RMSE | 진단 |
|---|---|---|---|
| 선형 회귀 | 68,973 | 70,003 | 둘 다 나쁨 → **과소적합** (너무 단순) |
| 결정 트리 | 0 | 66,574 | 차이가 엄청 큼 → **심한 과대적합** (외움) |
| 랜덤 포레스트 | 17,551 | **47,038** | 가장 좋음, 그래도 약간 과대적합 |

> 💡 **진단하는 법**
> - 훈련 점수도 나쁘고 모의고사도 나쁘다 → 과소적합 → 더 복잡한 모델, 더 좋은 특성
> - 훈련 점수는 좋은데 모의고사가 나쁘다 → 과대적합 → 모델 단순화, 데이터 추가
>
> 이 단계에서는 한 모델을 오래 붙잡지 말고, **가능성 있는 모델 2~5개를 추리는 것**이 목표예요.

---

## 6. 모델 미세 튜닝 (2.7)

랜덤 포레스트가 가장 유망해 보이니까, 이제 **설정값(하이퍼파라미터)을 조정**해서 성능을 더 끌어올려 볼게요.

### 6-1. 그리드 서치: 가능한 조합을 전부 시도

설정값 후보를 정해 주면, **모든 조합**을 하나씩 교차 검증해서 가장 좋은 걸 찾아줘요.

```python
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),                         # 전처리
    ("random_forest", RandomForestRegressor(random_state=42)),   # 모델
])

param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],      # 대도시 중심을 몇 개 찾을지
     'random_forest__max_features': [4, 6, 8]},         # 트리가 질문할 때 볼 특성 개수
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,      # 3조각 교차 검증
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```

**설정값 이름 읽는 법** — 밑줄 두 개(`__`)로 안쪽으로 들어가요.

```
preprocessing__geo__n_clusters
     ↓           ↓       ↓
full_pipeline의 "preprocessing" 단계
             → 그 안의 "geo" (ClusterSimilarity)
                      → 그것의 n_clusters 설정
```

> 💡 **전처리 설정까지 같이 튜닝할 수 있다**는 게 파이프라인의 큰 장점이에요! "대도시 중심을 몇 개로 잡을까?"도 자동으로 찾아줘요.

**몇 번이나 학습하나요?** 첫 번째 묶음 3×3 = 9가지, 두 번째 2×3 = 6가지, 합쳐서 15가지 조합. 각각 3번씩 교차 검증하니까 **45번 학습**해요. 그래서 시간이 꽤 걸려요.

결과 확인:

```python
grid_search.best_params_
# {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}

grid_search.best_estimator_    # 가장 좋은 설정으로 전체 훈련 세트를 다시 학습한 모델
```

> 💡 `n_clusters=15`는 후보 중 **가장 큰 값**이에요. 끝에 걸렸다는 건 "더 크게 하면 더 좋아질 수도 있다"는 신호예요. 후보 범위를 넓혀서 다시 찾아봐야 해요.

모든 조합의 점수를 표로 보기:

```python
cv_res = pd.DataFrame(grid_search.cv_results_)                   # 결과를 표로
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)   # 좋은 순 정렬
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols    # 열 이름 짧게
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)   # 마이너스 떼고 정수로
cv_res.head()
```

| n_clusters | max_features | 1회 | 2회 | 3회 | 평균 RMSE |
|---|---|---|---|---|---|
| 15 | 6 | 42725 | 43708 | 44335 | **43590** |
| 15 | 8 | 43486 | 43820 | 44900 | 44069 |
| 10 | 4 | 43798 | 44036 | 44961 | 44265 |

기본 랜덤 포레스트(47,038)보다 좋아졌어요!

### 6-2. 랜덤 서치: 무작위로 뽑아서 시도

후보가 많아지면 모든 조합을 다 해 보기엔 너무 오래 걸려요. **랜덤 서치**는 범위만 정해 주면 **무작위로 조합을 뽑아서** 정해진 횟수만큼만 시도해요.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),   # 3~49 중 아무거나
                  'random_forest__max_features': randint(low=2, high=20)}      # 2~19 중 아무거나

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,   # 10가지만 시도
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)
```

10가지 × 3번 = **30번 학습**으로 그리드 서치(45번)보다 적어요.

| n_clusters | max_features | 1회 | 2회 | 3회 | 평균 RMSE |
|---|---|---|---|---|---|
| 45 | 9 | 41342 | 42242 | 43057 | **42214** |
| 32 | 7 | 41825 | 42275 | 43241 | 42447 |

**더 적게 시도하고도 더 좋은 결과**를 얻었어요(42,214)! 그리드 서치는 정해진 몇 개 값만 반복하지만, 랜덤 서치는 매번 다른 값을 시도하니까 더 넓은 범위를 탐색할 수 있어요.

**그리드 서치 vs 랜덤 서치**

| | 그리드 서치 | 랜덤 서치 |
|---|---|---|
| 방식 | 후보 조합을 **전부** 시도 | 범위에서 **무작위로** 정해진 횟수만 |
| 좋을 때 | 후보가 적을 때 | 후보가 많거나 범위가 넓을 때 |
| 시간 조절 | 후보 수에 따라 정해짐 | `n_iter`로 직접 정할 수 있음 |

```python
from sklearn.experimental import enable_halving_search_cv   # 실험 기능 켜기
from sklearn.model_selection import HalvingRandomSearchCV
```

`HalvingRandomSearchCV`는 더 효율적인 버전이에요. 처음엔 많은 조합을 **적은 데이터로 빠르게** 평가하고, 성적 좋은 조합만 남겨서 데이터를 늘려 가며 다시 평가해요. 토너먼트 방식이에요.

#### 범위는 어떻게 정하나요?

| 분포 | 이럴 때 써요 |
|---|---|
| `randint(a, b+1)` | a~b 정수, 어느 값이든 비슷하게 그럴듯할 때 |
| `uniform(a, b)` | 위와 같은데 소수일 때 |
| `expon(scale)` | 대충 이 정도 크기일 것 같을 때 (작은 값 위주로 탐색) |
| `loguniform(a, b)` | **얼마가 좋을지 감도 안 올 때** (0.01~0.1이든 100~1000이든 골고루 탐색) |

### 6-3. 최고 모델 들여다보기: 어떤 특성이 중요했나?

```python
final_model = rnd_search.best_estimator_                              # 최고 모델 (전처리 포함)
feature_importances = final_model["random_forest"].feature_importances_   # 특성 중요도

sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),   # (중요도, 이름) 짝짓기
       reverse=True)                                                # 중요한 순으로
```

```
(0.186, 'log__median_income'),            ← 소득이 압도적 1위
(0.073, 'cat__ocean_proximity_INLAND'),   ← 내륙인가?
(0.066, 'bedrooms__ratio'),               ← 우리가 만든 침실 비율!
(0.054, 'rooms_per_house__ratio'),        ← 우리가 만든 집당 방 수!
(0.046, 'people_per_house__ratio'),       ← 우리가 만든 집당 인원!
(0.042, 'geo__Cluster 30 similarity'),    ← 대도시 근접도
...
```

**3-3에서 만든 비율 특성들이 상위권**에 있어요. 특성 공학이 효과가 있었다는 증거예요! 그리고 `ocean_proximity` 중엔 **INLAND만** 중요해서, 나머지 종류는 빼는 것도 고려해 볼 수 있어요.

### 6-4. 드디어 시험 (테스트 세트로 최종 평가)

맨 처음에 서랍에 넣어 둔 테스트 세트를 꺼낼 시간이에요.

```python
X_test = strat_test_set.drop("median_house_value", axis=1)   # 시험 문제
y_test = strat_test_set["median_house_value"].copy()          # 시험 정답

final_predictions = final_model.predict(X_test)               # ⚠️ predict만! fit 절대 금지

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)    # 41445.53
```

> ⚠️ 테스트 세트에는 **`predict`만** 해요. `fit`을 하면 시험 문제로 공부하는 셈이에요. 파이프라인이 훈련 세트에서 배운 중간값·평균·대도시 중심을 그대로 써서 변환해 줘요.

최종 점수는 **약 $41,446**이에요. 그런데 이 숫자 하나만으로는 얼마나 믿을 만한지 알 수 없어요. 다른 테스트 세트였다면 40,000이 나왔을 수도, 43,000이 나왔을 수도 있죠. 그래서 **신뢰구간**을 구해요.

```python
from scipy.stats import bootstrap

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2          # 구역별 오차 제곱
boot_result = bootstrap([squared_errors], rmse, confidence_level=confidence,
                        random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
print(f"95% CI for RMSE: ({rmse_lower:.4f}, {rmse_upper:.4f})")
# 95% CI for RMSE: (39520.9572, 43701.7681)
```

**부트스트랩**은 이렇게 동작해요.
1. 테스트 세트의 오차들 중에서 **중복을 허용해서** 같은 개수만큼 무작위로 다시 뽑아요.
2. 그걸로 RMSE를 계산해요.
3. 이걸 수천 번 반복하면 RMSE 값이 수천 개 생겨요.
4. 그중 하위 2.5% ~ 상위 2.5%를 뺀 범위가 95% 신뢰구간이에요.

결과 해석: **"실제 RMSE는 95% 확률로 $39,521 ~ $43,702 사이에 있다"**

> ⚠️ **절대 하면 안 되는 것**: 이 점수를 보고 "조금만 더 올려 볼까?" 하며 설정값을 다시 조정하는 것. 그 순간 테스트 세트에 맞춘 모델이 돼서, 진짜 새 데이터에선 이 점수가 안 나와요. 테스트 세트는 **딱 한 번만** 써요.

---

## 7. 론칭·모니터링·유지 보수 (2.8)

### 7-1. 모델 저장하기

```python
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")   # 파일로 저장
```

전처리 + 모델이 담긴 파이프라인을 **통째로** 파일 하나에 저장해요. `joblib`은 숫자 배열이 많은 객체를 저장하는 데 특화된 도구예요.

### 7-2. 다른 곳에서 불러와 쓰기

```python
import joblib

# ⚠️ 직접 만든 함수·클래스는 여기서 다시 정의하거나 import 해야 해요
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

#class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]   (위의 클래스 코드 전체)

final_model_reloaded = joblib.load("my_california_housing_model.pkl")   # 불러오기

new_data = housing.iloc[:5]                          # 새 구역 데이터라고 칠게요
predictions = final_model_reloaded.predict(new_data) # 바로 예측!
```

> ⚠️ **자주 하는 실수**: 저장 파일에는 `column_ratio`나 `ClusterSimilarity`의 **코드가 들어 있지 않아요.** "column_ratio라는 함수를 써라"라는 **이름표만** 들어 있어요. 그래서 불러오는 쪽에서 같은 이름의 함수와 클래스가 없으면 에러(`AttributeError`)가 나요.
>
> 실무에서는 직접 만든 변환기를 `my_transformers.py` 같은 별도 파일로 만들어 두고, 저장하는 쪽과 불러오는 쪽 모두에서 `import`해서 써요.

### 7-3. 배포한 뒤에 할 일

모델을 만들고 끝이 아니에요. 오히려 시작이에요.

| 할 일 | 왜? |
|---|---|
| **배포** | 웹 서비스에 넣거나, 별도 API 서버로 띄우거나, 클라우드(Vertex AI 등)에 올려요 |
| **성능 감시** | 세상은 계속 변해요. 1990년 데이터로 만든 모델은 2026년 집값을 못 맞히겠죠. 이렇게 성능이 서서히 떨어지는 걸 **모델 부패**라고 해요 |
| **입력 데이터 감시** | 갑자기 빈칸이 많아지거나 이상한 값이 들어오면 알림이 오도록 |
| **정기적 재학습** | 새 데이터로 주기적으로 다시 학습. 가능하면 자동화 |
| **버전 관리** | 새 모델이 더 나쁘면 이전 모델로 바로 되돌릴 수 있게 저장해 두기 |

---

## 8. 연습문제 (2.9)

### 8-1. 다른 모델(SVR) 써 보기

**SVR**(서포트 벡터 회귀)은 랜덤 포레스트와는 전혀 다른 방식의 모델이에요(부록에서 자세히). 그리드 서치로 설정값을 찾아봅니다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = [
        {'svr__kernel': ['linear'],                                    # 직선 방식일 때
         'svr__C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'svr__kernel': ['rbf'],                                       # 곡선 방식일 때
         'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])   # 5,000개만 사용

svr_grid_search_rmse = -grid_search.best_score_   # 70059.93
grid_search.best_params_    # {'svr__C': 10000.0, 'svr__kernel': 'linear'}
```

- SVR은 데이터가 많으면 **아주 느려져서** 5,000개만 썼어요.
- `linear`(직선) 방식에는 `gamma` 설정이 의미가 없어서, 두 묶음으로 나눠서 쓸데없는 조합을 줄였어요.
- 결과는 70,060으로 랜덤 포레스트보다 훨씬 나빠요. (데이터를 적게 쓴 탓도 있어요.)
- `C=10000`이 후보 범위 끝 근처예요 → 더 큰 값도 시도해 봐야 해요.

### 8-2. 랜덤 서치로 바꿔 보기

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, loguniform

param_distribs = {
        'svr__kernel': ['linear', 'rbf'],          # 둘 중 하나 무작위
        'svr__C': loguniform(20, 200_000),         # 20 ~ 200,000을 골고루 (감이 안 오니까)
        'svr__gamma': expon(scale=1.0),            # 1 근처 위주로
    }

rnd_search = RandomizedSearchCV(svr_pipeline,
                                param_distributions=param_distribs,
                                n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=42)
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

svr_rnd_search_rmse = -rnd_search.best_score_    # 56152.05  ← 70,060에서 크게 개선!
```

- `C`는 좋은 값이 10인지 10만인지 감이 안 와서 `loguniform`을 썼어요. 20~200, 200~2000, 2000~20000을 **똑같은 비율**로 탐색해 줘요.
- `gamma`는 1 근처일 거라 보고 `expon(scale=1)`을 썼어요. 뽑힌 값의 약 80%가 0.1~2.3 사이예요.

```python
s = expon(scale=1).rvs(100_000, random_state=42)    # 10만 개 뽑아 보기
((s > 0.105) & (s < 2.29)).sum() / 100_000          # 약 0.8 → 80%가 이 범위
```

같은 시간이면 랜덤 서치가 그리드 서치보다 좋은 설정을 찾는 경우가 많다는 걸 보여주는 예예요.

### 8-3. 중요한 특성만 골라 쓰기

```python
from sklearn.feature_selection import SelectFromModel

selector_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),   # 중요도 0.005 미만인 특성은 버림
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),   # 8-2에서 찾은 최고 설정
])

selector_rmses = -cross_val_score(selector_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
pd.Series(selector_rmses).describe()
```

`SelectFromModel`은 랜덤 포레스트로 특성 중요도를 먼저 계산하고, **별로 안 중요한 특성은 버리는** 변환기예요. 이번엔 효과가 없었어요. 기준값(0.005)이 적절하지 않았을 수도 있으니 이것도 튜닝해 볼 만해요.

### 8-4. 다른 모델의 예측값을 특성으로 쓰기

아이디어: "**주변 구역들의 소득**"을 특성으로 쓰면 어떨까? 위치가 비슷한 이웃 구역들의 소득을 KNN(가장 가까운 k개 이웃의 평균)으로 예측해서 새 특성으로 넣어 보는 거예요.

```python
from sklearn.base import MetaEstimatorMixin, clone

class FeatureFromRegressor(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    def __init__(self, regressor, target_features):
        self.regressor = regressor               # 쓸 모델 (KNN 등 아무거나)
        self.target_features = target_features   # 예측할 열 (예: 소득)

    def fit(self, X, y=None):
        if hasattr(X, "columns"):                        # 표로 들어왔으면
            self.feature_names_in_ = list(X.columns)     # 열 이름 기억
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)

        X_target = X_df[self.target_features]                  # 예측할 것: 소득
        X_rest = X_df.drop(columns=self.target_features)       # 입력: 나머지(위도, 경도)
        self.regressor_ = clone(self.regressor).fit(X_rest, X_target)   # 복사본으로 학습
        return self

    def transform(self, X):
        columns = self.feature_names_in_
        if hasattr(X, "columns"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=columns)
        X_rest = X_df.drop(columns=self.target_features)
        predictions = self.regressor_.predict(X_rest)      # 위치로 소득 예측
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)        # 세로 표 모양으로
        return predictions

    def get_feature_names_out(self, feature_names=None):
        check_is_fitted(self)
        estimator_class_name = self.regressor_.__class__.__name__
        return [f"{estimator_class_name.lower()}_prediction_{t}"
                for t in self.target_features]
```

> 💡 **`clone()`을 쓰는 이유**: 사용자가 넘겨준 모델을 직접 학습시키면 원본이 바뀌어 버려요. 그래서 **복사본을 만들어 학습**하고, 원본(설정값)은 그대로 둬요. "설정값은 `regressor`, 학습된 모델은 `regressor_`"로 구분하는 거예요.

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")   # 가까운 이웃 3곳, 가까울수록 가중치
knn_transformer = FeatureFromRegressor(knn_reg, ["median_income"])
geo_features = housing[["latitude", "longitude", "median_income"]]
knn_transformer.fit_transform(geo_features, housing_labels)
knn_transformer.get_feature_names_out()
# ['kneighborsregressor_prediction_median_income']
```

기존 전처리에서 `geo`(대도시 근접도) 부분을 이걸로 바꿔 봅니다.

```python
transformers = [(name, clone(transformer), columns)               # 기존 단계들 복사
                for name, transformer, columns in preprocessing.transformers]
geo_index = [name for name, _, _ in transformers].index("geo")    # "geo"가 몇 번째인지 찾기
transformers[geo_index] = ("geo", knn_transformer,                # 그 자리를 KNN으로 교체
                           ["latitude", "longitude", "median_income"])
new_geo_preprocessing = ColumnTransformer(transformers)

new_geo_pipeline = Pipeline([
    ('preprocessing', new_geo_preprocessing),
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])
new_pipe_rmses = -cross_val_score(new_geo_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
```

결과는 훨씬 나빴어요. 원래의 **대도시 근접도 특성이 더 좋았다**는 걸 알 수 있어요. 실험해 보고 안 되면 버리는 것도 중요한 과정이에요.

### 8-5. 전처리 설정까지 랜덤 서치

```python
param_distribs = {
    "preprocessing__geo__regressor__n_neighbors": range(1, 30),         # 이웃 몇 곳?
    "preprocessing__geo__regressor__weights": ["distance", "uniform"],  # 거리 가중치?
    "svr__C": loguniform(20, 200_000),
    "svr__gamma": expon(scale=1.0),
}

new_geo_rnd_search = RandomizedSearchCV(new_geo_pipeline,
                                        param_distributions=param_distribs,
                                        n_iter=50, cv=3,
                                        scoring='neg_root_mean_squared_error',
                                        random_state=42)
new_geo_rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
```

`preprocessing__geo__regressor__n_neighbors`는 **4단계 깊이**까지 들어가는 이름이에요.

```
파이프라인 → preprocessing → geo (FeatureFromRegressor) → regressor (KNN) → n_neighbors
```

`BaseEstimator`를 물려받아 만들었기 때문에 이렇게 깊은 곳의 설정도 자동 튜닝할 수 있어요. 다만 튜닝을 해도 대도시 근접도 방식을 이기진 못했어요.

### 8-6. `StandardScaler` 완벽하게 다시 만들기

4-4에서 만든 간단한 버전에 기능을 더 추가해요.

```python
from sklearn.utils.validation import check_is_fitted, validate_data

class StandardScalerClone(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_2d=True)   # 검사 + 열 개수·이름 자동 기억
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)   # reset=False: 훈련 때와 같은지 검사
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):                  # ← 새로 추가: 원래대로 되돌리기
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        X = X * self.scale_                          # 나눴던 걸 다시 곱하고
        return X + self.mean_ if self.with_mean else X   # 뺐던 평균을 다시 더하기

    def get_feature_names_out(self, input_features=None):   # ← 새로 추가: 열 이름
        if input_features is None:
            return getattr(self, "feature_names_in_",
                           [f"x{i}" for i in range(self.n_features_in_)])   # 이름 없으면 x0, x1, ...
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Invalid number of features")
            if hasattr(self, "feature_names_in_") and not np.all(
                self.feature_names_in_ == input_features
            ):
                raise ValueError("input_features ≠ feature_names_in_")
            return input_features
```

| 추가된 것 | 하는 일 |
|---|---|
| `validate_data` | 검사도 해 주고, 열 개수(`n_features_in_`)와 열 이름(`feature_names_in_`)을 **자동으로 기억** |
| `inverse_transform` | 변환한 걸 원래대로 되돌리기 (곱하고 → 더하기) |
| `get_feature_names_out` | 결과 열 이름 알려주기 |

제대로 만들었는지 테스트:

```python
from sklearn.utils.estimator_checks import check_estimator
check_estimator(StandardScalerClone())     # 사이킷런 규칙을 다 지켰는지 자동 검사 (수십 가지)

rng = np.random.default_rng(seed=42)
X = rng.random((1000, 3))                  # 테스트용 데이터

scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(X)
assert np.allclose(X_scaled, (X - X.mean(axis=0)) / X.std(axis=0))   # 계산이 맞나?

scaler = StandardScalerClone(with_mean=False)
X_scaled_uncentered = scaler.fit_transform(X)
assert np.allclose(X_scaled_uncentered, X / X.std(axis=0))           # 평균 안 빼는 옵션도 맞나?

scaler = StandardScalerClone()
X_back = scaler.inverse_transform(scaler.fit_transform(X))
assert np.allclose(X, X_back)                                         # 변환했다 되돌리면 원래대로?

assert np.all(scaler.get_feature_names_out() == ["x0", "x1", "x2"])   # 이름 없을 때
assert np.all(scaler.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c"])

df = pd.DataFrame({"a": rng.random(100), "b": rng.random(100)})
scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(df)
assert np.all(scaler.feature_names_in_ == ["a", "b"])                 # 표로 넣으면 이름 기억하나?
assert np.all(scaler.get_feature_names_out() == ["a", "b"])
```

> 💡 `np.allclose`는 "거의 같으면 OK"로 비교해요. 컴퓨터의 소수 계산엔 아주 작은 오차가 생겨서, `==`로 비교하면 0.30000000000000004 ≠ 0.3처럼 실패할 수 있거든요.

---

## 9. 전체 흐름 요약

### 9-1. 한눈에 보는 코드 흐름

```
load_housing_data()                          ← 데이터 받기
   ↓
pd.cut + train_test_split(stratify=...)      ← 시험용 데이터 먼저 떼어 두기 (소득 비율 유지)
   ↓
strat_train_set.copy() → 그래프, 상관관계      ← 훈련 데이터랑 친해지기
   ↓
preprocessing = ColumnTransformer([...])     ← 빈칸·글자·크기·비율·로그·대도시 근접도 손질을 하나로
   ↓
make_pipeline(preprocessing, 모델)            ← 여러 모델을 교차 검증으로 비교
   ↓
RandomizedSearchCV(full_pipeline, ...)       ← 전처리 + 모델 설정값을 함께 자동 튜닝
   ↓
final_model.predict(X_test) + bootstrap      ← 시험용 데이터로 딱 한 번 최종 채점
   ↓
joblib.dump / joblib.load                    ← 저장하고 운영
```

### 9-2. 성적이 어떻게 좋아졌나

| 단계 | RMSE (평균 오차) |
|---|---|
| 선형 회귀 | $70,003 |
| 결정 트리 | $66,574 |
| 랜덤 포레스트 | $47,038 |
| + 그리드 서치 | $43,590 |
| + 랜덤 서치 | $42,214 |
| **최종 테스트 세트** | **$41,446** (95% 확률로 $39,521 ~ $43,702) |

처음 전문가 추정치가 20% 넘게 틀렸던 것과 비교하면, 집값 중간값이 대략 $180,000~$200,000 수준이니까 꽤 쓸 만한 모델이 됐어요.

### 9-3. 이 장에서 꼭 기억할 7가지

1. **테스트 세트는 맨 처음에 떼어 두고, 맨 마지막에 딱 한 번만 쓴다.**
2. 중요한 특성이 있으면 **계층 샘플링**으로 테스트 세트의 비율을 맞춘다.
3. 원래 열보다 **의미 있는 비율 특성**이 더 강력할 수 있다.
4. **`fit`은 훈련 세트로만**, 테스트·새 데이터에는 `transform`/`predict`만.
5. 전처리는 **파이프라인**으로 묶어서 자동화한다. 실수도 줄고, 설정값 튜닝도 한 번에 된다.
6. 모델 비교는 훈련 세트 점수가 아니라 **교차 검증 점수**로 한다.
7. 직접 만든 함수·클래스는 모델을 불러오는 쪽에서도 **똑같이 있어야** 한다.
