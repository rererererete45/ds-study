# 2장. 머신러닝 프로젝트 처음부터 끝까지

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `02_end_to_end_machine_learning_project.ipynb`
>
> 실제 데이터로 머신러닝 프로젝트의 전 과정을 한 번 수행한다. 이후 모든 장은 이 장의 작업 흐름(워크플로) 위에서 진행된다.

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
| 1. 문제 정의 | 2.1–2.2 | 비즈니스 요구 | 문제 유형, 성능 지표 |
| 2. 데이터 수집·분할 | 2.3 | 원본 데이터 | 훈련 세트, 테스트 세트 |
| 3. 탐색 | 2.4 | 훈련 세트 | 특성에 대한 가설 |
| 4. 전처리 | 2.5 | 훈련 세트 | 전처리 파이프라인 |
| 5. 모델 선택 | 2.6 | 전처리된 데이터 | 유망 모델 후보 |
| 6. 튜닝·평가 | 2.7 | 후보 모델 | 최종 모델, 일반화 오차 |
| 7. 배포·운영 | 2.8 | 최종 모델 | 운영 시스템 |

## 데이터: 캘리포니아 주택 가격

1990년 캘리포니아 인구조사. 샘플 단위 = 구역(district), 20,640개, 특성 9개 + 타깃 1개.

| 컬럼 | 타입 | 설명 | 주의 |
|---|---|---|---|
| `longitude`, `latitude` | float | 경도, 위도 | |
| `housing_median_age` | float | 주택 연식 중간값 | 상한 52 |
| `total_rooms` | float | 구역 전체 방 수 | 구역 합계 |
| `total_bedrooms` | float | 구역 전체 침실 수 | 결측 207개 |
| `population` | float | 인구 | |
| `households` | float | 가구 수 | |
| `median_income` | float | 소득 중간값 | 약 $10,000 단위 |
| `median_house_value` | float | **타깃**: 주택 가격 중간값 | 상한 $500,000 |
| `ocean_proximity` | object | 해안 근접도 | 범주형 5종 |

## 목차

1. [문제 정의 (2.1–2.2)](#1-문제-정의-2122)
   - [1-1. 문제 유형 분류](#1-1-문제-유형-분류)
   - [1-2. 성능 지표](#1-2-성능-지표)
   - [1-3. 환경 확인](#1-3-환경-확인)
2. [데이터 수집과 분할 (2.3)](#2-데이터-수집과-분할-23)
   - [2-1. 데이터 다운로드](#2-1-데이터-다운로드)
   - [2-2. 데이터 구조 확인](#2-2-데이터-구조-확인)
   - [2-3. 테스트 세트 분리 원칙](#2-3-테스트-세트-분리-원칙)
   - [2-4. 무작위 분할](#2-4-무작위-분할)
   - [2-5. 해시 기반 분할](#2-5-해시-기반-분할)
   - [2-6. 계층 샘플링](#2-6-계층-샘플링)
3. [탐색과 시각화 (2.4)](#3-탐색과-시각화-24)
   - [3-1. 지리 정보 시각화](#3-1-지리-정보-시각화)
   - [3-2. 상관관계 분석](#3-2-상관관계-분석)
   - [3-3. 특성 조합](#3-3-특성-조합)
4. [데이터 전처리 (2.5)](#4-데이터-전처리-25)
   - [4-1. 사이킷런 API 설계](#4-1-사이킷런-api-설계)
   - [4-2. 결측치 처리](#4-2-결측치-처리)
   - [4-3. 범주형 특성 인코딩](#4-3-범주형-특성-인코딩)
   - [4-4. 특성 스케일링](#4-4-특성-스케일링)
   - [4-5. 분포 변환](#4-5-분포-변환)
   - [4-6. 타깃 변환](#4-6-타깃-변환)
   - [4-7. 사용자 정의 변환기](#4-7-사용자-정의-변환기)
   - [4-8. 변환 파이프라인](#4-8-변환-파이프라인)
5. [모델 선택과 훈련 (2.6)](#5-모델-선택과-훈련-26)
   - [5-1. 훈련 세트 평가](#5-1-훈련-세트-평가)
   - [5-2. 교차 검증](#5-2-교차-검증)
   - [5-3. 모델 비교와 진단](#5-3-모델-비교와-진단)
6. [모델 튜닝과 최종 평가 (2.7)](#6-모델-튜닝과-최종-평가-27)
   - [6-1. 그리드 서치](#6-1-그리드-서치)
   - [6-2. 랜덤 서치](#6-2-랜덤-서치)
   - [6-3. 특성 중요도 분석](#6-3-특성-중요도-분석)
   - [6-4. 테스트 세트 평가](#6-4-테스트-세트-평가)
7. [배포와 운영 (2.8)](#7-배포와-운영-28)
8. [연습문제 (2.9)](#8-연습문제-29)
9. [정리](#9-정리)

---

## 1. 문제 정의 (2.1–2.2)

### 1-1. 문제 유형 분류

**목적**
코드 작성 전에 문제의 종류를 확정한다. 문제 유형이 알고리즘·평가 방식·데이터 처리 방식을 결정한다.

**개념·원리**

요구사항: 구역의 특성으로 그 구역의 주택 가격 중간값을 예측한다. 예측값은 투자 판단 시스템(하위 시스템)의 입력으로 사용된다. 현재는 전문가가 수동으로 추정하며 오차가 20%를 넘는 경우가 있다.

| 분류 기준 | 선택지 | 이 문제 | 근거 |
|---|---|---|---|
| 레이블 유무 | 지도 / 비지도 / 준지도 / 강화 | **지도 학습** | 타깃(`median_house_value`)이 데이터에 있음 |
| 출력 형태 | 회귀 / 분류 | **회귀** | 연속값(가격) 예측 |
| 입력 특성 수 | 단순 / 다중 | **다중 회귀** | 특성 여러 개 사용 |
| 출력 개수 | 단변량 / 다변량 | **단변량 회귀** | 예측값 1개 |
| 학습 방식 | 배치 / 온라인 | **배치 학습** | 데이터가 고정되어 있고 메모리에 들어감 |

**가정 검증**: 하위 시스템이 가격 대신 범주("저가/중가/고가")를 사용한다면 분류 문제로 바뀐다. 문제 정의 단계에서 확인해야 한다.

**정리**
- 이 프로젝트 = 지도 학습 · 다중 회귀 · 단변량 · 배치 학습
- 모델의 용도(하위 시스템)를 먼저 확인해야 출력 형태와 요구 정확도가 정해진다

---

### 1-2. 성능 지표

**목적**
모델을 비교·평가할 기준을 정한다.

**개념·원리**

**RMSE** (Root Mean Squared Error, 평균 제곱근 오차)

$$
\text{RMSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}\left(\hat{y}^{(i)} - y^{(i)}\right)^2}
$$

**MAE** (Mean Absolute Error, 평균 절대 오차)

$$
\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}\left|\hat{y}^{(i)} - y^{(i)}\right|
$$

- $m$: 샘플 수, $y^{(i)}$: 실제값, $\hat{y}^{(i)}$: 예측값

| 구분 | RMSE | MAE |
|---|---|---|
| 노름 | ℓ2 (유클리드 노름) | ℓ1 (맨해튼 노름) |
| 큰 오차 | 제곱하므로 더 크게 반영 | 그대로 반영 |
| 이상치 | 민감 | 덜 민감 |
| 사용 | 오차가 정규분포에 가까울 때 (기본값) | 이상치가 많을 때 |

일반화: ℓk 노름 $\|\mathbf{v}\|_k = \left(\sum |v_i|^k\right)^{1/k}$. $k$가 클수록 큰 값에 집중하고 작은 값을 무시한다.

**정리**
- 회귀의 기본 지표는 RMSE, 단위는 타깃과 같다 (달러)
- RMSE 50,000 = "평균적으로 약 $50,000 오차"

---

### 1-3. 환경 확인

**목적**
코드 실행에 필요한 버전 조건을 미리 검사한다.

**코드**

```python
import sys
assert sys.version_info >= (3, 10)      # Python 3.10 이상

from packaging.version import Version
import sklearn
assert Version(sklearn.__version__) >= Version("1.6.1")   # scikit-learn 1.6.1 이상
```

**원리**
- `assert 조건`: 조건이 거짓이면 `AssertionError` 발생 → 실행 중단
- `Version()`: 버전 문자열을 숫자 단위로 비교. 문자열 비교는 `"1.10" < "1.6"`이 되는 문제가 있음
- 버전이 필요한 이유: `root_mean_squared_error`(1.4+), `sparse_output`(1.2+), `validate_data`(1.6+)

---

## 2. 데이터 수집과 분할 (2.3)

### 2-1. 데이터 다운로드

**목적**
데이터 수집을 함수로 만들어 재실행·갱신이 가능하게 한다.

**코드**

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():                              # ① 파일이 없을 때만
        Path("datasets").mkdir(parents=True, exist_ok=True)     # ② 폴더 생성
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)           # ③ 다운로드
        with tarfile.open(tarball_path) as housing_tarball:     # ④ 압축 해제
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))    # ⑤ DataFrame 반환

housing_full = load_housing_data()
```

**원리**

| 단계 | 코드 | 설명 |
|---|---|---|
| ① | `is_file()` | 캐싱. 이미 받은 파일이 있으면 다운로드 생략 |
| ② | `mkdir(parents=True, exist_ok=True)` | 상위 폴더까지 생성, 이미 있으면 무시 |
| ③ | `urlretrieve(url, path)` | URL → 로컬 파일 |
| ④ | `with tarfile.open()` | 컨텍스트 매니저로 자동 close. `filter="data"`는 경로 조작 공격(`../`) 차단 |
| ⑤ | `pd.read_csv()` | CSV → DataFrame |

**정리**
- 데이터 수집 자동화 → 데이터가 갱신되면 함수만 재호출

---

### 2-2. 데이터 구조 확인

**목적**
데이터의 크기, 타입, 결측치, 분포를 파악한다.

**코드**

```python
housing_full.head()                               # 상위 5행
housing_full.info()                               # 행 수, 열별 non-null 수, dtype
housing_full["ocean_proximity"].value_counts()    # 범주별 빈도
housing_full.describe()                           # 수치형 열 요약 통계

import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12, 8))       # 수치형 열 히스토그램 (구간 50개)
plt.show()
```

**결과 해석**

`info()`:

```
RangeIndex: 20640 entries, 0 to 20639
 4   total_bedrooms      20433 non-null  float64   → 결측치 207개
 9   ocean_proximity     20640 non-null  object    → 범주형
```

`describe()` 항목: count(결측 제외 개수), mean, std(표준편차), min, 25%/50%/75%(사분위수, 50% = 중간값), max

히스토그램에서 확인되는 문제:

| 관찰 | 의미 | 대응 (이후 절) |
|---|---|---|
| `median_income` 범위 0.5~15 | 단위가 스케일 조정됨 | 출처 확인, 그대로 사용 |
| 타깃·연식 최댓값에 막대 집중 | 상한 처리(clipping) | 해당 샘플 제거 고려 |
| 특성마다 범위가 크게 다름 | 스케일 불일치 | 4-4 스케일링 |
| 오른쪽 꼬리가 긴 분포 | 비대칭 분포 | 4-5 로그 변환 |

**정리**
- 처리할 문제 목록: 결측치, 범주형, 스케일 차이, 비대칭 분포, 타깃 상한

---

### 2-3. 테스트 세트 분리 원칙

**목적**
모델의 일반화 성능을 공정하게 측정할 데이터를 확보한다.

**개념·원리**

- **일반화 오차**(generalization error): 처음 보는 데이터에 대한 오차. 머신러닝의 실제 목표
- **데이터 스누핑 편향**(data snooping bias): 테스트 세트를 보고 나서 모델·특성을 고르면, 그 선택이 테스트 세트에 맞춰져 일반화 오차를 낙관적으로 추정하게 됨

**원칙**: 데이터를 자세히 탐색하기 **전에** 테스트 세트(보통 20%)를 분리하고, 최종 평가 전까지 사용하지 않는다.

| 분할 방법 | 소절 | 재현성 | 데이터 갱신 시 안정성 | 분포 보존 |
|---|---|---|---|---|
| 무작위 분할 | 2-4 | 시드 고정 시 O | X | 확률적 |
| 해시 기반 분할 | 2-5 | O | O | 확률적 |
| 계층 샘플링 | 2-6 | 시드 고정 시 O | X | O (기준 특성) |

---

### 2-4. 무작위 분할

**개념·원리**
인덱스를 무작위로 섞은 뒤 앞쪽 일정 비율을 테스트 세트로 사용한다.

**코드**

```python
import numpy as np

def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))     # 0 ~ n-1 무작위 순열
    test_set_size = int(len(data) * test_ratio)       # 20640 × 0.2 = 4128
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]   # iloc: 위치 기반 인덱싱

rng = np.random.default_rng(seed=42)                  # 시드 고정 → 재현 가능
train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
len(train_set), len(test_set)                         # (16512, 4128)
```

사이킷런 버전:

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)
```

**한계**
데이터셋이 갱신되어 행 수가 바뀌면 순열 전체가 달라진다 → 이전 테스트 샘플이 훈련 세트로 유입된다.

---

### 2-5. 해시 기반 분할

**개념·원리**
각 샘플의 불변 식별자(ID)를 해시 함수에 넣고, 해시값이 일정 임계값 미만이면 테스트 세트로 보낸다.

- `crc32(id)`: 0 ~ 2³²−1 범위의 정수, 거의 균등 분포
- 조건: `crc32(id) < test_ratio × 2³²` → 약 `test_ratio` 비율이 테스트 세트
- 해시는 결정적(같은 입력 → 같은 출력) → 데이터가 추가되어도 기존 샘플의 소속이 바뀌지 않음

**코드**

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))  # 불리언 Series
    return data.loc[~in_test_set], data.loc[in_test_set]                    # ~ : 논리 부정
```

ID 선택:

```python
# (a) 행 번호를 ID로 사용
housing_with_id = housing_full.reset_index()      # 'index' 열 추가
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# (b) 불변 특성으로 ID 생성
housing_with_id["id"] = (housing_full["longitude"] * 1000
                         + housing_full["latitude"])
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
```

| ID | 조건 | 문제 |
|---|---|---|
| 행 번호 | 새 데이터는 끝에만 추가, 삭제 금지 | 중간 삭제 시 번호가 밀림 |
| 위경도 조합 | 위치는 불변 | 같은 위치의 구역끼리 ID 중복 가능 |

---

### 2-6. 계층 샘플링

**목적**
테스트 세트가 중요한 특성의 분포를 전체 데이터와 같은 비율로 갖도록 한다.

**개념·원리**

**샘플링 편향**: 무작위 표본도 확률적으로 모집단과 다른 분포를 가질 수 있다.

예: 여성 비율 $p = 0.516$인 모집단에서 $n = 1000$명 추출. 표본의 여성 수 $X \sim \text{Binomial}(1000, 0.516)$.

$$
P(X < 490) + P(X > 540) = F(489) + \left(1 - F(540)\right) \approx 0.107
$$

($F$: 누적분포함수)

**계층 샘플링**(stratified sampling): 모집단을 동질적인 하위 그룹(계층)으로 나누고, 각 계층에서 비율에 맞게 추출한다.

**코드: 편향 확률 계산**

```python
from scipy.stats import binom

sample_size = 1000
ratio_female = 0.516
proba_too_small = binom(sample_size, ratio_female).cdf(490 - 1)     # P(X ≤ 489)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)     # P(X > 540)
print(proba_too_small + proba_too_large)                            # ≈ 0.107

# 시뮬레이션 검증
rng = np.random.default_rng(seed=42)
samples = (rng.random((100_000, sample_size)) < ratio_female).sum(axis=1)  # 10만 회 설문의 여성 수
((samples < 490) | (samples > 540)).mean()                                 # ≈ 0.107
```

**코드: 계층 생성**

소득이 가장 중요한 특성이라는 전제 → 연속값인 소득을 5개 구간으로 이산화한다.

```python
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
```

| 구간 | (0, 1.5] | (1.5, 3] | (3, 4.5] | (4.5, 6] | (6, ∞) |
|---|---|---|---|---|---|
| 계층 | 1 | 2 | 3 | 4 | 5 |

계층 수 설계 기준: 계층마다 충분한 샘플이 있어야 비율 추정이 안정적이다 → 과도하게 세분하지 않는다.

**코드: 계층 분할**

```python
# 분할 1개
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)

# 분할 여러 개 (교차 검증용)
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full,
                                              housing_full["income_cat"]):  # 2번째 인자 = 계층 기준
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])
strat_train_set, strat_test_set = strat_splits[0]
```

**코드: 효과 검증**

```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing_full),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)    # 상대 오차
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)
```

**결과 해석**

| 계층 | 전체 % | 계층 % | 무작위 % | 계층 오차 % | 무작위 오차 % |
|---|---|---|---|---|---|
| 1 | 3.98 | 4.00 | 4.24 | 0.36 | **6.45** |
| 2 | 31.88 | 31.88 | 30.74 | −0.02 | −3.59 |
| 3 | 35.06 | 35.05 | 34.52 | −0.01 | −1.53 |
| 4 | 17.63 | 17.64 | 18.41 | 0.03 | 4.42 |
| 5 | 11.44 | 11.43 | 12.09 | −0.08 | 5.63 |

- 계층 샘플링: 최대 상대 오차 0.36%
- 무작위 분할: 최대 6.45%, 비율이 작은 계층(1)일수록 오차가 큼

**후처리**: 분할용 열 제거

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

**정리**
- 중요 특성이 있으면 그 특성을 기준으로 계층 샘플링
- 연속 특성은 `pd.cut`으로 구간화 후 `stratify=`에 전달

---

## 3. 탐색과 시각화 (2.4)

**대상**: 훈련 세트의 복사본만 사용 (테스트 세트 격리 유지)

```python
housing = strat_train_set.copy()
```

### 3-1. 지리 정보 시각화

**목적**
위치와 집값의 관계를 파악한다.

**코드**

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)              # (a)
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)   # (b)
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,              # (c)
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
```

| 그림 | 인코딩 | 확인되는 것 |
|---|---|---|
| (a) | 위치 | 캘리포니아 형태 |
| (b) | 위치 + 투명도 0.2 | 샘플 밀집 지역 |
| (c) | 위치 + 크기(`s`=인구/100) + 색(`c`=집값, `cmap="jet"`) | 집값의 공간 분포 |

**결과 해석**
- 해안 및 대도시(베이 에어리어, LA, 샌디에이고) 주변에서 집값이 높다
- 가설: "인구 밀집 중심까지의 거리"가 유용한 특성이다 → 4-7의 `ClusterSimilarity`로 구현

---

### 3-2. 상관관계 분석

**목적**
각 특성과 타깃의 관계 강도를 정량화한다.

**개념·원리**

**피어슨 상관계수**

$$
r_{XY} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2}\sqrt{\sum (y_i - \bar{y})^2}}, \quad -1 \le r \le 1
$$

| 값 | 해석 |
|---|---|
| $r \to 1$ | 강한 양의 선형 관계 |
| $r \to 0$ | 선형 관계 없음 |
| $r \to -1$ | 강한 음의 선형 관계 |

한계: **선형 관계만 측정**한다. 비선형 관계는 $r \approx 0$일 수 있다. 또한 상관계수는 기울기의 크기와 무관하다.

**코드**

```python
corr_matrix = housing.corr(numeric_only=True)       # 수치형 열 쌍별 상관계수 행렬
corr_matrix["median_house_value"].sort_values(ascending=False)

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))   # 4×4 산점도 행렬 (대각선 = 히스토그램)

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
```

**결과 해석**

| 특성 | $r$ |
|---|---|
| median_income | **0.688** |
| total_rooms | 0.137 |
| housing_median_age | 0.102 |
| households | 0.071 |
| total_bedrooms | 0.055 |
| population | −0.020 |
| longitude | −0.051 |
| latitude | −0.140 |

- 소득이 가장 강한 예측 변수
- 소득-집값 산점도에서 $500,000 수평선(상한), $450,000·$350,000·$280,000 부근 수평선 → 데이터 특이 패턴. 모델이 학습하지 않도록 해당 구역 제거를 고려

---

### 3-3. 특성 조합

**목적**
기존 특성을 조합해 타깃과 더 관련 있는 특성을 만든다 (특성 공학).

**개념·원리**
`total_rooms`, `total_bedrooms`, `population`은 **구역 단위 합계**라 구역 규모(가구 수)에 비례한다. 집의 특성을 나타내려면 가구 수로 정규화한 **비율**이 필요하다.

**코드**

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

**결과 해석**

| 새 특성 | 정의 | $r$ | 비교 대상 $r$ |
|---|---|---|---|
| `rooms_per_house` | 방 / 가구 | 0.144 | `total_rooms` 0.137 |
| `bedrooms_ratio` | 침실 / 방 | **−0.256** | `total_bedrooms` 0.055 |
| `people_per_house` | 인구 / 가구 | −0.038 | `population` −0.020 |

- `bedrooms_ratio`가 음의 상관: 침실 외 공간(거실·서재 등) 비중이 클수록 비싸다

**정리**
- 합계 특성은 규모로 나눈 비율로 바꾸면 의미가 명확해진다
- 이 세 비율 특성은 4-8 파이프라인에 포함된다

---

## 4. 데이터 전처리 (2.5)

**입력 X와 타깃 y 분리**

```python
housing = strat_train_set.drop("median_house_value", axis=1)   # X (새 DataFrame 반환)
housing_labels = strat_train_set["median_house_value"].copy()  # y
```

**처리 대상**

| 문제 | 대상 열 | 소절 |
|---|---|---|
| 결측치 | `total_bedrooms` | 4-2 |
| 범주형 | `ocean_proximity` | 4-3 |
| 스케일 차이 | 모든 수치형 | 4-4 |
| 비대칭·다봉 분포 | 인구, 방 수, 연식 등 | 4-5 |
| 비율 특성·지리 특성 생성 | 3-3, 3-1 가설 | 4-7 |
| 자동화 | 전체 | 4-8 |

---

### 4-1. 사이킷런 API 설계

**목적**
이후 모든 전처리 도구가 따르는 공통 인터페이스를 이해한다.

**개념·원리**

| 개념 | 메서드 | 역할 | 예 |
|---|---|---|---|
| **추정기** (estimator) | `fit(X, y)` | 데이터로부터 파라미터 학습 | 모든 클래스 |
| **변환기** (transformer) | `transform(X)`, `fit_transform(X)` | 학습한 파라미터로 데이터 변환 | `SimpleImputer`, `StandardScaler` |
| **예측기** (predictor) | `predict(X)`, `score(X, y)` | 예측, 성능 점수 | `LinearRegression` |

| 설계 원칙 | 규칙 |
|---|---|
| 일관성 | 모든 객체가 같은 인터페이스 사용 |
| 검사 가능 | 하이퍼파라미터 = 공개 속성 (`imputer.strategy`), 학습된 파라미터 = 이름 끝 `_` (`imputer.statistics_`) |
| 클래스 남용 방지 | 입출력은 NumPy 배열·SciPy 희소 행렬 |
| 조합성 | 여러 변환기 + 추정기 → `Pipeline` |
| 합리적 기본값 | 대부분 기본값으로 동작 |

**fit / transform 사용 규칙**

| 데이터 | 호출 |
|---|---|
| 훈련 세트 | `fit` → `transform` (또는 `fit_transform`) |
| 검증·테스트·신규 데이터 | `transform`만 |

테스트 세트로 `fit`하면 테스트 정보가 전처리 파라미터에 유입된다(데이터 누수).

---

### 4-2. 결측치 처리

**목적**
`total_bedrooms`의 결측치 207개를 처리한다.

**개념·원리**

| 방법 | 코드 | 장점 | 단점 |
|---|---|---|---|
| ① 행 삭제 | `dropna(subset=[...])` | 단순 | 샘플 손실 |
| ② 열 삭제 | `drop(열, axis=1)` | 단순 | 특성 정보 손실 |
| ③ 대체(imputation) | `fillna(값)` | 샘플·특성 보존 | 대체값은 실제값이 아님 |

③을 쓸 때는 **훈련 세트의 통계량을 저장해 두고** 이후 모든 데이터에 같은 값을 적용해야 한다 → `SimpleImputer`.

**코드: 세 방법 비교**

```python
null_rows_idx = housing.isnull().any(axis=1)        # 결측이 있는 행 마스크

housing_option1 = housing.copy()
housing_option1.dropna(subset=["total_bedrooms"], inplace=True)          # ①

housing_option2 = housing.copy()
housing_option2.drop("total_bedrooms", axis=1, inplace=True)             # ②

housing_option3 = housing.copy()
median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"] = housing_option3["total_bedrooms"].fillna(median)  # ③
```

**코드: `SimpleImputer`**

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")                 # 하이퍼파라미터: 대체 전략
housing_num = housing.select_dtypes(include=[np.number])   # 중간값은 수치형에만 정의됨
imputer.fit(housing_num)                                   # 열별 중간값 학습
imputer.statistics_                                        # 학습된 파라미터
X = imputer.transform(housing_num)                         # NumPy 배열 반환
housing_tr = pd.DataFrame(X, columns=housing_num.columns,  # DataFrame 복원
                          index=housing_num.index)
```

| `strategy` | 대체값 | 대상 |
|---|---|---|
| `"median"` | 중간값 | 수치형 |
| `"mean"` | 평균 | 수치형 |
| `"most_frequent"` | 최빈값 | 수치형·범주형 |
| `"constant"` | `fill_value` | 수치형·범주형 |

**설계 포인트**: 현재 결측은 한 열뿐이지만, 운영 데이터에서는 어느 열이든 결측이 생길 수 있으므로 **모든 수치형 열**에 적용한다.

**참고: 이상치 탐지**

```python
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)     # 1: 정상, -1: 이상치
# 제거 시 X와 y에 같은 마스크 적용
#housing = housing.iloc[outlier_pred == 1]
#housing_labels = housing_labels.iloc[outlier_pred == 1]
```

원리: 무작위 분할을 반복할 때 적은 분할로 고립되는 샘플일수록 이상치로 판정.

**정리**
- 대체(imputation)를 기본으로 사용하고, 훈련 세트로만 `fit`
- 변환기 출력은 NumPy 배열 (`set_config(transform_output="pandas")`로 DataFrame 출력 가능)

---

### 4-3. 범주형 특성 인코딩

**목적**
텍스트 범주 `ocean_proximity`를 수치로 변환한다.

**개념·원리**

| 방법 | 출력 | 암묵적 가정 | 적합한 경우 |
|---|---|---|---|
| 순서 인코딩 (OrdinalEncoder) | 정수 1열 (0~4) | 범주 간 순서·거리 존재 | 순서형 범주 (나쁨<보통<좋음) |
| 원-핫 인코딩 (OneHotEncoder) | 0/1 열 k개 | 범주 간 거리 동일 | 명목형 범주 |

`ocean_proximity`는 명목형이다. 순서 인코딩 시 `<1H OCEAN`(0)과 `NEAR OCEAN`(4)이 가장 먼 범주로 표현되지만 실제로는 유사하다 → 원-핫 인코딩 사용.

**코드: OrdinalEncoder**

```python
housing_cat = housing[["ocean_proximity"]]      # [[...]] → 2D DataFrame (변환기 입력 형식)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]          # [[3.], [0.], [1.], [1.], [4.], [1.], [0.], [3.]]
ordinal_encoder.categories_      # [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])]
```

**코드: OneHotEncoder**

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)   # 16512×5 희소 행렬
housing_cat_1hot.toarray()                                  # 밀집 배열 변환

cat_encoder = OneHotEncoder(sparse_output=False)            # 밀집 배열로 바로 출력
```

- **희소 행렬**(sparse matrix): 0이 아닌 원소의 위치·값만 저장. 원-핫 결과는 행당 1개만 1이므로 메모리 효율적

**`pd.get_dummies()`와의 차이**

```python
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)          # 2개 열 (현재 데이터의 범주만)
cat_encoder.transform(df_test)   # 5개 열 (학습한 범주 전체)
```

| 구분 | `pd.get_dummies` | `OneHotEncoder` |
|---|---|---|
| 범주 기억 | X (호출 시점 데이터 기준) | O (`categories_`) |
| 출력 열 수 | 데이터마다 다를 수 있음 | 항상 동일 |
| 파이프라인 사용 | 부적합 | 적합 |

**미지 범주 처리**

```python
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
cat_encoder.handle_unknown = "ignore"            # 기본값 "error"
cat_encoder.transform(df_test_unknown)
# [[0., 0., 0., 0., 0.],    ← 미지 범주 → 모두 0
#  [0., 0., 1., 0., 0.]]

cat_encoder.feature_names_in_          # ['ocean_proximity']
cat_encoder.get_feature_names_out()    # ['ocean_proximity_<1H OCEAN', ..., 'ocean_proximity_NEAR OCEAN']
```

**정리**
- 명목형 → `OneHotEncoder(handle_unknown="ignore")`
- 범주 수가 매우 많으면 원-핫 대신 임베딩·대체 수치 특성 고려

---

### 4-4. 특성 스케일링

**목적**
특성 간 값 범위 차이를 제거한다.

**개념·원리**
대부분의 알고리즘(선형 모델, 거리 기반, 경사 하강법 등)은 값 범위가 큰 특성에 지배된다. (예: 방 수 6~39,320 vs 소득 0~15) 트리 기반 모델은 영향이 적다.

**min-max 스케일링**

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \;\;(\to [0, 1]), \quad \text{feature\_range로 } [a, b] \text{ 지정 가능}
$$

**표준화**

$$
z = \frac{x - \mu}{\sigma}
$$

| 구분 | MinMaxScaler | StandardScaler |
|---|---|---|
| 결과 범위 | 고정 (예: [−1, 1]) | 제한 없음 (평균 0, 표준편차 1) |
| 이상치 영향 | 큼 (min·max가 이상치에 의해 결정) | 상대적으로 작음 |
| 주 사용처 | 신경망 (입력 범위가 고정되어야 할 때) | 일반적인 기본값 |

**코드**

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

**정리**
- 기본은 `StandardScaler`
- 스케일러도 훈련 세트로만 `fit`

---

### 4-5. 분포 변환

**목적**
스케일링 전에 분포의 **모양**을 다듬는다. 스케일러는 위치·크기만 바꾸고 모양은 바꾸지 못한다.

**개념·원리**

| 분포 형태 | 문제 | 변환 | 효과 |
|---|---|---|---|
| 오른쪽 꼬리가 긺 (heavy tail) | 큰 값이 압축되지 못해 대부분 값이 좁은 범위에 몰림 | 로그 $\log x$ (또는 제곱근, 거듭제곱 0~1) | 종 모양에 가까워짐 |
| 임의 분포 | 형태 불명 | 버킷화 (백분위 구간) | 균등 분포 |
| 다봉 분포 (봉우리 여러 개) | 단일 단조 관계로 설명 불가 | 봉우리 기준 RBF 유사도 | 봉우리 근접도 특성 |

**RBF(가우스 방사 기저 함수) 유사도**

$$
\text{RBF}(x, c) = \exp\left(-\gamma \|x - c\|^2\right)
$$

- $x = c$이면 1, $c$에서 멀어질수록 0에 수렴
- $\gamma$가 클수록 감소가 빠름 (좁은 종 모양)

**코드: 로그 변환**

```python
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
```

**코드: 백분위 버킷화**

```python
percentiles = [np.percentile(housing["median_income"], p) for p in range(1, 100)]  # 1~99 백분위
flattened_median_income = pd.cut(housing["median_income"],
                                 bins=[-np.inf] + percentiles + [np.inf],
                                 labels=range(1, 100 + 1))   # 100개 구간, 각 1% → 균등
```

**코드: RBF 유사도**

```python
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

`housing_median_age`의 봉우리(35년) 근접도를 새 특성으로 만든다. 단순 모델도 "특정 연식대의 효과"를 표현할 수 있게 된다.

**정리**
- 변환 순서: 분포 변환(로그 등) → 스케일링
- 다봉 분포는 봉우리별 RBF 특성으로 분해

---

### 4-6. 타깃 변환

**목적**
타깃(y)을 변환해 학습한 경우, 예측값을 원래 단위로 되돌린다.

**개념·원리**
변환된 타깃으로 학습하면 `predict()` 출력도 변환된 공간의 값이다 → 역변환(`inverse_transform`) 필요.

**코드: 수동 처리**

```python
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())   # Series → 2D

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)                     # 변환된 공간의 값
predictions = target_scaler.inverse_transform(scaled_predictions)     # 원래 단위 (달러)
```

**코드: 자동 처리**

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)    # 내부에서 y 변환 후 학습
predictions = model.predict(some_new_data)               # 내부에서 역변환 후 반환
```

**정리**
- 타깃 변환 시 `TransformedTargetRegressor`로 역변환 누락 방지

---

### 4-7. 사용자 정의 변환기

**목적**
사이킷런에 없는 변환(비율 특성, 지리 군집 특성)을 파이프라인에 넣을 수 있는 형태로 만든다.

**개념·원리**

| 변환 유형 | 학습 필요 | 구현 방법 |
|---|---|---|
| 상태 없는 변환 (로그, 나눗셈 등) | X | `FunctionTransformer` |
| 상태 있는 변환 (평균, 군집 중심 등) | O | 클래스 (`BaseEstimator` + `TransformerMixin`) |

#### (1) `FunctionTransformer`

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)   # 역변환 함수 지정 가능
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))   # kw_args: 추가 인자
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41                                         # 샌프란시스코 좌표
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])   # 2D RBF

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))          # [[0.5], [0.75]]
```

- `X[:, [0]]`: 리스트 인덱싱 → 2D 열 벡터 유지 (`X[:, 0]`은 1D)

#### (2) 클래스 구현 규약

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):          # *args, **kwargs 금지
        self.with_mean = with_mean               # 저장만

    def fit(self, X, y=None):                    # y 인자 필수
        X = check_array(X)                       # 유한 실수 배열 검사
        self.mean_ = X.mean(axis=0)              # 학습 파라미터: 이름 끝 _
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]
        return self                              # self 반환

    def transform(self, X):
        check_is_fitted(self)                    # _로 끝나는 속성 존재 확인
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
```

| 규약 | 이유 |
|---|---|
| `BaseEstimator` 상속 | `get_params()` / `set_params()` 제공 → 하이퍼파라미터 탐색에 필요 |
| `TransformerMixin` 상속 | `fit_transform()` 제공 |
| `__init__`에 `*args`, `**kwargs` 금지 | `get_params()`가 생성자 시그니처로 하이퍼파라미터 목록을 만듦 |
| `__init__`은 인자 저장만 | 검증·계산은 `fit`에서 |
| `fit(X, y=None)` | 파이프라인이 y를 전달하므로 시그니처 통일 |
| 학습 속성 `name_` | 학습 여부 판별 기준 |
| `n_features_in_` | 입력 열 수 검증 |
| `return self` | 메서드 체이닝 |

#### (3) `ClusterSimilarity`

**목적**: 3-1의 가설("인구 밀집 중심과의 거리")을 특성으로 구현한다.

**원리**
1. `fit`: 위경도에 k-평균 군집을 적용해 군집 중심 k개를 학습
2. `transform`: 각 샘플과 각 군집 중심 사이의 RBF 유사도 계산 → k개 특성

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)    # 샘플 가중치 → 중심 위치에 반영
        return self

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)   # (n, k)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

```python
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
similarities[:3].round(2)
# [[0.  , 0.14, 0.  , 0.  , 0.  , 0.08, 0.  , 0.99, 0.  , 0.6 ],
#  [0.63, 0.  , 0.99, 0.  , 0.  , 0.  , 0.04, 0.  , 0.11, 0.  ], ...]
```

**결과 해석**: 첫 샘플은 7번 군집 중심에 거의 위치(0.99), 9번과도 가까움(0.6).

| 하이퍼파라미터 | 의미 | 튜닝 대상 |
|---|---|---|
| `n_clusters` | 군집 중심 수 = 생성 특성 수 | O (6장 그리드 서치) |
| `gamma` | 유사도 감소 속도 | O |

**정리**
- 상태 없는 변환 → `FunctionTransformer`, 상태 있는 변환 → 규약을 지킨 클래스
- `get_feature_names_out` 구현 → 파이프라인 전체의 특성 이름 추적 가능

---

### 4-8. 변환 파이프라인

**목적**
모든 전처리를 하나의 객체로 묶어 재현 가능하고 튜닝 가능한 형태로 만든다.

**개념·원리**

**`Pipeline`**: 단계의 순차 연결

| 동작 | 처리 |
|---|---|
| `fit(X, y)` | 단계 1~(n−1): `fit_transform` 후 출력 전달, 단계 n: `fit` |
| `transform` / `predict` | 마지막 단계가 가진 메서드를 노출 |
| 제약 | 마지막 단계 외에는 모두 변환기 |

**`ColumnTransformer`**: 열 집합별 병렬 처리 후 수평 결합(hstack)

```
            ┌─ [열 A, B] → 변환기 1 ─┐
입력 X ─────┼─ [열 C]    → 변환기 2 ─┼─→ hstack → 출력
            └─ 나머지    → remainder ─┘
```

**파라미터 접근 문법**: `단계이름__파라미터이름` (이중 밑줄로 중첩 경로 표현)

**코드: Pipeline 기본**

```python
from sklearn.pipeline import Pipeline, make_pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())  # 이름 자동

housing_num_prepared = num_pipeline.fit_transform(housing_num)
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)
```

| 접근 | 코드 |
|---|---|
| 다이어그램 표시 | `set_config(display='diagram')` |
| 단계 목록 | `num_pipeline.steps` |
| 인덱스 접근 | `num_pipeline[1]`, `num_pipeline[:-1]` |
| 이름 접근 | `num_pipeline.named_steps["simpleimputer"]` |
| 파라미터 설정 | `num_pipeline.set_params(simpleimputer__strategy="median")` |

**코드: ColumnTransformer 기본**

```python
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# 타입 기반 열 선택
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
```

**코드: 최종 전처리 파이프라인**

```python
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]                       # 출력 특성 이름

def ratio_pipeline():                      # 호출마다 새 객체 생성
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)        # 남은 열: housing_median_age
```

**설계 근거**

| 설계 | 근거 |
|---|---|
| `ratio_pipeline`을 함수로 정의 | 같은 객체를 여러 단계에서 공유하면 학습 상태가 덮어써짐 → 단계마다 독립 객체 필요 |
| 대체 → 나눗셈 → 표준화 순서 | 결측 상태로 나누면 결과도 결측 |
| `feature_names_out` 지정 | `FunctionTransformer`는 출력 이름을 알 수 없음 → 명시 |
| 같은 열을 여러 단계에 사용 | 허용됨 (각 단계가 독립적으로 열을 선택) |
| `remainder` | 미지정 열 처리. 기본 `"drop"`, `"passthrough"` 가능 |

**결과 해석**

```python
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape                 # (16512, 24)
preprocessing.get_feature_names_out()
```

| 단계 | 입력 열 | 출력 열 수 | 출력 이름 예 |
|---|---|---|---|
| bedrooms | total_bedrooms, total_rooms | 1 | `bedrooms__ratio` |
| rooms_per_house | total_rooms, households | 1 | `rooms_per_house__ratio` |
| people_per_house | population, households | 1 | `people_per_house__ratio` |
| log | 5개 | 5 | `log__median_income` |
| geo | latitude, longitude | 10 | `geo__Cluster 0 similarity` |
| cat | ocean_proximity | 5 | `cat__ocean_proximity_INLAND` |
| remainder | housing_median_age | 1 | `remainder__housing_median_age` |
| **합계** | | **24** | |

**정리**
- 원본 DataFrame → `preprocessing` → 24차원 특성 행렬
- 이름 규칙 `단계__원래이름`으로 모든 특성 추적 가능
- 파이프라인 내부 하이퍼파라미터는 `단계__파라미터`로 접근 → 6장 튜닝에 사용

---

## 5. 모델 선택과 훈련 (2.6)

### 5-1. 훈련 세트 평가

**목적**
여러 모델을 학습시키고 기본 성능을 확인한다.

**개념·원리**
전처리 파이프라인 뒤에 모델을 연결하면 원본 데이터로 바로 학습·예측할 수 있다.

```
원본 X ─→ preprocessing ─→ 모델 ─→ ŷ
```

**코드: 선형 회귀**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2)                   # 100 단위 반올림
housing_labels.iloc[:5].values

error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))
# -46.3%, -23.0%, 33.4%, -4.9%, -8.5%

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)   # 68972.89
```

**코드: 결정 트리**

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)
housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)  # 0.0
```

**결과 해석**

| 모델 | 훈련 RMSE | 해석 |
|---|---|---|
| 선형 회귀 | 68,973 | 타깃 대부분이 $120k~$265k → 오차가 큼. **과소적합** 의심 |
| 결정 트리 | 0 | 훈련 데이터를 완전히 암기. **과대적합** 의심 |

**한계**: 훈련 세트 오차는 일반화 성능을 나타내지 못한다. 테스트 세트는 최종 평가용이므로 사용할 수 없다 → 검증 방법 필요.

---

### 5-2. 교차 검증

**목적**
테스트 세트를 쓰지 않고 일반화 성능을 추정한다.

**개념·원리**

**k-폴드 교차 검증** (k-fold cross-validation)

1. 훈련 세트를 k개 폴드로 분할
2. i번째 반복: 폴드 i를 검증용, 나머지 k−1개로 학습
3. k번 반복 → 검증 점수 k개
4. 평균 = 성능 추정치, 표준편차 = 추정의 정밀도

```
반복 1: [검증][훈련][훈련]...[훈련]
반복 2: [훈련][검증][훈련]...[훈련]
  ⋮
반복 k: [훈련][훈련][훈련]...[검증]
```

**점수 부호 규약**: 사이킷런의 `scoring`은 "클수록 좋음"을 전제한다. 오차 지표는 음수로 반환된다(`neg_root_mean_squared_error`) → 부호를 반전해 RMSE로 사용.

**코드**

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_rmses).describe()

lin_rmses = -cross_val_score(lin_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_rmses).describe()

from sklearn.ensemble import RandomForestRegressor
forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
pd.Series(forest_rmses).describe()

forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)
forest_rmse = root_mean_squared_error(housing_labels, housing_predictions)
```

**랜덤 포레스트**: 샘플·특성을 무작위로 추출해 학습한 결정 트리 여러 개의 예측을 평균하는 앙상블 모델 (6장).

**결과 해석**

| 모델 | CV 평균 | CV 표준편차 |
|---|---|---|
| 결정 트리 | 66,574 | 1,103 |
| 선형 회귀 | 70,003 | 4,182 |
| 랜덤 포레스트 | **47,038** | 1,021 |

---

### 5-3. 모델 비교와 진단

**개념·원리**

| 훈련 오차 | 검증 오차 | 진단 | 대응 |
|---|---|---|---|
| 높음 | 높음 | **과소적합** (편향 큼) | 더 복잡한 모델, 더 좋은 특성, 규제 완화 |
| 낮음 | 높음 | **과대적합** (분산 큼) | 모델 단순화, 규제, 데이터 추가 |
| 낮음 | 낮음 | 적절 | |

**결과 해석**

| 모델 | 훈련 RMSE | CV RMSE | 차이 | 진단 |
|---|---|---|---|---|
| 선형 회귀 | 68,973 | 70,003 | 작음 | 과소적합 |
| 결정 트리 | 0 | 66,574 | 매우 큼 | 심한 과대적합 |
| 랜덤 포레스트 | 17,551 | 47,038 | 큼 | 과대적합 (가장 우수) |

**정리**
- 모델 비교는 교차 검증 점수로
- 이 단계 목표: 하이퍼파라미터를 깊게 조정하지 않고 유망 모델 2~5개 선정
- 선정 결과: 랜덤 포레스트

---

## 6. 모델 튜닝과 최종 평가 (2.7)

### 6-1. 그리드 서치

**목적**
하이퍼파라미터 후보의 모든 조합을 교차 검증으로 평가해 최적 조합을 찾는다.

**개념·원리**
- 탐색 공간: 사용자가 지정한 이산 후보값의 데카르트 곱
- 평가 횟수 = (조합 수) × (폴드 수)
- 파이프라인 내부 파라미터를 `단계__파라미터`로 지정 → **전처리와 모델 하이퍼파라미터를 동시에 탐색**

**코드**

```python
from sklearn.model_selection import GridSearchCV

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42)),
])
param_grid = [
    {'preprocessing__geo__n_clusters': [5, 8, 10],
     'random_forest__max_features': [4, 6, 8]},
    {'preprocessing__geo__n_clusters': [10, 15],
     'random_forest__max_features': [6, 8, 10]},
]
grid_search = GridSearchCV(full_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing, housing_labels)
```

| 파라미터 | 경로 | 의미 |
|---|---|---|
| `preprocessing__geo__n_clusters` | full_pipeline → preprocessing → geo(ClusterSimilarity) → n_clusters | 군집 중심 수 |
| `random_forest__max_features` | full_pipeline → random_forest → max_features | 분할 시 고려할 특성 수 |

평가 횟수: (3×3 + 2×3) × 3 = **45회**

```python
full_pipeline.get_params().keys()     # 탐색 가능한 전체 파라미터 목록
grid_search.best_params_              # {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}
grid_search.best_estimator_           # 최적 조합으로 전체 훈련 세트 재학습된 모델 (refit=True 기본)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
```

**결과 해석**

| n_clusters | max_features | split0 | split1 | split2 | 평균 RMSE |
|---|---|---|---|---|---|
| 15 | 6 | 42,725 | 43,708 | 44,335 | **43,590** |
| 15 | 8 | 43,486 | 43,820 | 44,900 | 44,069 |
| 10 | 4 | 43,798 | 44,036 | 44,961 | 44,265 |

- 47,038 → 43,590으로 개선
- 최적값 15가 후보 범위의 **경계(최댓값)** → 범위를 확장해 재탐색해야 함

---

### 6-2. 랜덤 서치

**목적**
넓은 탐색 공간을 제한된 예산으로 효율적으로 탐색한다.

**개념·원리**
- 각 하이퍼파라미터에 **확률 분포**를 지정하고, `n_iter`개 조합을 무작위 추출해 평가
- 평가 횟수 = `n_iter` × 폴드 수

| 구분 | 그리드 서치 | 랜덤 서치 |
|---|---|---|
| 탐색 공간 | 이산 후보값 | 확률 분포 (연속 가능) |
| 평가 횟수 | 조합 수에 의해 결정 | `n_iter`로 직접 지정 |
| 하이퍼파라미터별 시도 값 수 | 후보 개수로 고정 | 반복마다 다른 값 |
| 효과 없는 하이퍼파라미터 | 시간 낭비 큼 | 영향 작음 |
| 적합 | 탐색 공간이 작을 때 | 탐색 공간이 크거나 연속일 때 |

**분포 선택 기준**

| 분포 | 용도 |
|---|---|
| `randint(a, b+1)` | 정수, 범위 내 값이 모두 동등하게 그럴듯함 |
| `uniform(a, b)` | 실수, 위와 동일 |
| `geom(1/scale)` | 정수, 대략적 스케일을 알고 작은 값 선호 |
| `expon(scale)` | 실수, 위와 동일 |
| `loguniform(a, b)` | 스케일 자체를 모를 때 (각 자릿수 구간을 동일 확률로 탐색) |

**코드**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)
rnd_search.fit(housing, housing_labels)      # 10 × 3 = 30회
```

**결과 해석**

| n_clusters | max_features | split0 | split1 | split2 | 평균 RMSE |
|---|---|---|---|---|---|
| 45 | 9 | 41,342 | 42,242 | 43,057 | **42,214** |
| 32 | 7 | 41,825 | 42,275 | 43,241 | 42,447 |

- 그리드 서치(45회, 43,590)보다 적은 평가(30회)로 더 좋은 결과(42,214)

**참고: 연속 반감 탐색**

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
```

원리: 1라운드에 많은 후보를 적은 자원(샘플 수)으로 평가 → 상위 후보만 남기고 자원을 늘려 재평가 → 반복.

---

### 6-3. 특성 중요도 분석

**목적**
최적 모델이 어떤 특성에 의존하는지 확인해 특성 선택·개선 방향을 정한다.

**코드**

```python
final_model = rnd_search.best_estimator_                                   # 전처리 포함
feature_importances = final_model["random_forest"].feature_importances_   # 합 = 1
sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
       reverse=True)
```

**결과 해석**

| 순위 | 특성 | 중요도 | 출처 |
|---|---|---|---|
| 1 | `log__median_income` | 0.186 | 원본 (로그 변환) |
| 2 | `cat__ocean_proximity_INLAND` | 0.073 | 원본 (원-핫) |
| 3 | `bedrooms__ratio` | 0.066 | **3-3 특성 조합** |
| 4 | `rooms_per_house__ratio` | 0.054 | **3-3 특성 조합** |
| 5 | `people_per_house__ratio` | 0.046 | **3-3 특성 조합** |
| 6 | `geo__Cluster 30 similarity` | 0.042 | **4-7 ClusterSimilarity** |

- 특성 공학으로 만든 특성이 상위권 → 탐색 단계 가설이 유효했음
- `ocean_proximity` 중 INLAND만 중요 → 나머지 범주 제거 검토 가능

---

### 6-4. 테스트 세트 평가

**목적**
최종 모델의 일반화 오차를 추정한다.

**개념·원리**

- 테스트 세트에는 `predict`만 적용 (전처리 파라미터는 훈련 세트에서 학습된 값 사용)
- 점 추정치 하나로는 신뢰도를 알 수 없음 → **신뢰구간** 계산

**부트스트랩 신뢰구간**
1. 테스트 샘플의 제곱 오차 집합에서 같은 크기로 **복원 추출**
2. 재표본의 RMSE 계산
3. 1~2를 다수 반복 → RMSE의 경험적 분포
4. 분포의 2.5%, 97.5% 분위수 → 95% 신뢰구간

**코드**

```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)
final_rmse = root_mean_squared_error(y_test, final_predictions)    # 41445.53

from scipy.stats import bootstrap

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = bootstrap([squared_errors], rmse, confidence_level=confidence,   # 데이터는 시퀀스로 전달
                        random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
print(f"95% CI for RMSE: ({rmse_lower:.4f}, {rmse_upper:.4f})")
```

**결과 해석**

| 항목 | 값 |
|---|---|
| 테스트 RMSE | **41,446** |
| 95% 신뢰구간 | **(39,521, 43,702)** |

- 테스트 RMSE가 교차 검증 RMSE(42,214)와 비슷 → 교차 검증 추정이 타당했음
- 일반적으로 튜닝을 거친 모델은 교차 검증 점수보다 테스트 점수가 약간 나쁘게 나올 수 있음

**규칙**: 테스트 결과를 보고 하이퍼파라미터를 재조정하지 않는다. 재조정하면 테스트 세트에 과대적합되어 일반화 오차 추정이 무효가 된다.

---

## 7. 배포와 운영 (2.8)

### 7-1. 모델 저장과 로드

**목적**
학습된 파이프라인을 운영 환경에서 재사용한다.

**개념·원리**
- `joblib`: NumPy 배열이 많은 객체의 직렬화에 효율적인 pickle 기반 도구
- 저장 대상: 전처리 + 모델 파이프라인 전체
- **직렬화의 한계**: pickle은 함수·클래스의 **코드가 아닌 참조 경로**(예: `__main__.column_ratio`)만 저장한다 → 로드하는 환경에 같은 이름의 정의가 있어야 한다

**코드**

```python
import joblib
joblib.dump(final_model, "my_california_housing_model.pkl")      # 저장
```

```python
# 운영 스크립트
import joblib
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):                                   # 사용자 정의 함수 재정의 필요
    return X[:, [0]] / X[:, [1]]

#class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]                                             # 사용자 정의 클래스 재정의 필요

final_model_reloaded = joblib.load("my_california_housing_model.pkl")
new_data = housing.iloc[:5]
predictions = final_model_reloaded.predict(new_data)
```

**실무 구성**: 사용자 정의 변환기를 별도 모듈(`.py`)로 분리 → 학습 코드와 운영 코드가 같은 모듈을 import

### 7-2. 운영 단계 작업

| 영역 | 내용 |
|---|---|
| 배포 방식 | 웹 앱 내부 로드 / 별도 예측 서비스(REST API) / 클라우드 플랫폼(Vertex AI 등) |
| 성능 모니터링 | 실시간 성능 측정. **모델 부패**(데이터 분포 변화로 인한 성능 저하) 감지 |
| 입력 품질 모니터링 | 결측·이상값·분포 변화 감지 및 알림 |
| 재학습 | 새 데이터 수집 → 재학습 → 평가 → 교체 과정 자동화 |
| 버전 관리 | 모델·데이터셋 버전 보관, 이전 모델로 롤백 가능하게 |

---

## 8. 연습문제 (2.9)

| 번호 | 과제 | 핵심 개념 | 결과 |
|---|---|---|---|
| 1 | SVR 그리드 서치 | 커널별 탐색 공간 분리 | RMSE 70,060 |
| 2 | SVR 랜덤 서치 | `loguniform`, `expon` 분포 | RMSE 56,152 |
| 3 | `SelectFromModel` 특성 선택 | 중요도 기반 선택 변환기 | 개선 없음 |
| 4 | KNN 예측을 특성으로 사용 | 메타 추정기, `clone` | 악화 |
| 5 | 전처리 옵션 랜덤 서치 | 다단계 중첩 파라미터 | 군집 특성보다 열세 |
| 6 | `StandardScalerClone` 완성 | `validate_data`, 역변환, 특성 이름 | 규약 검사 통과 |

### 8-1. SVR 그리드 서치

**목적**: 다른 계열 모델(서포트 벡터 회귀)의 성능을 확인한다.

```python
from sklearn.svm import SVR

param_grid = [
        {'svr__kernel': ['linear'], 'svr__C': [10., 30., 100., 300., 1000.,
                                               3000., 10000., 30000.0]},
        {'svr__kernel': ['rbf'], 'svr__C': [1.0, 3.0, 10., 30., 100., 300.,
                                            1000.0],
         'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]
svr_pipeline = Pipeline([("preprocessing", preprocessing), ("svr", SVR())])
grid_search = GridSearchCV(svr_pipeline, param_grid, cv=3,
                           scoring='neg_root_mean_squared_error')
grid_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])

-grid_search.best_score_        # 70059.93
grid_search.best_params_        # {'svr__C': 10000.0, 'svr__kernel': 'linear'}
```

- 샘플 5,000개만 사용: SVM의 훈련 복잡도가 샘플 수의 제곱~세제곱에 비례
- 딕셔너리 분리: `linear` 커널에는 `gamma`가 무의미 → 불필요한 조합 제거
- 최적 `C`가 범위 경계 근처 → 범위 확장 필요

### 8-2. SVR 랜덤 서치

```python
from scipy.stats import expon, loguniform

param_distribs = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': loguniform(20, 200_000),
        'svr__gamma': expon(scale=1.0),
    }
rnd_search = RandomizedSearchCV(svr_pipeline,
                                param_distributions=param_distribs,
                                n_iter=50, cv=3,
                                scoring='neg_root_mean_squared_error',
                                random_state=42)
rnd_search.fit(housing.iloc[:5000], housing_labels.iloc[:5000])
-rnd_search.best_score_         # 56152.05

s = expon(scale=1).rvs(100_000, random_state=42)
((s > 0.105) & (s < 2.29)).sum() / 100_000      # ≈ 0.8
```

| 하이퍼파라미터 | 분포 | 근거 |
|---|---|---|
| `C` | `loguniform(20, 200000)` | 적절한 스케일을 모름 → 자릿수 구간 균등 탐색 |
| `gamma` | `expon(scale=1)` | 1 근처로 예상 → 약 80%가 0.1~2.3 |

결과: RBF 커널 선택, 70,060 → 56,152

### 8-3. 특성 선택

```python
from sklearn.feature_selection import SelectFromModel

selector_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),          # 중요도 하한
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])
selector_rmses = -cross_val_score(selector_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
```

원리: 내부 모델의 `feature_importances_`가 `threshold` 미만인 특성을 제거하는 변환기. 결과: 개선 없음 → `threshold`도 튜닝 대상.

### 8-4. 회귀 예측을 특성으로 사용하는 변환기

**목적**: 위경도로 주변 구역의 소득을 예측(KNN)해 특성으로 추가한다.

**원리**
- 메타 추정기: 다른 추정기를 하이퍼파라미터로 받는 추정기 (`MetaEstimatorMixin`)
- `clone(self.regressor)`: 하이퍼파라미터로 받은 추정기를 복제해 학습 → 원본 보존. 학습된 객체는 `regressor_`에 저장

```python
from sklearn.base import MetaEstimatorMixin, clone

class FeatureFromRegressor(BaseEstimator, TransformerMixin, MetaEstimatorMixin):
    def __init__(self, regressor, target_features):
        self.regressor = regressor
        self.target_features = target_features

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        X_target = X_df[self.target_features]               # 예측 대상 (소득)
        X_rest = X_df.drop(columns=self.target_features)    # 입력 (위경도)
        self.regressor_ = clone(self.regressor).fit(X_rest, X_target)
        return self

    def transform(self, X):
        columns = self.feature_names_in_
        if hasattr(X, "columns"):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=columns)
        X_rest = X_df.drop(columns=self.target_features)
        predictions = self.regressor_.predict(X_rest)
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def get_feature_names_out(self, feature_names=None):
        check_is_fitted(self)
        estimator_class_name = self.regressor_.__class__.__name__
        return [f"{estimator_class_name.lower()}_prediction_{t}"
                for t in self.target_features]
```

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = FeatureFromRegressor(knn_reg, ["median_income"])

# 기존 preprocessing의 "geo" 단계를 교체
transformers = [(name, clone(transformer), columns)
                for name, transformer, columns in preprocessing.transformers]
geo_index = [name for name, _, _ in transformers].index("geo")
transformers[geo_index] = ("geo", knn_transformer,
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

결과: 성능 악화 → `ClusterSimilarity` 특성이 더 유효.

### 8-5. 전처리 옵션 랜덤 서치

```python
param_distribs = {
    "preprocessing__geo__regressor__n_neighbors": range(1, 30),
    "preprocessing__geo__regressor__weights": ["distance", "uniform"],
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

파라미터 경로: `Pipeline → preprocessing(ColumnTransformer) → geo(FeatureFromRegressor) → regressor(KNN) → n_neighbors`. `BaseEstimator`의 `get_params(deep=True)`가 중첩 구조를 재귀적으로 노출하기 때문에 가능하다. 결과: 튜닝 후에도 군집 유사도 특성보다 열세.

### 8-6. `StandardScalerClone` 완성

**추가 요구사항**: `inverse_transform`, `feature_names_in_` 설정, `get_feature_names_out`

```python
from sklearn.utils.validation import check_is_fitted, validate_data

class StandardScalerClone(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_2d=True)              # n_features_in_, feature_names_in_ 자동 설정
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False) # reset=False: 학습 시와 일치 검사
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_

    def inverse_transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        X = X * self.scale_
        return X + self.mean_ if self.with_mean else X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return getattr(self, "feature_names_in_",
                           [f"x{i}" for i in range(self.n_features_in_)])
        else:
            if len(input_features) != self.n_features_in_:
                raise ValueError("Invalid number of features")
            if hasattr(self, "feature_names_in_") and not np.all(
                self.feature_names_in_ == input_features
            ):
                raise ValueError("input_features ≠ feature_names_in_")
            return input_features
```

| 메서드 | 수식 / 동작 |
|---|---|
| `transform` | $z = (x - \mu) / \sigma$ |
| `inverse_transform` | $x = z \cdot \sigma + \mu$ |
| `get_feature_names_out` | 학습 시 열 이름 → 없으면 `x0, x1, …` |

**검증**

```python
from sklearn.utils.estimator_checks import check_estimator
check_estimator(StandardScalerClone())          # 사이킷런 추정기 규약 일괄 검사

rng = np.random.default_rng(seed=42)
X = rng.random((1000, 3))

scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(X)
assert np.allclose(X_scaled, (X - X.mean(axis=0)) / X.std(axis=0))     # 변환 정확성

scaler = StandardScalerClone(with_mean=False)
X_scaled_uncentered = scaler.fit_transform(X)
assert np.allclose(X_scaled_uncentered, X / X.std(axis=0))             # with_mean=False

scaler = StandardScalerClone()
X_back = scaler.inverse_transform(scaler.fit_transform(X))
assert np.allclose(X, X_back)                                           # 역변환 왕복

assert np.all(scaler.get_feature_names_out() == ["x0", "x1", "x2"])
assert np.all(scaler.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c"])

df = pd.DataFrame({"a": rng.random(100), "b": rng.random(100)})
scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(df)
assert np.all(scaler.feature_names_in_ == ["a", "b"])                   # DataFrame 열 이름 기억
assert np.all(scaler.get_feature_names_out() == ["a", "b"])
```

- `np.allclose`: 부동소수점 허용 오차 내 비교

---

## 9. 정리

### 9-1. 워크플로와 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| 데이터 수집 | `load_housing_data()` | 수집 자동화 |
| 테스트 분리 | `train_test_split(stratify=income_cat)` | 탐색 전에 분리, 계층 샘플링 |
| 탐색 | `corr()`, 산점도, 비율 특성 | 훈련 세트만, 가설 도출 |
| 전처리 | `ColumnTransformer([...])` | 훈련 세트로만 `fit`, 파이프라인화 |
| 모델 선택 | `cross_val_score(..., cv=10)` | 교차 검증으로 비교 |
| 튜닝 | `RandomizedSearchCV(full_pipeline, ...)` | 전처리·모델 동시 탐색 |
| 최종 평가 | `final_model.predict(X_test)`, `bootstrap` | 테스트 세트 1회 사용, 신뢰구간 |
| 배포 | `joblib.dump` / `joblib.load` | 사용자 정의 코드 동반 배포 |

### 9-2. 성능 변화

| 단계 | RMSE |
|---|---|
| 선형 회귀 (CV) | 70,003 |
| 결정 트리 (CV) | 66,574 |
| 랜덤 포레스트 (CV) | 47,038 |
| 그리드 서치 최적 (CV) | 43,590 |
| 랜덤 서치 최적 (CV) | 42,214 |
| **최종 테스트** | **41,446** (95% CI: 39,521 ~ 43,702) |

### 9-3. 핵심 원칙

1. 테스트 세트는 탐색 전에 분리하고, 최종 평가에서 한 번만 사용한다.
2. 중요 특성이 있으면 계층 샘플링으로 분포를 보존한다.
3. 합계 특성은 규모로 나눈 비율로 바꾸면 정보가 명확해진다.
4. 모든 변환기는 훈련 세트로만 `fit`하고, 이후 데이터에는 `transform`만 적용한다.
5. 전처리는 파이프라인으로 묶는다 → 재현성 확보, 전처리 하이퍼파라미터도 튜닝 가능.
6. 모델 비교는 훈련 오차가 아닌 교차 검증 오차로 한다.
7. 사용자 정의 변환기는 사이킷런 규약을 지키고, 배포 시 코드도 함께 제공한다.
