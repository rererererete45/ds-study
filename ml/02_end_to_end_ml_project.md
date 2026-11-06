# 2장. 머신러닝 프로젝트 처음부터 끝까지

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `02_end_to_end_machine_learning_project.ipynb`
>
> 실제 데이터로 **문제 정의 → 데이터 수집 → 탐색 → 전처리 파이프라인 → 모델 비교 → 튜닝 → 평가 → 배포**까지 한 바퀴 도는 장. 이후 모든 장의 뼈대가 된다.
>
> 실습 데이터: 캘리포니아 주택 가격 `housing.csv` (1990년 인구조사, 구역(district) 단위, 20,640행)
>
> | 컬럼 | 타입 | 설명 |
> |---|---|---|
> | `longitude`, `latitude` | float | 경도·위도 |
> | `housing_median_age` | float | 주택 연식 중간값 (상한 있음) |
> | `total_rooms`, `total_bedrooms` | float | 구역 전체 방·침실 수 (`total_bedrooms` 결측 207개) |
> | `population`, `households` | float | 인구·가구 수 |
> | `median_income` | float | 소득 중간값 (약 $10,000 단위로 스케일 조정됨) |
> | `median_house_value` | float | **타깃** — 주택 가격 중간값 ($500,000에서 상한) |
> | `ocean_proximity` | object | 해안 근접도 (범주형 5개) |

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

상황 설정: "머신러닝 주택 회사"에 입사 → 구역 특성으로 **그 구역의 중간 주택 가격**을 예측하는 모델을 만든다. 예측값은 투자 판단 시스템의 입력으로 쓰인다. 지금은 전문가가 수작업으로 추정하는데 오차가 20% 이상 나기도 한다.

### 1-1. 문제 정의

| 질문 | 답 |
|---|---|
| 레이블이 있나? | 있음 → **지도 학습** |
| 무엇을 예측? | 값(가격) → **회귀**. 특성 여러 개 → 다중 회귀, 출력 하나 → 단변량 회귀 |
| 데이터가 계속 들어오나? | 아님, 크기도 작음 → **배치 학습** |

### 1-2. 성능 지표

- **RMSE** = √(평균((y − ŷ)²)) — 큰 오차에 더 큰 벌점. 회귀의 기본 지표 (ℓ2 노름)
- **MAE** = 평균(|y − ŷ|) — 이상치가 많을 때 유리 (ℓ1 노름)
- 노름 지수가 클수록 큰 값에 민감해진다

> 💡 **가정 검사**: 하위 시스템이 정말 "가격 숫자"를 원하는지 확인. "싸다/보통/비싸다" 범주를 원하면 분류 문제가 된다.

### 1-3. 환경 확인

```python
import sys
assert sys.version_info >= (3, 10)

from packaging.version import Version
import sklearn
assert Version(sklearn.__version__) >= Version("1.6.1")
```

- `assert 조건` → 거짓이면 즉시 에러
- 뒤에서 쓰는 `root_mean_squared_error`(1.4+), `sparse_output`(1.2+) 때문에 버전 확인
- `Version()`으로 감싸야 `"1.10" > "1.6"`을 숫자로 비교 (문자열 비교면 반대가 됨)

---

## 2. 데이터 가져오기 (2.3)

### 2-1. 다운로드 함수

```python
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing_full = load_housing_data()
```

| 줄 | 의미 |
|---|---|
| `Path(...)` | 경로를 객체로 다룸. OS별 `/`·`\` 차이 자동 처리 |
| `if not ...is_file()` | 이미 받았으면 건너뜀 → 여러 번 실행해도 다운로드 1회 |
| `mkdir(parents=True, exist_ok=True)` | 상위 폴더까지 생성, 이미 있어도 에러 없음 |
| `urlretrieve(url, 경로)` | URL 파일을 로컬에 저장 |
| `with tarfile.open(...)` | `.tgz` 열고 블록 끝나면 자동으로 닫음 |
| `extractall(..., filter="data")` | 압축 해제. `filter="data"`는 `../../` 같은 악성 경로를 막는 보안 옵션 |

→ 데이터 수집을 **함수로 자동화**하는 습관. 데이터가 갱신되면 함수만 다시 호출.

### 2-2. 데이터 구조 훑어보기

```python
housing_full.head()                              # 처음 5행
housing_full.info()                              # 열별 타입·결측치
housing_full["ocean_proximity"].value_counts()   # 범주별 개수
housing_full.describe()                          # 수치형 요약 통계
```

```
RangeIndex: 20640 entries, 0 to 20639
 4   total_bedrooms      20433 non-null  float64   ← 207개 결측
 9   ocean_proximity     20640 non-null  object    ← 유일한 텍스트(범주형) 열
```

```python
import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full.hist(bins=50, figsize=(12, 8))   # 모든 수치형 열 히스토그램, 막대 50개
plt.show()
```

히스토그램에서 읽을 것:
- `median_income`은 달러가 아님 (0.5~15, 약 $10k 단위)
- `housing_median_age`, `median_house_value`는 최댓값에 막대가 솟음 → **상한으로 잘린 데이터**. 타깃이 $500k에서 잘려 있으면 모델은 그 이상을 예측 못 함
- 오른쪽 꼬리가 긴 분포가 많음 → 나중에 로그 변환

### 2-3. 테스트 세트 만들기

> ⚠️ 데이터를 더 들여다보기 **전에** 테스트 세트부터 떼어 놓는다. 먼저 보면 그 패턴에 맞춰 모델을 고르게 된다 → **데이터 스누핑 편향**

#### ① 직접 만든 무작위 분할

```python
import numpy as np

def shuffle_and_split_data(data, test_ratio, rng):
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

rng = np.random.default_rng(seed=42)
train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
len(train_set)   # 16512
len(test_set)    # 4128
```

- `rng.permutation(n)` → 0~n−1을 섞은 배열
- `iloc[]` → **위치(행 번호)** 기반 인덱싱
- 문제점: **데이터셋이 갱신되면** 시드가 같아도 분할이 바뀜 → 예전 테스트 샘플이 훈련 세트로 새어 들어감

#### ② ID 해시 기반 분할

```python
from zlib import crc32

def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

- `crc32()` → 0 ~ 2³²−1의 해시 (사실상 균등 분포)
- 해시 < `0.2 × 2³²`이면 테스트 → 약 20%
- **같은 ID는 항상 같은 해시** → 데이터가 추가돼도 기존 샘플 소속 유지
- `~` = 불리언 NOT

```python
housing_with_id = housing_full.reset_index()      # 행 번호를 'index' 열로 추가
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

housing_with_id["id"] = (housing_full["longitude"] * 1000
                         + housing_full["latitude"])
train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "id")
```

- 행 번호를 ID로 쓰면 → 새 데이터는 끝에만 붙이고 중간 삭제 금지
- 그래서 **위경도처럼 변하지 않는 값**으로 ID 생성 (같은 위치끼리는 겹칠 수 있음)

#### ③ 사이킷런 `train_test_split`

```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)
```

①을 한 줄로. X, y 여러 배열을 넣으면 **같은 인덱스로 함께** 나눠 준다.

#### ④ 계층 샘플링이 필요한 이유

모집단 여성 비율 51.6%에서 1,000명을 무작위로 뽑았을 때, 표본이 49% 미만 또는 54% 초과일 확률:

```python
from scipy.stats import binom

sample_size = 1000
ratio_female = 0.516
proba_too_small = binom(sample_size, ratio_female).cdf(490 - 1)     # P(X ≤ 489)
proba_too_large = 1 - binom(sample_size, ratio_female).cdf(540)     # P(X > 540)
print(proba_too_small + proba_too_large)   # 약 0.107
```

시뮬레이션으로도 같은 값:

```python
rng = np.random.default_rng(seed=42)
samples = (rng.random((100_000, sample_size)) < ratio_female).sum(axis=1)
((samples < 490) | (samples > 540)).mean()
```

- 10만 × 1000 난수 행렬 = 설문 10만 회. `< 0.516`이면 여성 → 행별 합계 = 여성 수
- 결과 약 **10.7%** → 순수 무작위 표본은 10번에 1번꼴로 편향된다

#### ⑤ 소득 범주 만들기

```python
housing_full["income_cat"] = pd.cut(housing_full["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

cat_counts = housing_full["income_cat"].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
```

- `pd.cut` → 연속값을 구간으로: (0, 1.5] → 1, …, (6, ∞) → 5
- "소득이 가장 중요한 특성"이라는 가정 → 테스트 세트가 소득 분포를 대표하도록
- 계층이 너무 많으면 계층당 샘플이 부족 → 5개로 제한

#### ⑥ 계층 분할 실행

```python
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full,
                                              housing_full["income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]
```

- 계층 비율을 유지한 서로 다른 분할 `n_splits`개 (교차 검증용)
- `.split(X, y)`의 두 번째 인자 = **비율을 맞출 기준 열**
- 인덱스를 돌려주므로 `iloc`으로 꺼냄

분할 하나만 필요하면:

```python
strat_train_set, strat_test_set = train_test_split(
    housing_full, test_size=0.2, stratify=housing_full["income_cat"],
    random_state=42)
```

#### ⑦ 계층 vs 무작위 비교

```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

compare_props = pd.DataFrame({
    "Overall %": income_cat_proportions(housing_full),
    "Stratified %": income_cat_proportions(strat_test_set),
    "Random %": income_cat_proportions(test_set),
}).sort_index()
compare_props["Strat. Error %"] = (compare_props["Stratified %"] /
                                   compare_props["Overall %"] - 1)
compare_props["Rand. Error %"] = (compare_props["Random %"] /
                                  compare_props["Overall %"] - 1)
(compare_props * 100).round(2)
```

| 소득 범주 | 전체 % | 계층 % | 무작위 % | 계층 오차 % | 무작위 오차 % |
|---|---|---|---|---|---|
| 1 | 3.98 | 4.00 | 4.24 | 0.36 | **6.45** |
| 2 | 31.88 | 31.88 | 30.74 | −0.02 | −3.59 |
| 3 | 35.06 | 35.05 | 34.52 | −0.01 | −1.53 |
| 4 | 17.63 | 17.64 | 18.41 | 0.03 | 4.42 |
| 5 | 11.44 | 11.43 | 12.09 | −0.08 | 5.63 |

→ 계층 분할은 최대 0.36%, 무작위는 최대 6.45%. 소수 계층(1범주, 3.98%)일수록 무작위 분할이 크게 흔들린다.

#### ⑧ 임시 열 제거

```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

- `axis=1` 열 방향, `inplace=True` 원본 직접 수정
- `set_` — 내장 함수 `set`을 덮어쓰지 않으려는 관례

---

## 3. 탐색과 시각화 (2.4)

```python
housing = strat_train_set.copy()
```

**훈련 세트만** 탐색. `.copy()`로 원본 보호.

### 3-1. 지리 정보 시각화

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True)
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
```

`alpha=0.2` → 겹친 곳이 진해져 **밀집 지역**이 드러남

```python
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
```

| 인자 | 의미 |
|---|---|
| `s=population/100` | 점 크기 = 인구 (100으로 나눠야 화면을 안 덮음) |
| `c="median_house_value"` | 점 색 = 집값 |
| `cmap="jet"` | 파랑(낮음) → 빨강(높음) |
| `colorbar=True` | 색 범례 막대 |
| `sharex=False` | pandas x축 라벨 버그 회피 |

→ 집값은 **해안·인구 밀집 지역**에 좌우됨 → "도시 중심과의 거리" 특성 아이디어 → 뒤의 `ClusterSimilarity`

### 3-2. 상관관계

```python
corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

```
median_income         0.688380
total_rooms           0.137455
housing_median_age    0.102175
households            0.071426
total_bedrooms        0.054635
population           -0.020153
longitude            -0.050859
latitude             -0.139584
```

- 피어슨 상관계수 (−1~1). `numeric_only=True`로 텍스트 열 제외
- 소득 0.69로 압도적
- ⚠️ 상관계수는 **선형 관계만** 잡는다

```python
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)
```

- `scatter_matrix` → 모든 쌍의 산점도 (4×4). 대각선은 히스토그램
- 소득-집값 산점도에서 **$500k 수평선**(상한 흔적), $450k·$350k·$280k 부근에도 희미한 선 → 이런 구역을 제거하는 것도 고려

### 3-3. 특성 조합 실험

```python
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
```

```
median_income         0.688380
rooms_per_house       0.143663   ← total_rooms(0.137)보다 높음
...
people_per_house     -0.038224
bedrooms_ratio       -0.256397   ← total_bedrooms(0.055)보다 훨씬 강함
```

- `total_rooms`는 **구역 전체** 방 수 → 구역 크기에 좌우. 가구 수로 나눠야 집 한 채 크기
- `bedrooms_ratio` −0.26 → 침실 비율이 낮을수록(거실·서재가 많을수록) 비쌈
- **도메인 지식으로 만든 비율 특성**이 원래 특성보다 나을 수 있다

---

## 4. 데이터 준비 (2.5)

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

- **입력 X / 정답 y 분리**
- `drop()`은 새 DataFrame 반환 → 원본 유지. 3-3에서 추가한 비율 열은 여기서 사라짐 (뒤에서 파이프라인이 다시 만듦)

### 4-1. 결측치 처리

```python
null_rows_idx = housing.isnull().any(axis=1)   # 결측이 하나라도 있는 행 마스크
housing.loc[null_rows_idx].head()

housing_option1 = housing.copy()
housing_option1.dropna(subset=["total_bedrooms"], inplace=True)  # 옵션 1: 행 삭제

housing_option2 = housing.copy()
housing_option2.drop("total_bedrooms", axis=1, inplace=True)     # 옵션 2: 열 삭제

housing_option3 = housing.copy()
median = housing["total_bedrooms"].median()
housing_option3["total_bedrooms"] = housing_option3["total_bedrooms"].fillna(median)  # 옵션 3: 대체
```

| 옵션 | 장점 | 단점 |
|---|---|---|
| 1. 행 삭제 | 간단 | 데이터 감소 |
| 2. 열 삭제 | 간단 | 정보 손실 |
| 3. **대체** | 데이터 보존 | 진짜 값은 아님 |

옵션 3을 직접 하면 훈련 세트 중간값을 **저장해 뒀다가** 테스트·새 데이터에도 같은 값으로 채워야 함 → `SimpleImputer`

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])   # 수치형만
imputer.fit(housing_num)
imputer.statistics_                  # 학습된 각 열의 중간값
housing_num.median().values          # 직접 계산한 값과 동일

X = imputer.transform(housing_num)                  # NumPy 배열 반환
imputer.feature_names_in_                           # 학습 때 본 열 이름
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)  # 다시 DataFrame으로
```

- 텍스트 열은 중간값이 없으므로 수치형만
- 지금은 결측이 `total_bedrooms`뿐이지만 미래 데이터는 모르니 **모든 열**에 적용
- 변환기 출력은 기본 **NumPy 배열** → 열 이름·인덱스가 사라짐
- `set_config(transform_output="pandas")` 쓰면 DataFrame으로 바로 받음

#### 📌 사이킷런 API 설계 원칙 (이 장 전체의 핵심)

| 개념 | 메서드 | 예 |
|---|---|---|
| **추정기**(estimator) | `fit()` — 데이터에서 학습 | `SimpleImputer`, `LinearRegression` |
| **변환기**(transformer) | `transform()`, `fit_transform()` | `SimpleImputer`, `StandardScaler` |
| **예측기**(predictor) | `predict()`, `score()` | `LinearRegression` |
| 하이퍼파라미터 | 생성자 인자 | `strategy="median"` |
| 학습된 파라미터 | 이름이 **밑줄(_)로 끝남** | `statistics_`, `mean_` |

#### 이상치 탐지

```python
from sklearn.ensemble import IsolationForest

isolation_forest = IsolationForest(random_state=42)
outlier_pred = isolation_forest.fit_predict(X)   # 1 = 정상, -1 = 이상치

#housing = housing.iloc[outlier_pred == 1]
#housing_labels = housing_labels.iloc[outlier_pred == 1]
```

- 무작위 트리로 샘플을 고립 → 쉽게 고립되면 이상치
- 이 장에서는 실제로 지우지 않음. 지운다면 **X와 y를 같은 마스크로 함께**

### 4-2. 범주형 특성

```python
housing_cat = housing[["ocean_proximity"]]   # [[ ]] → DataFrame(2D). 변환기는 2D 입력 필요
```

#### OrdinalEncoder

```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:8]          # [[3.], [0.], [1.], [1.], [4.], [1.], [0.], [3.]]
ordinal_encoder.categories_
# [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])]
```

- 알파벳 순서로 0~4
- 문제: 모델이 **0과 1이 0과 4보다 비슷하다**고 오해. 실제론 0(<1H OCEAN)과 4(NEAR OCEAN)가 더 비슷
- 순서가 있는 범주(나쁨/보통/좋음)에만 적합

#### OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot               # <16512x5 sparse matrix ...>
housing_cat_1hot.toarray()     # 밀집 배열로

cat_encoder = OneHotEncoder(sparse_output=False)   # 처음부터 밀집 배열
```

- 범주마다 0/1 열 → 16,512 × 5
- 행마다 1이 하나뿐 → **희소 행렬**(0 아닌 값 위치만 저장)로 메모리 절약

#### ⚠️ `pd.get_dummies` vs `OneHotEncoder`

```python
df_test = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
pd.get_dummies(df_test)          # 열 2개만: INLAND, NEAR BAY
cat_encoder.transform(df_test)   # 학습한 5개 열 그대로
```

- `get_dummies`는 **지금 데이터에 있는 범주만** 열로 만듦 → 훈련 때와 열 개수가 달라져 모델이 깨짐
- `OneHotEncoder`는 학습한 범주를 기억 → 항상 같은 열 구조

```python
df_test_unknown = pd.DataFrame({"ocean_proximity": ["<2H OCEAN", "ISLAND"]})
cat_encoder.handle_unknown = "ignore"
cat_encoder.transform(df_test_unknown)
# [[0., 0., 0., 0., 0.],   ← 처음 보는 범주는 전부 0
#  [0., 0., 1., 0., 0.]]

cat_encoder.feature_names_in_       # ['ocean_proximity']
cat_encoder.get_feature_names_out() # ['ocean_proximity_<1H OCEAN', ..., 'ocean_proximity_NEAR OCEAN']
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)
```

### 4-3. 특성 스케일링과 변환

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)
```

| 스케일러 | 공식 | 특징 |
|---|---|---|
| `MinMaxScaler` | (x − min) / (max − min) → 지정 범위 | 이상치 하나가 스케일을 망칠 수 있음. 신경망은 −1~1 선호 |
| `StandardScaler` | (x − 평균) / 표준편차 | 범위 제한 없음, 이상치에 덜 민감 |

- 방 개수 6~39,320 vs 소득 0~15 → 범위가 큰 특성에 알고리즘이 끌려감
- ⚠️ **훈련 세트로만 `fit`**, 테스트 세트는 `transform`만

#### 꼬리가 긴 분포 → 로그

```python
fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
housing["population"].hist(ax=axs[0], bins=50)
housing["population"].apply(np.log).hist(ax=axs[1], bins=50)
```

스케일러는 분포 **모양**은 못 바꿈 → 스케일링 **전에** 로그로 종 모양에 가깝게

#### 백분위 버킷화

```python
percentiles = [np.percentile(housing["median_income"], p) for p in range(1, 100)]
flattened_median_income = pd.cut(housing["median_income"],
                                 bins=[-np.inf] + percentiles + [np.inf],
                                 labels=range(1, 100 + 1))
```

1~99 백분위를 경계로 100구간 → 각 1%씩 → **균등 분포**

#### 다봉 분포 → RBF 유사도

```python
from sklearn.metrics.pairwise import rbf_kernel
age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
```

- **exp(−γ · (x − 35)²)** → x = 35면 1, 멀어질수록 0
- "35년과 얼마나 가까운가" 특성 → 선형 모델도 특정 연식대 효과를 잡을 수 있음
- `gamma`가 클수록 종 모양이 좁아짐

#### 타깃 변환

```python
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]   # 새 데이터라고 가정

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
```

- y를 변환했으면 예측도 변환된 스케일 → `inverse_transform`으로 달러로 복원
- `to_frame()` → Series를 1열 DataFrame으로 (2D 요구)

자동으로:

```python
from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)    # 이미 원래 단위
```

### 4-4. 사용자 정의 변환기

#### `FunctionTransformer` — 학습이 필요 없는 변환

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing[["population"]])

rbf_transformer = FunctionTransformer(rbf_kernel,
                                      kw_args=dict(Y=[[35.]], gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])

sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(rbf_kernel,
                                     kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(housing[["latitude", "longitude"]])   # 샌프란시스코와의 근접도

ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))   # [[0.5], [0.75]]
```

- `kw_args` → 함수에 추가로 넘길 인자
- `X[:, [0]]`(리스트 인덱싱) → 2D 유지. `X[:, 0]`은 1D

#### 학습이 필요한 변환기 → 클래스로

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        assert self.n_features_in_ == X.shape[1]
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_
```

| 규칙 | 이유 |
|---|---|
| `BaseEstimator` 상속 | `get_params()`·`set_params()` 자동 → 그리드 서치가 이걸로 하이퍼파라미터를 바꿈 |
| `TransformerMixin` 상속 | `fit_transform()` 공짜 |
| `__init__`에 `*args`·`**kwargs` 금지 | `get_params()`가 생성자 시그니처로 하이퍼파라미터 목록을 만듦 |
| `__init__`은 **저장만** | 검증·계산 금지, 같은 이름으로 저장 |
| `fit(X, y=None)` | 안 써도 `y`를 받아야 파이프라인에서 동작 |
| 학습 결과는 `이름_` | `check_is_fitted`가 밑줄로 학습 여부 판단 |
| `n_features_in_` | `transform` 때 열 개수 검증 |
| `return self` | 메서드 체이닝 `fit(...).transform(...)` |

#### `ClusterSimilarity` — 이 장의 핵심 변환기

```python
from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
```

- **fit**: KMeans로 위경도를 `n_clusters`개 군집화. 중심은 `kmeans_.cluster_centers_`
- `sample_weight`: 샘플 가중치. 집값을 주면 비싼 지역 쪽으로 중심이 끌려감
- **transform**: 각 구역이 **각 군집 중심과 얼마나 가까운지**(RBF) → (샘플 수 × n_clusters)
- **get_feature_names_out**: 출력 열 이름 → 파이프라인 전체의 특성 이름 추적 가능

```python
cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(housing[["latitude", "longitude"]])
similarities[:3].round(2)
# [[0.  , 0.14, 0.  , 0.  , 0.  , 0.08, 0.  , 0.99, 0.  , 0.6 ],
#  [0.63, 0.  , 0.99, 0.  , 0.  , 0.  , 0.04, 0.  , 0.11, 0.  ], ...]
```

첫 구역은 7번 군집 중심과 거의 같은 위치(0.99). 3-1의 "도시 중심과의 거리" 직관을 자동화한 특성.

### 4-5. 변환 파이프라인

#### `Pipeline` 기본

```python
from sklearn.pipeline import Pipeline, make_pipeline

num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())  # 이름 자동
```

- `(이름, 추정기)` 리스트. 이름에 `__` 금지
- 마지막 단계 빼고는 모두 **변환기**
- `fit` → 각 단계 `fit_transform` 후 다음 단계로, 마지막만 `fit`
- 파이프라인은 **마지막 단계의 메서드**를 노출 (스케일러면 `transform`, 회귀 모델이면 `predict`)
- `make_pipeline` 이름 = 클래스명 소문자 (`simpleimputer`, `standardscaler`)

```python
from sklearn import set_config
set_config(display='diagram')
num_pipeline                    # 주피터에서 다이어그램 표시

housing_num_prepared = num_pipeline.fit_transform(housing_num)
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_num.index)

num_pipeline.steps                        # [(이름, 추정기), ...]
num_pipeline[1]                           # 두 번째 단계
num_pipeline[:-1]                         # 마지막 뺀 서브 파이프라인
num_pipeline.named_steps["simpleimputer"] # 이름으로 접근
num_pipeline.set_params(simpleimputer__strategy="median")  # 단계__파라미터
```

> 💡 `단계이름__파라미터` 문법 → 뒤의 그리드 서치가 이걸로 파이프라인 내부 하이퍼파라미터를 바꾼다

#### `ColumnTransformer` — 열마다 다른 처리

```python
from sklearn.compose import ColumnTransformer

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
```

- `(이름, 변환기, 열 목록)` 리스트. 결과는 옆으로 이어 붙임(hstack)
- 범주형은 중간값이 없으니 `most_frequent`

열 이름 나열 없이 타입으로 선택:

```python
from sklearn.compose import make_column_selector, make_column_transformer

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)
housing_prepared = preprocessing.fit_transform(housing)
```

#### ⭐ 최종 전처리 파이프라인

```python
def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
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
    remainder=default_num_pipeline)  # one column remaining: housing_median_age
```

- `column_ratio` — 3-3에서 손으로 만든 비율 특성을 함수로
- `ratio_name` — `FunctionTransformer`는 출력 이름을 모르므로 알려줌. 시그니처는 `(변환기, 입력이름들)`
- `ratio_pipeline()`을 **함수로** 만든 이유 → 객체 하나를 세 군데 쓰면 학습 상태를 공유해 꼬임. 호출마다 새 객체
- 순서: 결측 대체 → 나누기 → 표준화 (`total_bedrooms` 결측을 나누기 전에 채움)
- `feature_names_out="one-to-one"` → 입력 이름을 그대로 출력 이름으로

| 이름 | 입력 열 | 출력 | 의미 |
|---|---|---|---|
| bedrooms | 침실, 방 | 1 | 침실 비율 |
| rooms_per_house | 방, 가구 | 1 | 집당 방 수 |
| people_per_house | 인구, 가구 | 1 | 집당 인원 |
| log | 5개 | 5 | 로그 → 표준화 |
| geo | 위도, 경도 | 10 | 군집 중심 유사도 |
| cat | ocean_proximity | 5 | 원-핫 |
| remainder | `housing_median_age` | 1 | 기본 처리 |

- **같은 열을 여러 변환기에서 써도 됨** (`total_rooms`는 비율 2개 + 로그)
- `remainder` — 지정 안 된 열 처리. 기본 `"drop"`, `"passthrough"`도 가능

```python
housing_prepared = preprocessing.fit_transform(housing)
housing_prepared.shape          # (16512, 24)
preprocessing.get_feature_names_out()
# ['bedrooms__ratio', 'rooms_per_house__ratio', 'people_per_house__ratio',
#  'log__total_bedrooms', ..., 'log__median_income',
#  'geo__Cluster 0 similarity', ..., 'geo__Cluster 9 similarity',
#  'cat__ocean_proximity_<1H OCEAN', ..., 'cat__ocean_proximity_NEAR OCEAN',
#  'remainder__housing_median_age']
```

3 + 5 + 10 + 5 + 1 = **24개 특성**, 이름은 `변환기이름__원래이름`. 이 객체 하나로 원본 DataFrame → 모델 입력.

---

## 5. 모델 선택과 훈련 (2.6)

### 5-1. 선형 회귀

```python
from sklearn.linear_model import LinearRegression

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)
housing_predictions[:5].round(-2)   # -2 = 100의 자리로 반올림
housing_labels.iloc[:5].values

error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))
# -46.3%, -23.0%, 33.4%, -4.9%, -8.5%
```

전처리 + 모델을 **파이프라인 하나**로 → 원본 DataFrame을 그대로 `fit`

```python
from sklearn.metrics import root_mean_squared_error

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
lin_rmse      # 68972.89
```

집값 대부분이 $120k~$265k인데 오차 $69k → **과소적합** (특성 정보 부족 or 모델이 너무 단순)

### 5-2. 결정 트리

```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)
tree_rmse     # 0.0
```

0 = 완벽이 아니라 **데이터를 외움** → 과대적합. 테스트 세트는 아직 못 쓰니 다른 검증 방법이 필요.

### 5-3. 교차 검증

```python
from sklearn.model_selection import cross_val_score

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(tree_rmses).describe()
# mean 66573.73, std 1103.40, min 64607.90, max 68532.21

lin_rmses = -cross_val_score(lin_reg, housing, housing_labels,
                              scoring="neg_root_mean_squared_error", cv=10)
pd.Series(lin_rmses).describe()      # mean 70003.40, std 4182.19
```

**k-폴드 교차 검증**: 훈련 세트를 10폴드로 → 9개로 학습, 1개로 평가 → 10번 반복 → 점수 10개

- 사이킷런 `scoring`은 "클수록 좋다" 규칙 → RMSE를 음수로 돌려줌 → 앞에 `-`
- `std` = 추정의 안정성. 결정 트리 66,574 ± 1,103 → 선형 회귀보다 아주 조금 나을 뿐

### 5-4. 랜덤 포레스트

```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))
forest_rmses = -cross_val_score(forest_reg, housing, housing_labels,
                                scoring="neg_root_mean_squared_error", cv=10)
pd.Series(forest_rmses).describe()   # mean 47038.09, std 1021.49

forest_reg.fit(housing, housing_labels)
housing_predictions = forest_reg.predict(housing)
forest_rmse = root_mean_squared_error(housing_labels, housing_predictions)
forest_rmse   # 17551.21
```

- 무작위 샘플·특성으로 트리 여러 개 → 예측 **평균** (앙상블, 7장)
- 훈련 17,551 vs 검증 47,038 → **여전히 과대적합** → 규제·데이터 추가·다른 모델

| 모델 | 훈련 RMSE | CV RMSE | 진단 |
|---|---|---|---|
| 선형 회귀 | 68,973 | 70,003 | 과소적합 |
| 결정 트리 | 0 | 66,574 | 심한 과대적합 |
| 랜덤 포레스트 | 17,551 | **47,038** | 가장 좋음, 다소 과대적합 |

> 💡 이 단계 목표는 하이퍼파라미터를 오래 만지는 게 아니라 **유망한 모델 2~5개 추리기**

---

## 6. 모델 미세 튜닝 (2.7)

### 6-1. 그리드 서치

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

파라미터 이름 해석:

```
preprocessing__geo__n_clusters
  full_pipeline → "preprocessing"(ColumnTransformer)
               → "geo"(ClusterSimilarity)
               → n_clusters
```

- **전처리 하이퍼파라미터도 모델과 함께 튜닝** → 파이프라인의 큰 장점
- `max_features` = 랜덤 포레스트 각 분할에서 고려할 특성 수
- 조합 3×3 + 2×3 = 15, `cv=3` → **45번 학습**

```python
print(str(full_pipeline.get_params().keys())[:1000] + "...")  # 튜닝 가능한 파라미터 목록
grid_search.best_params_
# {'preprocessing__geo__n_clusters': 15, 'random_forest__max_features': 6}
grid_search.best_estimator_     # 최적 조합으로 전체 훈련 세트에 재학습된 파이프라인
```

- `refit=True`(기본) → 최적 조합으로 **전체 훈련 세트 재학습**
- `n_clusters=15`가 탐색 범위 **최댓값** → 범위를 넓혀 다시 탐색해야 함

```python
cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)
cv_res.head()
```

| n_clusters | max_features | split0 | split1 | split2 | mean_test_rmse |
|---|---|---|---|---|---|
| 15 | 6 | 42725 | 43708 | 44335 | **43590** |
| 15 | 8 | 43486 | 43820 | 44900 | 44069 |
| 10 | 4 | 43798 | 44036 | 44961 | 44265 |

### 6-2. 랜덤 서치

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50),
                  'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3,
    scoring='neg_root_mean_squared_error', random_state=42)

rnd_search.fit(housing, housing_labels)
```

- 값 목록 대신 **확률 분포**. `randint(3, 50)` → 3~49 균등
- `n_iter=10` × 3폴드 = 30번 학습
- 그리드 대비 장점: 하이퍼파라미터마다 서로 다른 값을 많이 시도 / 효과 없는 파라미터에 시간 낭비 적음 / 반복 횟수로 예산 조절

| n_clusters | max_features | split0 | split1 | split2 | mean_test_rmse |
|---|---|---|---|---|---|
| 45 | 9 | 41342 | 42242 | 43057 | **42214** |
| 32 | 7 | 41825 | 42275 | 43241 | 42447 |

→ 더 적은 학습(30 vs 45)으로 더 좋은 결과

```python
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
```

많은 조합을 적은 자원으로 평가 → 좋은 조합만 남기며 자원 증가. 실험 기능이라 `enable_halving_search_cv` 먼저 임포트.

#### 분포 고르는 법

| 분포 | 언제 |
|---|---|
| `randint(a, b+1)` | a~b 정수, 모든 값이 비슷하게 그럴듯할 때 |
| `uniform(a, b)` | 위의 실수 버전 |
| `geom(1/scale)` | 이산값, 대략 스케일은 알고 작은 값 선호 |
| `expon(scale)` | 위의 연속 버전 |
| `loguniform(a, b)` | **스케일을 전혀 모를 때** (0.01~0.1과 10~100을 같은 비율로) |

### 6-3. 최상 모델 분석

```python
final_model = rnd_search.best_estimator_  # includes preprocessing
feature_importances = final_model["random_forest"].feature_importances_
feature_importances.round(2)

sorted(zip(feature_importances,
           final_model["preprocessing"].get_feature_names_out()),
       reverse=True)
```

```
(0.186, 'log__median_income'),
(0.073, 'cat__ocean_proximity_INLAND'),
(0.066, 'bedrooms__ratio'),
(0.054, 'rooms_per_house__ratio'),
(0.046, 'people_per_house__ratio'),
(0.042, 'geo__Cluster 30 similarity'), ...
```

- `final_model["random_forest"]` → 이름으로 단계 꺼내기
- `zip(중요도, 이름)` → `sorted(reverse=True)` (튜플은 첫 원소 기준 정렬)
- 소득 압도적 → 내륙 여부 → 비율 특성. `ocean_proximity` 중 INLAND만 중요 → 나머지 범주 제거 고려

### 6-4. 테스트 세트로 평가

```python
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

final_predictions = final_model.predict(X_test)

final_rmse = root_mean_squared_error(y_test, final_predictions)
print(final_rmse)    # 41445.53
```

테스트 세트는 **`predict`만**. 파이프라인이 훈련 세트에서 배운 중간값·평균·군집 중심을 그대로 적용.

```python
from scipy.stats import bootstrap

def rmse(squared_errors):
    return np.sqrt(np.mean(squared_errors))

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
boot_result = bootstrap([squared_errors], rmse, confidence_level=confidence,
                        random_state=42)
rmse_lower, rmse_upper = boot_result.confidence_interval
print(f"95% CI for RMSE: ({rmse_lower:.4f}, {rmse_upper:.4f})")
# 95% CI for RMSE: (39520.9572, 43701.7681)
```

- 숫자 하나로는 신뢰도를 모름 → **신뢰구간**
- **부트스트랩**: 제곱 오차에서 복원 추출로 재표본 수천 번 → 매번 RMSE → 분포의 2.5%·97.5% 지점
- `bootstrap` 첫 인자는 **튜플/리스트로 감싼** 데이터 → `[squared_errors]`

> ⚠️ 이 결과를 보고 테스트 점수를 올리려고 다시 튜닝하면 안 된다. 테스트 세트에 맞춰진 모델이 되어 실제 성능을 낙관적으로 추정하게 된다.

---

## 7. 론칭·모니터링·유지 보수 (2.8)

```python
import joblib

joblib.dump(final_model, "my_california_housing_model.pkl")
```

**전처리 + 모델 파이프라인 전체**를 파일 하나로. joblib은 NumPy 배열이 많은 객체를 pickle보다 효율적으로 저장.

운영 환경에서 불러오기:

```python
import joblib

from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

#class ClusterSimilarity(BaseEstimator, TransformerMixin):
#    [...]

final_model_reloaded = joblib.load("my_california_housing_model.pkl")

new_data = housing.iloc[:5]  # pretend these are new districts
predictions = final_model_reloaded.predict(new_data)
```

> ⚠️ pickle(joblib) 파일에는 함수·클래스 **코드가 저장되지 않는다**. "`__main__.column_ratio`를 참조하라" 같은 **이름**만 저장된다. 불러오는 쪽에서 `column_ratio`, `ratio_name`, `ClusterSimilarity`를 **똑같이 정의하거나 임포트**해야 한다 (안 하면 `AttributeError`). 실무에선 사용자 정의 변환기를 별도 `.py` 모듈로 두고 양쪽에서 임포트.

배포 이후:
- **배포 방식**: 웹 앱 내부 로드 / 별도 REST API 서비스 / 클라우드(Vertex AI 등)
- **모니터링**: 세상이 바뀌면 성능이 서서히 떨어짐(**모델 부패**, 데이터 드리프트) → 실시간 성능 측정, 입력 품질 감시, 알림
- **유지 보수**: 새 데이터 수집·재학습 자동화, 모델·데이터 버전 관리, 이전 모델로 롤백 준비

---

## 8. 연습문제 (2.9)

### 8-1. SVR 그리드 서치

```python
from sklearn.model_selection import GridSearchCV
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

svr_grid_search_rmse = -grid_search.best_score_   # 70059.93
grid_search.best_params_    # {'svr__C': 10000.0, 'svr__kernel': 'linear'}
```

- SVM은 데이터가 커지면 매우 느림 → 5,000개만
- `linear`엔 `gamma`가 무의미 → 딕셔너리를 나눠 쓸데없는 조합 회피
- C=10000이 범위 경계 근처 → 더 큰 값 시도 필요

### 8-2. 랜덤 서치로 교체

```python
from sklearn.model_selection import RandomizedSearchCV
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

svr_rnd_search_rmse = -rnd_search.best_score_    # 56152.05
```

- `C` → 스케일을 모르니 `loguniform(20, 200000)`
- `gamma` → 대략 1 근처라 보고 `expon(scale=1)` (약 80%가 0.1~2.3)
- RBF 커널 선택, RMSE 70,060 → **56,152**

```python
s = expon(scale=1).rvs(100_000, random_state=42)
((s > 0.105) & (s < 2.29)).sum() / 100_000      # 약 0.8
```

### 8-3. `SelectFromModel` 특성 선택

```python
from sklearn.feature_selection import SelectFromModel

selector_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('selector', SelectFromModel(RandomForestRegressor(random_state=42),
                                 threshold=0.005)),  # min feature importance
    ('svr', SVR(C=rnd_search.best_params_["svr__C"],
                gamma=rnd_search.best_params_["svr__gamma"],
                kernel=rnd_search.best_params_["svr__kernel"])),
])

selector_rmses = -cross_val_score(selector_pipeline,
                                  housing.iloc[:5000],
                                  housing_labels.iloc[:5000],
                                  scoring="neg_root_mean_squared_error",
                                  cv=3)
pd.Series(selector_rmses).describe()
```

- 내부 랜덤 포레스트의 `feature_importances_`가 0.005 미만인 특성을 버리는 **변환기**
- 결과: 효과 없음 → 임계값도 튜닝해 볼 만함

### 8-4. 회귀 모델 예측을 특성으로 쓰는 변환기

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

        X_target = X_df[self.target_features]
        X_rest = X_df.drop(columns=self.target_features)
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

- 문제는 KNN으로 "주변 구역의 소득"을 예측해 특성으로 쓰기. 정답은 **어떤 회귀 모델이든** 받도록 일반화
- `MetaEstimatorMixin` → 다른 추정기를 감싸는 메타 추정기 표시
- **`clone(self.regressor)`** → 원본(하이퍼파라미터)을 건드리지 않고 복제본을 학습해 `regressor_`에 저장
- `target_features`(소득) = 예측 대상, 나머지(위경도) = 입력

```python
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_transformer = FeatureFromRegressor(knn_reg, ["median_income"])
geo_features = housing[["latitude", "longitude", "median_income"]]
knn_transformer.fit_transform(geo_features, housing_labels)
knn_transformer.get_feature_names_out()
# ['kneighborsregressor_prediction_median_income']
```

기존 `geo` 단계를 교체:

```python
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

결과: 훨씬 나쁨 → 군집 유사도 특성이 더 좋다

### 8-5. 전처리 옵션까지 랜덤 서치

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

`preprocessing__geo__regressor__n_neighbors` → 파이프라인 → ColumnTransformer → FeatureFromRegressor → KNN → n_neighbors, **4단계 깊이**. `BaseEstimator` 상속 덕분. 그래도 군집 유사도를 못 이김.

### 8-6. `StandardScalerClone` 완전판

```python
from sklearn.utils.validation import check_is_fitted, validate_data

class StandardScalerClone(TransformerMixin, BaseEstimator):
    def __init__(self, with_mean=True):
        self.with_mean = with_mean

    def fit(self, X, y=None):
        X = validate_data(self, X, ensure_2d=True)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
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

4-4의 간단 버전과 차이:
- **`validate_data`** — `fit`에서 `n_features_in_`, (DataFrame이면) `feature_names_in_`을 **자동 설정**. `transform`에선 `reset=False`로 열 개수·이름 일치 검사
- **`inverse_transform`** — 역순: 표준편차 곱하고 평균 더하기
- **`get_feature_names_out`** — 학습 때 이름이 있으면 그것, 없으면 `x0, x1, …`

검증:

```python
from sklearn.utils.estimator_checks import check_estimator
check_estimator(StandardScalerClone())     # 사이킷런 API 규약 자동 검사

rng = np.random.default_rng(seed=42)
X = rng.random((1000, 3))

scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(X)
assert np.allclose(X_scaled, (X - X.mean(axis=0)) / X.std(axis=0))

scaler = StandardScalerClone(with_mean=False)
X_scaled_uncentered = scaler.fit_transform(X)
assert np.allclose(X_scaled_uncentered, X / X.std(axis=0))

scaler = StandardScalerClone()
X_back = scaler.inverse_transform(scaler.fit_transform(X))
assert np.allclose(X, X_back)

assert np.all(scaler.get_feature_names_out() == ["x0", "x1", "x2"])
assert np.all(scaler.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c"])

df = pd.DataFrame({"a": rng.random(100), "b": rng.random(100)})
scaler = StandardScalerClone()
X_scaled = scaler.fit_transform(df)
assert np.all(scaler.feature_names_in_ == ["a", "b"])
assert np.all(scaler.get_feature_names_out() == ["a", "b"])
```

- `check_estimator` → 수십 가지 규약 테스트를 한 번에. 파이프라인·그리드 서치에서 문제없이 동작함을 보증
- `np.allclose` → 부동소수점 오차 허용 비교 (`==`는 미세 오차로 실패 가능)

---

## 9. 전체 흐름 요약

```
load_housing_data()                         ← 2.3 데이터 수집
   ↓ pd.cut + train_test_split(stratify)    ← 2.3 테스트 세트 격리
strat_train_set.copy() → 시각화/상관관계      ← 2.4 탐색
   ↓
preprocessing = ColumnTransformer([...])    ← 2.5 비율·로그·군집·원핫·결측
   ↓
make_pipeline(preprocessing, 모델)           ← 2.6 cross_val_score로 비교
   ↓
RandomizedSearchCV(full_pipeline, ...)      ← 2.7 전처리 + 모델 동시 튜닝
   ↓ final_model.predict(X_test) + bootstrap ← 2.7 최종 평가 (단 한 번)
joblib.dump / joblib.load                   ← 2.8 배포
```

| 단계 | RMSE |
|---|---|
| 선형 회귀 (CV) | 70,003 |
| 결정 트리 (CV) | 66,574 |
| 랜덤 포레스트 (CV) | 47,038 |
| 그리드 서치 최적 (CV) | 43,590 |
| 랜덤 서치 최적 (CV) | 42,214 |
| **최종 테스트 세트** | **41,446** (95% CI 39,521 ~ 43,702) |
