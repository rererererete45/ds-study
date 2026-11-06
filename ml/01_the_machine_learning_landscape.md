# 1장. 한눈에 보는 머신러닝

> 교재: 『핸즈온 머신러닝 with 사이킷런, 파이토치』 — 노트북 `01_the_machine_learning_landscape.ipynb`
>
> 머신러닝이 **무엇이고, 왜 쓰고, 어떤 종류가 있고, 무엇이 어렵고, 어떻게 평가하는지** 큰 지도를 그리는 장. 코드는 짧지만 이후 모든 장에서 반복되는 `fit → predict` 흐름과 **과대적합·대표성·규제·검증** 개념이 여기서 처음 나온다.
>
> 실습 데이터: OECD 삶의 만족도(BLI 2020) + 세계은행 1인당 GDP(2020)를 국가 단위로 합친 표
>
> | 파일 | 행 수 | 설명 |
> |---|---|---|
> | `lifesat.csv` | 27개국 | 1인당 GDP **$23,500 ~ $62,500** 구간만 남긴 **일부 데이터** (본문 예제용) |
> | `lifesat_full.csv` | 36개국 | 구간 제한 없는 **전체 데이터** (대표성·과대적합 설명용) |
>
> | 컬럼 | 타입 | 설명 |
> |---|---|---|
> | `Country` | object | 국가명 |
> | `GDP per capita (USD)` | float | **특성** — 1인당 GDP (달러) |
> | `Life satisfaction` | float | **타깃** — 삶의 만족도 (0~10점) |

## 목차

1. [머신러닝이란? (1.1)](#1-머신러닝이란-11)
2. [왜 머신러닝을 사용하나요? (1.2)](#2-왜-머신러닝을-사용하나요-12)
3. [애플리케이션 사례 (1.3)](#3-애플리케이션-사례-13)
4. [머신러닝 시스템의 종류 (1.4)](#4-머신러닝-시스템의-종류-14)
5. [머신러닝의 주요 도전 과제 (1.5)](#5-머신러닝의-주요-도전-과제-15)
6. [테스트와 검증 (1.6)](#6-테스트와-검증-16)
7. [연습문제](#7-연습문제)
8. [전체 흐름 요약](#8-전체-흐름-요약)

> 📌 노트북 구성: **코드 예제 1-1**(선형 회귀) + **k-최근접 이웃 버전** → 나머지는 `lifesat.csv`를 만들고 본문 그림을 그리는 코드("Generating the data and figures — please skip"). 그림 코드도 1.4·1.5절 개념(모델 선택, 대표성, 과대적합, 규제)을 그대로 보여 주므로 해당 절에 나눠 붙였다. 1.6절은 노트북에 코드가 없어 **보충 코드**를 직접 만들어 붙였다.

---

## 0. 준비

```python
import sys
assert sys.version_info >= (3, 10)

from packaging.version import Version
import sklearn
assert Version(sklearn.__version__) >= Version("1.6.1")

import matplotlib.pyplot as plt
plt.rc('font', size=12)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=12)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

- `assert 조건` → 거짓이면 즉시 `AssertionError`. 파이썬 3.10+, 사이킷런 1.6.1+ 확인
- `Version()`으로 감싸야 `"1.10"`과 `"1.6"`을 **숫자 순서**로 비교한다 (문자열 비교면 `"1.10" < "1.6"`)
- `plt.rc(그룹, 옵션=값)` → 이후 모든 그래프의 기본 글꼴 크기 설정

---

## 1. 머신러닝이란? (1.1)

### 1-1. 정의

- **데이터에서 학습**하도록 컴퓨터를 프로그래밍하는 것 — 규칙을 사람이 직접 쓰지 않고, 데이터에서 규칙을 찾게 한다
- 공학적 정의(톰 미첼): 어떤 **작업 T**에 대한 **성능 P**가 **경험 E**로 인해 좋아지면, 그 프로그램은 E로부터 학습한다고 말한다

| 요소 | 스팸 필터 예시 |
|---|---|
| 작업 T | 새 메일이 스팸인지 판별 |
| 경험 E | 사용자가 표시한 스팸 메일 + 정상 메일(햄) = **훈련 세트** |
| 성능 P | 올바르게 분류한 비율 = **정확도** |

### 1-2. 용어

| 용어 | 뜻 |
|---|---|
| 훈련 세트 (training set) | 학습에 쓰는 샘플 모음 |
| 훈련 샘플 / 인스턴스 (sample, instance) | 훈련 세트의 한 행 |
| 모델 (model) | 예측을 실제로 하는 부분 (예: 신경망, 랜덤 포레스트) |
| 특성 (feature) | 입력 변수 (예: 1인당 GDP) |
| 레이블 / 타깃 (label, target) | 정답 (예: 삶의 만족도, 스팸 여부) |

> 💡 위키피디아를 통째로 내려받는다고 컴퓨터가 "학습"하는 것은 아니다. 데이터가 늘었을 뿐, **어떤 작업을 더 잘하게** 된 게 아니기 때문. → 학습 = 성능 향상.

---

## 2. 왜 머신러닝을 사용하나요? (1.2)

### 2-1. 전통적 프로그래밍 vs 머신러닝 (스팸 필터)

| | 전통적 방식 | 머신러닝 방식 |
|---|---|---|
| 방법 | "4U", "신용카드", "무료" 같은 패턴을 사람이 찾아 **규칙 목록** 작성 | 스팸·햄 예시를 주면 **어떤 단어·구절이 스팸을 잘 가르는지** 자동으로 학습 |
| 규칙 | 길고 복잡해짐 → 유지 보수 어려움 | 프로그램이 짧고 유지 보수 쉬움, 정확도도 높은 편 |
| 스패머가 "4U" → "For U"로 바꾸면 | 사람이 규칙을 계속 고쳐야 함 | 사용자가 표시한 새 스팸에서 **자동으로** 다시 학습 |

### 2-2. 머신러닝이 특히 유리한 문제

1. **규칙이 너무 길고 복잡한 문제** — 수많은 수작업 규칙을 하나의 모델로 대체
2. **전통적 알고리즘으로 풀 방법이 없는 문제** — 음성 인식, 이미지 인식 등
3. **환경이 계속 바뀌는 문제** — 새 데이터로 재훈련해 적응
4. **사람이 배우는 데 도움** — 학습된 모델을 들여다보면 몰랐던 상관관계·패턴을 발견 (**데이터 마이닝**)

> 💡 4번이 데이터 분석과 머신러닝이 만나는 지점. 예: 스팸 필터가 어떤 단어를 중요하게 보는지 살펴보면 스팸의 특징을 사람이 새로 알게 된다.

---

## 3. 애플리케이션 사례 (1.3)

노트북 코드는 없음. 작업 → 대표 기법 → 이 책에서 다루는 장으로 정리.

| 작업 사례 | 문제 유형 | 대표 기법 | 다루는 장 |
|---|---|---|---|
| 생산 라인 제품 이미지로 불량 자동 분류 | 이미지 분류 | CNN, 비전 트랜스포머 | 12, 16장 |
| 뇌 스캔에서 종양 탐지 | 시맨틱 분할 (픽셀 단위 분류) | CNN, 트랜스포머 | 12, 16장 |
| 뉴스 기사 자동 분류 / 포럼의 공격적 댓글 표시 | 텍스트 분류 (NLP) | RNN, 트랜스포머 | 14, 15장 |
| 긴 문서 자동 요약, 챗봇·개인 비서 | 텍스트 요약·생성, 자연어 이해 | 트랜스포머, LLM | 15장 |
| 여러 지표로 회사의 내년 매출 예측 | **회귀** | 선형·다항 회귀, SVM, 랜덤 포레스트, 신경망 (시계열이면 RNN·CNN·트랜스포머) | 4, 5, 6, 9, 13장 |
| 음성 명령에 반응하는 앱 | 음성 인식 | RNN, CNN, 트랜스포머 | 13~15장 |
| 신용카드 부정 거래 감지 | **이상치 탐지** | 가우스 혼합, 아이솔레이션 포레스트, 오토인코더 | 8, 18장 |
| 구매 이력으로 고객 세분화 → 그룹별 마케팅 | **군집** | k-평균, DBSCAN | 8장 |
| 고차원 데이터를 이해하기 쉬운 그래프로 | 데이터 시각화, **차원 축소** | PCA, t-SNE | 7장 |
| 구매 이력으로 관심 상품 추천 | 추천 시스템 | 신경망 | — |
| 게임 지능형 봇 | **강화 학습** | 정책·가치 학습 (예: 알파고) | 19장 |

> 💡 같은 "예측"이라도 출력이 **값이면 회귀, 범주면 분류**. 1장의 삶의 만족도 예측은 값(점수)을 예측하므로 회귀다.

---

## 4. 머신러닝 시스템의 종류 (1.4)

분류 기준은 세 가지이고, **서로 배타적이지 않아** 조합할 수 있다.

| 기준 | 질문 | 종류 |
|---|---|---|
| ① 훈련 지도 방식 | 훈련 중에 어떤 감독(정답)을 받나? | 지도 / 비지도 / 준지도 / 자기 지도 / 강화 |
| ② 배치 vs 온라인 | 데이터가 들어오는 대로 점진적으로 학습하나? | 배치 학습 / 온라인 학습 |
| ③ 사례 기반 vs 모델 기반 | 새 데이터를 아는 데이터와 비교하나, 패턴으로 모델을 만드나? | 사례 기반 / 모델 기반 |

예: 최신 스팸 필터 = 심층 신경망을 스팸·햄 예시로 실시간 학습 → **온라인 + 모델 기반 + 지도 학습**.

### 4-1. 훈련 지도 방식

| 방식 | 훈련 데이터 | 대표 작업 | 예 |
|---|---|---|---|
| **지도 학습** | 레이블 있음 | **분류**, **회귀** | 스팸 분류, 중고차 가격 예측 |
| **비지도 학습** | 레이블 없음 | 군집, 시각화·차원 축소, 이상치·특이치 탐지, 연관 규칙 학습 | 블로그 방문자 그룹 나누기, 장바구니 분석 |
| **준지도 학습** | 일부만 레이블 | 비지도 + 지도 조합 | 사진 앱: 얼굴 군집(비지도) 후 사람마다 이름 하나만 붙이면 전체 태깅 |
| **자기 지도 학습** | 레이블 없는 데이터로 **레이블을 스스로 생성** | 사전 훈련 후 다른 작업으로 **전이** | 이미지 일부를 가리고 원본 복원 학습 → 그 모델을 분류에 재사용 |
| **강화 학습** | 레이블 없음, **보상/벌점** | 정책(policy) 학습 | 보행 로봇, 알파고 |

- **회귀**: 특성으로 **수치**를 예측. 레이블 = 타깃 값 (1장 실습이 이 경우)
- **분류**: 특성으로 **범주**를 예측. 로지스틱 회귀처럼 이름은 회귀지만 분류에 쓰는 것도 있다(4장)
- **차원 축소**: 정보를 크게 잃지 않으면서 상관된 특성을 하나로 합치기(**특성 추출**). 지도 학습 앞단에 두면 속도·메모리 이득
- **이상치 탐지** vs **특이치 탐지**: 이상치는 학습한 정상 분포에서 벗어난 샘플을 찾고, 특이치는 **훈련 세트에 깨끗한(정상) 샘플만** 있다고 보고 새로운 패턴을 찾는다
- **강화 학습 용어**: 에이전트(agent)가 환경을 관찰 → 행동(action) → 보상(reward) → 시간에 따라 가장 큰 보상을 얻는 전략인 정책을 스스로 학습

> 💡 자기 지도 학습은 레이블을 스스로 만들기 때문에 엄밀히는 "레이블이 있는" 학습이다. 그래서 비지도 학습과 구분한다. LLM의 사전 훈련(다음 단어 맞히기)이 대표적인 자기 지도 학습.

### 4-2. 배치 학습 vs 온라인 학습

| | 배치 학습 (오프라인) | 온라인 학습 |
|---|---|---|
| 훈련 방식 | 가용 데이터 **전부**로 한 번에 훈련 → 배포 후 더 이상 학습 안 함 | 데이터를 **하나씩 또는 미니배치**로 넣어 점진적 학습 |
| 새 데이터 반영 | 전체 데이터로 **처음부터 재훈련** 후 교체 (자동화는 가능) | 들어오는 즉시 학습 → 빠른 적응 |
| 자원 | 시간·계산 자원 많이 필요 | 학습한 데이터는 버려도 됨 → 자원 절약 |
| 맞는 상황 | 데이터가 천천히 변함, 자원 충분 | 데이터가 빠르게 변함(주가), 자원 제한(스마트폰·로봇), 데이터가 너무 큼 |

- **모델 부패 / 데이터 드리프트**: 세상은 계속 변하는데 모델은 그대로 → 성능이 점점 떨어짐. 배치 학습이면 주기적으로 재훈련 필요
- **외부 메모리 학습 (out-of-core)**: 메모리에 안 들어가는 대용량 데이터를 미니배치로 쪼개 온라인 학습 기법으로 학습. 실제론 오프라인으로 돌기 때문에 "점진적 학습"이라는 말이 더 정확하다
- **학습률 (learning rate)**: 새 데이터에 얼마나 빠르게 적응할지
  - 크면 → 빨리 적응하지만 예전 데이터를 금방 잊음
  - 작으면 → 관성이 커져 느리게 학습하지만 노이즈·대표성 없는 샘플에 덜 민감
- **위험**: 나쁜 데이터(센서 고장, 검색 순위 조작 스팸)가 들어오면 성능이 서서히 떨어진다 → 성능 모니터링, 이상 감지 시 학습 중단·이전 상태로 되돌리기, 입력 데이터 이상치 탐지

### 4-3. 사례 기반 학습 vs 모델 기반 학습

**일반화**: 훈련 데이터에서 잘하는 것만으로는 부족하고, **처음 보는 샘플**에서 잘 예측해야 한다. 일반화 방식이 두 가지다.

| | 사례 기반 | 모델 기반 |
|---|---|---|
| 방식 | 훈련 샘플을 **기억** → 새 샘플과의 **유사도**로 예측 | 샘플로 **모델(수식)**을 만들고 그 모델로 예측 |
| 스팸 예 | 알려진 스팸과 공통 단어가 많으면 스팸 | 단어 빈도 → 스팸 확률 수식 |
| 1장 코드 | `KNeighborsRegressor` | `LinearRegression` |

#### 모델 기반 학습 절차 — "돈이 사람을 행복하게 만드는가?"

1. 데이터 연구 → 2. 모델 선택 → 3. 훈련 데이터로 학습(비용 함수 최소화 파라미터 찾기) → 4. 새 데이터에 추론(일반화 기대)

**(0) 데이터 만들기 (노트북 "please skip" 부분)**

```python
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

datapath = Path() / "datasets" / "lifesat"
datapath.mkdir(parents=True, exist_ok=True)

data_root = "https://github.com/ageron/data/raw/main/"
for filename in ("oecd_bli.csv", "gdp_per_capita.csv"):
    if not (datapath / filename).is_file():
        print("Downloading", filename)
        url = data_root + "lifesat/" + filename
        urllib.request.urlretrieve(url, datapath / filename)

oecd_bli = pd.read_csv(datapath / "oecd_bli.csv")
gdp_per_capita = pd.read_csv(datapath / "gdp_per_capita.csv")
```

- `Path() / "datasets" / "lifesat"` → `/` 연산자로 경로 이어 붙이기 (OS별 구분자 자동 처리)
- 파일이 이미 있으면 다운로드 건너뜀 → 여러 번 실행해도 1회만 받는다 (2장의 `load_housing_data()`와 같은 습관)

**GDP 데이터 → 2020년만, 국가를 인덱스로**

```python
gdp_year = 2020
gdppc_col = "GDP per capita (USD)"
lifesat_col = "Life satisfaction"

gdp_per_capita = gdp_per_capita[gdp_per_capita["Year"] == gdp_year]   # 2020년 행만
gdp_per_capita = gdp_per_capita.drop(["Code", "Year"], axis=1)       # 불필요한 열 삭제
gdp_per_capita.columns = ["Country", gdppc_col]                      # 열 이름 정리
gdp_per_capita.set_index("Country", inplace=True)                    # 국가명을 인덱스로
```

```
                             GDP per capita (USD)
Country
Afghanistan                           1978.961579
Africa Eastern and Southern           3387.594670   ← 국가가 아닌 "지역 집계"도 섞여 있음
...
```

**OECD BLI → 전체 인구 기준만, 지표를 열로 펼치기**

```python
oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
```

- `INEQUALITY` 열에는 남성·여성·고소득층 등 하위 집단 값도 있음 → `"TOT"`(전체)만 남김
- `pivot` → **긴 형식**(국가·지표·값 3열) → **넓은 형식**(국가 × 지표 행렬). 대기오염, 고용률, 삶의 만족도 등 지표가 각각 열이 된다

| 원래 (긴 형식) | | | → | pivot 후 (넓은 형식) | | |
|---|---|---|---|---|---|---|
| Country | Indicator | Value | | Country | Air pollution | Life satisfaction … |
| Australia | Air pollution | 5.0 | | Australia | 5.0 | 7.3 |
| Australia | Life satisfaction | 7.3 | | | | |

**두 데이터 병합 → 필요한 두 열만**

```python
full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                              left_index=True, right_index=True)
full_country_stats.sort_values(by=gdppc_col, inplace=True)
full_country_stats = full_country_stats[[gdppc_col, lifesat_col]]
```

- `left_index=True, right_index=True` → 양쪽 **인덱스(국가명)** 기준으로 조인. 기본 `how="inner"`라 양쪽에 다 있는 국가만 남음 → "Africa Eastern and Southern" 같은 지역 집계는 자연스럽게 빠진다
- GDP 순으로 정렬 → 36개국

**일부 구간만 잘라 `lifesat.csv` 저장**

```python
min_gdp = 23_500
max_gdp = 62_500

country_stats = full_country_stats[(full_country_stats[gdppc_col] >= min_gdp) &
                                   (full_country_stats[gdppc_col] <= max_gdp)]

country_stats.to_csv(datapath / "lifesat.csv")            # 27개국
full_country_stats.to_csv(datapath / "lifesat_full.csv")  # 36개국
```

- `23_500` → 숫자 사이 `_`는 자릿수 구분용 (파이썬 3.6+), 값은 23500
- 조건 여러 개는 `&`로 묶고 **각 조건을 괄호로** 감싸야 한다 (`and` 쓰면 에러)
- ⚠️ 저자가 일부러 **가난한 나라·부유한 나라를 뺀** 데이터를 만든 것. 5장에서 이게 "대표성 없는 데이터" 문제였음이 드러난다

**(1) 데이터 연구 — 산점도**

```python
country_stats.plot(kind='scatter', figsize=(5, 3), grid=True,
                   x=gdppc_col, y=lifesat_col)

position_text = {
    "Turkey": (29_500, 4.2),
    "Hungary": (28_000, 6.9),
    "France": (40_000, 5),
    "New Zealand": (28_000, 8.2),
    "Australia": (50_000, 5.5),
    "United States": (59_000, 5.3),
    "Denmark": (46_000, 8.5)
}

for country, pos_text in position_text.items():
    pos_data_x = country_stats[gdppc_col].loc[country]
    pos_data_y = country_stats[lifesat_col].loc[country]
    country = "US" if country == "United States" else country
    plt.annotate(country, xy=(pos_data_x, pos_data_y),
                 xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5,
                                 shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "ro")

plt.axis([min_gdp, max_gdp, 4, 9])
plt.show()
```

| 코드 | 의미 |
|---|---|
| `df.plot(kind='scatter', x=, y=)` | pandas에서 바로 산점도 |
| `.loc[country]` | 인덱스(국가명)로 값 꺼내기 |
| `plt.annotate(텍스트, xy=점, xytext=글자위치, arrowprops=...)` | 점에 화살표 달린 설명 붙이기 |
| `plt.plot(x, y, "ro")` | 빨간(r) 원(o) 마커로 강조 |
| `plt.axis([xmin, xmax, ymin, ymax])` | 축 범위 고정 |

강조한 7개국:

| 국가 | 1인당 GDP | 삶의 만족도 |
|---|---|---|
| Turkey | 28,385 | 5.5 |
| Hungary | 31,008 | 5.6 |
| France | 42,026 | 6.5 |
| New Zealand | 42,404 | 7.3 |
| Australia | 48,698 | 7.3 |
| Denmark | 55,938 | 7.6 |
| United States | 60,236 | 6.9 |

→ 노이즈는 있지만 **GDP가 오를수록 만족도도 대체로 선형으로 오르는** 경향.

**(2) 모델 선택 — 선형 모델**

$$\text{삶의 만족도} = \theta_0 + \theta_1 \times \text{1인당 GDP}$$

θ₀(절편), θ₁(기울기)이 **모델 파라미터**. 값에 따라 어떤 직선이든 표현할 수 있다.

```python
X = np.linspace(min_gdp, max_gdp, 1000)

w1, w2 = 4.2, 0
plt.plot(X, w1 + w2 * 1e-5 * X, "r")      # 빨강: 수평선
w1, w2 = 10, -9
plt.plot(X, w1 + w2 * 1e-5 * X, "g")      # 초록: 우하향
w1, w2 = 3, 8
plt.plot(X, w1 + w2 * 1e-5 * X, "b")      # 파랑: 우상향 → 데이터에 가장 가까움
```

- `np.linspace(시작, 끝, 개수)` → 그래프용으로 x축을 1000등분
- GDP 단위가 수만 달러라 기울기가 매우 작음 → `w2 * 1e-5`로 표시 (θ₁ = 8 × 10⁻⁵)
- `fr"$\theta_0 = {w1}$"` → `f`(변수 삽입) + `r`(백슬래시 그대로) 문자열. `$...$`는 matplotlib의 LaTeX 수식 표기. `{{-5}}`는 f-문자열 안에서 중괄호 자체를 쓰려고 두 번 씀

**어떤 직선이 가장 좋은가?** → 성능 측정 지표가 필요

| 함수 | 측정하는 것 | 목표 |
|---|---|---|
| **효용 함수** (적합도 함수) | 모델이 얼마나 **좋은지** | 최대화 |
| **비용 함수** | 모델이 얼마나 **나쁜지** | 최소화 |

선형 회귀는 보통 **예측과 실제값 사이 거리(오차)를 재는 비용 함수**를 최소화하는 θ를 찾는다 = **훈련**.

**(3) 훈련 — 코드 예제 1-1**

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 데이터 다운로드 및 준비
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# 데이터 시각화
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# 선형 모델 선택
model = LinearRegression()

# 모델 훈련
model.fit(X, y)

# 푸에르토리코에 대한 예측
X_new = [[33_442.8]]          # 2020년 푸에르토리코 1인당 GDP
print(model.predict(X_new))   # [[6.01610329]]
```

| 줄 | 의미 |
|---|---|
| `pd.read_csv(URL)` | URL을 바로 읽을 수 있음 |
| `lifesat[["GDP..."]]` | **이중 대괄호** → 1열짜리 DataFrame (2차원). 단일 대괄호면 Series(1차원) |
| `.values` | NumPy 배열로 변환 → `X.shape == (27, 1)` |
| `LinearRegression()` | 모델 객체 생성 (아직 학습 전) |
| `model.fit(X, y)` | 비용 함수(MSE)를 최소화하는 θ₀, θ₁을 찾음 |
| `model.predict(X_new)` | 학습한 θ로 새 샘플 예측 |
| `[[33_442.8]]` | 샘플 1개 × 특성 1개 → 반드시 **2차원** |

> ⚠️ 사이킷런의 `X`는 항상 **(샘플 수, 특성 수)** 2차원이어야 한다. 특성이 하나여도 `[[값]]` 또는 `X.reshape(-1, 1)` 형태. 1차원으로 넣으면 `Expected 2D array` 에러.

**학습된 파라미터 확인 (그림 코드)**

```python
from sklearn import linear_model

X_sample = country_stats[[gdppc_col]].values
y_sample = country_stats[[lifesat_col]].values

lin1 = linear_model.LinearRegression()
lin1.fit(X_sample, y_sample)

t0, t1 = lin1.intercept_[0], lin1.coef_[0][0]
print(f"θ0={t0:.2f}, θ1={t1:.2e}")
```

```
θ0=3.75, θ1=6.78e-05
```

- 학습된 값은 `intercept_`(θ₀), `coef_`(θ₁)에 저장. 이름 끝 `_` = **훈련으로 얻은 속성** (사이킷런 규칙)
- `y`가 2차원 `(27, 1)`이라 `intercept_`는 `(1,)`, `coef_`는 `(1, 1)` → `[0]`, `[0][0]`으로 스칼라 추출
- `:.2e` → 지수 표기 소수 둘째 자리

**(4) 추론 — 푸에르토리코 예측**

```python
puerto_rico_gdp_per_capita = gdp_per_capita[gdppc_col].loc["Puerto Rico"]
# 33442.8315702748   ← OECD 회원국이 아니라 삶의 만족도 데이터에 없음

puerto_rico_predicted_life_satisfaction = lin1.predict(
    [[puerto_rico_gdp_per_capita]])[0, 0]
# 6.016105434473317
```

손으로 검산: 3.749 + 6.779 × 10⁻⁵ × 33,442.8 = 3.749 + 2.267 = **6.016** ✔

그림에서는 `plt.plot([x, x], [4, 예측값], "r--")`로 x축에서 직선까지 빨간 점선을 그어 예측값을 표시한다.

#### 사례 기반으로 바꾸기 — k-최근접 이웃

`import` 한 줄과 모델 생성 한 줄만 바꾸면 된다.

```python
from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)
print(model.predict(X_new))   # [[5.73333333]]
```

GDP가 가장 가까운 3개국의 만족도 **평균**:

| 이웃 | 1인당 GDP | 삶의 만족도 | 푸에르토리코와 거리 |
|---|---|---|---|
| Poland | 32,238 | 6.1 | 1,205 |
| Portugal | 32,181 | 5.4 | 1,262 |
| Estonia | 35,638 | 5.7 | 2,196 |

(6.1 + 5.4 + 5.7) / 3 = **5.733** ✔ (`model.kneighbors(X_new)`로 직접 확인 가능)

| | 선형 회귀 (모델 기반) | 3-NN (사례 기반) |
|---|---|---|
| 예측값 | 6.016 | 5.733 |
| `fit`이 하는 일 | θ₀, θ₁ 계산 | 훈련 데이터를 **저장**만 함 |
| `predict`가 하는 일 | θ₀ + θ₁x 계산 | 가까운 3개 찾아 평균 |

> 💡 모델이 바뀌어도 `fit` / `predict` 인터페이스가 같다 → 사이킷런의 핵심 설계(**일관성**). 2장 이후 모델 교체가 쉬운 이유.

#### 모델 기반 학습 요약

1. 데이터를 분석한다
2. 모델을 선택한다
3. 훈련 데이터로 훈련한다 (비용 함수를 최소화하는 파라미터 탐색)
4. 새 데이터에 모델을 적용해 예측한다 (**추론**) — 이 모델이 잘 일반화되기를 기대

결과가 나쁘면 → 특성 추가(고용률, 건강, 대기오염 등), 더 좋은 훈련 데이터 확보, 더 강력한 모델(다항 회귀 등) 선택.

---

## 5. 머신러닝의 주요 도전 과제 (1.5)

문제는 크게 둘: **나쁜 데이터**, **나쁜 모델(알고리즘)**.

### 5-1. 충분하지 않은 양의 훈련 데이터

- 간단한 문제도 보통 수천 개, 이미지·음성 같은 복잡한 문제는 수백만 개 샘플이 필요 (사전 훈련 모델을 재사용하면 줄일 수 있음)
- **데이터의 불합리한 효과** (2001년 MS 연구, 2009년 피터 노르빅): 충분한 데이터가 있으면 아주 다른 알고리즘들도 비슷하게 잘 동작 → **알고리즘보다 데이터가 더 중요할 수 있다**
- 단, 중소 규모 데이터셋이 여전히 흔하고, 데이터를 더 모으는 건 비싸다 → 알고리즘을 버리면 안 됨

### 5-2. 대표성 없는 훈련 데이터

일반화하려면 훈련 데이터가 **일반화하려는 새 사례를 잘 대표**해야 한다. 노트북 그림 코드가 바로 이 상황을 재현한다.

**빠졌던 9개국 꺼내기**

```python
missing_data = full_country_stats[(full_country_stats[gdppc_col] < min_gdp) |
                                  (full_country_stats[gdppc_col] > max_gdp)]
```

- `|` = OR. 구간 **밖**의 나라

| 국가 | 1인당 GDP | 삶의 만족도 |
|---|---|---|
| South Africa | 11,466 | 4.7 |
| Colombia | 13,441 | 6.3 |
| Brazil | 14,064 | 6.4 |
| Mexico | 17,888 | 6.5 |
| Chile | 23,325 | 6.5 |
| Norway | 63,586 | 7.6 |
| Switzerland | 68,393 | 7.5 |
| Ireland | 89,689 | 7.0 |
| Luxembourg | 110,261 | 6.9 |

**일부 데이터 모델 vs 전체 데이터 모델**

```python
full_country_stats.plot(kind='scatter', figsize=(8, 3),
                        x=gdppc_col, y=lifesat_col, grid=True)

for country, pos_text in position_text_missing_countries.items():
    pos_data_x, pos_data_y = missing_data.loc[country]        # 한 행 → 두 값 언패킹
    plt.annotate(country, xy=(pos_data_x, pos_data_y), xytext=pos_text, fontsize=12,
                 arrowprops=dict(facecolor='black', width=0.5, shrink=0.08, headwidth=5))
    plt.plot(pos_data_x, pos_data_y, "rs")                    # 빨간 사각형

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1 * X, "b:")                              # 일부 데이터 모델 (점선)

lin_reg_full = linear_model.LinearRegression()
Xfull = np.c_[full_country_stats[gdppc_col]]
yfull = np.c_[full_country_stats[lifesat_col]]
lin_reg_full.fit(Xfull, yfull)

t0full, t1full = lin_reg_full.intercept_[0], lin_reg_full.coef_[0][0]
plt.plot(X, t0full + t1full * X, "k")                       # 전체 데이터 모델 (검은 실선)
```

- `np.c_[Series]` → 1차원을 **열 벡터 (n, 1)** 로 만드는 단축 표기. `[[col]].values`와 같은 효과
- `"b:"` = 파란 점선, `"k"` = 검은 실선, `"rs"` = 빨간 사각형

| 모델 | θ₀ | θ₁ | 훈련 R² | 푸에르토리코 예측 |
|---|---|---|---|---|
| 일부 데이터 (27개국) | 3.75 | 6.78 × 10⁻⁵ | 0.727 | 6.02 |
| 전체 데이터 (36개국) | 5.58 | 2.33 × 10⁻⁵ | 0.361 | 6.36 |

- 빠진 나라를 넣자 **기울기가 1/3 수준**으로 꺾임 → 선형 모델은 애초에 잘 맞지 않는다
- 아주 부유한 나라가 중간 소득 나라보다 꼭 행복하지 않고, 일부 가난한 나라는 더 행복해 보임
- 대표성 없는 데이터로 학습한 모델은 특히 **가난한 나라·부유한 나라**에서 정확한 예측을 하기 어렵다

| 원인 | 설명 |
|---|---|
| **샘플링 잡음** | 샘플이 작으면 **우연히** 대표성 없는 데이터가 됨 |
| **샘플링 편향** | 샘플이 커도 **추출 방법이 잘못**되면 대표성 없음 |

> 💡 샘플링 편향 고전 사례: 1936년 미국 대선 Literary Digest 여론조사. 1천만 명에게 우편 → 240만 응답, **랜던 57% 승리 예측 → 실제는 루스벨트 62% 당선**. ① 전화번호부·잡지 구독자 명단 등 부유층 위주 주소(공화당 지지 성향) ② 응답자 25% 미만 = **비응답 편향**(정치 관심 많은 특정 집단만 응답). 다른 예: 펑크 음악 영상 분류기를 만들려고 유튜브에서 "펑크 음악" 검색 → 인기 아티스트 위주로 편향됨.

### 5-3. 낮은 품질의 데이터

- 오류·이상치·잡음이 많으면 패턴을 찾기 어려움 → **데이터 정제에 시간을 쏟을 가치가 충분**
- 이상치가 명확하면 무시하거나 수동으로 수정
- 일부 특성이 빠졌다면(예: 고객 5%가 나이 미기입) → 특성 자체를 무시 / 해당 샘플 무시 / 중간값 등으로 채우기 / 채운 모델과 안 채운 모델 둘 다 훈련 → 2장 `SimpleImputer`로 이어짐

### 5-4. 관련 없는 특성

**Garbage in, garbage out.** 좋은 특성을 찾는 것 = **특성 공학**

| 작업 | 의미 |
|---|---|
| 특성 선택 | 가지고 있는 특성 중 가장 유용한 것 고르기 |
| 특성 추출 | 특성을 결합해 더 유용한 특성 만들기 (차원 축소가 도움) |
| 새 특성 생성 | 새 데이터 수집 |

### 5-5. 훈련 데이터 과대적합

**과대적합 (overfitting)**: 훈련 데이터에는 너무 잘 맞지만 **일반성이 떨어지는** 것. 사람이 한 번 바가지 쓴 경험으로 "그 나라 택시 기사는 다 도둑"이라고 일반화하는 것과 같다.

**① 고차 다항 모델로 과대적합 재현**

```python
from sklearn import preprocessing
from sklearn import pipeline

poly = preprocessing.PolynomialFeatures(degree=10, include_bias=False)
scaler = preprocessing.StandardScaler()
lin_reg2 = linear_model.LinearRegression()

pipeline_reg = pipeline.Pipeline([
    ('poly', poly),
    ('scal', scaler),
    ('lin', lin_reg2)])
pipeline_reg.fit(Xfull, yfull)
curve = pipeline_reg.predict(X[:, np.newaxis])
plt.plot(X, curve)
```

| 단계 | 하는 일 |
|---|---|
| `PolynomialFeatures(degree=10, include_bias=False)` | x → x, x², …, x¹⁰ 의 10개 특성으로 확장 (`include_bias=False`: 상수항 1 열 생략, 선형 회귀가 절편을 따로 학습하므로) |
| `StandardScaler()` | 평균 0, 표준편차 1로 표준화. GDP¹⁰ ≈ 10⁵⁰ 수준이라 스케일을 맞추지 않으면 수치적으로 불안정 |
| `LinearRegression()` | 확장된 특성에 선형 회귀 → 결과적으로 10차 다항식 |
| `Pipeline([(이름, 변환기), ..., (이름, 모델)])` | 여러 단계를 하나로 묶어 `fit`·`predict` 한 번에. 2장에서 본격적으로 다룸 |
| `X[:, np.newaxis]` | 1차원 `(1000,)` → 2차원 `(1000, 1)`. `reshape(-1, 1)`과 같음 |

- 훈련 R²: 다항 모델 **0.855** vs 선형 **0.361** → 훈련 데이터엔 훨씬 잘 맞음
- 하지만 곡선이 크게 출렁거림 → 이 모델의 예측을 **믿을 수 없다** (뒤의 6-3에서 검증 오차로 확인)

**② 우연한 패턴 — 이름에 W가 들어간 나라**

```python
w_countries = [c for c in full_country_stats.index if "W" in c.upper()]
full_country_stats.loc[w_countries][lifesat_col]
```

```
New Zealand    7.3
Sweden         7.3
Norway         7.6
Switzerland    7.5
```

- 36개국 중 이름에 W가 들어간 나라는 모두 만족도 7 이상 → "W가 있으면 행복하다"는 **패턴처럼 보인다**

```python
all_w_countries = [c for c in gdp_per_capita.index if "W" in c.upper()]
gdp_per_capita.loc[all_w_countries].sort_values(by=gdppc_col)
```

- 범위를 GDP 데이터 전체로 넓히면 Malawi(1,487), Rwanda(2,099), Zimbabwe(2,745) 같은 저소득국도 다 W가 들어간다 → **순전히 우연**
- 복잡한 모델은 이런 잡음 속 우연한 패턴까지 찾아낸다. 모델은 그게 진짜인지 잡음인지 구분하지 못한다

> 💡 `c.upper()`로 대문자로 바꾼 뒤 `"W" in` → 대소문자 무시 검색 (Norway의 w, Sweden의 w 모두 잡힘). 리스트 컴프리헨션 `[c for c in ... if 조건]`은 이후 장에서도 자주 나온다.

**해결 방법** — 과대적합은 모델이 데이터 양·잡음에 비해 너무 복잡할 때 생긴다

1. **모델 단순화**: 파라미터가 적은 모델 선택(고차 다항 → 선형), 특성 수 줄이기, **모델 규제**
2. **훈련 데이터 더 모으기**
3. **잡음 줄이기**: 오류 수정, 이상치 제거

**③ 규제 (regularization)**

모델을 단순하게 하고 과대적합 위험을 줄이도록 **제약**을 가하는 것. 선형 모델의 자유도는 2(θ₀, θ₁)인데, θ₁을 작게 유지하도록 강제하면 자유도가 1~2 사이가 된다.

```python
country_stats.plot(kind='scatter', x=gdppc_col, y=lifesat_col, figsize=(8, 3))
missing_data.plot(kind='scatter', x=gdppc_col, y=lifesat_col,
                  marker="s", color="r", grid=True, ax=plt.gca())

X = np.linspace(0, 115_000, 1000)
plt.plot(X, t0 + t1 * X, "b:", label="Linear model on partial data")
plt.plot(X, t0full + t1full * X, "k-", label="Linear model on all data")

ridge = linear_model.Ridge(alpha=10**9.5)
X_sample = country_stats[[gdppc_col]]
y_sample = country_stats[[lifesat_col]]
ridge.fit(X_sample, y_sample)
t0ridge, t1ridge = ridge.intercept_[0], ridge.coef_[0]
plt.plot(X, t0ridge + t1ridge * X, "b--",
         label="Regularized linear model on partial data")
plt.legend(loc="lower right")
```

| 코드 | 의미 |
|---|---|
| `ax=plt.gca()` | **gca = get current axes**. 두 번째 DataFrame 산점도를 같은 그래프 위에 겹쳐 그림 |
| `Ridge(alpha=...)` | 기울기(θ₁)가 커지는 것에 벌점을 주는 선형 회귀 (4장 릿지 회귀) |
| `alpha=10**9.5` | 규제 강도 ≈ 3.16 × 10⁹. GDP 값이 수만 단위라 θ₁이 매우 작고, 그만큼 alpha가 커야 효과가 보인다 |
| `label=` + `plt.legend()` | 범례 표시 |

| 모델 (학습 데이터) | θ₀ | θ₁ |
|---|---|---|
| 선형, 일부 데이터 (파란 점선) | 3.75 | 6.78 × 10⁻⁵ |
| **규제 선형, 일부 데이터** (파란 파선) | 5.35 | 2.93 × 10⁻⁵ |
| 선형, 전체 데이터 (검은 실선) | 5.58 | 2.33 × 10⁻⁵ |

→ 일부 데이터만 썼는데도 규제를 걸자 기울기가 작아져 **전체 데이터 모델에 훨씬 가까워졌다**. 훈련 데이터엔 덜 맞지만 새 샘플엔 더 잘 일반화된다.

- **하이퍼파라미터**: 규제 강도(`alpha`)처럼 **학습 알고리즘의** 파라미터. 훈련 전에 정하고 훈련 중엔 상수
  - 아주 크게 → 기울기 ≈ 0, 수평선 → 과대적합은 없지만 좋은 해도 못 찾음
  - 튜닝이 머신러닝 시스템 구축의 중요한 부분 → 6장에서 검증 세트로 고른다

| | 모델 파라미터 | 하이퍼파라미터 |
|---|---|---|
| 예 | θ₀, θ₁ (`intercept_`, `coef_`) | `alpha`, `n_neighbors`, `degree` |
| 누가 정하나 | **학습 알고리즘**이 데이터로 찾음 | **사람**이 훈련 전에 지정 |
| 사이킷런 위치 | 훈련 후 `_` 붙은 속성 | 생성자 인자 |

### 5-6. 훈련 데이터 과소적합

**과소적합 (underfitting)**: 모델이 너무 단순해서 데이터의 구조를 학습하지 못함. 예: 삶의 만족도의 실제 구조는 선형보다 복잡 → 선형 모델은 **훈련 샘플에서도** 부정확

해결:
1. 파라미터가 더 많은 **강력한 모델** 선택
2. 더 좋은 특성 제공 (특성 공학)
3. **규제 줄이기** (하이퍼파라미터 감소)

### 5-7. 핵심 요약

- 머신러닝 = 규칙을 명시하지 않고 **데이터에서 학습**해 작업을 더 잘하게 만드는 것
- 종류: 지도/비지도/준지도/자기 지도/강화, 배치/온라인, 사례 기반/모델 기반
- 모델 기반이면 파라미터를 조정해 훈련 세트에 맞추고, 그 모델로 새 사례를 예측
- 훈련 세트가 너무 작거나, 대표성이 없거나, 잡음이 많거나, 관련 없는 특성이 많으면 잘 동작하지 않는다
- 모델은 **너무 단순(과소적합)해도, 너무 복잡(과대적합)해도** 안 된다

---

## 6. 테스트와 검증 (1.6)

모델이 새 샘플에 얼마나 잘 일반화될지 아는 방법은 **새 샘플에 실제로 적용해 보는 것**뿐. 실서비스에 바로 올리는 대신 데이터를 나눈다.

> 📌 노트북에는 1.6절 코드가 없다. 아래 6-3은 개념을 확인하려고 `lifesat_full.csv`(36개국)로 직접 만든 **보충 코드**다.

### 6-1. 훈련 세트와 테스트 세트

- 데이터를 **훈련 세트**와 **테스트 세트**로 분할 → 훈련 세트로 학습, 테스트 세트로 평가
- 새 샘플에 대한 오류 비율 = **일반화 오차** (외부 샘플 오차, out-of-sample error)
- 테스트 세트로 추정한 일반화 오차 → 처음 보는 샘플에 대한 성능을 미리 가늠

| 훈련 오차 | 일반화 오차 | 진단 |
|---|---|---|
| 낮음 | **높음** | **과대적합** |
| 높음 | 높음 | 과소적합 |
| 낮음 | 낮음 | 좋음 |

- 보통 **80%를 훈련, 20%를 테스트**. 데이터가 아주 크면(예: 1천만 개) 1%만 떼어도 테스트 세트로 충분(10만 개)

### 6-2. 하이퍼파라미터 튜닝과 모델 선택

문제 상황: 선형 vs 다항 모델 고민 → 둘 다 훈련해 테스트 세트로 비교 → 선형 선택. 규제 하이퍼파라미터 100개를 테스트 세트로 비교해 오차 5%인 최적값 선택 → **실서비스 오차 15%** 😱

**이유**: 테스트 세트로 일반화 오차를 **여러 번** 측정해 모델·하이퍼파라미터를 그 테스트 세트에 맞춰 버렸다 → 새 데이터에선 성능이 떨어짐 (**테스트 세트에 과대적합**)

**해결 — 홀드아웃 검증 (holdout validation)**

```
전체 데이터
├─ 훈련 세트 ──┬─ (축소된) 훈련 세트  → 여러 후보 모델·하이퍼파라미터로 훈련
│             └─ 검증 세트 (= 개발 세트, dev set) → 가장 좋은 후보 선택
│   → 선택된 모델을 (검증 세트 포함) 전체 훈련 세트로 다시 훈련 → 최종 모델
└─ 테스트 세트  → 최종 모델을 **딱 한 번** 평가 → 일반화 오차 추정
```

검증 세트 크기의 딜레마:

| 검증 세트가 너무 작으면 | 너무 크면 |
|---|---|
| 평가가 부정확 → 최적이 아닌 모델을 고를 수 있음 | 남은 훈련 세트가 작아짐 → 전체로 훈련할 최종 모델과 비교 조건이 달라짐 (마라톤 선수를 단거리 기록으로 뽑는 격) |

→ **교차 검증 (cross-validation)**: 작은 검증 세트 여러 개로 반복 평가 → 평균으로 정확한 성능 측정. 단점: 훈련 시간이 검증 세트 개수에 비례해 늘어남 (2장 `cross_val_score`)

### 6-3. 보충 코드 — 과대적합을 검증으로 잡아내기

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import root_mean_squared_error

full = pd.read_csv("datasets/lifesat/lifesat_full.csv", index_col=0)
X = full[["GDP per capita (USD)"]].values
y = full["Life satisfaction"].values

# 1) 테스트 세트를 먼저 떼어 놓고 손대지 않는다
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)          # 훈련 28개국, 테스트 8개국

# 2) 후보 모델
models = {
    "선형": LinearRegression(),
    "3-NN": KNeighborsRegressor(n_neighbors=3),
    "10차 다항": make_pipeline(PolynomialFeatures(10, include_bias=False),
                              StandardScaler(), LinearRegression()),
}

# 3) 훈련 세트 안에서만 5-폴드 교차 검증으로 비교
cv = KFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    model.fit(X_train, y_train)
    train_rmse = root_mean_squared_error(y_train, model.predict(X_train))
    cv_rmse = -cross_val_score(model, X_train, y_train, cv=cv,
                               scoring="neg_root_mean_squared_error")
    print(name, round(train_rmse, 3), round(cv_rmse.mean(), 3))

# 4) 검증 성적이 가장 좋은 모델만 테스트 세트로 한 번 평가
best = models["3-NN"]
root_mean_squared_error(y_test, best.predict(X_test))   # 0.565
```

| 모델 | 훈련 RMSE | 교차 검증 RMSE (5-폴드 평균) |
|---|---|---|
| 선형 | 0.571 | 0.641 |
| **3-NN** | 0.348 | **0.466** ← 선택 |
| 10차 다항 | **0.242** (가장 좋음) | **217.9** (폭발) |

- 10차 다항은 **훈련 오차가 가장 낮지만** 검증 오차가 폭발 → 과대적합을 숫자로 확인. 다섯 폴드 중 하나에서 RMSE가 1,000을 넘었다: 검증 폴드에 훈련 폴드의 GDP 범위 밖 나라가 들어가면 10차 곡선이 범위 밖에서 급격히 튀기 때문
- 선택한 3-NN을 **테스트 세트로 단 한 번** 평가 → RMSE 0.565 (만족도 0.57점 정도 오차)
- `cross_val_score`는 "클수록 좋은" 점수를 쓰므로 RMSE에 음수를 붙인 `neg_root_mean_squared_error`를 반환 → 앞에 `-`를 붙여 되돌림

> ⚠️ 36개 샘플은 너무 작아서 분할·폴드에 따라 결과가 크게 흔들린다. 여기선 "훈련 오차만 보면 속는다"는 흐름을 보는 용도. `random_state`를 바꾸면 순위가 달라질 수 있다.

### 6-4. 데이터 불일치

훈련 데이터는 많이 구할 수 있지만 **실서비스 데이터를 완벽히 대표하지 못하는** 경우.

예: 꽃 사진 분류 앱. 웹에서 꽃 사진 수백만 장을 받았지만, 실제로 **앱으로 찍은 사진**은 1,000장뿐

- **규칙: 검증 세트와 테스트 세트는 실서비스 데이터를 최대한 대표해야 한다** → 앱 사진 1,000장을 반씩 검증·테스트 세트로 (중복·유사 사진이 양쪽에 섞이지 않게 주의)
- 웹 사진으로 훈련한 모델이 검증 세트에서 성능이 나쁘면 원인이 둘 중 무엇인지 알 수 없다
  1. 훈련 세트에 **과대적합**
  2. 웹 사진과 앱 사진의 **데이터 불일치**

→ **훈련-개발 세트 (train-dev set)**: 웹 사진(훈련 데이터) 일부를 떼어 둔 세트

| 훈련-개발 세트 성능 | 검증 세트 성능 | 진단 | 대응 |
|---|---|---|---|
| 나쁨 | 나쁨 | 훈련 세트에 **과대적합** | 모델 단순화·규제, 데이터 추가, 정제 |
| **좋음** | 나쁨 | **데이터 불일치** | 웹 사진을 앱 사진처럼 전처리 후 재훈련 |
| 좋음 | 좋음 | 문제 없음 | 테스트 세트로 최종 평가 |

> 💡 데이터 불일치 대응 예: 웹 이미지에 앱 카메라 특성(흐림, 조명 등)을 흉내 내는 변환을 적용. 생성 모델로 웹 이미지를 앱 사진 스타일로 바꾸는 방법도 있다.

### 6-5. 공짜 점심 없음 (No Free Lunch)

- 모델 = 데이터의 **간소화된 표현**. 일반화되지 않을 세부 사항(잡음)은 버린다
- 무엇을 버리고 무엇을 남길지 정하려면 **가정**이 필요 (예: 선형 모델은 "데이터가 근본적으로 선형이고 직선과의 차이는 잡음"이라고 가정)
- 데이터에 대해 **아무 가정도 하지 않으면**, 어떤 모델이 다른 모델보다 낫다고 말할 근거가 없다 (1996 데이비드 월퍼트) → 어떤 데이터셋에선 선형, 다른 데이터셋에선 신경망이 최선
- 모든 모델을 다 평가할 수는 없으므로 → 데이터에 대해 **합리적인 가정**을 하고 **적절한 몇 개 모델만** 평가 (간단한 작업 → 규제 강도가 다른 선형 모델들, 복잡한 문제 → 여러 신경망)

---

## 7. 연습문제

노트북 답안을 정리. 모두 개념 문제.

| 번호 | 질문 | 답 요약 |
|---|---|---|
| 1 | 머신러닝을 어떻게 정의? | **데이터에서 학습**하는 시스템을 만드는 것. 학습 = 어떤 성능 지표 기준으로 작업을 더 잘하게 되는 것 |
| 2 | 머신러닝이 도움이 되는 네 가지 문제 유형 | ① 알고리즘 해법이 없는 복잡한 문제 ② 긴 수작업 규칙 목록 대체 ③ 변화하는 환경에 적응 ④ 사람의 학습 보조(데이터 마이닝) |
| 3 | 레이블된 훈련 세트란? | 각 샘플에 **원하는 정답(레이블)**이 들어 있는 훈련 세트 |
| 4 | 가장 일반적인 지도 학습 작업 두 가지 | **회귀**, **분류** |
| 5 | 일반적인 비지도 학습 작업 네 가지 | **군집, 시각화, 차원 축소, 연관 규칙 학습** |
| 6 | 처음 보는 지형에서 로봇이 걷게 하려면? | **강화 학습**. 지도·준지도로 표현할 수도 있지만 덜 자연스럽다 |
| 7 | 고객을 여러 그룹으로 나누려면? | 그룹 정의를 모르면 **군집**(비지도). 원하는 그룹을 알면 그룹별 예시를 주고 **분류**(지도) |
| 8 | 스팸 감지는 지도/비지도? | 전형적인 **지도 학습** — 이메일과 레이블(스팸/정상)을 함께 학습 |
| 9 | 온라인 학습 시스템이란? | 배치 학습과 달리 **점진적으로** 학습. 변화하는 데이터와 자율 시스템에 빠르게 적응, 매우 큰 데이터 훈련 가능 |
| 10 | 외부 메모리 학습이란? | 메인 메모리에 안 들어가는 대용량 데이터를 **미니배치로 쪼개** 온라인 학습 기법으로 학습 |
| 11 | 유사도 측정으로 예측하는 학습은? | **사례 기반 학습** — 훈련 데이터를 기억했다가 새 샘플과 가장 비슷한 샘플들로 예측 |
| 12 | 모델 파라미터 vs 하이퍼파라미터 | 모델 파라미터: 새 샘플에 대한 예측을 결정 (예: 선형 모델의 기울기), 학습 알고리즘이 최적값을 찾음. 하이퍼파라미터: 모델이 아니라 **학습 알고리즘 자체**의 파라미터 (예: 규제 강도) |
| 13 | 모델 기반 알고리즘이 찾는 것·전략·예측 방법 | 새 샘플에 잘 일반화되는 **최적의 파라미터 값**을 찾음. 훈련 데이터에서 예측이 얼마나 나쁜지 재는 **비용 함수**(규제 시 복잡도 벌점 추가)를 최소화해 훈련. 예측은 새 샘플의 특성을 찾은 파라미터로 된 예측 함수에 넣음 |
| 14 | 머신러닝의 주요 도전 과제 | 데이터 부족, 낮은 데이터 품질, 대표성 없는 데이터, 정보가 없는 특성, 과소적합(너무 단순한 모델), 과대적합(너무 복잡한 모델) |
| 15 | 훈련 데이터에선 좋은데 새 샘플에서 나쁘면? 해결책 3가지 | **과대적합**(또는 훈련 데이터 운이 아주 좋았음). ① 데이터 더 모으기 ② 모델 단순화(더 간단한 알고리즘, 파라미터·특성 수 줄이기, 규제) ③ 훈련 데이터 잡음 줄이기 |
| 16 | 테스트 세트는 무엇이고 왜 쓰나? | 실서비스 투입 **전에** 새 샘플에 대한 **일반화 오차를 추정** |
| 17 | 검증 세트의 목적은? | **모델 비교**. 최선의 모델 선택과 하이퍼파라미터 튜닝 |
| 18 | 훈련-개발 세트란? 언제, 어떻게? | 훈련 데이터와 검증·테스트 데이터(= 실서비스와 비슷해야 함) 사이에 **불일치 위험**이 있을 때 사용. 훈련 세트 일부를 떼어 둔 세트(훈련엔 안 씀). 훈련 세트는 좋은데 훈련-개발 세트에서 나쁘면 → 과대적합. 둘 다 좋은데 검증 세트에서 나쁘면 → **데이터 불일치** → 훈련 데이터를 검증·테스트 데이터와 비슷하게 개선 |
| 19 | 테스트 세트로 하이퍼파라미터를 튜닝하면? | **테스트 세트에 과대적합** → 측정한 일반화 오차가 낙관적 → 기대보다 성능 나쁜 모델을 출시할 위험 |

---

## 8. 전체 흐름 요약

```
[1.1 정의] 작업 T의 성능 P가 경험 E(훈련 세트)로 향상 = 학습
[1.2 이유] 복잡한 규칙 대체 / 알고리즘 없는 문제 / 변하는 환경 / 데이터 마이닝
[1.3 사례] 이미지·텍스트 분류, 회귀, 이상치 탐지, 군집, 차원 축소, 추천, 강화 학습

[1.4 종류] (조합 가능)
  ├ 지도 방식: 지도(분류·회귀) / 비지도(군집·차원 축소·이상치) / 준지도 / 자기 지도 / 강화
  ├ 배치 vs 온라인 (학습률, 외부 메모리 학습, 데이터 드리프트)
  └ 사례 기반 vs 모델 기반
       KNeighborsRegressor(n_neighbors=3) → 5.73   ← 이웃 평균
       LinearRegression()                 → 6.02   ← θ0 + θ1·x
       공통: model.fit(X, y) → model.predict(X_new)

[1.5 도전 과제]
  나쁜 데이터: 양 부족 / 대표성 없음(샘플링 잡음·편향) / 낮은 품질 / 관련 없는 특성
  나쁜 모델:  과대적합 → 단순화·규제(Ridge alpha)·데이터 추가·잡음 제거
             과소적합 → 강력한 모델·좋은 특성·규제 완화

[1.6 테스트와 검증]
  훈련 | 검증(모델·하이퍼파라미터 선택, 교차 검증) | 테스트(최종 1회)
  데이터 불일치 → 훈련-개발 세트로 과대적합과 구분
  공짜 점심 없음 → 합리적 가정 + 몇 개 모델만 평가
```

### 사례 기반 vs 모델 기반 / 과대적합 vs 과소적합

| | 원인 | 증상 | 처방 |
|---|---|---|---|
| 과대적합 | 데이터 양·잡음 대비 모델이 너무 복잡 | 훈련 오차 ↓, 검증 오차 ↑ | 단순한 모델, 특성 축소, **규제 ↑**, 데이터 추가, 잡음 제거 |
| 과소적합 | 모델이 너무 단순 | 훈련 오차부터 ↑ | 강력한 모델, 특성 공학, **규제 ↓** |

### 주요 결과

| 실험 | 지표 | 값 |
|---|---|---|
| 선형 회귀, 일부 데이터 27개국 | θ₀ / θ₁ | 3.75 / 6.78 × 10⁻⁵ |
| 〃 | 푸에르토리코 예측 | **6.016** |
| 3-NN, 일부 데이터 | 푸에르토리코 예측 (폴란드·포르투갈·에스토니아 평균) | **5.733** |
| 선형 회귀, 전체 데이터 36개국 | θ₀ / θ₁ | 5.58 / 2.33 × 10⁻⁵ |
| 선형 (일부 / 전체 데이터) | 훈련 R² | 0.727 / 0.361 |
| 10차 다항, 전체 데이터 | 훈련 R² | 0.855 (과대적합) |
| 릿지 alpha=10^9.5, 일부 데이터 | θ₀ / θ₁ | 5.35 / 2.93 × 10⁻⁵ (전체 데이터 모델에 근접) |
| 보충: 5-폴드 CV RMSE (선형 / 3-NN / 10차 다항) | 검증 오차 | 0.641 / **0.466** / 217.9 |
| 보충: 3-NN 테스트 세트 | RMSE | 0.565 |

### 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `df[["열"]].values` / `np.c_[series]` | 특성 1개를 2차원 `(n, 1)` 배열로 | 4-3, 5-2 |
| `X[:, np.newaxis]` | 1차원 → 열 벡터 | 5-5 |
| `model.fit(X, y)` → `model.predict(X_new)` | 모든 사이킷런 모델의 공통 인터페이스 | 4-3 |
| `model.intercept_`, `model.coef_` | 훈련으로 얻은 파라미터 (`_` 접미사) | 4-3, 5-5 |
| `df[df["Year"] == 2020]`, `(조건1) & (조건2)` | 불리언 인덱싱 필터 | 4-3, 5-2 |
| `df.pivot(index=, columns=, values=)` | 긴 형식 → 넓은 형식 | 4-3 |
| `pd.merge(left, right, left_index=True, right_index=True)` | 인덱스 기준 조인 | 4-3 |
| `Pipeline([("이름", 변환기), ..., ("이름", 모델)])` | 다항 특성 → 스케일링 → 회귀 묶기 | 5-5 |
| `Ridge(alpha=...)` | 규제 선형 회귀 | 5-5 |
| `np.linspace(a, b, n)` + `plt.plot(X, θ0 + θ1 * X)` | 모델 직선 그리기 | 4-3, 5-2 |
| `plt.annotate(..., arrowprops=dict(...))` | 그래프 점에 화살표 설명 | 4-3, 5-2 |
| `df.plot(..., ax=plt.gca())` | DataFrame 그래프 겹쳐 그리기 | 5-5 |
| `train_test_split` + `cross_val_score(..., scoring="neg_root_mean_squared_error")` | 테스트 분리 + 교차 검증 (보충) | 6-3 |
