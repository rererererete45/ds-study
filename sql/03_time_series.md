# 3장. 시계열 분석

활용 사례는 나의 Velog를 확인해주세요~~
url:

> 시간 순서대로 쌓인 데이터를 **트렌드 → 롤링(이동평균·누적) → 계절성(기간 비교)** 순서로 뜯어보는 과정
>
> 실습 데이터: 미국 소매 판매 데이터 `retail_sales` (월별 × 업종별 매출)
>
> | 컬럼 | 타입 | 설명 |
> |---|---|---|
> | `sales_month` | date | 매월 1일 (예: 1992-01-01) |
> | `naics_code` | text | 업종 코드 |
> | `kind_of_business` | text | 업종명 |
> | `reason_for_null` | text | 매출이 NULL인 이유 |
> | `sales` | 숫자 | 월 매출 (백만 달러) |

## 목차

1. [날짜·시간 다루기 (PostgreSQL)](#1-날짜시간-다루기-postgresql)
2. [트렌드 분석](#2-트렌드-분석)
3. [요소 비교](#3-요소-비교)
4. [전체 대비 비율](#4-전체-대비-비율)
5. [인덱싱](#5-인덱싱)
6. [시간 윈도우 롤링](#6-시간-윈도우-롤링)
7. [누적값](#7-누적값)
8. [계절성 분석: 기간 비교](#8-계절성-분석-기간-비교)
9. [SQL 문법 메모](#9-sql-문법-메모)

---

## 1. 날짜·시간 다루기 (PostgreSQL)

책·블로그 예제가 MySQL 기준이면 날짜 함수 이름이 달라서 에러가 난다.

| 용도 | MySQL | PostgreSQL |
|---|---|---|
| 날짜 → 문자열 포맷 | `DATE_FORMAT(d, '%Y-%m')` | `TO_CHAR(d, 'YYYY-MM')` |
| 월 초로 자르기 | `DATE_FORMAT(d, '%Y-%m-01')` | `DATE_TRUNC('month', d)` |
| 일수 차이 | `DATEDIFF(끝, 시작)` | `끝 - 시작` (date끼리 빼기) |
| 날짜 더하기 | `DATE_ADD(d, INTERVAL 1 DAY)` | `d + INTERVAL '1 day'` |
| 연·월 추출 | `YEAR(d)`, `MONTH(d)` | `DATE_PART('year', d)`, `EXTRACT(MONTH FROM d)` |

> 💡 `DATEDIFF('day', 시작, 끝)` 형태는 SQL Server·Redshift 문법. PostgreSQL에는 없다.

### 1-1. 자르기: `DATE_TRUNC` vs `TO_CHAR`

```sql
SELECT DATE_TRUNC('month', '2020-10-04 12:33:35'::timestamp);        -- 2020-10-01 00:00:00 (timestamp)
SELECT DATE_TRUNC('month', '2020-10-04 12:33:35'::timestamp)::date;  -- 2020-10-01 (date)
SELECT TO_CHAR('2020-10-04 12:33:35'::timestamp, 'YYYY-MM');          -- '2020-10' (text)
```

- 월별 `GROUP BY`에는 **`DATE_TRUNC`** 가 낫다 → 결과가 날짜 타입이라 정렬·비교·롤링이 그대로 된다
- `TO_CHAR`는 화면 표시용

| 포맷 | MySQL | PostgreSQL |
|---|---|---|
| 연 | `%Y` | `YYYY` |
| 월 | `%m` | `MM` |
| 일 | `%d` | `DD` |
| 시 | `%H` | `HH24` |
| 분 | `%i` | `MI` |
| 초 | `%s` | `SS` |

### 1-2. 날짜 차이

```sql
SELECT DATE '2020-06-30' - DATE '2020-05-31';            -- 30 (integer)
SELECT '2020-06-30 12:00'::timestamp
     - '2020-05-31 00:00'::timestamp;                    -- 30 days 12:00:00 (interval)
SELECT AGE(DATE '2021-08-15', DATE '2020-05-31');        -- 1 year 2 mons 15 days
```

| 빼는 대상 | 결과 타입 |
|---|---|
| date − date | integer (일수) |
| timestamp − timestamp | interval |
| `AGE(끝, 시작)` | interval (연·월·일 단위) |

### 1-3. 타임존

```sql
-- UTC로 저장된 timestamptz를 한국 시간 기준 월로 자르기
DATE_TRUNC('month', ts AT TIME ZONE 'Asia/Seoul')
```

> ⚠️ 로그 데이터는 보통 UTC. 그대로 월별 집계하면 **월말·월초 오전 9시 이전** 기록이 다른 달로 잡힌다.

> ⚠️ `'2020-10-04'`처럼 따옴표만 쓴 리터럴은 PostgreSQL이 `unknown` 타입으로 본다. 함수에 넣을 땐 `::timestamp`, `::date`로 캐스팅. (테이블 컬럼은 이미 타입이 있어서 불필요)

---

## 2. 트렌드 분석

### 2-1. 간단한 트렌드

```sql
-- 월별 그대로
SELECT sales_month, sales
FROM retail_sales
WHERE kind_of_business = 'Retail and food services sales, total'
ORDER BY 1;
```

```sql
-- 연 단위로 집계 → 월별 노이즈(계절성)가 사라지고 장기 추세가 보인다
SELECT date_part('year', sales_month) AS sales_year
     , sum(sales) AS sales
FROM retail_sales
WHERE kind_of_business = 'Retail and food services sales, total'
GROUP BY 1
ORDER BY 1;
```

> 💡 시계열은 **집계 단위(일·주·월·연)** 를 바꿔 보는 것만으로도 보이는 게 달라진다. 너무 촘촘하면 노이즈, 너무 크면 변화가 묻힌다.

---

## 3. 요소 비교

여러 항목(업종)의 추세를 나란히 놓고 비교.

### 3-1. 세로로 나열

```sql
SELECT date_part('year', sales_month) AS sales_year
     , kind_of_business
     , sum(sales) AS sales
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
GROUP BY 1, 2
ORDER BY 1, 2;
```

> 💡 문자열 안의 작은따옴표는 `''` 두 번으로 이스케이프 (`'Men''s clothing stores'`)

### 3-2. 피벗해서 옆으로 나란히 (CASE 문)

같은 행에 두 값이 있어야 **차이·배율** 계산이 가능하다.

```sql
SELECT date_part('year', sales_month) AS sales_year
     , sum(CASE WHEN kind_of_business = 'Women''s clothing stores' THEN sales END) AS womens_sales
     , sum(CASE WHEN kind_of_business = 'Men''s clothing stores'   THEN sales END) AS mens_sales
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
GROUP BY 1
ORDER BY 1;
```

### 3-3. 차이 · 배율 · 비율 차이

피벗 결과를 서브쿼리로 감싸고 바깥에서 계산.

```sql
SELECT sales_year
     , womens_sales - mens_sales                 AS womens_minus_mens    -- 차이
     , womens_sales / mens_sales                 AS womens_times_of_mens -- 배율 (몇 배)
     , (womens_sales / mens_sales - 1) * 100     AS womens_pct_of_mens   -- 몇 % 더 큰가
FROM (
    SELECT date_part('year', sales_month) AS sales_year
         , sum(CASE WHEN kind_of_business = 'Women''s clothing stores' THEN sales END) AS womens_sales
         , sum(CASE WHEN kind_of_business = 'Men''s clothing stores'   THEN sales END) AS mens_sales
    FROM retail_sales
    WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
      AND sales_month <= '2019-12-01'
    GROUP BY 1
) a
ORDER BY 1;
```

| 지표 | 식 | 읽는 법 |
|---|---|---|
| 차이 | `A - B` | A가 B보다 얼마나 많은가 (절대값) |
| 배율 | `A / B` | A가 B의 몇 배인가 |
| 비율 차이 | `(A / B - 1) * 100` | A가 B보다 몇 % 더 큰가 |

- `sales_month <= '2019-12-01'` : 2020년(코로나)은 남성복 매출 일부가 NULL이라 제외
- 서브쿼리 없이 `sum(CASE…) - sum(CASE…)`처럼 한 번에 써도 결과는 같다

> ⚠️ B가 0일 수 있으면 `A / NULLIF(B, 0)` 으로 0 나누기 에러를 막는다.

---

## 4. 전체 대비 비율

"이 항목이 전체에서 몇 %인가?"

### 4-1. 같은 달 전체 대비 (셀프 조인)

```sql
SELECT sales_month
     , kind_of_business
     , sales * 100 / total_sales AS pct_total_sales
FROM (
    SELECT a.sales_month, a.kind_of_business, a.sales
         , sum(b.sales) AS total_sales
    FROM retail_sales a
    JOIN retail_sales b
      ON a.sales_month = b.sales_month                 -- 같은 달끼리 붙여서
     AND b.kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
    WHERE a.kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
    GROUP BY 1, 2, 3                                    -- b쪽을 합산
) aa
ORDER BY 1, 2;
```

### 4-2. 같은 달 전체 대비 (윈도우 함수) ✅ 더 간단

```sql
SELECT sales_month, kind_of_business, sales
     , sum(sales) OVER (PARTITION BY sales_month)              AS total_sales
     , sales * 100 / sum(sales) OVER (PARTITION BY sales_month) AS pct_total
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
ORDER BY 1, 2;
```

- `PARTITION BY sales_month` → 같은 달끼리 묶어서 합계를 **각 행 옆에** 붙인다 (행이 줄지 않음)

### 4-3. 연간 합계 대비 (그 해에서 이 달이 차지하는 비중)

```sql
SELECT sales_month, kind_of_business, sales
     , sum(sales) OVER (PARTITION BY date_part('year', sales_month), kind_of_business) AS yearly_sales
     , sales * 100
       / sum(sales) OVER (PARTITION BY date_part('year', sales_month), kind_of_business) AS pct_yearly
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
ORDER BY 1, 2;
```

| PARTITION BY | 의미 |
|---|---|
| `sales_month` | 같은 달의 모든 업종 합 → 업종 간 점유율 |
| `연도, 업종` | 같은 업종의 그 해 합 → 월별 비중 (계절 패턴) |

> 💡 셀프 조인 방식도 같은 결과. 윈도우 함수가 없던 DB나 조건이 복잡할 때 쓰는 방법으로 알아두기.

### 4-4. GROUP BY 결과에 윈도우 함수 쓰기

원본이 집계 전 데이터라면 `SUM(SUM(…)) OVER` 패턴.

```sql
SELECT date_part('year', sales_month) AS sales_year
     , kind_of_business
     , sum(sales) AS sales
     , sum(sales) * 100.0
       / SUM(sum(sales)) OVER (PARTITION BY date_part('year', sales_month)) AS pct_of_year
FROM retail_sales
WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
GROUP BY 1, 2;
```

- 안쪽 `sum(sales)` = GROUP BY로 묶은 업종별 연 합계
- 바깥 `SUM(…) OVER` = 그 결과들을 같은 연도끼리 다시 더한 연도 전체 합계
- 실행 순서상 윈도우 함수는 GROUP BY **다음**에 계산되기 때문에 가능

---

## 5. 인덱싱

**기준 시점(보통 첫 해)을 100으로 놓고** 이후 변화를 % 로 본다. 규모가 다른 항목끼리도 성장 속도를 비교할 수 있다.

### 5-1. `first_value()`로 기준값 붙이기

```sql
SELECT sales_year, sales
     , first_value(sales) OVER (ORDER BY sales_year) AS index_sales
FROM (
    SELECT date_part('year', sales_month) AS sales_year
         , sum(sales) AS sales
    FROM retail_sales
    WHERE kind_of_business = 'Women''s clothing stores'
    GROUP BY 1
) a;
```

### 5-2. 기준 대비 증감률

```sql
SELECT sales_year, sales
     , (sales / first_value(sales) OVER (ORDER BY sales_year) - 1) * 100 AS pct_from_index
FROM (
    SELECT date_part('year', sales_month) AS sales_year
         , sum(sales) AS sales
    FROM retail_sales
    WHERE kind_of_business = 'Women''s clothing stores'
    GROUP BY 1
) a;
```

### 5-3. 윈도우 함수 없이 (조인 버전)

```sql
SELECT sales_year, sales
     , (sales / index_sales - 1) * 100 AS pct_from_index
FROM (
    SELECT date_part('year', aa.sales_month) AS sales_year
         , bb.index_sales
         , sum(aa.sales) AS sales
    FROM retail_sales aa
    JOIN (
        -- 첫 해 매출 합계 1행
        SELECT first_year, sum(a.sales) AS index_sales
        FROM retail_sales a
        JOIN (
            SELECT min(date_part('year', sales_month)) AS first_year
            FROM retail_sales
            WHERE kind_of_business = 'Women''s clothing stores'
        ) b ON date_part('year', a.sales_month) = b.first_year
        WHERE a.kind_of_business = 'Women''s clothing stores'
        GROUP BY 1
    ) bb ON 1 = 1                    -- 모든 행에 기준값 1행을 붙임 (크로스 조인)
    WHERE aa.kind_of_business = 'Women''s clothing stores'
    GROUP BY 1, 2
) aaa
ORDER BY 1;
```

> ⚠️ 배포된 예제 코드에는 이 쿼리 앞부분 4줄이 **중복으로 붙어 있어** 그대로 돌리면 `syntax error at or near "SELECT"` 가 난다. 위 코드가 정리한 버전. 결과는 5-2와 같으므로 실무에서는 `first_value()` 쪽을 쓰면 된다.

### 5-4. 여러 항목을 각각의 기준으로 인덱싱

```sql
SELECT sales_year, kind_of_business, sales
     , (sales / first_value(sales) OVER (PARTITION BY kind_of_business ORDER BY sales_year) - 1) * 100
       AS pct_from_index
FROM (
    SELECT date_part('year', sales_month) AS sales_year
         , kind_of_business
         , sum(sales) AS sales
    FROM retail_sales
    WHERE kind_of_business IN ('Men''s clothing stores', 'Women''s clothing stores')
      AND sales_month <= '2019-12-31'
    GROUP BY 1, 2
) a
ORDER BY 1, 2;
```

- `PARTITION BY kind_of_business` → 업종마다 **자기 첫 해**를 기준으로 삼는다

---

## 6. 시간 윈도우 롤링

**이동평균**: 최근 N기간의 평균을 한 칸씩 밀면서 계산 → 월별 들쭉날쭉함을 부드럽게 만든다.

### 6-1. 셀프 조인으로 윈도우 직접 만들기

```sql
-- 2019-12 기준 최근 12개월(2019-01 ~ 2019-12)이 어떻게 붙는지 확인
SELECT a.sales_month, a.sales
     , b.sales_month AS rolling_sales_month
     , b.sales       AS rolling_sales
FROM retail_sales a
JOIN retail_sales b
  ON a.kind_of_business = b.kind_of_business
 AND b.sales_month BETWEEN a.sales_month - INTERVAL '11 months' AND a.sales_month
 AND b.kind_of_business = 'Women''s clothing stores'
WHERE a.kind_of_business = 'Women''s clothing stores'
  AND a.sales_month = '2019-12-01';
```

```sql
-- 12개월 이동평균
SELECT a.sales_month, a.sales
     , avg(b.sales)   AS moving_avg
     , count(b.sales) AS records_count   -- 12가 아니면 데이터가 빠진 구간
FROM retail_sales a
JOIN retail_sales b
  ON a.kind_of_business = b.kind_of_business
 AND b.sales_month BETWEEN a.sales_month - INTERVAL '11 months' AND a.sales_month
 AND b.kind_of_business = 'Women''s clothing stores'
WHERE a.kind_of_business = 'Women''s clothing stores'
  AND a.sales_month >= '1993-01-01'
GROUP BY 1, 2
ORDER BY 1;
```

- `- INTERVAL '11 months'` + 현재 달 = 12개월
- 1993-01부터 보는 이유: 1992년은 앞쪽 12개월이 다 안 채워짐

### 6-2. 윈도우 프레임으로 ✅ 더 간단

```sql
SELECT sales_month
     , avg(sales)   OVER (ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS moving_avg
     , count(sales) OVER (ORDER BY sales_month ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS records_count
FROM retail_sales
WHERE kind_of_business = 'Women''s clothing stores';
```

**프레임 문법**

```sql
ROWS BETWEEN [시작] AND [끝]
```

| 표현 | 의미 |
|---|---|
| `UNBOUNDED PRECEDING` | 파티션의 맨 처음 행 |
| `N PRECEDING` | 현재 행에서 N행 앞 |
| `CURRENT ROW` | 현재 행 |
| `N FOLLOWING` | 현재 행에서 N행 뒤 |
| `UNBOUNDED FOLLOWING` | 파티션의 맨 마지막 행 |

> ⚠️ `ROWS`는 **행 개수** 기준. 중간에 빠진 달이 있으면 "11행 앞"이 11개월 전이 아니게 된다 → 6-3

### 6-3. 희소 데이터와 롤링

데이터가 없는 기간은 **행 자체가 없다**. 이럴 땐 빈틈 없는 날짜 테이블을 기준으로 조인한다.

```sql
-- date_dim: 날짜가 하루도 빠짐없이 들어있는 테이블 (책에서 따로 생성)
-- 1월·7월 데이터만 있다고 가정했을 때 매월 12개월 이동평균
SELECT a.date
     , avg(b.sales)   AS moving_avg
     , count(b.sales) AS records
FROM date_dim a
JOIN (
    SELECT sales_month, sales
    FROM retail_sales
    WHERE kind_of_business = 'Women''s clothing stores'
      AND date_part('month', sales_month) IN (1, 7)
) b ON b.sales_month BETWEEN a.date - INTERVAL '11 months' AND a.date
WHERE a.date = a.first_day_of_month
  AND a.date BETWEEN '1993-01-01' AND '2020-12-01'
GROUP BY 1
ORDER BY 1;
```

날짜 테이블이 없으면 PostgreSQL의 `generate_series`로 즉석에서 만든다.

```sql
SELECT generate_series('1993-01-01'::date, '2020-12-01'::date, INTERVAL '1 month')::date AS month;
```

```sql
-- 위 쿼리의 date_dim 자리를 generate_series로 대체
SELECT a.date
     , avg(b.sales)   AS moving_avg
     , count(b.sales) AS records
FROM (
    SELECT generate_series('1993-01-01'::date, '2020-12-01'::date, INTERVAL '1 month')::date AS date
) a
JOIN (
    SELECT sales_month, sales
    FROM retail_sales
    WHERE kind_of_business = 'Women''s clothing stores'
      AND date_part('month', sales_month) IN (1, 7)
) b ON b.sales_month BETWEEN a.date - INTERVAL '11 months' AND a.date
GROUP BY 1
ORDER BY 1;
```

> 💡 날짜 테이블 대신 **원본 테이블의 DISTINCT 월 목록**을 기준으로 써도 된다 (다른 업종이라도 그 달 데이터가 있으면 월 목록이 채워짐)

```sql
SELECT a.sales_month, avg(b.sales) AS moving_avg
FROM (
    SELECT DISTINCT sales_month
    FROM retail_sales
    WHERE sales_month BETWEEN '1993-01-01' AND '2020-12-01'
) a
JOIN retail_sales b
  ON b.sales_month BETWEEN a.sales_month - INTERVAL '11 months' AND a.sales_month
 AND b.kind_of_business = 'Women''s clothing stores'
GROUP BY 1
ORDER BY 1;
```

---

## 7. 누적값

**YTD(Year-To-Date)**: 그 해 1월부터 현재 달까지의 합.

```sql
SELECT sales_month, sales
     , sum(sales) OVER (PARTITION BY date_part('year', sales_month)
                        ORDER BY sales_month) AS sales_ytd
FROM retail_sales
WHERE kind_of_business = 'Women''s clothing stores';
```

- `PARTITION BY 연도` → 매년 1월에 0부터 다시 시작
- `ORDER BY`가 들어간 집계 윈도우는 기본 프레임이 **처음 ~ 현재 행** → 자동으로 누적합이 된다

```sql
-- 셀프 조인 버전
SELECT a.sales_month, a.sales
     , sum(b.sales) AS sales_ytd
FROM retail_sales a
JOIN retail_sales b
  ON date_part('year', a.sales_month) = date_part('year', b.sales_month)
 AND b.sales_month <= a.sales_month
 AND b.kind_of_business = 'Women''s clothing stores'
WHERE a.kind_of_business = 'Women''s clothing stores'
GROUP BY 1, 2;
```

| 비교 | ORDER BY 있음 | ORDER BY 없음 |
|---|---|---|
| `sum(x) OVER (PARTITION BY y …)` | 누적합 | 파티션 전체 합 (4장의 전체 대비 비율) |

---

## 8. 계절성 분석: 기간 비교

계절성: 매년 같은 시기에 반복되는 패턴 (예: 서점 매출은 1월·8월 개학 시즌에 급증).

### 8-1. 직전 기간 대비 (MoM, YoY)

```sql
-- lag(): 이전 행 값을 현재 행에 가져온다
SELECT kind_of_business, sales_month, sales
     , lag(sales_month) OVER (PARTITION BY kind_of_business ORDER BY sales_month) AS prev_month
     , lag(sales)       OVER (PARTITION BY kind_of_business ORDER BY sales_month) AS prev_month_sales
FROM retail_sales
WHERE kind_of_business = 'Book stores';
```

```sql
-- MoM 증감률
SELECT kind_of_business, sales_month, sales
     , (sales / lag(sales) OVER (PARTITION BY kind_of_business ORDER BY sales_month) - 1) * 100
       AS pct_growth_from_previous
FROM retail_sales
WHERE kind_of_business = 'Book stores';
```

```sql
-- YoY 증감률 (연 단위 집계 후 lag)
SELECT sales_year, yearly_sales
     , lag(yearly_sales) OVER (ORDER BY sales_year) AS prev_year_sales
     , (yearly_sales / lag(yearly_sales) OVER (ORDER BY sales_year) - 1) * 100 AS pct_growth_from_previous
FROM (
    SELECT date_part('year', sales_month) AS sales_year
         , sum(sales) AS yearly_sales
    FROM retail_sales
    WHERE kind_of_business = 'Book stores'
    GROUP BY 1
) a;
```

| 함수 | 의미 |
|---|---|
| `lag(x)` | 1행 앞의 x |
| `lag(x, N)` | N행 앞의 x |
| `lead(x)` | 1행 뒤의 x |

> ⚠️ 계절성이 강한 데이터에서 MoM은 **계절 효과**까지 섞여 나온다 (1월이 12월보다 높은 건 당연). → 작년 같은 달과 비교해야 함

### 8-2. 작년 같은 달과 비교

**핵심 아이디어**: `PARTITION BY 월` 로 묶으면 같은 월끼리 한 그룹이 되고, 그 안에서 `lag`는 **작년 같은 달**을 가리킨다.

```sql
SELECT sales_month, sales
     , lag(sales_month) OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) AS prev_year_month
     , lag(sales)       OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) AS prev_year_sales
FROM retail_sales
WHERE kind_of_business = 'Book stores';
```

```sql
-- 작년 같은 달 대비 절대 차이와 증감률
SELECT sales_month, sales
     , sales - lag(sales) OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) AS absolute_diff
     , (sales / lag(sales) OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) - 1) * 100 AS pct_diff
FROM retail_sales
WHERE kind_of_business = 'Book stores';
```

**연도를 열로 피벗해서 월별로 나란히 보기**

```sql
SELECT date_part('month', sales_month) AS month_number
     , to_char(sales_month, 'Month')   AS month_name
     , max(CASE WHEN date_part('year', sales_month) = 1992 THEN sales END) AS sales_1992
     , max(CASE WHEN date_part('year', sales_month) = 1993 THEN sales END) AS sales_1993
     , max(CASE WHEN date_part('year', sales_month) = 1994 THEN sales END) AS sales_1994
FROM retail_sales
WHERE kind_of_business = 'Book stores'
  AND sales_month BETWEEN '1992-01-01' AND '1994-12-01'
GROUP BY 1, 2
ORDER BY 1;
```

- 한 칸에 값이 하나뿐이라 `max` 대신 `sum`을 써도 같다 (행 → 열 변환용 집계)
- `to_char(…, 'Month')`는 공백으로 9자리 패딩된다 → 깔끔하게 하려면 `'FMMonth'`

### 8-3. 여러 이전 기간과 비교

작년 한 해만 보면 그 해가 이상치일 수 있다 → **최근 3년 같은 달 평균**과 비교.

```sql
SELECT sales_month, sales
     , sales / ((prev_sales_1 + prev_sales_2 + prev_sales_3) / 3) * 100 AS pct_of_3_prev
FROM (
    SELECT sales_month, sales
         , lag(sales, 1) OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) AS prev_sales_1
         , lag(sales, 2) OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) AS prev_sales_2
         , lag(sales, 3) OVER (PARTITION BY date_part('month', sales_month) ORDER BY sales_month) AS prev_sales_3
    FROM retail_sales
    WHERE kind_of_business = 'Book stores'
) a;
```

```sql
-- 윈도우 프레임으로 한 줄에 ✅
SELECT sales_month, sales
     , sales / avg(sales) OVER (PARTITION BY date_part('month', sales_month)
                                ORDER BY sales_month
                                ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) * 100 AS pct_of_prev_3
FROM retail_sales
WHERE kind_of_business = 'Book stores';
```

- `3 PRECEDING AND 1 PRECEDING` → 현재 행은 **빼고** 직전 3개 (3년 전·2년 전·1년 전 같은 달)
- 결과가 100보다 크면 "최근 3년 같은 달 평균보다 잘 팔렸다"

---

## 9. SQL 문법 메모

| 문법 | 의미 |
|---|---|
| 같은 SELECT 안에서 별칭 재사용 ❌ | `SUM(x) AS total, total / 2` 불가 → 식을 다시 쓰거나 서브쿼리/CTE로 감싸기 |
| ORDER BY에서는 별칭 사용 ⭕ | SELECT보다 **나중에** 실행되기 때문 |
| 정수 나눗셈 | `int / int`는 소수점이 잘린다 → `100.0`, `60000.0`처럼 한쪽을 소수로 |
| 서브쿼리 별칭 | PostgreSQL 15 이하는 `FROM ( … ) a` 처럼 별칭 **필수** (16부터 생략 가능) |
| `NULLIF(b, 0)` | 0으로 나누기 방지 |
| `AVG(x)` | `SUM(x) / COUNT(*)`와 같지만 NULL 행은 분모에서 빠진다 |
| CTE (`WITH`) | 서브쿼리에 이름을 붙여 위로 빼낸 것. 여러 번 참조하거나 단계를 나눌 때 읽기 쉬움 |
| 실행 순서 | FROM → WHERE → GROUP BY → HAVING → **윈도우 함수** → SELECT → ORDER BY |

**별칭 재사용: 식 반복 vs CTE**

```sql
-- ① 식을 다시 쓰기
SELECT count(*) AS cnt
     , sum(x) AS total
     , sum(x) / count(*) AS avg_x
FROM t;

-- ② CTE로 감싸기 (식이 길거나 여러 번 쓸 때)
WITH s AS (
    SELECT count(*) AS cnt, sum(x) AS total
    FROM t
)
SELECT cnt, total, total / cnt AS avg_x
FROM s;
```

> ⚠️ CTE 안에서 `ROUND` 해놓고 바깥에서 다시 나누면 반올림 오차가 쌓인다. 반올림은 **마지막에 한 번만**.

**셀프 조인 vs 윈도우 함수**

| 하고 싶은 것 | 셀프 조인 | 윈도우 함수 |
|---|---|---|
| 전체 대비 비율 | 같은 달끼리 JOIN 후 SUM | `sum() OVER (PARTITION BY 월)` |
| 인덱싱 | 첫 해 합계를 1=1로 조인 | `first_value() OVER (ORDER BY 연도)` |
| 이동평균 | `BETWEEN 월 - 11개월 AND 월` 로 JOIN | `avg() OVER (ROWS BETWEEN 11 PRECEDING …)` |
| 누적합 | `b.월 <= a.월` 로 JOIN | `sum() OVER (ORDER BY 월)` |

→ 대부분 윈도우 함수가 짧고 빠르다. 셀프 조인은 **날짜 범위 기준**이라 빠진 달이 있어도 정확하다는 장점이 있다.
