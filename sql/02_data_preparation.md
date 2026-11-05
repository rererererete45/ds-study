# 2장. 데이터 준비

활용 사례는 나의 Velog를 확인해주세요~~
url:

> 분석 전에 데이터를 **이해하고(프로파일링) → 정제하고 → 원하는 모양으로 바꾸는(셰이핑)** 과정

## 목차

1. [데이터 타입](#1-데이터-타입)
2. [프로파일링: 데이터 분포](#2-프로파일링-데이터-분포)
3. [프로파일링: 데이터 품질](#3-프로파일링-데이터-품질)
4. [준비: 데이터 정제](#4-준비-데이터-정제)
5. [준비: 데이터 셰이핑](#5-준비-데이터-셰이핑)
6. [SQL 문법 메모](#6-sql-문법-메모)

---

## 1. 데이터 타입

### 측정 방식에 따른 구분

| 구분 | 설명 | 예시 |
|---|---|---|
| 정량 데이터 | 특정 수치로 정량화한 데이터 | 주문 금액, 방문 횟수 |
| 정성 데이터 | 명확한 수치로 측정할 수 없는 데이터 | 리뷰 텍스트, 설문 서술형 응답 |

### 출처에 따른 구분

| 구분 | 설명 |
|---|---|
| 퍼스트 파티 | 기관이 **직접** 수집한 데이터 |
| 세컨드 파티 | 기관에 서비스를 제공하는 **업체가** 수집한 데이터 |
| 서드 파티 | 비용을 내고 **구매**한 데이터, 또는 정부 공개 데이터 등 외부에서 무료로 얻은 데이터 |

### 희소 데이터 (Sparse Data)

빈 값이나 중요하지 않은 정보가 많아, 크기에 비해 의미 있는 정보가 적은 데이터.

---

## 2. 프로파일링: 데이터 분포

데이터셋을 다룰 때 **가장 먼저 하는 일**. 탐색적 데이터 분석(EDA)과 연결된다.

### 2-1. 히스토그램과 빈도

**① 값별 빈도**

```sql
SELECT fruit, count(*) AS quantity
FROM fruit_inventory
GROUP BY fruit;
```

**② 빈도의 빈도 — "주문을 N번 한 고객은 몇 명인가?"**

```sql
SELECT order_cnt, count(*) AS num_customers
FROM (
    -- 1단계: 고객별 주문 수 (고객 1명 = 1행)
    SELECT customer_id, count(order_id) AS order_cnt
    FROM orders
    GROUP BY customer_id
) per_customer
-- 2단계: 주문 수가 같은 고객끼리 묶어서 고객 수 집계
GROUP BY order_cnt
ORDER BY order_cnt;
```

동작 예시

| 원본 (orders) | → | 1단계 결과 | → | 최종 결과 |
|---|---|---|---|---|
| A, A, B, C, C, D | | A=2, B=1, C=2, D=1 | | 1회: 2명 / 2회: 2명 |

> 💡 안쪽 컬럼 별칭을 테이블명과 같은 `orders`로 지으면 헷갈린다. `order_cnt`처럼 구분되게 짓자.

### 2-2. 구간화 (Binning)

연속값을 프로파일링할 때 유용. 사람이 **기준값을 직접 정해서** 구간을 나눈다.

**CASE 문 구조**

```sql
CASE
    WHEN condition_1 THEN return_value_1
    WHEN condition_2 THEN return_value_2
    ...
    ELSE return_value_default
END
```

- 위에서부터 순서대로 검사하고, **처음 만족하는 조건**의 값을 반환
- 그래서 `<= 100` 다음 `<= 500`은 자동으로 "100 초과 500 이하"가 됨

```sql
SELECT
    CASE
        WHEN order_amount <= 100 THEN 'up to 100'
        WHEN order_amount <= 500 THEN '100 - 500'
        ELSE '500+'
    END AS amount_bin,
    CASE
        WHEN order_amount <= 100 THEN 'small'
        WHEN order_amount <= 500 THEN 'medium'
        ELSE 'large'
    END AS amount_category,
    count(customer_id) AS orders
FROM orders
GROUP BY 1, 2;
```

- `GROUP BY 1, 2` = SELECT의 1번째·2번째 컬럼(`amount_bin`, `amount_category`) **조합**으로 묶기
- 두 CASE의 기준이 같으므로 결과는 3행 (기준이 다르면 조합이 더 쪼개짐)
- 집계 함수가 아닌 SELECT 컬럼은 **모두** GROUP BY에 있어야 하므로 2번도 넣어야 함

> ⚠️ `count(customer_id)`는 **주문 건수**를 센다. 고객 수를 원하면 `count(DISTINCT customer_id)`.

### 2-3. N분위수 (NTILE)

데이터를 **순서대로 줄 세워 N등분**한다. 구간화와 달리 기준값을 데이터가 정해준다.

**윈도우 함수 구조**

```sql
function(컬럼) OVER (PARTITION BY 컬럼 ORDER BY 컬럼)
```

- `function`: count, sum, avg, min, max, rank, first_value, ntile 등
- `PARTITION BY`: 그룹을 나눠서 그룹 안에서 계산 (생략 시 전체)
- `ORDER BY`: 그룹 안에서의 정렬 기준
- GROUP BY와 달리 **행을 줄이지 않고** 각 행 옆에 계산값을 붙인다

```sql
SELECT decile,
       min(order_amount) AS lower_bound,
       max(order_amount) AS upper_bound,
       count(order_id)   AS orders
FROM (
    -- 1단계: 주문 금액 순으로 10등분해서 1~10 번호 부여
    SELECT order_id, order_amount,
           ntile(10) OVER (ORDER BY order_amount) AS decile
    FROM orders
) a
-- 2단계: 구간별 최소·최대 금액과 건수
GROUP BY decile
ORDER BY decile;
```

| 비교 | 구간화 (CASE) | N분위수 (NTILE) |
|---|---|---|
| 구간 기준 | 사람이 정한 값 (100, 500) | 데이터 순서 |
| 구간별 건수 | 제각각 | 거의 균등 |

> ⚠️ 주의
> - 건수가 N으로 안 나누어떨어지면 **앞 번호부터 1건씩 더** 들어간다 (23건 → 3,3,3,2,2,…)
> - 같은 금액이 경계에 걸리면 **다른 구간으로 갈라질 수 있다** (값이 아니라 순서로 자르기 때문)
> - 별칭을 `ntile`처럼 함수명과 같게 지으면 DB에 따라 에러가 날 수 있다

---

## 3. 프로파일링: 데이터 품질

### 3-1. 중복 탐지

```sql
-- 중복된 조합이 몇 개나 있는지
SELECT count(*) AS duplicated_groups
FROM (
    SELECT column_a, column_b, column_c, count(*) AS records
    FROM some_table
    GROUP BY 1, 2, 3
) a
WHERE records > 1;
```

```sql
-- 어떤 조합이 몇 번씩 중복됐는지 직접 보기
SELECT column_a, column_b, column_c, count(*) AS records
FROM some_table
GROUP BY 1, 2, 3
HAVING count(*) > 1
ORDER BY records DESC;
```

- `WHERE`는 그룹화 **전** 행 필터, `HAVING`은 그룹화 **후** 집계 결과 필터

### 3-2. 중복 제거

JOIN 결과로 같은 고객이 여러 번 나올 때(1:N 조인) 사용.

```sql
-- DISTINCT
SELECT DISTINCT a.customer_id, a.customer_name, a.customer_email
FROM customers a
JOIN transactions b ON a.customer_id = b.customer_id;

-- GROUP BY (결과 동일)
SELECT a.customer_id, a.customer_name, a.customer_email
FROM customers a
JOIN transactions b ON a.customer_id = b.customer_id
GROUP BY 1, 2, 3;
```

---

## 4. 준비: 데이터 정제

### 4-1. CASE 변환

값을 표준화하거나 범주를 다시 묶을 때.

```sql
SELECT
    CASE
        WHEN gender IN ('M', 'male', '남') THEN 'Male'
        WHEN gender IN ('F', 'female', '여') THEN 'Female'
        ELSE 'Unknown'
    END AS gender_clean
FROM users;
```

### 4-2. 타입 변환과 캐스팅

```sql
CAST(order_amount AS integer)   -- 표준 SQL
order_amount::integer           -- PostgreSQL 축약형
CAST('2026-11-03' AS date)
```

### 4-3. NULL 값 다루기

| 함수 | 동작 | 예시 |
|---|---|---|
| `coalesce(a, b, …)` | 앞에서부터 처음으로 NULL이 **아닌** 값 반환 | `coalesce(num_orders, 0)` → NULL이면 0 |
| `nullif(a, b)` | a와 b가 같으면 NULL, 다르면 a | `nullif(date, '1970-01-01')` → 기본값을 NULL로 |

```sql
-- 0으로 나누기 방지에도 nullif를 자주 쓴다
SELECT revenue / nullif(orders, 0) AS avg_order_value
FROM summary;
```

> ⚠️ NULL은 `= NULL`이 아니라 `IS NULL` / `IS NOT NULL`로 비교한다.

### 4-4. 결측 데이터

- 결측 여부 확인: `count(*) - count(column)` = 해당 컬럼의 NULL 개수
- 처리 방법: 행 제외, 기본값 대체(`coalesce`), 평균·이전값 등으로 보간

---

## 5. 준비: 데이터 셰이핑

분석·시각화에 맞게 **행과 열의 모양을 바꾸는** 작업.

### 5-1. CASE 문을 활용한 피벗 (행 → 열)

```sql
SELECT order_date,
       sum(CASE WHEN product = 'shirt' THEN order_amount ELSE 0 END) AS shirt_amount,
       sum(CASE WHEN product = 'shoes' THEN order_amount ELSE 0 END) AS shoes_amount
FROM orders
GROUP BY order_date;
```

### 5-2. UNION 문을 활용한 언피벗 (열 → 행)

```sql
SELECT country, '2024' AS year, pop_2024 AS population FROM country_pop
UNION ALL
SELECT country, '2025' AS year, pop_2025 AS population FROM country_pop
UNION ALL
SELECT country, '2026' AS year, pop_2026 AS population FROM country_pop;
```

- `UNION`은 중복 제거, `UNION ALL`은 그대로 이어붙임 (보통 `UNION ALL`이 빠름)

### 5-3. PIVOT / UNPIVOT 함수

일부 DB(SQL Server, Oracle, Snowflake 등)는 전용 함수를 지원. 지원 여부가 DB마다 달라서 CASE·UNION 방식을 먼저 익혀두는 게 좋다.

---

## 6. SQL 문법 메모

| 문법 | 의미 |
|---|---|
| `GROUP BY 1, 2` | SELECT 절 1·2번째 컬럼으로 묶기 (컬럼명으로 쓰는 게 더 읽기 쉬움) |
| 서브쿼리 `( … ) a` | 쿼리 결과를 임시 테이블 `a`처럼 사용 |
| 쿼리 읽는 순서 | **안쪽 서브쿼리 → 바깥 쿼리** 순으로 결과 테이블을 상상하며 읽기 |
| 실행 순서 | FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY |
