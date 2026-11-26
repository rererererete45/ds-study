# 07. EASE

[← SASRec](../02_sequential_core/06_SASRec.md) · [📚 전체 목차](../README.md) · [SLIMElastic →](../03_mf_similarity/08_SLIMElastic.md)

> 📌 **분류** 일반 추천 · **블록** B3 행렬 분해·유사도 · **깊이** A 정독 (2주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [EASE](https://recbole.io/docs/user_guide/model/general/ease.html) · **소스** `recbole/model/general_recommender/ease.py`
>
> 🗂️ **추천 데이터** `ml-1m` · `Netflix(샘플)` · `Amazon-Books`
>
> 💡 닫힌 형태 해를 갖는 강한 선형 모델

## 0. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 논문 | 제목 · 학회 · 연도 |
| 한 줄 요약 |  |
| 이전 모델과 다른 점 |  |
| 입력 데이터 |  |

## 1. 논문 요약

### 풀려는 문제

### 핵심 아이디어

## 2. 핵심 수식

### 모델 구조

### 손실 함수

## 3. RecBole 코드 해설

| 수식·개념 | 코드 위치 | 설명 |
|---|---|---|
|  |  |  |

## 4. 직접 구현 (파이토치)

- 코드: [`code/ease.py`](../code/ease.py)
- RecBole 구현과 다른 점:

## 5. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='EASE', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |
| Netflix(샘플) |  |  |  |  |  |
| Amazon-Books |  |  |  |  |  |

- 앞선 모델과 비교:

## 6. 회고

- 막혔던 점:
- 다음 모델로 넘길 질문:
