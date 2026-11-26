# 69. FEARec

[← CORE](../08_transformer_ssl/68_CORE.md) · [📚 전체 목차](../README.md) · [SINE →](../08_transformer_ssl/70_SINE.md)

> 📌 **분류** 순차 추천 · **블록** B8 트랜스포머·자기지도 · **깊이** B 보통 (1주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [FEARec](https://recbole.io/docs/user_guide/model/sequential/fearec.html) · **소스** `recbole/model/sequential_recommender/fearec.py`
>
> 🗂️ **추천 데이터** `ml-1m` · `Amazon-Beauty`

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

## 2. RecBole 코드 해설

| 수식·개념 | 코드 위치 | 설명 |
|---|---|---|
|  |  |  |

## 3. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='FEARec', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |
| Amazon-Beauty |  |  |  |  |  |

## 4. 회고

- 다음 모델로 넘길 질문:
