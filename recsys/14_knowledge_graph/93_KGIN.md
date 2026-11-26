# 93. KGIN

[← KGAT](../14_knowledge_graph/92_KGAT.md) · [📚 전체 목차](../README.md) · [MCCLK →](../14_knowledge_graph/94_MCCLK.md)

> 📌 **분류** 지식 그래프 추천 · **블록** B14 지식 그래프 · **깊이** B 보통 (1주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [KGIN](https://recbole.io/docs/user_guide/model/knowledge/kgin.html) · **소스** `recbole/model/knowledge_aware_recommender/kgin.py`
>
> 🗂️ **추천 데이터** `ml-1m(+KG)` · `Amazon-Books(+KG)`

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

run_recbole(model='KGIN', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m(+KG) |  |  |  |  |  |
| Amazon-Books(+KG) |  |  |  |  |  |

## 4. 회고

- 다음 모델로 넘길 질문:
