# 92. KGAT

[← KGNNLS](../14_knowledge_graph/91_KGNNLS.md) · [📚 전체 목차](../README.md) · [KGIN →](../14_knowledge_graph/93_KGIN.md)

> 📌 **분류** 지식 그래프 추천 · **블록** B14 지식 그래프 · **깊이** A 정독 (2주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [KGAT](https://recbole.io/docs/user_guide/model/knowledge/kgat.html) · **소스** `recbole/model/knowledge_aware_recommender/kgat.py`
>
> 🗂️ **추천 데이터** `ml-1m(+KG)` · `Amazon-Books(+KG)` · `Lastfm-track(+KG)`
>
> 💡 지식 그래프 추천의 대표 모델

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

- 코드: [`code/kgat.py`](../code/kgat.py)
- RecBole 구현과 다른 점:

## 5. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='KGAT', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m(+KG) |  |  |  |  |  |
| Amazon-Books(+KG) |  |  |  |  |  |
| Lastfm-track(+KG) |  |  |  |  |  |

- 앞선 모델과 비교:

## 6. 회고

- 막혔던 점:
- 다음 모델로 넘길 질문:
