# 03. ItemKNN

[← Pop](../01_baseline/02_Pop.md) · [📚 전체 목차](../README.md) · [AsymKNN →](../01_baseline/04_AsymKNN.md)

> 📌 **분류** 일반 추천 · **블록** B1 베이스라인 · **깊이** B 보통 (1주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [ItemKNN](https://recbole.io/docs/user_guide/model/general/itemknn.html) · **소스** `recbole/model/general_recommender/itemknn.py`
>
> 🗂️ **추천 데이터** `ml-1m` · `Amazon-Books`
>
> 💡 아이템 유사도 베이스라인

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

run_recbole(model='ItemKNN', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |
| Amazon-Books |  |  |  |  |  |

## 4. 회고

- 다음 모델로 넘길 질문:
