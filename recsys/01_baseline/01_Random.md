# 01. Random

[📚 전체 목차](../README.md) · [Pop →](../01_baseline/02_Pop.md)

> 📌 **분류** 일반 추천 · **블록** B1 베이스라인 · **깊이** C 가볍게 (0.5주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [Random](https://recbole.io/docs/user_guide/model/general/random.html) · **소스** `recbole/model/general_recommender/random.py`
>
> 💡 성능 하한선. 평가 지표와 RecBole 실행 흐름 익히기

## 요약

(한 문단: 무엇을 풀었고, 핵심 아이디어는 무엇이며, 이전 모델과 무엇이 다른가)

## 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='Random', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |

## 메모

-
