# 02. Pop

[← Random](../01_baseline/01_Random.md) · [📚 전체 목차](../README.md) · [ItemKNN →](../01_baseline/03_ItemKNN.md)

> 📌 **분류** 일반 추천 · **블록** B1 베이스라인 · **깊이** C 가볍게 (0.5주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [Pop](https://recbole.io/docs/user_guide/model/general/pop.html) · **소스** `recbole/model/general_recommender/pop.py`
>
> 🗂️ **추천 데이터** `ml-1m`
>
> 💡 인기도 베이스라인

## 요약

(한 문단: 무엇을 풀었고, 핵심 아이디어는 무엇이며, 이전 모델과 무엇이 다른가)

## 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='Pop', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |

## 메모

-
