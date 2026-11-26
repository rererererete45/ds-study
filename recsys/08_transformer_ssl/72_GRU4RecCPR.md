# 72. GRU4RecCPR

[← SASRecCPR](../08_transformer_ssl/71_SASRecCPR.md) · [📚 전체 목차](../README.md) · [SRGNN →](../09_graph_session/73_SRGNN.md)

> 📌 **분류** 순차 추천 · **블록** B8 트랜스포머·자기지도 · **깊이** C 가볍게 (0.5주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [GRU4RecCPR](https://recbole.io/docs/user_guide/model/sequential/gru4reccpr.html) · **소스** `recbole/model/sequential_recommender/gru4reccpr.py`

## 요약

(한 문단: 무엇을 풀었고, 핵심 아이디어는 무엇이며, 이전 모델과 무엇이 다른가)

## 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='GRU4RecCPR', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |

## 메모

-
