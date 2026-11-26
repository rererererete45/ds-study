# 20. NNCF

[← ConvNCF](../04_neural_cf/19_ConvNCF.md) · [📚 전체 목차](../README.md) · [SimpleX →](../04_neural_cf/21_SimpleX.md)

> 📌 **분류** 일반 추천 · **블록** B4 신경망 CF · **깊이** C 가볍게 (0.5주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [NNCF](https://recbole.io/docs/user_guide/model/general/nncf.html) · **소스** `recbole/model/general_recommender/nncf.py`
>
> 🗂️ **추천 데이터** `ml-1m`

## 요약

(한 문단: 무엇을 풀었고, 핵심 아이디어는 무엇이며, 이전 모델과 무엇이 다른가)

## 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='NNCF', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |

## 메모

-
