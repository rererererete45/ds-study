# 49. XGBOOST

[← LR](../11_ctr_basic/48_LR.md) · [📚 전체 목차](../README.md) · [LIGHTGBM →](../11_ctr_basic/50_LIGHTGBM.md)

> 📌 **분류** 컨텍스트(CTR) 추천 · **블록** B11 CTR 기초 · **깊이** C 가볍게 (0.5주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [XGBOOST](https://recbole.io/docs/user_guide/model/context/xgboost.html) · **소스** `recbole/model/exlib_recommender/xgboost.py`
>
> 🗂️ **추천 데이터** `ml-1m`
>
> 💡 외부 라이브러리(XGBoost). 트리 모델 비교 기준. 실행 이름·설정은 RecBole 문서 확인

## 요약

(한 문단: 무엇을 풀었고, 핵심 아이디어는 무엇이며, 이전 모델과 무엇이 다른가)

## 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='XGBOOST', dataset='ml-1m')
```

| 데이터 | AUC | LogLoss | 학습 시간 | 비고 |
|---|---|---|---|---|
| ml-1m |  |  |  |  |

## 메모

-
