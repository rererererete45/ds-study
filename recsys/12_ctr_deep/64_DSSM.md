# 64. DSSM

[← AutoInt](../12_ctr_deep/63_AutoInt.md) · [📚 전체 목차](../README.md) · [BERT4Rec →](../08_transformer_ssl/65_BERT4Rec.md)

> 📌 **분류** 컨텍스트(CTR) 추천 · **블록** B12 CTR 딥러닝 · **깊이** B 보통 (1주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [DSSM](https://recbole.io/docs/user_guide/model/context/dssm.html) · **소스** `recbole/model/context_aware_recommender/dssm.py`
>
> 🗂️ **추천 데이터** `ml-1m` · `MIND`
>
> 💡 투 타워 구조. 후보 생성 단계의 기본

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

run_recbole(model='DSSM', dataset='ml-1m')
```

| 데이터 | AUC | LogLoss | 학습 시간 | 비고 |
|---|---|---|---|---|
| ml-1m |  |  |  |  |
| MIND |  |  |  |  |

## 4. 회고

- 다음 모델로 넘길 질문:
