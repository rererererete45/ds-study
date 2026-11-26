# 56. WideDeep

[← PNN](../11_ctr_basic/55_PNN.md) · [📚 전체 목차](../README.md) · [NFM →](../12_ctr_deep/57_NFM.md)

> 📌 **분류** 컨텍스트(CTR) 추천 · **블록** B12 CTR 딥러닝 · **깊이** A 정독 (2주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [WideDeep](https://recbole.io/docs/user_guide/model/context/widedeep.html) · **소스** `recbole/model/context_aware_recommender/widedeep.py`
>
> 🗂️ **추천 데이터** `ml-1m` · `Criteo(샘플)` · `Avazu(샘플)`
>
> 💡 기억(Wide) + 일반화(Deep) 구조

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

- 코드: [`code/widedeep.py`](../code/widedeep.py)
- RecBole 구현과 다른 점:

## 5. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='WideDeep', dataset='ml-1m')
```

| 데이터 | AUC | LogLoss | 학습 시간 | 비고 |
|---|---|---|---|---|
| ml-1m |  |  |  |  |
| Criteo(샘플) |  |  |  |  |
| Avazu(샘플) |  |  |  |  |

- 앞선 모델과 비교:

## 6. 회고

- 막혔던 점:
- 다음 모델로 넘길 질문:
