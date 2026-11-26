# 16. GRU4Rec

[← FPMC](../02_sequential_core/15_FPMC.md) · [📚 전체 목차](../README.md) · [NeuMF →](../04_neural_cf/17_NeuMF.md)

> 📌 **분류** 순차 추천 · **블록** B2 순차 핵심 · **깊이** A 정독 (2주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [GRU4Rec](https://recbole.io/docs/user_guide/model/sequential/gru4rec.html) · **소스** `recbole/model/sequential_recommender/gru4rec.py`
>
> 🗂️ **추천 데이터** `ml-1m` · `YOOCHOOSE(1/64)` · `DIGINETICA`
>
> 💡 RNN 기반 세션 추천

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

- 코드: [`code/gru4rec.py`](../code/gru4rec.py)
- RecBole 구현과 다른 점:

## 5. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='GRU4Rec', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |
| YOOCHOOSE(1/64) |  |  |  |  |  |
| DIGINETICA |  |  |  |  |  |

- 앞선 모델과 비교:

## 6. 회고

- 막혔던 점:
- 다음 모델로 넘길 질문:
