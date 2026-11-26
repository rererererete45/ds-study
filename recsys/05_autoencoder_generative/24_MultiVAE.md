# 24. MultiVAE

[← MultiDAE](../05_autoencoder_generative/23_MultiDAE.md) · [📚 전체 목차](../README.md) · [MacridVAE →](../05_autoencoder_generative/25_MacridVAE.md)

> 📌 **분류** 일반 추천 · **블록** B5 오토인코더·생성 · **깊이** A 정독 (2주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [MultiVAE](https://recbole.io/docs/user_guide/model/general/multivae.html) · **소스** `recbole/model/general_recommender/multivae.py`
>
> 💡 변분 오토인코더 추천의 기본

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

- 코드: [`code/multivae.py`](../code/multivae.py)
- RecBole 구현과 다른 점:

## 5. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='MultiVAE', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |

- 앞선 모델과 비교:

## 6. 회고

- 막혔던 점:
- 다음 모델로 넘길 질문:
