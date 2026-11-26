# 17. NeuMF

[← GRU4Rec](../02_sequential_core/16_GRU4Rec.md) · [📚 전체 목차](../README.md) · [DMF →](../04_neural_cf/18_DMF.md)

> 📌 **분류** 일반 추천 · **블록** B4 신경망 CF · **깊이** A 정독 (2주) · **상태** ⬜ 예정
>
> 📄 **RecBole 문서** [NeuMF](https://recbole.io/docs/user_guide/model/general/neumf.html) · **소스** `recbole/model/general_recommender/neumf.py`
>
> 💡 신경망 협업 필터링의 기본

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

- 코드: [`code/neumf.py`](../code/neumf.py)
- RecBole 구현과 다른 점:

## 5. 실험

```python
from recbole.quick_start import run_recbole

run_recbole(model='NeuMF', dataset='ml-1m')
```

| 데이터 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 학습 시간 |
|---|---|---|---|---|---|
| ml-1m |  |  |  |  |  |

- 앞선 모델과 비교:

## 6. 회고

- 막혔던 점:
- 다음 모델로 넘길 질문:
