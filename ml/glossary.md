# 용어집

[📚 전체 목차](../README.md)

> 노트에 나오는 핵심 용어를 주제별로 모았다. **뜻**은 한 줄 요약이고, 자세한 설명·코드·그림은 **처음 설명하는 곳** 링크로 이동해서 본다.
> 모르는 말이 나오면 이 페이지에서 `Ctrl + F`로 찾아보면 된다.

## 목차

- [1. 기본 개념과 데이터 (1~2장)](#1-기본-개념과-데이터-12장) — 36개
- [2. 전처리와 사이킷런 (2장)](#2-전처리와-사이킷런-2장) — 13개
- [3. 분류와 평가 지표 (3장)](#3-분류와-평가-지표-3장) — 13개
- [4. 선형 모델과 최적화 (4장)](#4-선형-모델과-최적화-4장) — 24개
- [5. 트리와 앙상블 (5~6장)](#5-트리와-앙상블-56장) — 17개
- [6. 차원 축소와 비지도 학습 (7~8장)](#6-차원-축소와-비지도-학습-78장) — 21개
- [7. 신경망 기초와 파이토치 (9~10장)](#7-신경망-기초와-파이토치-910장) — 14개
- [8. 심층 신경망 훈련 (11장)](#8-심층-신경망-훈련-11장) — 16개
- [9. CNN과 컴퓨터 비전 (12장)](#9-cnn과-컴퓨터-비전-12장) — 13개
- [10. 시퀀스와 RNN (13장)](#10-시퀀스와-rnn-13장) — 11개
- [11. 자연어 처리와 어텐션 (14장)](#11-자연어-처리와-어텐션-14장) — 10개
- [12. 트랜스포머 (15~16장)](#12-트랜스포머-1516장) — 21개
- [13. 서포트 벡터 머신 (부록 C)](#13-서포트-벡터-머신-부록-c) — 8개


## 1. 기본 개념과 데이터 (1~2장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **머신러닝** | machine learning | 명시적인 규칙 없이 데이터에서 패턴을 배워 작업 성능이 좋아지는 시스템을 만드는 분야 | [1장 1-1. 정의](01_the_machine_learning_landscape.md#1-1-정의) |
| **샘플** | sample, instance | 데이터의 한 행. 예측의 단위(집 한 채, 이미지 한 장) | [1장 1-2. 용어](01_the_machine_learning_landscape.md#1-2-용어) |
| **특성** | feature | 샘플을 설명하는 입력 값 하나(열 하나). 모델의 X | [1장 1-2. 용어](01_the_machine_learning_landscape.md#1-2-용어) |
| **레이블** | label | 샘플의 정답. 분류에서는 클래스, 회귀에서는 타깃 값 | [1장 1-2. 용어](01_the_machine_learning_landscape.md#1-2-용어) |
| **타깃** | target | 모델이 예측해야 하는 값(y). 회귀에서 레이블을 부르는 말 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **훈련 세트** | training set | 모델이 학습에 쓰는 데이터 | [1장 1-2. 용어](01_the_machine_learning_landscape.md#1-2-용어) |
| **지도 학습** | supervised learning | 레이블이 있는 데이터로 입력 → 정답의 관계를 배우는 방식 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **비지도 학습** | unsupervised learning | 레이블 없이 데이터의 구조(군집, 저차원 구조, 밀도)를 찾는 방식 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **준지도 학습** | semi-supervised learning | 레이블이 일부에만 있는 데이터로 학습하는 방식 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **자기 지도 학습** | self-supervised learning | 데이터 자체에서 레이블을 만들어(가리고 맞히기 등) 학습하는 방식. 사전 훈련에 많이 쓴다 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **강화 학습** | reinforcement learning | 에이전트가 행동 → 보상을 반복하며 보상을 최대화하는 정책을 배우는 방식 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **분류** | classification | 샘플이 어느 클래스(범주)에 속하는지 예측하는 작업 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **회귀** | regression | 숫자 값(타깃)을 예측하는 작업 | [1장 4-1. 훈련 지도 방식](01_the_machine_learning_landscape.md#4-1-훈련-지도-방식) |
| **배치 학습** | batch learning | 가진 데이터 전부로 한 번에 훈련하고, 새 데이터가 생기면 처음부터 다시 훈련하는 방식 | [1장 4-2. 배치 학습 vs 온라인 학습](01_the_machine_learning_landscape.md#4-2-배치-학습-vs-온라인-학습) |
| **온라인 학습** | online learning | 데이터를 하나씩(또는 미니배치로) 넣어 가며 점진적으로 학습하는 방식 | [1장 4-2. 배치 학습 vs 온라인 학습](01_the_machine_learning_landscape.md#4-2-배치-학습-vs-온라인-학습) |
| **학습률** | learning rate | 한 번의 갱신에서 파라미터를 얼마나 크게 바꿀지 정하는 값. 크면 빠르지만 불안정, 작으면 안정적이지만 느림 | [4장 2-3. 학습률의 영향](04_training_linear_models.md#2-3-학습률의-영향) |
| **사례 기반 학습** | instance-based learning | 훈련 샘플을 기억해 두고, 새 샘플과 비슷한 샘플들로 예측하는 방식(예: k-최근접 이웃) | [1장 4-3. 사례 기반 학습 vs 모델 기반 학습](01_the_machine_learning_landscape.md#4-3-사례-기반-학습-vs-모델-기반-학습) |
| **모델 기반 학습** | model-based learning | 데이터로 수식(모델)의 파라미터를 학습하고 그 수식으로 예측하는 방식 | [1장 4-3. 사례 기반 학습 vs 모델 기반 학습](01_the_machine_learning_landscape.md#4-3-사례-기반-학습-vs-모델-기반-학습) |
| **모델 파라미터** | model parameter | 훈련으로 정해지는 모델 내부 값(예: 선형 모델의 기울기·절편, 신경망의 가중치) | [1장 4-3. 사례 기반 학습 vs 모델 기반 학습](01_the_machine_learning_landscape.md#4-3-사례-기반-학습-vs-모델-기반-학습) |
| **하이퍼파라미터** | hyperparameter | 훈련 전에 사람이 정하는 학습 알고리즘의 설정값(예: 규제 강도, 학습률, 트리 깊이) | [1장 5-5. 훈련 데이터 과대적합](01_the_machine_learning_landscape.md#5-5-훈련-데이터-과대적합) |
| **비용 함수** | cost function | 모델이 얼마나 나쁜지 재는 함수. 훈련 = 비용 함수를 최소화하는 파라미터 찾기 | [1장 4-3. 사례 기반 학습 vs 모델 기반 학습](01_the_machine_learning_landscape.md#4-3-사례-기반-학습-vs-모델-기반-학습) |
| **일반화** | generalization | 훈련 때 보지 못한 새 샘플에서도 잘 예측하는 것 | [1장 4-3. 사례 기반 학습 vs 모델 기반 학습](01_the_machine_learning_landscape.md#4-3-사례-기반-학습-vs-모델-기반-학습) |
| **일반화 오차** | generalization error | 새 샘플에 대한 오차. 테스트 세트로 추정한다 | [1장 6-1. 훈련 세트와 테스트 세트](01_the_machine_learning_landscape.md#6-1-훈련-세트와-테스트-세트) |
| **과대적합** | overfitting | 모델이 훈련 데이터의 잡음까지 외워, 훈련 성능은 좋은데 새 데이터 성능이 나쁜 상태 | [1장 5-5. 훈련 데이터 과대적합](01_the_machine_learning_landscape.md#5-5-훈련-데이터-과대적합) |
| **과소적합** | underfitting | 모델이 너무 단순해 훈련 데이터의 구조조차 못 배운 상태 | [1장 5-6. 훈련 데이터 과소적합](01_the_machine_learning_landscape.md#5-6-훈련-데이터-과소적합) |
| **규제** | regularization | 모델의 자유도를 제한해 과대적합을 줄이는 모든 방법(가중치 벌점, 트리 깊이 제한, 드롭아웃 등) | [1장 5-5. 훈련 데이터 과대적합](01_the_machine_learning_landscape.md#5-5-훈련-데이터-과대적합) |
| **테스트 세트** | test set | 최종 모델의 일반화 오차를 추정하려고 마지막까지 떼어 두는 데이터 | [1장 6-1. 훈련 세트와 테스트 세트](01_the_machine_learning_landscape.md#6-1-훈련-세트와-테스트-세트) |
| **검증 세트** | validation set, dev set | 여러 모델·하이퍼파라미터 중 무엇을 고를지 비교하는 데 쓰는 데이터 | [1장 6-2. 하이퍼파라미터 튜닝과 모델 선택](01_the_machine_learning_landscape.md#6-2-하이퍼파라미터-튜닝과-모델-선택) |
| **교차 검증** | cross-validation | 데이터를 k개 폴드로 나눠, 폴드를 돌아가며 검증에 쓰고 점수를 평균하는 방법 | [2장 5-2. 교차 검증](02_end_to_end_machine_learning_project.md#5-2-교차-검증) |
| **훈련-개발 세트** | train-dev set | 훈련 데이터 일부를 떼어 둔 세트. 과대적합과 데이터 불일치를 구분할 때 쓴다 | [1장 6-4. 데이터 불일치](01_the_machine_learning_landscape.md#6-4-데이터-불일치) |
| **데이터 불일치** | data mismatch | 훈련 데이터와 실서비스 데이터의 분포가 달라 생기는 성능 저하 | [1장 6-4. 데이터 불일치](01_the_machine_learning_landscape.md#6-4-데이터-불일치) |
| **샘플링 편향** | sampling bias | 표본을 뽑는 방법이 잘못돼 데이터가 모집단을 대표하지 못하는 것 | [1장 5-2. 대표성 없는 훈련 데이터](01_the_machine_learning_landscape.md#5-2-대표성-없는-훈련-데이터) |
| **계층 샘플링** | stratified sampling | 중요한 특성의 구간(계층)마다 비율에 맞게 뽑아 분포를 보존하는 샘플링 | [2장 2-6. 계층 샘플링](02_end_to_end_machine_learning_project.md#2-6-계층-샘플링) |
| **데이터 스누핑 편향** | data snooping bias | 테스트 세트를 보고 모델을 고르는 바람에 성능을 낙관적으로 추정하게 되는 편향 | [2장 2-3. 테스트 세트 분리 원칙](02_end_to_end_machine_learning_project.md#2-3-테스트-세트-분리-원칙) |
| **특성 공학** | feature engineering | 좋은 특성을 고르고(선택), 합치고(추출), 새로 만드는(생성) 작업 | [1장 5-4. 관련 없는 특성](01_the_machine_learning_landscape.md#5-4-관련-없는-특성) |
| **k-최근접 이웃 (KNN)** | k-nearest neighbors | 새 샘플과 가장 가까운 훈련 샘플 k개의 레이블(평균 또는 다수결)로 예측하는 사례 기반 모델 | [1장 4-3. 사례 기반 학습 vs 모델 기반 학습](01_the_machine_learning_landscape.md#4-3-사례-기반-학습-vs-모델-기반-학습) |

## 2. 전처리와 사이킷런 (2장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **상관계수** | correlation coefficient | 두 변수의 선형 관계 강도(−1~1). 비선형 관계는 잡지 못한다 | [2장 3-2. 상관관계 분석](02_end_to_end_machine_learning_project.md#3-2-상관관계-분석) |
| **결측치 대체** | imputation | 빠진 값을 중앙값·평균 등으로 채우는 것 (`SimpleImputer`) | [2장 4-2. 결측치 처리](02_end_to_end_machine_learning_project.md#4-2-결측치-처리) |
| **원-핫 인코딩** | one-hot encoding | 범주 하나를 '해당 범주만 1, 나머지 0'인 이진 특성 여러 개로 바꾸는 방법 | [2장 4-3. 범주형 특성 인코딩](02_end_to_end_machine_learning_project.md#4-3-범주형-특성-인코딩) |
| **표준화** | standardization | 평균을 빼고 표준편차로 나눠 평균 0, 표준편차 1로 만드는 스케일링 (`StandardScaler`) | [2장 4-4. 특성 스케일링](02_end_to_end_machine_learning_project.md#4-4-특성-스케일링) |
| **min-max 스케일링** | min-max scaling | 최솟값·최댓값을 기준으로 값을 정해진 범위(보통 0~1)로 옮기는 스케일링 | [2장 4-4. 특성 스케일링](02_end_to_end_machine_learning_project.md#4-4-특성-스케일링) |
| **추정기** | estimator | `fit()`으로 데이터에서 무언가를 학습하는 모든 사이킷런 객체 | [2장 4-1. 사이킷런 API 설계](02_end_to_end_machine_learning_project.md#4-1-사이킷런-api-설계) |
| **변환기** | transformer | `transform()`으로 데이터를 바꾸는 추정기(스케일러, 인코더 등). 트랜스포머 모델과는 다른 뜻 | [2장 4-1. 사이킷런 API 설계](02_end_to_end_machine_learning_project.md#4-1-사이킷런-api-설계) |
| **예측기** | predictor | `predict()`로 예측하는 추정기(모델) | [2장 4-1. 사이킷런 API 설계](02_end_to_end_machine_learning_project.md#4-1-사이킷런-api-설계) |
| **파이프라인** | pipeline | 전처리 단계와 모델을 순서대로 묶어 하나의 추정기처럼 쓰는 객체 (`Pipeline`) | [2장 4-8. 변환 파이프라인](02_end_to_end_machine_learning_project.md#4-8-변환-파이프라인) |
| **RMSE** | root mean squared error | 오차 제곱의 평균에 제곱근을 씌운 회귀 지표. 큰 오차에 민감 | [2장 1-2. 성능 지표](02_end_to_end_machine_learning_project.md#1-2-성능-지표) |
| **MAE** | mean absolute error | 오차 절댓값의 평균. 이상치에 RMSE보다 덜 민감 | [2장 1-2. 성능 지표](02_end_to_end_machine_learning_project.md#1-2-성능-지표) |
| **그리드 서치** | grid search | 하이퍼파라미터 후보의 모든 조합을 교차 검증으로 평가하는 탐색 | [2장 6-1. 그리드 서치](02_end_to_end_machine_learning_project.md#6-1-그리드-서치) |
| **랜덤 서치** | randomized search | 하이퍼파라미터를 분포에서 무작위로 뽑아 정해진 횟수만 평가하는 탐색 | [2장 6-2. 랜덤 서치](02_end_to_end_machine_learning_project.md#6-2-랜덤-서치) |

## 3. 분류와 평가 지표 (3장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **오차 행렬** | confusion matrix | 실제 클래스(행) × 예측 클래스(열) 조합별 샘플 수를 센 표 | [3장 3-3. 오차 행렬](03_classification.md#3-3-오차-행렬) |
| **정확도** | accuracy | 전체 중 맞힌 비율. 클래스가 불균형하면 쓸모없을 수 있다 | [3장 3-1. 교차 검증으로 정확도 측정](03_classification.md#3-1-교차-검증으로-정확도-측정) |
| **정밀도** | precision | 양성이라고 예측한 것 중 실제 양성의 비율 = TP / (TP + FP) | [3장 3-4. 정밀도·재현율·F1 점수](03_classification.md#3-4-정밀도재현율f1-점수) |
| **재현율** | recall, TPR | 실제 양성 중 양성으로 찾아낸 비율 = TP / (TP + FN) | [3장 3-4. 정밀도·재현율·F1 점수](03_classification.md#3-4-정밀도재현율f1-점수) |
| **F1 점수** | F1 score | 정밀도와 재현율의 조화 평균. 둘 다 높아야 높다 | [3장 3-4. 정밀도·재현율·F1 점수](03_classification.md#3-4-정밀도재현율f1-점수) |
| **결정 임곗값** | decision threshold | 결정 점수(또는 확률)가 이 값보다 크면 양성으로 판정하는 기준 | [3장 3-5. 결정 점수와 임곗값](03_classification.md#3-5-결정-점수와-임곗값) |
| **PR 곡선** | precision-recall curve | 임곗값을 바꿔 가며 재현율(가로)과 정밀도(세로)를 그린 곡선 | [3장 3-6. PR 곡선과 임곗값 선택](03_classification.md#3-6-pr-곡선과-임곗값-선택) |
| **ROC 곡선** | receiver operating characteristic | 임곗값을 바꿔 가며 FPR(가로)과 TPR(세로)을 그린 곡선 | [3장 3-7. ROC 곡선과 AUC](03_classification.md#3-7-roc-곡선과-auc) |
| **AUC** | area under the curve | ROC 곡선 아래 면적. 1이면 완벽, 0.5면 무작위 수준 | [3장 3-7. ROC 곡선과 AUC](03_classification.md#3-7-roc-곡선과-auc) |
| **OvR / OvO** | one-versus-rest / one-versus-one | 이진 분류기로 다중 분류를 하는 두 전략. 클래스마다 하나 / 클래스 쌍마다 하나 | [3장 4-1. 다중 분류 전략](03_classification.md#4-1-다중-분류-전략) |
| **다중 레이블 분류** | multilabel classification | 샘플 하나에 레이블 여러 개(예: 사진 속 여러 사람)를 예측 | [3장 6. 다중 레이블 분류](03_classification.md#6-다중-레이블-분류-36) |
| **다중 출력 분류** | multioutput classification | 레이블 여러 개 × 레이블마다 여러 값을 예측(예: 픽셀마다 밝기) | [3장 7. 다중 출력 분류](03_classification.md#7-다중-출력-분류-37) |
| **데이터 증강** | data augmentation | 훈련 샘플을 살짝 변형(이동, 회전, 뒤집기 등)해 데이터를 늘리는 방법 | [3장 8-2. 데이터 증강](03_classification.md#8-2-데이터-증강) |

## 4. 선형 모델과 최적화 (4장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **정규 방정식** | normal equation | 선형 회귀의 비용 함수를 최소화하는 θ를 한 번에 계산하는 공식 | [4장 1-3. 정규 방정식](04_training_linear_models.md#1-3-정규-방정식) |
| **MSE** | mean squared error | 오차 제곱의 평균. 선형 회귀의 기본 비용 함수 | [4장 1-1. 모델과 비용 함수](04_training_linear_models.md#1-1-모델과-비용-함수) |
| **경사 하강법** | gradient descent | 비용 함수의 기울기(그레이디언트) 반대 방향으로 파라미터를 조금씩 옮겨 최솟값을 찾는 방법 | [4장 2-1. 경사 하강법의 원리](04_training_linear_models.md#2-1-경사-하강법의-원리) |
| **그레이디언트** | gradient | 각 파라미터로 비용 함수를 편미분한 값의 벡터. 비용이 가장 빠르게 커지는 방향 | [4장 2-2. 배치 경사 하강법](04_training_linear_models.md#2-2-배치-경사-하강법) |
| **에포크** | epoch | 훈련 세트 전체를 한 번 다 사용하는 것 | [4장 2-2. 배치 경사 하강법](04_training_linear_models.md#2-2-배치-경사-하강법) |
| **확률적 경사 하강법** | stochastic gradient descent, SGD | 매 스텝 샘플 하나로 그레이디언트를 계산하는 경사 하강법 | [4장 2-4. 확률적 경사 하강법](04_training_linear_models.md#2-4-확률적-경사-하강법) |
| **미니배치** | mini-batch | 매 스텝 그레이디언트를 계산하는 데 쓰는 작은 샘플 묶음 | [4장 2-6. 미니배치 경사 하강법](04_training_linear_models.md#2-6-미니배치-경사-하강법) |
| **학습 스케줄** | learning schedule | 훈련이 진행될수록 학습률을 바꾸는 규칙(보통 점점 줄임) | [4장 2-4. 확률적 경사 하강법](04_training_linear_models.md#2-4-확률적-경사-하강법) |
| **볼록 함수** | convex function | 어디서 출발해도 경사 하강법이 하나뿐인 최솟값에 도달하는 그릇 모양 함수 | [4장 2-1. 경사 하강법의 원리](04_training_linear_models.md#2-1-경사-하강법의-원리) |
| **다항 회귀** | polynomial regression | 특성의 거듭제곱을 새 특성으로 추가해 선형 모델로 곡선을 맞추는 방법 | [4장 3-1. 다항 특성 변환](04_training_linear_models.md#3-1-다항-특성-변환) |
| **학습 곡선** | learning curve | 훈련 세트 크기(또는 에포크)에 따른 훈련·검증 오차 그래프. 과소/과대적합 진단에 쓴다 | [4장 4-2. 학습 곡선: 과소적합 모델](04_training_linear_models.md#4-2-학습-곡선-과소적합-모델) |
| **편향/분산 트레이드오프** | bias/variance trade-off | 단순한 모델은 편향(잘못된 가정)이 크고, 복잡한 모델은 분산(데이터 변화에 민감)이 크다는 상충 관계 | [4장 4-4. 편향/분산 트레이드오프](04_training_linear_models.md#4-4-편향분산-트레이드오프) |
| **릿지 회귀** | ridge regression | 가중치 제곱합(ℓ2)에 벌점을 주는 선형 회귀 | [4장 5-2. 릿지 회귀](04_training_linear_models.md#5-2-릿지-회귀) |
| **라쏘 회귀** | lasso regression | 가중치 절댓값 합(ℓ1)에 벌점을 주는 선형 회귀. 덜 중요한 가중치를 정확히 0으로 만든다 | [4장 5-3. 라쏘 회귀](04_training_linear_models.md#5-3-라쏘-회귀) |
| **엘라스틱넷** | elastic net | ℓ1과 ℓ2 벌점을 섞은 선형 회귀 | [4장 5-5. 엘라스틱넷](04_training_linear_models.md#5-5-엘라스틱넷) |
| **조기 종료** | early stopping | 검증 오차가 더 나빠지기 시작하면 훈련을 멈추는 규제 방법 | [4장 5-6. 조기 종료](04_training_linear_models.md#5-6-조기-종료) |
| **로지스틱 회귀** | logistic regression | 선형 점수를 시그모이드에 통과시켜 양성 확률을 내는 이진 분류 모델 | [4장 6-1. 확률 추정](04_training_linear_models.md#6-1-확률-추정) |
| **시그모이드** | sigmoid, logistic function | 어떤 실수든 0~1로 바꾸는 S자 함수 σ(t) = 1 / (1 + e⁻ᵗ) | [4장 6-1. 확률 추정](04_training_linear_models.md#6-1-확률-추정) |
| **로짓** | logit | 시그모이드(또는 소프트맥스)에 넣기 전의 점수. 로그 오즈 | [4장 6-1. 확률 추정](04_training_linear_models.md#6-1-확률-추정) |
| **교차 엔트로피 (크로스 엔트로피)** | cross entropy | 예측 확률이 정답 클래스에 낮을수록 커지는 분류용 비용 함수(로그 손실) | [4장 6-2. 훈련과 비용 함수](04_training_linear_models.md#6-2-훈련과-비용-함수) |
| **결정 경계** | decision boundary | 모델이 예측 클래스를 바꾸는 경계. 로지스틱 회귀에서는 확률 50%인 곳 | [4장 6-4. 결정 경계: 특성 1개](04_training_linear_models.md#6-4-결정-경계-특성-1개) |
| **소프트맥스 회귀** | softmax regression | 클래스마다 점수를 계산하고 소프트맥스로 확률로 바꾸는 다중 분류 모델 | [4장 6-6. 소프트맥스 회귀](04_training_linear_models.md#6-6-소프트맥스-회귀) |
| **소프트맥스** | softmax | 점수 벡터를 합이 1인 확률 벡터로 바꾸는 함수. exp(점수) / Σexp | [4장 6-6. 소프트맥스 회귀](04_training_linear_models.md#6-6-소프트맥스-회귀) |
| **배치 크기** | batch size | 경사 하강법 한 스텝에 쓰는 샘플 수. 크면 그레이디언트가 안정적이고 GPU 효율이 좋지만 메모리를 많이 쓴다 | [9장 3-4. 배치 크기](09_artificial_neural_networks.md#3-4-배치-크기) |

## 5. 트리와 앙상블 (5~6장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **결정 트리** | decision tree | "특성 ≤ 임곗값?" 질문을 반복해 데이터를 나누는 모델 | [5장 1. 결정 트리 학습과 시각화](05_decision_trees.md#1-결정-트리-학습과-시각화-51) |
| **지니 불순도** | Gini impurity | 노드에 여러 클래스가 섞인 정도. 한 클래스만 있으면 0 | [5장 1-3. 노드 정보와 지니 불순도](05_decision_trees.md#1-3-노드-정보와-지니-불순도) |
| **CART** | classification and regression tree | 불순도(또는 MSE)를 가장 많이 줄이는 분할을 탐욕적으로 고르는 트리 훈련 알고리즘 | [5장 4. CART 훈련 알고리즘](05_decision_trees.md#4-cart-훈련-알고리즘-54) |
| **엔트로피** | entropy | 불확실성(무질서도)의 척도. 지니 불순도 대신 쓸 수 있다 | [5장 6. 지니 불순도 또는 엔트로피?](05_decision_trees.md#6-지니-불순도-또는-엔트로피-56) |
| **앙상블** | ensemble | 여러 모델의 예측을 모아 하나의 예측을 만드는 방법 | [6장 1. 투표 기반 분류기](06_ensemble_learning_and_random_forests.md#1-투표-기반-분류기-61) |
| **직접 투표 / 간접 투표** | hard / soft voting | 예측 클래스의 다수결 / 클래스 확률의 평균으로 합치는 투표 방식 | [6장 1-1. 직접 투표 (hard voting)](06_ensemble_learning_and_random_forests.md#1-1-직접-투표-hard-voting) |
| **배깅** | bagging, bootstrap aggregating | 같은 모델을 중복 허용 무작위 샘플(부트스트랩)로 여러 번 훈련해 합치는 방법 | [6장 2. 배깅과 페이스팅](06_ensemble_learning_and_random_forests.md#2-배깅과-페이스팅-62) |
| **페이스팅** | pasting | 배깅과 같지만 중복 없이 샘플을 뽑는 방법 | [6장 2. 배깅과 페이스팅](06_ensemble_learning_and_random_forests.md#2-배깅과-페이스팅-62) |
| **OOB 평가** | out-of-bag evaluation | 배깅에서 각 예측기가 훈련에 쓰지 않은 샘플로 성능을 추정하는 방법 | [6장 2-3. OOB 평가 (out-of-bag)](06_ensemble_learning_and_random_forests.md#2-3-oob-평가-out-of-bag) |
| **랜덤 포레스트** | random forest | 결정 트리 배깅 + 노드마다 특성 일부만 고려하는 앙상블 | [6장 4. 랜덤 포레스트](06_ensemble_learning_and_random_forests.md#4-랜덤-포레스트-64) |
| **엑스트라 트리** | extremely randomized trees | 랜덤 포레스트에서 임곗값까지 무작위로 정하는 앙상블. 훈련이 빠르다 | [6장 4-3. 엑스트라 트리 (Extremely Randomized Trees)](06_ensemble_learning_and_random_forests.md#4-3-엑스트라-트리-extremely-randomized-trees) |
| **특성 중요도** | feature importance | 각 특성이 불순도를 평균적으로 얼마나 줄였는지로 잰 중요도 | [6장 4-4. 특성 중요도](06_ensemble_learning_and_random_forests.md#4-4-특성-중요도) |
| **부스팅** | boosting | 약한 모델을 순차적으로 훈련하며 앞 모델의 실수를 보완해 나가는 앙상블 | [6장 5. 부스팅](06_ensemble_learning_and_random_forests.md#5-부스팅-65) |
| **AdaBoost** | adaptive boosting | 앞 모델이 틀린 샘플의 가중치를 높여 다음 모델을 훈련하는 부스팅 | [6장 5-1. AdaBoost (Adaptive Boosting)](06_ensemble_learning_and_random_forests.md#5-1-adaboost-adaptive-boosting) |
| **그레이디언트 부스팅** | gradient boosting | 앞 모델의 잔차(오차)를 다음 모델이 학습하는 부스팅 | [6장 5-2. 그레이디언트 부스팅 (Gradient Boosting)](06_ensemble_learning_and_random_forests.md#5-2-그레이디언트-부스팅-gradient-boosting) |
| **스태킹** | stacking | 예측들을 합치는 방법 자체를 블렌더(메타 모델)로 학습하는 앙상블 | [6장 6. 스태킹](06_ensemble_learning_and_random_forests.md#6-스태킹-66) |
| **리프 노드** | leaf node | 자식이 없는 맨 아래 노드. 예측값(다수 클래스·평균)이 여기서 나온다 | [5장 1-3. 노드 정보와 지니 불순도](05_decision_trees.md#1-3-노드-정보와-지니-불순도) |

## 6. 차원 축소와 비지도 학습 (7~8장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **차원의 저주** | curse of dimensionality | 차원이 커질수록 데이터가 희박해지고 거리가 멀어져 학습이 어려워지는 현상 | [7장 1. 차원의 저주](07_dimensionality_reduction.md#1-차원의-저주-71) |
| **투영** | projection | 고차원 데이터를 저차원 부분 공간(평면 등)에 수직으로 내리는 것 | [7장 2-1. 투영 (Projection)](07_dimensionality_reduction.md#2-1-투영-projection) |
| **매니폴드 학습** | manifold learning | 데이터가 놓인 휘어진 저차원 공간(매니폴드)을 펼쳐 차원을 줄이는 방법 | [7장 2-2. 매니폴드 학습 (Manifold Learning)](07_dimensionality_reduction.md#2-2-매니폴드-학습-manifold-learning) |
| **PCA (주성분 분석)** | principal component analysis | 분산이 가장 큰 축(주성분)들을 찾아 그 축으로 투영하는 선형 차원 축소 | [7장 3. 주성분 분석](07_dimensionality_reduction.md#3-주성분-분석-73) |
| **설명된 분산 비율** | explained variance ratio | 각 주성분이 전체 분산의 몇 %를 담는지 | [7장 3-4. 설명된 분산의 비율](07_dimensionality_reduction.md#3-4-설명된-분산의-비율) |
| **재구성 오차** | reconstruction error | 압축했다가 복원한 데이터와 원본의 차이. 정보 손실의 척도 | [7장 3-6. 압축을 위한 PCA](07_dimensionality_reduction.md#3-6-압축을-위한-pca) |
| **랜덤 투영** | random projection | 무작위 행렬을 곱해 차원을 줄이는 방법. 거리를 대략 보존한다 | [7장 4. 랜덤 투영](07_dimensionality_reduction.md#4-랜덤-투영-74) |
| **LLE** | locally linear embedding | 각 샘플을 이웃의 선형 결합으로 표현하고 그 관계를 저차원에서 보존하는 매니폴드 학습 | [7장 5. 지역 선형 임베딩](07_dimensionality_reduction.md#5-지역-선형-임베딩-75) |
| **t-SNE** | t-distributed stochastic neighbor embedding | 가까운 샘플은 가깝게 유지하며 2~3차원으로 줄이는 시각화용 기법 | [7장 6. 다른 차원 축소 기법](07_dimensionality_reduction.md#6-다른-차원-축소-기법-76) |
| **군집** | clustering | 레이블 없이 비슷한 샘플끼리 묶는 작업 | [8장 1-1. 분류 vs 군집](08_unsupervised_learning.md#1-1-분류-vs-군집) |
| **k-평균** | k-means | 샘플을 가장 가까운 중심에 배정 ↔ 중심을 평균으로 갱신을 반복해 군집 k개를 찾는 알고리즘 | [8장 1-2. k-평균 훈련과 예측](08_unsupervised_learning.md#1-2-k-평균-훈련과-예측) |
| **이너셔** | inertia | 각 샘플과 자기 군집 중심 사이 거리 제곱의 합. k-평균의 비용 | [8장 1-4. 이너셔와 다중 초기화](08_unsupervised_learning.md#1-4-이너셔와-다중-초기화) |
| **실루엣 점수** | silhouette score | 샘플이 자기 군집 안에 잘 있고 다른 군집과 멀리 떨어진 정도의 평균(−1~1) | [8장 1-6. 최적의 군집 수 찾기](08_unsupervised_learning.md#1-6-최적의-군집-수-찾기) |
| **레이블 전파** | label propagation | 소수의 레이블을 같은 군집의 다른 샘플로 복사해 레이블을 늘리는 방법 | [8장 1-9. 활용 ② 준지도 학습](08_unsupervised_learning.md#1-9-활용--준지도-학습) |
| **DBSCAN** | density-based spatial clustering of applications with noise | 밀집 영역이 이어진 곳을 군집으로 보고, 어디에도 속하지 않는 샘플을 이상치로 보는 알고리즘 | [8장 1-10. DBSCAN](08_unsupervised_learning.md#1-10-dbscan) |
| **가우스 혼합** | Gaussian mixture model | 데이터가 여러 정규분포의 혼합에서 나왔다고 보고 각 분포를 추정하는 확률 모델 | [8장 2. 가우스 혼합](08_unsupervised_learning.md#2-가우스-혼합-82) |
| **EM 알고리즘** | expectation-maximization | 샘플의 군집 소속 확률 추정(E)과 분포 파라미터 갱신(M)을 반복하는 알고리즘 | [8장 기댓값-최대화(EM) 알고리즘](08_unsupervised_learning.md#기댓값-최대화em-알고리즘) |
| **이상치 탐지 / 특이치 탐지** | anomaly / novelty detection | 정상에서 벗어난 샘플 찾기. 특이치 탐지는 깨끗한 데이터로 훈련하고 새 샘플만 판정 | [8장 2-4. 가우스 혼합으로 이상치 탐지](08_unsupervised_learning.md#2-4-가우스-혼합으로-이상치-탐지) |
| **BIC / AIC** | Bayesian / Akaike information criterion | 모델 적합도와 파라미터 수를 함께 고려하는 지표. 작을수록 좋다. 군집 수 선택에 쓴다 | [8장 2-5. 군집 수 선택: BIC · AIC](08_unsupervised_learning.md#2-5-군집-수-선택-bic--aic) |
| **SVD (특잇값 분해)** | singular value decomposition | 행렬을 UΣVᵀ로 나누는 분해. V의 열이 주성분, 특잇값²이 분산에 비례 | [7장 3-1. 주성분 찾기 — SVD](07_dimensionality_reduction.md#3-1-주성분-찾기--svd) |
| **공분산 행렬** | covariance matrix | 특성 쌍마다 함께 변하는 정도를 모은 행렬. 대각선은 각 특성의 분산. PCA·가우스 혼합의 핵심 | [7장 3-1. 주성분 찾기 — SVD](07_dimensionality_reduction.md#3-1-주성분-찾기--svd) |

## 7. 신경망 기초와 파이토치 (9~10장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **퍼셉트론** | perceptron | 입력의 가중합에 계단 함수를 적용하는 가장 단순한 인공 신경망 | [9장 1-3. 퍼셉트론](09_artificial_neural_networks.md#1-3-퍼셉트론) |
| **다층 퍼셉트론** | multilayer perceptron, MLP | 입력층 – 은닉층 여러 개 – 출력층으로 이루어진 완전 연결 신경망 | [9장 1-6. 다층 퍼셉트론과 역전파](09_artificial_neural_networks.md#1-6-다층-퍼셉트론과-역전파) |
| **은닉 층** | hidden layer | 입력층과 출력층 사이의 층 | [9장 1-6. 다층 퍼셉트론과 역전파](09_artificial_neural_networks.md#1-6-다층-퍼셉트론과-역전파) |
| **활성화 함수** | activation function | 뉴런의 가중합에 적용하는 비선형 함수. 없으면 층을 쌓아도 선형 모델과 같다 | [9장 1-7. 활성화 함수](09_artificial_neural_networks.md#1-7-활성화-함수) |
| **ReLU** | rectified linear unit | max(0, z). 계산이 빠르고 그레이디언트가 잘 전달돼 기본으로 쓰는 활성화 함수 | [9장 1-7. 활성화 함수](09_artificial_neural_networks.md#1-7-활성화-함수) |
| **역전파** | backpropagation | 출력의 오차를 연쇄 법칙으로 거꾸로 전파해 모든 가중치의 그레이디언트를 한 번에 구하는 알고리즘 | [9장 1-6. 다층 퍼셉트론과 역전파](09_artificial_neural_networks.md#1-6-다층-퍼셉트론과-역전파) |
| **순전파** | forward pass | 입력에서 출력 방향으로 층을 차례로 계산하는 것 | [9장 1-6. 다층 퍼셉트론과 역전파](09_artificial_neural_networks.md#1-6-다층-퍼셉트론과-역전파) |
| **텐서** | tensor | 파이토치의 다차원 배열. 넘파이 배열과 비슷하지만 GPU 연산과 자동 미분을 지원 | [10장 1-2. 텐서](10_neural_nets_with_pytorch.md#1-2-텐서) |
| **자동 미분** | autograd | 연산 기록을 바탕으로 그레이디언트를 자동으로 계산하는 파이토치 기능 (`backward()`) | [10장 1-6. 자동 미분 autograd](10_neural_nets_with_pytorch.md#1-6-자동-미분-autograd) |
| **옵티마이저** | optimizer | 그레이디언트로 파라미터를 갱신하는 알고리즘(SGD, Adam 등) | [10장 2-3. 고수준 API로 선형 회귀](10_neural_nets_with_pytorch.md#2-3-고수준-api로-선형-회귀) |
| **손실 함수** | loss function | 훈련 중 최소화하는 함수. 비용 함수와 같은 뜻으로 쓴다 | [10장 1-7. autograd로 경사 하강법 구현](10_neural_nets_with_pytorch.md#1-7-autograd로-경사-하강법-구현) |
| **DataLoader** | data loader | 데이터셋을 미니배치로 나눠(필요하면 섞어) 꺼내 주는 파이토치 객체 | [10장 4. DataLoader로 미니배치 경사 하강법 구현하기](10_neural_nets_with_pytorch.md#4-dataloader로-미니배치-경사-하강법-구현하기-104) |
| **Optuna** | Optuna | 좋았던 영역을 더 자주 시도(TPE)하고 가망 없는 시도는 중간에 자르는(가지치기) 하이퍼파라미터 탐색 라이브러리 | [10장 8-1. 목적 함수와 스터디](10_neural_nets_with_pytorch.md#8-1-목적-함수와-스터디) |
| **체크포인트** | checkpoint | 훈련 중간의 모델 가중치(와 옵티마이저 상태)를 저장한 파일. 최고 성능 모델 보관·훈련 재개에 쓴다 | [10장 9-2. 가중치만 저장](10_neural_nets_with_pytorch.md#9-2-가중치만-저장) |

## 8. 심층 신경망 훈련 (11장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **그레이디언트 소실 / 폭주** | vanishing / exploding gradients | 역전파 중 그레이디언트가 층을 지날수록 0에 가까워지거나 폭발적으로 커지는 문제 | [11장 1-1. 불안정한 그레이디언트](11_training_deep_neural_networks.md#1-1-불안정한-그레이디언트) |
| **가중치 초기화 (글로럿·He)** | Glorot / He initialization | 층의 입출력 개수에 맞춰 가중치 분산을 정해 신호가 층을 지나도 크기를 유지하게 하는 초기화 | [11장 1-2. 글로럿과 He 초기화](11_training_deep_neural_networks.md#1-2-글로럿과-he-초기화) |
| **배치 정규화** | batch normalization | 층 입력을 미니배치 평균·표준편차로 정규화한 뒤 학습되는 스케일·이동을 적용하는 층 | [11장 1-7. 배치 정규화](11_training_deep_neural_networks.md#1-7-배치-정규화) |
| **층 정규화** | layer normalization | 배치가 아니라 샘플 하나의 특성들로 정규화하는 층. RNN·트랜스포머에서 쓴다 | [11장 1-8. 층 정규화](11_training_deep_neural_networks.md#1-8-층-정규화) |
| **그레이디언트 클리핑** | gradient clipping | 그레이디언트가 일정 크기를 넘지 않게 잘라 폭주를 막는 방법 | [11장 1-9. 그레이디언트 클리핑](11_training_deep_neural_networks.md#1-9-그레이디언트-클리핑) |
| **전이 학습** | transfer learning | 비슷한 작업에서 미리 훈련된 모델의 층을 가져와 재사용하는 방법 | [11장 2. 사전 훈련된 층 재사용하기](11_training_deep_neural_networks.md#2-사전-훈련된-층-재사용하기-112) |
| **미세 튜닝** | fine-tuning | 사전 훈련된 모델을 새 작업 데이터로 (작은 학습률로) 추가 훈련하는 것 | [11장 2-5. 층 재사용 · 동결 · 미세 튜닝](11_training_deep_neural_networks.md#2-5-층-재사용--동결--미세-튜닝) |
| **동결** | freezing | 일부 층의 가중치를 훈련 중 바뀌지 않게 고정하는 것 | [11장 2-5. 층 재사용 · 동결 · 미세 튜닝](11_training_deep_neural_networks.md#2-5-층-재사용--동결--미세-튜닝) |
| **모멘텀** | momentum | 이전 갱신 방향을 관성처럼 누적해 경사 하강을 가속하는 방법 | [11장 3-2. 모멘텀 최적화](11_training_deep_neural_networks.md#3-2-모멘텀-최적화) |
| **Adam** | adaptive moment estimation | 모멘텀과 파라미터별 적응형 학습률(RMSProp)을 합친 옵티마이저 | [11장 3. 고속 옵티마이저](11_training_deep_neural_networks.md#3-고속-옵티마이저-113) |
| **학습률 스케줄링 (스케줄러)** | learning rate scheduling | 훈련 중 학습률을 바꾸는 전략(지수 감소, 코사인, 1사이클 등) | [11장 4. 학습률 스케줄링](11_training_deep_neural_networks.md#4-학습률-스케줄링-114) |
| **웜업** | warmup | 훈련 초반에 학습률을 작게 시작해 점점 올리는 것 | [11장 4-4. 웜업](11_training_deep_neural_networks.md#4-4-웜업) |
| **드롭아웃** | dropout | 훈련 중 매 스텝 뉴런 일부를 무작위로 꺼서 과대적합을 줄이는 규제 | [11장 5-2. 드롭아웃](11_training_deep_neural_networks.md#5-2-드롭아웃) |
| **가중치 감쇠** | weight decay | 매 스텝 가중치를 조금씩 줄이는 규제. ℓ2 규제와 비슷한 효과 | [11장 5-1. ℓ1과 ℓ2 규제](11_training_deep_neural_networks.md#5-1-ℓ1과-ℓ2-규제) |
| **SELU** | scaled exponential linear unit | 조건(표준화 입력, 르쿤 초기화, 순차 완전 연결)을 지키면 층 출력이 평균 0·분산 1을 스스로 유지하는 활성화 함수 | [11장 1-5. ELU와 SELU](11_training_deep_neural_networks.md#1-5-elu와-selu) |
| **GELU** | Gaussian error linear unit | ReLU를 부드럽게 만든 활성화 함수. 트랜스포머(BERT, GPT)의 기본값 | [11장 1-6. GELU, Swish, Mish와 최신 활성화 함수](11_training_deep_neural_networks.md#1-6-gelu-swish-mish와-최신-활성화-함수) |

## 9. CNN과 컴퓨터 비전 (12장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **합성곱 층** | convolutional layer | 작은 필터를 이미지 위로 밀며 국부 패턴을 찾는 층 | [12장 2. 합성곱 층](12_deep_computer_vision_with_cnns.md#2-합성곱-층-122) |
| **국부 수용장** | local receptive field | 뉴런 하나가 보는 입력의 작은 영역 | [12장 1. 시각 피질 구조](12_deep_computer_vision_with_cnns.md#1-시각-피질-구조-121) |
| **필터 (커널)** | filter, kernel | 합성곱 층이 학습하는 작은 가중치 행렬. 특정 패턴에 반응한다 | [12장 2-2. 필터와 특성 맵](12_deep_computer_vision_with_cnns.md#2-2-필터와-특성-맵) |
| **특성 맵** | feature map | 필터 하나를 입력 전체에 적용한 출력. 그 패턴이 어디 있는지 보여 준다 | [12장 2-2. 필터와 특성 맵](12_deep_computer_vision_with_cnns.md#2-2-필터와-특성-맵) |
| **패딩 / 스트라이드** | padding / stride | 입력 가장자리에 0을 덧대는 것 / 필터가 한 번에 이동하는 칸 수 | [12장 2-5. 패딩과 스트라이드](12_deep_computer_vision_with_cnns.md#2-5-패딩과-스트라이드) |
| **풀링 층** | pooling layer | 영역마다 최댓값(또는 평균)만 남겨 특성 맵을 줄이는 층 | [12장 4. 파이토치로 풀링 층 구현하기](12_deep_computer_vision_with_cnns.md#4-파이토치로-풀링-층-구현하기-124) |
| **전역 평균 풀링** | global average pooling | 특성 맵마다 전체 평균 하나만 남기는 풀링 | [12장 4. 파이토치로 풀링 층 구현하기](12_deep_computer_vision_with_cnns.md#4-파이토치로-풀링-층-구현하기-124) |
| **잔차 연결 (스킵 연결)** | residual / skip connection | 층의 입력을 출력에 그대로 더하는 연결. 아주 깊은 네트워크도 훈련되게 한다 (ResNet) | [12장 5. CNN 구조](12_deep_computer_vision_with_cnns.md#5-cnn-구조-125) |
| **IoU** | intersection over union | 예측 박스와 정답 박스의 겹친 면적 / 합친 면적 | [12장 9-2. 손실과 평가 지표: IoU](12_deep_computer_vision_with_cnns.md#9-2-손실과-평가-지표-iou) |
| **NMS** | non-max suppression | 같은 물체를 가리키는 겹치는 박스 중 가장 확신하는 것만 남기는 후처리 | [12장 10. 객체 탐지](12_deep_computer_vision_with_cnns.md#10-객체-탐지-1210) |
| **mAP** | mean average precision | 클래스별 PR 곡선 아래 면적의 평균. 객체 탐지 지표 | [12장 10. 객체 탐지](12_deep_computer_vision_with_cnns.md#10-객체-탐지-1210) |
| **시맨틱 분할** | semantic segmentation | 픽셀마다 클래스를 예측하는 작업 | [12장 12-1. 시맨틱 분할과 업샘플링](12_deep_computer_vision_with_cnns.md#12-1-시맨틱-분할과-업샘플링) |
| **YOLO** | You Only Look Once | 이미지를 한 번 통과시켜 격자 칸마다 박스와 클래스를 예측하는 실시간 객체 탐지 모델 | [12장 10-3. YOLO로 객체 탐지](12_deep_computer_vision_with_cnns.md#10-3-yolo로-객체-탐지) |

## 10. 시퀀스와 RNN (13장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **순환 뉴런** | recurrent neuron | 현재 입력과 이전 타임 스텝의 자기 출력을 함께 받는 뉴런 | [13장 1. 순환 뉴런과 순환 층](13_processing_sequences_using_rnns_and_cnns.md#1-순환-뉴런과-순환-층-131) |
| **메모리 셀** | memory cell | 타임 스텝을 넘어 상태(기억)를 유지하는 신경망 구성 요소 | [13장 1-2. 메모리 셀](13_processing_sequences_using_rnns_and_cnns.md#1-2-메모리-셀) |
| **BPTT** | backpropagation through time | 시간 축으로 펼친 RNN에 역전파를 적용하는 방법 | [13장 2-1. BPTT](13_processing_sequences_using_rnns_and_cnns.md#2-1-bptt) |
| **시계열** | time series | 시간 순서로 측정한 값의 시퀀스 | [13장 3. 시계열 예측하기](13_processing_sequences_using_rnns_and_cnns.md#3-시계열-예측하기-133) |
| **계절성 / 차분** | seasonality / differencing | 일정 주기로 반복되는 패턴 / 현재 값에서 일정 시차 전 값을 빼서 추세·계절성을 없애는 것 | [13장 3-2. 주간 계절성과 나이브 예측](13_processing_sequences_using_rnns_and_cnns.md#3-2-주간-계절성과-나이브-예측) |
| **나이브 예측** | naive forecasting | "직전(또는 한 주기 전) 값이 그대로 반복된다"고 보는 기준 예측 | [13장 3-2. 주간 계절성과 나이브 예측](13_processing_sequences_using_rnns_and_cnns.md#3-2-주간-계절성과-나이브-예측) |
| **LSTM** | long short-term memory | 게이트로 무엇을 기억·삭제·출력할지 학습해 긴 의존성을 다루는 RNN 셀 | [13장 4-2. 단기 기억 문제와 LSTM](13_processing_sequences_using_rnns_and_cnns.md#4-2-단기-기억-문제와-lstm) |
| **GRU** | gated recurrent unit | LSTM을 단순화한 게이트 RNN 셀 | [13장 4-3. GRU](13_processing_sequences_using_rnns_and_cnns.md#4-3-gru) |
| **인과적 합성곱 / 팽창 합성곱** | causal / dilated convolution | 미래 타임 스텝을 보지 않는 합성곱 / 입력을 건너뛰며 보는 합성곱 (WaveNet) | [13장 4. 긴 시퀀스 다루기](13_processing_sequences_using_rnns_and_cnns.md#4-긴-시퀀스-다루기-134) |
| **ARIMA / SARIMA** | autoregressive integrated moving average (seasonal) | 과거 값(AR)과 과거 오차(MA)의 선형 결합으로 예측하고, 차분(I)으로 추세를, 계절 항(S)으로 계절성을 다루는 통계 시계열 모델 | [13장 3-4. ARMA 모델 계열](13_processing_sequences_using_rnns_and_cnns.md#3-4-arma-모델-계열) |
| **시퀀스-투-시퀀스 (seq2seq)** | sequence-to-sequence | 입력 시퀀스를 받아 매 타임 스텝 출력 시퀀스를 내는 구조. 13장에서는 매 스텝 다음 14일을 예측 | [13장 3-12. 시퀀스-투-시퀀스 모델로 예측하기](13_processing_sequences_using_rnns_and_cnns.md#3-12-시퀀스-투-시퀀스-모델로-예측하기) |

## 11. 자연어 처리와 어텐션 (14장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **토큰 / 토큰화** | token / tokenization | 텍스트를 모델이 다루는 단위(단어, 서브워드, 글자)로 자르는 것 | [14장 2-2. 토큰화의 세 가지 단위](14_nlp_with_rnns_and_attention.md#2-2-토큰화의-세-가지-단위) |
| **서브워드 토큰화 (BPE)** | subword tokenization, byte pair encoding | 자주 붙어 나오는 글자 쌍을 합쳐 가며 단어보다 작은 단위의 어휘를 만드는 토큰화 | [14장 2-3. BPE 토크나이저 훈련](14_nlp_with_rnns_and_attention.md#2-3-bpe-토크나이저-훈련) |
| **임베딩** | embedding | 토큰 같은 범주 값을 학습되는 밀집 벡터로 표현한 것 (`nn.Embedding`) | [14장 1-2. 임베딩](14_nlp_with_rnns_and_attention.md#1-2-임베딩) |
| **온도** | temperature | 소프트맥스 전에 로짓을 나누는 값. 높으면 다양한, 낮으면 확실한 쪽을 고른다 | [14장 1. char-RNN으로 셰익스피어 문체 생성하기](14_nlp_with_rnns_and_attention.md#1-char-rnn으로-셰익스피어-문체-생성하기-141) |
| **어텐션 마스크 (패딩 마스크)** | attention / padding mask | 패딩 토큰을 계산에서 무시하도록 표시하는 마스크 | [14장 2-6. 배치 처리: 패딩·잘라내기·어텐션 마스크](14_nlp_with_rnns_and_attention.md#2-6-배치-처리-패딩잘라내기어텐션-마스크) |
| **인코더–디코더** | encoder–decoder | 입력 시퀀스를 표현으로 바꾸는 인코더와 그 표현으로 출력 시퀀스를 만드는 디코더 구조 | [14장 3-1. 인코더–디코더 구조](14_nlp_with_rnns_and_attention.md#3-1-인코더디코더-구조) |
| **교사 강요** | teacher forcing | 디코더 훈련 때 이전 스텝의 예측 대신 정답 토큰을 입력으로 넣는 방법 | [14장 3-3. 배치 만들기와 교사 강요](14_nlp_with_rnns_and_attention.md#3-3-배치-만들기와-교사-강요) |
| **빔 검색** | beam search | 가장 그럴듯한 후보 문장 k개를 유지하며 생성하는 디코딩 방법 | [14장 3. 신경망 기계 번역을 위한 인코더-디코더 네트워크](14_nlp_with_rnns_and_attention.md#3-신경망-기계-번역을-위한-인코더-디코더-네트워크-143) |
| **어텐션** | attention | 출력을 만들 때 입력의 각 위치에 가중치를 두어 필요한 부분에 집중하는 메커니즘 | [14장 4. 어텐션 메커니즘](14_nlp_with_rnns_and_attention.md#4-어텐션-메커니즘-144) |
| **쿼리·키·값** | query, key, value | 어텐션에서 '무엇을 찾는가(쿼리)'와 '각 위치의 색인(키)'을 비교해, 비슷한 위치의 '내용(값)'을 가중합하는 구조 | [14장 4-3. 점곱 어텐션 구현](14_nlp_with_rnns_and_attention.md#4-3-점곱-어텐션-구현) |

## 12. 트랜스포머 (15~16장)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **트랜스포머** | transformer | RNN 없이 어텐션만으로 시퀀스를 처리하는 구조. 병렬 훈련이 가능하다 | [15장 1. Attention Is All You Need: 원본 트랜스포머 아키텍처](15_transformers_for_nlp_and_chatbots.md#1-attention-is-all-you-need-원본-트랜스포머-아키텍처-151) |
| **셀프 어텐션** | self-attention | 한 시퀀스 안의 모든 위치가 서로에게 어텐션하는 것 | [15장 1-1. 전체 구조](15_transformers_for_nlp_and_chatbots.md#1-1-전체-구조) |
| **멀티 헤드 어텐션** | multi-head attention | 어텐션을 여러 개(헤드) 병렬로 수행해 여러 종류의 관계를 동시에 보는 층 | [15장 1-4. 멀티 헤드 어텐션 구현](15_transformers_for_nlp_and_chatbots.md#1-4-멀티-헤드-어텐션-구현) |
| **위치 인코딩 (위치 임베딩)** | positional encoding | 순서를 모르는 어텐션에 토큰 위치 정보를 더해 주는 벡터 | [15장 1. Attention Is All You Need: 원본 트랜스포머 아키텍처](15_transformers_for_nlp_and_chatbots.md#1-attention-is-all-you-need-원본-트랜스포머-아키텍처-151) |
| **인과 마스크** | causal mask | 디코더가 미래 토큰을 보지 못하게 가리는 마스크 | [15장 1. Attention Is All You Need: 원본 트랜스포머 아키텍처](15_transformers_for_nlp_and_chatbots.md#1-attention-is-all-you-need-원본-트랜스포머-아키텍처-151) |
| **크로스 어텐션** | cross-attention | 디코더가 인코더 출력에 어텐션하는 것(쿼리는 디코더, 키·값은 인코더) | [15장 1-7. 인코더 층과 디코더 층](15_transformers_for_nlp_and_chatbots.md#1-7-인코더-층과-디코더-층) |
| **BERT / MLM** | Bidirectional Encoder Representations from Transformers / masked language model | 가린 토큰을 양방향 문맥으로 맞히도록 사전 훈련한 인코더 전용 모델 | [15장 3-1. BERT의 구조와 사전 훈련](15_transformers_for_nlp_and_chatbots.md#3-1-bert의-구조와-사전-훈련) |
| **GPT / 인과적 언어 모델** | causal language model | 다음 토큰을 예측하도록 훈련한 디코더 전용 모델 | [15장 4-1. GPT와 인과적 언어 모델링](15_transformers_for_nlp_and_chatbots.md#4-1-gpt와-인과적-언어-모델링) |
| **top-p 샘플링** | nucleus sampling | 누적 확률이 p가 될 때까지의 상위 토큰들 중에서만 샘플링하는 생성 방법 | [15장 4. 디코더 기반 트랜스포머](15_transformers_for_nlp_and_chatbots.md#4-디코더-기반-트랜스포머-154) |
| **퓨샷 프롬프트** | few-shot prompting | 원하는 답 형식의 예시 몇 개를 프롬프트에 넣어 모델이 따라 하게 하는 방법 | [15장 4-4. 퓨샷 프롬프트](15_transformers_for_nlp_and_chatbots.md#4-4-퓨샷-프롬프트) |
| **SFT** | supervised fine-tuning | (지시, 좋은 답) 쌍으로 언어 모델을 미세 튜닝하는 것 | [15장 5-4. 정렬: SFT, RLHF, DPO](15_transformers_for_nlp_and_chatbots.md#5-4-정렬-sft-rlhf-dpo) |
| **RLHF / DPO** | reinforcement learning from human feedback / direct preference optimization | 사람이 선호한 답을 더 내도록 모델을 조정하는 방법. DPO는 보상 모델 없이 선호 쌍으로 직접 최적화 | [15장 5-4. 정렬: SFT, RLHF, DPO](15_transformers_for_nlp_and_chatbots.md#5-4-정렬-sft-rlhf-dpo) |
| **비전 트랜스포머** | vision transformer, ViT | 이미지를 패치로 잘라 토큰처럼 트랜스포머에 넣는 모델 | [16장 1-2. 원본 ViT: 이미지는 16×16 단어의 모음](16_vision_and_multimodal_transformers.md#1-2-원본-vit-이미지는-1616-단어의-모음) |
| **패치 임베딩** | patch embedding | 이미지 패치 하나를 벡터 하나로 바꾸는 층 | [16장 1-3. 패치 임베딩 구현](16_vision_and_multimodal_transformers.md#1-3-패치-임베딩-구현) |
| **지식 증류** | knowledge distillation | 큰 교사 모델의 출력을 작은 학생 모델이 따라 하도록 훈련하는 방법 (DistilBERT, DeiT) | [15장 3-7. BERT 이후의 인코더 모델](15_transformers_for_nlp_and_chatbots.md#3-7-bert-이후의-인코더-모델) |
| **대조 학습 (CLIP)** | contrastive learning | 짝이 맞는 (이미지, 텍스트)는 가깝게, 안 맞는 쌍은 멀게 임베딩을 학습하는 방법 | [16장 2. 멀티모달 트랜스포머](16_vision_and_multimodal_transformers.md#2-멀티모달-트랜스포머-162) |
| **제로샷 분류** | zero-shot classification | 해당 클래스로 훈련한 적 없이, 클래스 이름(텍스트)과의 유사도로 분류하는 것 | [16장 2-4. CLIP으로 제로샷 이미지 분류하기](16_vision_and_multimodal_transformers.md#2-4-clip으로-제로샷-이미지-분류하기) |
| **멀티모달** | multimodal | 이미지·텍스트·음성 등 여러 종류의 입력을 함께 다루는 것 | [16장 2. 멀티모달 트랜스포머](16_vision_and_multimodal_transformers.md#2-멀티모달-트랜스포머-162) |
| **피드포워드 층** | position-wise feed-forward network | 트랜스포머 블록 안에서 토큰마다 따로 적용하는 2층 완전 연결 네트워크(어텐션 다음) | [15장 1-7. 인코더 층과 디코더 층](15_transformers_for_nlp_and_chatbots.md#1-7-인코더-층과-디코더-층) |
| **LLM (대규모 언어 모델)** | large language model | 수십억 개 이상의 파라미터를 가진, 대량의 텍스트로 다음 토큰 예측을 사전 훈련한 언어 모델 | [15장 4-5. 더 큰 모델: Mistral-7B](15_transformers_for_nlp_and_chatbots.md#4-5-더-큰-모델-mistral-7b) |
| **허깅 페이스** | Hugging Face | 사전 훈련 모델·데이터셋·토크나이저를 공유하는 허브와 `transformers`·`datasets`·`trl` 라이브러리 | [14장 2-1. IMDB 데이터셋 적재](14_nlp_with_rnns_and_attention.md#2-1-imdb-데이터셋-적재) |

## 13. 서포트 벡터 머신 (부록 C)

| 용어 | 영어 | 뜻 | 처음 설명하는 곳 |
|---|---|---|---|
| **서포트 벡터 머신** | support vector machine, SVM | 두 클래스 사이에 가장 넓은 '도로'를 내는 분류(·회귀) 모델 | [부록 C 1. 선형 SVM 분류](C_support_vector_machines.md#1-선형-svm-분류) |
| **마진** | margin | 결정 경계에서 가장 가까운 샘플까지의 거리. 도로 폭의 절반 | [부록 C 1. 선형 SVM 분류](C_support_vector_machines.md#1-선형-svm-분류) |
| **서포트 벡터** | support vector | 도로 위나 경계에 걸친 샘플. 이 샘플들만 결정 경계를 정한다 | [부록 C 1. 선형 SVM 분류](C_support_vector_machines.md#1-선형-svm-분류) |
| **소프트 마진** | soft margin | 마진 오류를 일부 허용하며 도로 폭과의 균형을 `C`로 조절하는 SVM | [부록 C 2. 소프트 마진 분류](C_support_vector_machines.md#2-소프트-마진-분류) |
| **힌지 손실** | hinge loss | max(0, 1 − t·s). 올바른 쪽으로 충분히 떨어진 샘플은 손실 0 | [부록 C 5. SVM의 내부 동작](C_support_vector_machines.md#5-svm의-내부-동작) |
| **커널 트릭** | kernel trick | 고차원으로 변환한 뒤의 내적을, 변환하지 않고 원래 공간에서 바로 계산하는 기법 | [부록 C 3-2. 커널 트릭](C_support_vector_machines.md#3-2-커널-트릭) |
| **RBF 커널** | Gaussian RBF kernel | exp(−γ‖a − b‖²). `gamma`가 클수록 각 샘플의 영향 범위가 좁아 경계가 구불구불해진다 | [부록 C 3-3. 유사도 특성과 RBF 커널](C_support_vector_machines.md#3-3-유사도-특성과-rbf-커널) |
| **ε-둔감 손실 (SVM 회귀)** | epsilon-insensitive | 폭 2ε 도로 안의 샘플은 손실 0으로 보는 회귀 방식 | [부록 C 4. SVM 회귀](C_support_vector_machines.md#4-svm-회귀) |

---

[📚 전체 목차](../README.md)
