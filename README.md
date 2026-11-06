# ds-study

데이터 사이언스 공부 기록

## SQL

| 장 | 주제 | 키워드 |
|---|---|---|
| [2장](sql/02_data_preparation.md) | 데이터 준비 | 데이터 타입, 프로파일링(분포·품질), 정제, 셰이핑 |
| [3장](sql/03_time_series.md) | 시계열 분석 | 날짜 함수, 트렌드, 요소 비교, 전체 대비 비율, 인덱싱, 이동평균, 누적값, YoY·MoM |

## 머신러닝 (핸즈온 머신러닝 with 사이킷런, 파이토치)

| 장 | 주제 | 키워드 |
|---|---|---|
| [1장](ml/01_the_machine_learning_landscape.md) | 한눈에 보는 머신러닝 | 머신러닝 정의, 지도·비지도·준지도·자기 지도·강화 학습, 배치·온라인 학습, 사례 기반(KNN)·모델 기반(선형 회귀), fit·predict, 대표성 없는 데이터, 과대적합·과소적합, 규제(Ridge), 하이퍼파라미터, 훈련·검증·테스트 세트, 교차 검증, 데이터 불일치, 공짜 점심 없음 |
| [2장](ml/02_end_to_end_ml_project.md) | 머신러닝 프로젝트 처음부터 끝까지 | 계층 샘플링, 상관관계, SimpleImputer, OneHotEncoder, 스케일링, 사용자 정의 변환기, Pipeline·ColumnTransformer, 교차 검증, 그리드·랜덤 서치, 부트스트랩, joblib |
| [3장](ml/03_classification.md) | 분류 | MNIST, SGDClassifier, StratifiedKFold, 오차 행렬, 정밀도·재현율·F1, 임곗값, PR·ROC 곡선, AUC, OvR·OvO, 오류 분석, 다중 레이블(ClassifierChain), 다중 출력 |
| [4장](ml/04_training_models.md) | 모델 훈련 | 정규 방정식, 유사역행렬(SVD), 배치·확률적·미니배치 경사 하강법, 학습 스케줄, 다항 회귀, 학습 곡선, 편향·분산, 릿지·라쏘·엘라스틱넷, 조기 종료, 로지스틱 회귀, 결정 경계, 소프트맥스 회귀 |
| [5장](ml/05_decision_trees.md) | 결정 트리 | export_graphviz, 지니 불순도, predict_proba, CART 비용 함수, 계산 복잡도, 엔트로피, 규제(max_depth·min_samples_leaf), DecisionTreeRegressor, 축 방향 민감성·PCA, 분산 문제, tree_ 구조, 다수결 숲 |
| [6장](ml/06_ensemble_learning_and_random_forests.md) | 앙상블 학습과 랜덤 포레스트 | 큰 수의 법칙, 직접·간접 투표, 배깅·페이스팅, OOB 평가, 랜덤 패치·서브스페이스, 랜덤 포레스트, 엑스트라 트리, 특성 중요도, AdaBoost, 그레이디언트 부스팅(축소·조기 종료), HistGradientBoosting, 스태킹 |
| [7장](ml/07_dimensionality.md) | 차원 축소 | 차원의 저주, 투영·매니폴드 학습, PCA(SVD), 설명된 분산 비율, 압축, 랜덤·점진적 PCA, 랜덤 투영(존슨–린덴슈트라우스), LLE, MDS·Isomap·t-SNE, 커널 PCA |
| [8장](ml/08_unsupervised_learning.md) | 비지도 학습 | k-평균, 이너셔, k-평균++, 미니배치 k-평균, 엘보·실루엣, 이미지 분할, 준지도 학습(레이블 전파), DBSCAN, 가우스 혼합(EM), covariance_type, 이상치 탐지, BIC·AIC, 베이즈 가우스 혼합 |
