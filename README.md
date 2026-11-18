# ds-study

데이터 사이언스 공부 기록

## SQL

| 장 | 주제 | 키워드 |
|---|---|---|
| [2장](sql/02_data_preparation.md) | 데이터 준비 | 데이터 타입, 프로파일링(분포·품질), 정제, 셰이핑 |
| [3장](sql/03_time_series.md) | 시계열 분석 | 날짜 함수, 트렌드, 요소 비교, 전체 대비 비율, 인덱싱, 이동평균, 누적값, YoY·MoM |

## 머신러닝 (핸즈온 머신러닝 with 사이킷런, 파이토치)

1장부터 순서대로 읽으면 한 권의 노트처럼 이어지도록 정리했다. 각 장은 앞 장에서 남긴 질문을 받아 시작하고, 다음 장으로 넘길 질문을 남기며 끝난다.

### 학습 로드맵

```
1장 한눈에 보는 머신러닝 ─ 용어·종류·과대적합·검증의 큰 지도
   ↓ 지도를 실제 데이터로
2장 프로젝트 처음부터 끝까지 ─ 회귀 워크플로 (분할 → 탐색 → 파이프라인 → 튜닝 → 배포)
   ↓ 같은 워크플로를 분류로
3장 분류 ─ 정확도 너머의 성능 측정 (정밀도·재현율·ROC)
   ↓ 블랙박스였던 fit()의 내부로
4장 모델 훈련 ─ 경사 하강법 · 규제 · 로지스틱 회귀
   ↓ 선형 모델과 다른 비선형 모델
5장 결정 트리 ─ 해석 쉬움, 그러나 분산이 큼
   ↓ 분산 문제를 여러 모델로 해결
6장 앙상블과 랜덤 포레스트 ─ 투표 · 배깅 · 부스팅 · 스태킹
   ↓ 모델 대신 데이터(특성 수)를 줄이기
7장 차원 축소 ─ PCA · 랜덤 투영 · 매니폴드 학습
   ↓ 레이블 없이 구조 찾기
8장 비지도 학습 ─ k-평균 · DBSCAN · 가우스 혼합
   ↓ 사이킷런 모델에서 신경망으로
9장 인공 신경망 소개 ─ 퍼셉트론 · MLP와 역전파 · 사이킷런 MLP · 튜닝 가이드라인
   ↓ fit() 한 줄을 직접 쓰는 훈련 루프로
10장 파이토치로 신경망 구축하기 ─ 텐서 · autograd · nn.Module · DataLoader · Optuna · 저장과 컴파일
   ↓ 얕은 MLP에서 깊은 신경망으로
11장 심층 신경망 훈련 ─ 초기화 · 활성화 함수 · 배치 정규화 · 전이 학습 · 옵티마이저 · 학습률 스케줄 · 규제
   ↓ 완전 연결 층 대신 이미지 구조를 쓰는 층으로
12장 합성곱 신경망을 사용한 심층 컴퓨터 비전 ─ 합성곱 · 풀링 · ResNet · 전이 학습 · 객체 탐지 · 분할
   ↓ 공간 구조에서 시간 구조로
13장 RNN과 CNN을 사용한 시퀀스 처리 ─ 순환 뉴런 · BPTT · 시계열 예측(나이브·SARIMA·RNN·Seq2Seq) · LSTM · GRU · 1D 합성곱 · WaveNet
   ↓ 숫자 시퀀스에서 텍스트 시퀀스로
14장 RNN과 어텐션을 사용한 자연어 처리 ─ char-RNN · 서브워드 토큰화 · 감성 분석 · 허깅 페이스 · 인코더-디코더 · 빔 검색 · 어텐션
   ↓ RNN을 걷어 내고 어텐션만으로
15장 자연어 처리와 챗봇을 위한 트랜스포머 ─ 위치 임베딩 · 멀티 헤드 어텐션 · 마스크 · 번역 트랜스포머 · BERT(MLM) · GPT 생성(top-p·퓨샷) · 챗봇(SFT·DPO)
   ↓ 텍스트 토큰에서 이미지 패치로
16장~ 비전과 멀티모달 트랜스포머 (정리 예정)
```

### 장별 노트

| 장 | 주제 | 키워드 |
|---|---|---|
| [1장](ml/01_the_machine_learning_landscape.md) | 한눈에 보는 머신러닝 | 머신러닝 정의, 지도·비지도·준지도·자기 지도·강화 학습, 배치·온라인 학습, 사례 기반(KNN)·모델 기반(선형 회귀), fit·predict, 대표성 없는 데이터, 과대적합·과소적합, 규제(Ridge), 하이퍼파라미터, 훈련·검증·테스트 세트, 교차 검증, 데이터 불일치, 공짜 점심 없음 |
| [2장](ml/02_end_to_end_machine_learning_project.md) | 머신러닝 프로젝트 처음부터 끝까지 | 계층 샘플링, 상관관계, SimpleImputer, OneHotEncoder, 스케일링, 사용자 정의 변환기, Pipeline·ColumnTransformer, 교차 검증, 그리드·랜덤 서치, 부트스트랩, joblib |
| [3장](ml/03_classification.md) | 분류 | MNIST, SGDClassifier, StratifiedKFold, 오차 행렬, 정밀도·재현율·F1, 임곗값, PR·ROC 곡선, AUC, OvR·OvO, 오류 분석, 다중 레이블(ClassifierChain), 다중 출력, 데이터 증강, 타이타닉, 스팸 분류기 |
| [4장](ml/04_training_linear_models.md) | 모델 훈련 | 정규 방정식, 유사역행렬(SVD), 배치·확률적·미니배치 경사 하강법, 학습 스케줄, 다항 회귀, 학습 곡선, 편향·분산, 릿지·라쏘·엘라스틱넷, 조기 종료, 로지스틱 회귀, 결정 경계, 소프트맥스 회귀 |
| [5장](ml/05_decision_trees.md) | 결정 트리 | export_graphviz, 지니 불순도, predict_proba, CART 비용 함수, 계산 복잡도, 엔트로피, 규제(max_depth·min_samples_leaf), DecisionTreeRegressor, 축 방향 민감성·PCA, 분산 문제, tree_ 구조, 다수결 숲 |
| [6장](ml/06_ensemble_learning_and_random_forests.md) | 앙상블 학습과 랜덤 포레스트 | 큰 수의 법칙, 직접·간접 투표, 배깅·페이스팅, OOB 평가, 랜덤 패치·서브스페이스, 랜덤 포레스트, 엑스트라 트리, 특성 중요도, AdaBoost, 그레이디언트 부스팅(축소·조기 종료), HistGradientBoosting, 스태킹 |
| [7장](ml/07_dimensionality_reduction.md) | 차원 축소 | 차원의 저주, 투영·매니폴드 학습, PCA(SVD), 설명된 분산 비율, 압축, 랜덤·점진적 PCA, 랜덤 투영(존슨–린덴슈트라우스), LLE, MDS·Isomap·t-SNE, 커널 PCA |
| [8장](ml/08_unsupervised_learning.md) | 비지도 학습 | k-평균, 이너셔, k-평균++, 미니배치 k-평균, 엘보·실루엣, 이미지 분할, 준지도 학습(레이블 전파), DBSCAN, 가우스 혼합(EM), covariance_type, 이상치 탐지, BIC·AIC, 베이즈 가우스 혼합 |
| [9장](ml/09_artificial_neural_networks.md) | 인공 신경망 소개 | 인공 뉴런·논리 연산, TLU, 퍼셉트론 학습 규칙, XOR 문제, 다층 퍼셉트론(MLP), 역전파, 활성화 함수(시그모이드·tanh·ReLU), MLPRegressor, MLPClassifier, 패션 MNIST, 조기 종료, 과잉 확신, 은닉 층 수·뉴런 수·학습률·배치 크기 |
| [10장](ml/10_neural_nets_with_pytorch.md) | 파이토치로 신경망 구축하기 | 텐서, 하드웨어 가속(GPU), autograd, 훈련 루프, nn.Linear·nn.Sequential, 옵티마이저(SGD), DataLoader, 미니배치 경사 하강법, torchmetrics, 학습 곡선, nn.Module(와이드 & 딥·다중 입력·다중 출력), 토치비전, CrossEntropyLoss(로짓), Optuna(TPE·가지치기), state_dict, TorchScript, torch.compile |
| [11장](ml/11_training_deep_neural_networks.md) | 심층 신경망 훈련 | 그레이디언트 소실·폭주, 글로럿·He·르쿤 초기화, LeakyReLU·ELU·SELU·GELU·Swish·Mish, 배치 정규화·층 정규화, 그레이디언트 클리핑, 전이 학습(동결·미세 튜닝), 모멘텀·네스테로프·AdaGrad·RMSProp·Adam·AdamW, 학습률 스케줄링(지수·성능 기반·웜업·코사인·1사이클), ℓ1·ℓ2 규제·가중치 감쇠, 드롭아웃, MC 드롭아웃, 맥스-노름, 조기 종료, CIFAR10 |
| [12장](ml/12_deep_computer_vision_with_cnns.md) | 합성곱 신경망을 사용한 심층 컴퓨터 비전 | 국부 수용장, nn.Conv2d(필터·특성 맵·패딩·스트라이드), 메모리 요구량, 최대·평균 풀링, 깊이 방향 풀링, 전역 평균 풀링, LeNet-5·AlexNet·GoogLeNet(인셉션·1×1 병목)·VGGNet·ResNet(스킵 연결)·Xception(분리 합성곱)·SENet·EfficientNet·ConvNeXt, ResNet-34 구현, LazyLinear, 토치비전 사전 훈련 모델, 전이 학습(Flowers102), 데이터 증강, 위치 추정(BoundingBoxes·IoU), NMS, 완전 합성곱 신경망, YOLO, mAP, 객체 추적, 시맨틱 분할(FCN·전치 합성곱) |
| [13장](ml/13_processing_sequences_using_rnns_and_cnns.md) | RNN과 CNN을 사용한 시퀀스 처리 | 순환 뉴런·메모리 셀, 시퀀스-투-시퀀스·시퀀스-투-벡터, BPTT, 시카고 승객 수 시계열, 계절성·차분, 나이브 예측, MAE·MAPE, ARIMA·SARIMA, ACF·PACF, TimeSeriesDataset, HuberLoss, ReduceLROnPlateau, nn.RNN, 심층 RNN, 다변량·다중 작업 예측, 여러 타임 스텝 예측, unfold, 시퀀스-투-시퀀스, 층 정규화, LSTM·LSTMCell, GRU·GRUCell, nn.Conv1d 다운샘플링, 인과적·팽창 합성곱, WaveNet, 죽은 ReLU 진단, 바흐 코랄 생성(온도 샘플링) |
| [14장](ml/14_nlp_with_rnns_and_attention.md) | RNN과 어텐션을 사용한 자연어 처리 | char-RNN, CharDataset, nn.Embedding, GRU, permute(0, 2, 1), 온도 샘플링(torch.multinomial), 상태가 있는 RNN(StatefulCharDataset·detach), IMDB, tokenizers(BPE·BBPE·WordPiece·유니그램), 패딩·잘라내기·어텐션 마스크, AutoTokenizer(GPT-2·BERT·ALBERT), collate_fn, padding_idx, 패킹된 시퀀스(pack_padded_sequence), 양방향 GRU, 사전 훈련 임베딩(from_pretrained·freeze), BERT 문맥 임베딩(last_hidden_state·pooler_output), BertForSequenceClassification, Trainer·TrainingArguments, pipeline, 모델 편향, 인코더–디코더 NMT, 교사 강요, ignore_index, 탐욕적 디코딩, 빔 검색·길이 페널티, 바다나우·루옹 어텐션, 점곱 어텐션(쿼리·키·값), 어텐션 패딩 마스크 |
| [15장](ml/15_transformers_for_nlp_and_chatbots.md) | 자연어 처리와 챗봇을 위한 트랜스포머 | Attention Is All You Need, 셀프 어텐션, 위치 임베딩(훈련형·사인/코사인), 스케일드 점곱 어텐션(√d), 멀티 헤드 어텐션(split_heads), nn.MultiheadAttention(in_proj_weight·batch_first), 인과 마스크(torch.triu·generate_square_subsequent_mask)·key_padding_mask, 잔차 연결·층 정규화(Post-LN), 위치별 피드포워드, 인코더·디코더 층·크로스 어텐션, nn.Transformer, NmtTransformer, 인코더 전용·디코더 전용·인코더–디코더, BERT(MLM 80/10/10·NSP·[CLS]), BertForMaskedLM, DataCollatorForLanguageModeling, fill-mask, Sentence Transformers(코사인 유사도), RoBERTa·DistilBERT·ALBERT·ELECTRA·DeBERTa, GPT·인과적 언어 모델, AutoModelForCausalLM(device_map·dtype), generate(do_sample·top_p), 퓨샷 프롬프트, Mistral-7B, 액세스 토큰, BobTheChatbot, SFT·RLHF·DPO, 시퀀스 로그 확률(gather), DPO 손실(참조 모델·β), TRL(SFTTrainer·DPOTrainer), Alpaca, HH-RLHF, Instruct 모델·채팅 템플릿, T5·BART, 고정 위치 인코딩(register_buffer·상대 위치) |

### 노트 읽는 법

모든 장은 같은 순서로 구성된다.

| 순서 | 섹션 | 내용 |
|---|---|---|
| 머리말 | 교재·노트북, 장 요약 | ⬅️ 앞 장과의 연결, ➡️ 다음 장으로 넘어가는 이유 |
| 1 | 이 장의 흐름 | 단계 · 교재 절 · **핵심 질문** · 결과물 표 |
| 2 | 데이터 | 데이터 · 형태 · 사용 절 · 비고 표 |
| 3 | 목차 | 절(`n.`)과 소절(`n-m.`) |
| 4 | 0. 준비 | 버전 확인 · 그래프 설정 (각 줄 설명은 [1장](ml/01_the_machine_learning_landscape.md#0-준비)) |
| 5 | 본문 `1.` ~ | 제목 옆 괄호 = 교재 절 번호 (예: `(3.3)`) |
| 6 | 연습문제 | 교재 연습문제 풀이 |
| 7 | 정리 | 전체 흐름 → 개념과 코드 대응 → (방법 비교) → 주요 결과 → (자주 쓴 코드 패턴) → 핵심 원칙 |
| 꼬리말 | 다음 장 | 이전 · 전체 목차 · 다음 장 링크 |

본문에서 쓰는 표시

| 표시 | 뜻 |
|---|---|
| > 🎯 **목적** | 이 소절에서 해결하려는 것 |
| **개념·원리** / **코드** / **결과 해석** | 소절 안의 정의·수식 / 노트북 코드 / 출력값의 의미 |
| > 📌 | 정리·참고 |
| > 💡 | 이해를 돕는 통찰·연결 |
| > ⚠️ | 주의할 점, 버전 차이 |
| > ✅ | 실험 결과 확인 |
| 보충 코드 | 노트북에 없지만 개념 확인용으로 직접 만든 코드 |
