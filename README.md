# ds-study

데이터 사이언스 공부 기록

## SQL

| 장 | 주제 | 키워드 |
|---|---|---|
| [2장](sql/02_data_preparation.md) | 데이터 준비 | 데이터 타입, 프로파일링(분포·품질), 정제, 셰이핑 |
| [3장](sql/03_time_series.md) | 시계열 분석 | 날짜 함수, 트렌드, 요소 비교, 전체 대비 비율, 인덱싱, 이동평균, 누적값, YoY·MoM |

## 머신러닝 (핸즈온 머신러닝 with 사이킷런, 파이토치)

1장부터 순서대로 읽으면 한 권의 노트처럼 이어지도록 정리했다. 각 장은 앞 장에서 남긴 질문을 받아 시작하고, 다음 장으로 넘길 질문을 남기며 끝난다.

### 이 노트만으로 공부하는 법

| 순서 | 할 일 | 자료 |
|---|---|---|
| 0 | 넘파이 `shape`, 행렬 곱, 미분, 소프트맥스가 낯설면 먼저 훑는다 (막힐 때 돌아와도 된다) | [0장. 수학과 도구](ml/00_math_and_tools.md) |
| 1 | 장 맨 앞 **🎓 강의를 시작하며**로 큰 그림(한 문장 요약·비유·도달 목표)을 잡고, **이 장의 흐름** 표로 "이 장이 답할 질문"을 본다 | 각 장 맨 위 |
| 2 | 절마다 **💬 강의 박스**(비유·핵심 한 줄·헷갈림 주의)를 먼저 읽고, 소절 단위로 🎯 **목적** → 개념 → 코드 → 그림 → 결과 해석 순서로 읽는다 | 각 장 본문 |
| 3 | 그림은 바로 아래 괄호(축·색·기호 설명)를 먼저 읽고 본다 | 각 장 본문 |
| 4 | 모르는 용어가 나오면 용어집에서 찾고, 링크로 처음 설명한 곳에 가 본다 | [📖 용어집](ml/glossary.md) |
| 5 | 연습문제의 질문만 먼저 보고 스스로 답한 뒤 답 요약과 비교한다 | 각 장 연습문제 |
| 6 | 장 끝 **정리**의 전체 흐름·핵심 원칙으로 복습하고, **🧠 5분 셀프 테스트**를 정답을 가린 채 소리 내어 풀어 본다 | 각 장 정리 |
| 7 | 코드를 직접 돌려 보려면 교재 저장소의 노트북을 같은 장 번호로 연다 (Colab 가능). 노트 속 숫자는 노트북 출력값이라 버전·난수에 따라 조금 다를 수 있다 | [ageron/handson-mlp](https://github.com/ageron/handson-mlp) |

### 학습 로드맵

```
0장 수학과 도구 (부록) ─ 넘파이·판다스·맷플롯립, 행렬 곱, 미분, 확률
   ↓ 도구를 갖추고
1장 한눈에 보는 머신러닝 ─ 용어·종류·과대적합·검증의 큰 지도
   ↓ 지도를 실제 데이터로
2장 프로젝트 처음부터 끝까지 ─ 회귀 워크플로 (분할 → 탐색 → 파이프라인 → 튜닝 → 배포)
   ↓ 같은 워크플로를 분류로
3장 분류 ─ 정확도 너머의 성능 측정 (정밀도·재현율·ROC)
   ↓ 블랙박스였던 fit()의 내부로
4장 모델 훈련 ─ 경사 하강법 · 규제 · 로지스틱 회귀
   ⋯ 부록 C 서포트 벡터 머신 ─ 2·3·6장에서 쓰는 SVC·SVR의 원리 (여기서 읽으면 좋다)
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
16장 비전 및 멀티모달 트랜스포머 ─ 패치 임베딩 · ViT 구현 · ViT/DeiT 미세 튜닝 · Swin · DINO 비지도 분할 · CLIP 제로샷 · Perceiver · BLIP-2 · Gemini
   ↓ 크고 느린 트랜스포머를 빠르고 가볍게
17장 트랜스포머 속도 높이기 ─ KV 캐시 · 추측 디코딩 · 희소 어텐션(BigBird) · LSH · Performer · GQA/MLA · 플래시 어텐션 · MoE · LoRA · 체크포인팅 · 패킹 · 그레이디언트 누적 · 병렬화
   ↓ 이해하는 모델에서 만들어 내는 모델로
18장 오토인코더, GAN 그리고 확산 모델 ─ 선형 AE = PCA · 적층·합성곱 AE · 이상치 탐지 · 가중치 묶기 · 잡음 제거 · 희소(KL) · VAE(재매개변수화·시맨틱 보간) · 이산 VAE(Gumbel-softmax) · GAN·모드 붕괴 · DCGAN · StyleGAN · 확산(DDPM·DDIM·U-Net) · 스테이블 디퓨전 · 플로 매칭
   ↓ 데이터를 흉내 내는 모델에서 보상으로 행동을 배우는 모델로
19장 강화 학습 ─ 에이전트·보상·정책 · Gymnasium(CartPole) · 신경망 정책 · 신용 할당(감가율·반환값) · REINFORCE · MDP·벨만 방정식 · Q-가치 반복 · TD 학습 · Q-러닝 · ε-그리디 · DQN(재생 버퍼) · 목표 네트워크·더블·PER·듀얼링 · 액터-크리틱 · PPO(클리핑·GAE) · 아타리 브레이크아웃(SB3) · AlphaZero·SAC·RLHF
```

### 장별 노트

| 장 | 주제 | 키워드 |
|---|---|---|
| [0장](ml/00_math_and_tools.md) | 시작하기 전에 — 수학과 파이썬 도구 (부록) | 넘파이 shape·axis·브로드캐스팅, 판다스, 그림 읽는 법, 내적·행렬 곱, 도함수·그레이디언트·연쇄 법칙, 정규분포·로그·소프트맥스, 기호 정리 |
| [📖 용어집](ml/glossary.md) | 핵심 용어 286개 | 주제별 한 줄 뜻 + 처음 설명하는 곳 링크 |
| [1장](ml/01_the_machine_learning_landscape.md) | 한눈에 보는 머신러닝 | 머신러닝 정의, 지도·비지도·준지도·자기 지도·강화 학습, 배치·온라인 학습, 사례 기반(KNN)·모델 기반(선형 회귀), fit·predict, 대표성 없는 데이터, 과대적합·과소적합, 규제(Ridge), 하이퍼파라미터, 훈련·검증·테스트 세트, 교차 검증, 데이터 불일치, 공짜 점심 없음 |
| [2장](ml/02_end_to_end_machine_learning_project.md) | 머신러닝 프로젝트 처음부터 끝까지 | 계층 샘플링, 상관관계, SimpleImputer, OneHotEncoder, 스케일링, 사용자 정의 변환기, Pipeline·ColumnTransformer, 교차 검증, 그리드·랜덤 서치, 부트스트랩, joblib |
| [3장](ml/03_classification.md) | 분류 | MNIST, SGDClassifier, StratifiedKFold, 오차 행렬, 정밀도·재현율·F1, 임곗값, PR·ROC 곡선, AUC, OvR·OvO, 오류 분석, 다중 레이블(ClassifierChain), 다중 출력, 데이터 증강, 타이타닉, 스팸 분류기 |
| [4장](ml/04_training_linear_models.md) | 모델 훈련 | 정규 방정식, 유사역행렬(SVD), 배치·확률적·미니배치 경사 하강법, 학습 스케줄, 다항 회귀, 학습 곡선, 편향·분산, 릿지·라쏘·엘라스틱넷, 조기 종료, 로지스틱 회귀, 결정 경계, 소프트맥스 회귀 |
| [5장](ml/05_decision_trees.md) | 결정 트리 | export_graphviz, 지니 불순도, predict_proba, CART 비용 함수, 계산 복잡도, 엔트로피, 규제(max_depth·min_samples_leaf), DecisionTreeRegressor, 축 방향 민감성·PCA, 분산 문제, tree_ 구조, 다수결 숲 |
| [6장](ml/06_ensemble_learning_and_random_forests.md) | 앙상블 학습과 랜덤 포레스트 | 큰 수의 법칙, 직접·간접 투표, 배깅·페이스팅, OOB 평가, 랜덤 패치·서브스페이스, 랜덤 포레스트, 엑스트라 트리, 특성 중요도, AdaBoost, 그레이디언트 부스팅(축소·조기 종료), HistGradientBoosting, 스태킹 |
| [7장](ml/07_dimensionality_reduction.md) | 차원 축소 | 차원의 저주, 투영·매니폴드 학습, PCA(SVD), 설명된 분산 비율, 압축, 랜덤·점진적 PCA, 랜덤 투영(존슨–린덴슈트라우스), LLE, MDS·Isomap·t-SNE, 커널 PCA |
| [8장](ml/08_unsupervised_learning.md) | 비지도 학습 | k-평균, 이너셔, k-평균++, 미니배치 k-평균, 엘보·실루엣, 이미지 분할, 준지도 학습(레이블 전파), DBSCAN, 가우스 혼합(EM), covariance_type, 이상치 탐지, BIC·AIC, 베이즈 가우스 혼합, 올리베티 얼굴(군집 특성·얼굴 생성·재구성 오차) |
| [부록 C](ml/C_support_vector_machines.md) | 서포트 벡터 머신 (4장 다음에 읽기) | 라지 마진, 서포트 벡터, 소프트 마진(`C`), 힌지 손실, 커널 트릭(다항·RBF, `gamma`), SVM 회귀(ε), 목적 함수와 쌍대 문제, LinearSVC·SVC·SGD 선택 |
| [9장](ml/09_artificial_neural_networks.md) | 인공 신경망 소개 | 인공 뉴런·논리 연산, TLU, 퍼셉트론 학습 규칙, XOR 문제, 다층 퍼셉트론(MLP), 역전파, 활성화 함수(시그모이드·tanh·ReLU), MLPRegressor, MLPClassifier, 패션 MNIST, 조기 종료, 과잉 확신, 은닉 층 수·뉴런 수·학습률·배치 크기 |
| [10장](ml/10_neural_nets_with_pytorch.md) | 파이토치로 신경망 구축하기 | 텐서, 하드웨어 가속(GPU), autograd, 훈련 루프, nn.Linear·nn.Sequential, 옵티마이저(SGD), DataLoader, 미니배치 경사 하강법, torchmetrics, 학습 곡선, nn.Module(와이드 & 딥·다중 입력·다중 출력), 토치비전, CrossEntropyLoss(로짓), Optuna(TPE·가지치기), state_dict, TorchScript, torch.compile |
| [11장](ml/11_training_deep_neural_networks.md) | 심층 신경망 훈련 | 그레이디언트 소실·폭주, 글로럿·He·르쿤 초기화, LeakyReLU·ELU·SELU·GELU·Swish·Mish, 배치 정규화·층 정규화, 그레이디언트 클리핑, 전이 학습(동결·미세 튜닝), 모멘텀·네스테로프·AdaGrad·RMSProp·Adam·AdamW, 학습률 스케줄링(지수·성능 기반·웜업·코사인·1사이클), ℓ1·ℓ2 규제·가중치 감쇠, 드롭아웃, MC 드롭아웃, 맥스-노름, 조기 종료, CIFAR10 |
| [12장](ml/12_deep_computer_vision_with_cnns.md) | 합성곱 신경망을 사용한 심층 컴퓨터 비전 | 국부 수용장, nn.Conv2d(필터·특성 맵·패딩·스트라이드), 메모리 요구량, 최대·평균 풀링, 깊이 방향 풀링, 전역 평균 풀링, LeNet-5·AlexNet·GoogLeNet(인셉션·1×1 병목)·VGGNet·ResNet(스킵 연결)·Xception(분리 합성곱)·SENet·EfficientNet·ConvNeXt, ResNet-34 구현, LazyLinear, 토치비전 사전 훈련 모델, 전이 학습(Flowers102), 데이터 증강, 위치 추정(BoundingBoxes·IoU), NMS, 완전 합성곱 신경망, YOLO, mAP, 객체 추적, 시맨틱 분할(FCN·전치 합성곱) |
| [13장](ml/13_processing_sequences_using_rnns_and_cnns.md) | RNN과 CNN을 사용한 시퀀스 처리 | 순환 뉴런·메모리 셀, 시퀀스-투-시퀀스·시퀀스-투-벡터, BPTT, 시카고 승객 수 시계열, 계절성·차분, 나이브 예측, MAE·MAPE, ARIMA·SARIMA, ACF·PACF, TimeSeriesDataset, HuberLoss, ReduceLROnPlateau, nn.RNN, 심층 RNN, 다변량·다중 작업 예측, 여러 타임 스텝 예측, unfold, 시퀀스-투-시퀀스, 층 정규화, LSTM·LSTMCell, GRU·GRUCell, nn.Conv1d 다운샘플링, 인과적·팽창 합성곱, WaveNet, 죽은 ReLU 진단, 바흐 코랄 생성(온도 샘플링) |
| [14장](ml/14_nlp_with_rnns_and_attention.md) | RNN과 어텐션을 사용한 자연어 처리 | char-RNN, CharDataset, nn.Embedding, GRU, permute(0, 2, 1), 온도 샘플링(torch.multinomial), 상태가 있는 RNN(StatefulCharDataset·detach), IMDB, tokenizers(BPE·BBPE·WordPiece·유니그램), 패딩·잘라내기·어텐션 마스크, AutoTokenizer(GPT-2·BERT·ALBERT), collate_fn, padding_idx, 패킹된 시퀀스(pack_padded_sequence), 양방향 GRU, 사전 훈련 임베딩(from_pretrained·freeze), BERT 문맥 임베딩(last_hidden_state·pooler_output), BertForSequenceClassification, Trainer·TrainingArguments, pipeline, 모델 편향, 인코더–디코더 NMT, 교사 강요, ignore_index, 탐욕적 디코딩, 빔 검색·길이 페널티, 바다나우·루옹 어텐션, 점곱 어텐션(쿼리·키·값), 어텐션 패딩 마스크 |
| [15장](ml/15_transformers_for_nlp_and_chatbots.md) | 자연어 처리와 챗봇을 위한 트랜스포머 | Attention Is All You Need, 셀프 어텐션, 위치 임베딩(훈련형·사인/코사인), 스케일드 점곱 어텐션(√d), 멀티 헤드 어텐션(split_heads), nn.MultiheadAttention(in_proj_weight·batch_first), 인과 마스크(torch.triu·generate_square_subsequent_mask)·key_padding_mask, 잔차 연결·층 정규화(Post-LN), 위치별 피드포워드, 인코더·디코더 층·크로스 어텐션, nn.Transformer, NmtTransformer, 인코더 전용·디코더 전용·인코더–디코더, BERT(MLM 80/10/10·NSP·[CLS]), BertForMaskedLM, DataCollatorForLanguageModeling, fill-mask, Sentence Transformers(코사인 유사도), RoBERTa·DistilBERT·ALBERT·ELECTRA·DeBERTa, GPT·인과적 언어 모델, AutoModelForCausalLM(device_map·dtype), generate(do_sample·top_p), 퓨샷 프롬프트, Mistral-7B, 액세스 토큰, BobTheChatbot, SFT·RLHF·DPO, 시퀀스 로그 확률(gather), DPO 손실(참조 모델·β), TRL(SFTTrainer·DPOTrainer), Alpaca, HH-RLHF, Instruct 모델·채팅 템플릿, T5·BART, 고정 위치 인코딩(register_buffer·상대 위치) |
| [16장](ml/16_vision_and_multimodal_transformers.md) | 비전 및 멀티모달 트랜스포머 | 귀납적 편향, 시각 어텐션 RNN, DETR(객체 쿼리·헝가리안 매칭), ViT(패치 임베딩·Conv2d=패치별 선형 층·[CLS]·위치 임베딩·Pre-LN), interpolate_pos_encoding, Oxford-IIIT Pet, AutoImageProcessor(do_convert_rgb), collate_fn, Trainer(remove_unused_columns=False), DeiT(증류 토큰·하드 증류·강한 증강), PVT(공간 축소 어텐션), Swin(윈도·이동 윈도·패치 병합), DINO(자기 증류·EMA 교사·다중 크롭·센터링·샤프닝), 어텐션 맵 비지도 분할(output_attentions·eager), BEiT·MAE·DINOv2·SAM, VideoBERT, ViLBERT(공동 어텐션), CLIP(이중 인코더·대조 손실·logit_scale 온도·제로샷 분류·프롬프트 템플릿·get_image_features), DALL·E(dVAE 토큰), Perceiver(잠재 배열·크로스 어텐션·푸리에 위치 인코딩), Perceiver IO(출력 쿼리), Flamingo(Perceiver Resampler·tanh 게이트), BLIP(CapFilt)·BLIP-2(Q-Former·쿼리 32개·float16 generate), LLaVA·GPT-4o·Whisper·ImageBind·RT-2, Gemini API(google-genai) |
| [17장](ml/17_speeding_up_transformers.md) | 트랜스포머 속도 높이기 | 프리필·디코드(메모리 바운드), KV 캐시(use_cache·DynamicCache·past_key_values), 캐시 크기 계산, 추측 디코딩(assistant_model·수락 min(1, p/q)·잔여 분포·기대 토큰 수), 프롬프트 룩업·연속 배칭·페이지드 어텐션, 어텐션의 L² 비용, 희소 어텐션(Sparse Transformer·Longformer·BigBird·Routing Transformer), BigBird fill-mask(block_sparse → original_full), Reformer(각도 LSH·버킷), Linformer, Performer(양의 랜덤 특성 φ·직교화(QR)·FAVOR+·선형 어텐션), MQA·GQA(enable_gqa), MLA(잠재 캐시·흡수), 플래시 어텐션(HBM·SRAM·타일링·온라인 소프트맥스), SDPA 백엔드·attn_implementation, MoE(라우터·top-k·부하 균형 보조 손실·용량 계수·Mixtral·DeepSeek-V3), PEFT·LoRA(LoraConfig·get_peft_model·병합)·QLoRA, 활성화 체크포인팅(checkpoint), 버키팅·패킹(group_by_length), 그레이디언트 누적(토큰 수 정규화), 데이터·텐서·파이프라인 병렬(버블)·ZeRO·FSDP |
| [18장](ml/18_autoencoders_gans_and_diffusion_models.md) | 오토인코더, GAN 그리고 확산 모델 | 인코더·코딩·디코더, 재구성 손실, 과소완전·과대완전, 선형 오토인코더 = PCA(부분 공간), 적층 오토인코더, AutoencoderDataset, 재구성 오차 이상치 탐지(MNIST vs 패션 MNIST), t-SNE 시각화, 가중치 묶기(F.linear·weight.t()), 비지도 사전 훈련, 합성곱 오토인코더(ConvTranspose2d·output_padding), 순환 오토인코더(LSTM), 잡음 제거(Dropout·GaussianNoise·self.training), 희소 오토인코더(KL 발산·목표 희소도·namedtuple 출력), VAE(μ·log σ²·재매개변수화 트릭·잠재 손실·ELBO), 이미지 생성·시맨틱 보간(torch.lerp), 이산 VAE(Gumbel-softmax·hard=True·온도 어닐링), GAN(생성자·판별자·두 단계 훈련·BCELoss·detach·requires_grad), 모드 붕괴·내시 균형·경험 재생·미니배치 판별, DCGAN, ProGAN·StyleGAN(매핑 네트워크·AdaIN), 확산 모델(순방향·역방향·코사인 분산 스케줄·지름길 공식), DiffusionSample·DiffusionDataset, 시간 인코딩, U-Net(분리 합성곱·교차 스킵), DDPM·DDIM 샘플링, 잠재 확산·CFG, diffusers(sd-turbo·적대적 확산 증류), 플로 매칭(직선 경로·속도·오일러) |
| [19장](ml/19_reinforcement_learning.md) | 강화 학습 | 에이전트·환경·관측·행동·보상, 정책·정책 탐색(보상 지형), Gymnasium(make·reset·step·render·registry), CartPole, done·truncated, 하드코딩 정책, 신경망 정책(Bernoulli·log_prob·샘플링), 신용 할당 문제, 감가율·반환값(compute_returns)·행동 이득(정규화), REINFORCE(정책 그레이디언트 정리), 마르코프 연쇄(흡수 기대 스텝), MDP(전이 확률·보상), 벨만 최적 방정식, 가치 반복·Q-가치 반복, 감가율과 최적 정책, TD 학습(TD 타깃·TD 오차·부트스트래핑), Q-러닝(학습률 감소·오프-폴리시), ε-그리디·탐험 함수, SARSA vs Q-러닝(CliffWalking), 근사 Q-러닝·DQN, 재생 버퍼(deque·원형 버퍼), gather·inference_mode, 치명적 망각, 고정 Q-가치 타깃·더블 DQN(최대화 편향)·우선 순위 경험 재생·듀얼링 DQN·Rainbow, 시드 민감도, 액터-크리틱(공유 몸통·TD 오차·detach)·A2C·A3C, PPO(확률 비율·클리핑·GAE·엔트로피 보너스), ale_py·make_atari_env(AtariWrapper 전처리·VecEnv), VecFrameStack, CnnPolicy(NatureCNN), 텐서보드 지표, CheckpointCallback, PPO.load·predict, EpisodicLifeEnv와 게임 점수, AlphaGo·AlphaZero·MuZero·Dreamer, DDPG·TD3·SAC, 호기심·POET, 모방 학습·오프라인 RL·결정 트랜스포머·RLHF |

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
| 🎓 강의를 시작하며 | 장 전체를 한 문장·비유로 요약, 이 장이 끝나면 할 수 있어야 할 것 |
| > 💬 **굵은 제목** | 절의 직관(비유), 핵심 한 줄, 학생들이 자주 헷갈리는 포인트 |
| 🧠 5분 셀프 테스트 | 정답을 접어 둔 셀프 테스트 (▶를 눌러 정답 확인) |
| > 🎯 **목적** | 이 소절에서 해결하려는 것 |
| **개념·원리** / **코드** / **결과 해석** | 소절 안의 정의·수식 / 노트북 코드 / 출력값의 의미 |
| > 📌 | 정리·참고 |
| > 💡 | 이해를 돕는 통찰·연결 |
| > ⚠️ | 주의할 점, 버전 차이 |
| > ✅ | 실험 결과 확인 |
| 그림 `nb_…` / `fig…` | 노트북 출력 그림. 그림 아래 괄호 = 축·색·기호 읽는 법 |
| 그림 `extra_…` | 노트북에 없지만 개념 이해를 위해 직접 만든 보충 그림 |
| 보충 코드 | 노트북에 없지만 개념 확인용으로 직접 만든 코드 |
