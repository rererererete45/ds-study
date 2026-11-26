# 추천 시스템 — RecBole 94개 모델 정리

[RecBole](https://recbole.io/) 1.2.1에 구현된 추천 모델 94개를 하나씩 공부하고 정리하는 노트다. 모델마다 논문 요약, RecBole 코드 해설, 실험 결과를 남기고, 핵심 모델은 파이토치로 직접 다시 구현한다.

## 정리 깊이

| 깊이 | 모델 수 | 예상 | 하는 일 |
|---|---|---|---|
| **A 정독** | 18 | 2주 | 논문 정독 · 핵심 수식 정리 · RecBole 코드 해설 · 파이토치로 직접 재구현 · 실험 비교 |
| **B 보통** | 43 | 1주 | 논문 요약 · RecBole 코드 해설 · RecBole로 실행 · 비교표에 결과 추가 |
| **C 가볍게** | 33 | 0.5주 | 한 문단 요약 · RecBole로 실행 · 비교표에 결과 추가 |

전체 예상 95.5주. 상태 표시: ⬜ 예정 · 🟦 진행 중 · ✅ 완료

## 폴더 구조

```
recsys/
├── README.md                  # 이 문서: 커리큘럼과 진도표
├── 01_baseline/               # B1 베이스라인
├── 02_sequential_core/        # B2 순차 핵심
├── 03_mf_similarity/          # B3 행렬 분해·유사도
├── 04_neural_cf/              # B4 신경망 CF
├── 05_autoencoder_generative/ # B5 오토인코더·생성
├── 06_graph_cf/               # B6 그래프 CF
├── 07_sequential_extended/    # B7 순차 확장
├── 08_transformer_ssl/        # B8 트랜스포머·자기지도
├── 09_graph_session/          # B9 그래프 세션
├── 10_feature_sequential/     # B10 피처 결합 순차
├── 11_ctr_basic/              # B11 CTR 기초
├── 12_ctr_deep/               # B12 CTR 딥러닝
├── 13_interest_ctr/           # B13 관심사·최신 CTR
├── 14_knowledge_graph/        # B14 지식 그래프
├── code/                      # A 깊이 모델의 파이토치 직접 구현
└── results/                   # 모델별 실험 결과 비교표
```

## 학습 순서

앞 블록에서 배운 것을 다음 블록이 이어받도록 배치했다. 행렬 분해 → 신경망 → 오토인코더 → 그래프 → 순차 확장 → CTR → 트랜스포머 → 지식 그래프 순이다.

| 블록 | 주제 | 모델 수 | 예상(주) |
|---|---|---|---|
| [B1 베이스라인](01_baseline/) | 평가 기준선과 랭킹 손실(BPR)의 기본 | 5 | 4.5 |
| [B2 순차 핵심](02_sequential_core/) | 다음 아이템 예측의 기본 세 모델 | 3 | 5 |
| [B3 행렬 분해·유사도](03_mf_similarity/) | 아이템 간 유사도와 선형 모델 | 8 | 7 |
| [B4 신경망 CF](04_neural_cf/) | 내적을 신경망으로 바꾼 협업 필터링 | 5 | 5.5 |
| [B5 오토인코더·생성](05_autoencoder_generative/) | 사용자 벡터를 복원·생성하는 모델 | 8 | 9.5 |
| [B6 그래프 CF](06_graph_cf/) | 사용자-아이템 그래프 위의 메시지 전달 | 7 | 8.5 |
| [B7 순차 확장](07_sequential_extended/) | RNN·CNN·어텐션·반복 소비 등 다양한 순차 모델 | 11 | 8 |
| [B11 CTR 기초](11_ctr_basic/) | 클릭률 예측과 피처 교차의 기본 | 8 | 6.5 |
| [B12 CTR 딥러닝](12_ctr_deep/) | 피처 교차를 딥러닝으로 | 9 | 12 |
| [B8 트랜스포머·자기지도](08_transformer_ssl/) | 트랜스포머 기반 순차 추천과 자기지도 학습 | 8 | 9 |
| [B9 그래프 세션](09_graph_session/) | 세션을 그래프로 보는 모델 | 2 | 3 |
| [B10 피처 결합 순차](10_feature_sequential/) | 아이템 속성·지식을 순차 모델에 결합 | 5 | 4 |
| [B13 관심사·최신 CTR](13_interest_ctr/) | 사용자 관심 어텐션과 최신 피처 교차 | 5 | 4.5 |
| [B14 지식 그래프](14_knowledge_graph/) | 외부 지식 그래프로 추천 보강 | 10 | 8.5 |

## 진도표

| # | 모델 | 분류 | 블록 | 깊이 | 상태 |
|---|---|---|---|---|---|
| 01 | [Random](01_baseline/01_Random.md) | 일반 추천 | B1 베이스라인 | C | ⬜ |
| 02 | [Pop](01_baseline/02_Pop.md) | 일반 추천 | B1 베이스라인 | C | ⬜ |
| 03 | [ItemKNN](01_baseline/03_ItemKNN.md) | 일반 추천 | B1 베이스라인 | B | ⬜ |
| 04 | [AsymKNN](01_baseline/04_AsymKNN.md) | 일반 추천 | B1 베이스라인 | C | ⬜ |
| 05 | [BPR](01_baseline/05_BPR.md) | 일반 추천 | B1 베이스라인 | A | ⬜ |
| 06 | [SASRec](02_sequential_core/06_SASRec.md) | 순차 추천 | B2 순차 핵심 | A | ⬜ |
| 07 | [EASE](03_mf_similarity/07_EASE.md) | 일반 추천 | B3 행렬 분해·유사도 | A | ⬜ |
| 08 | [SLIMElastic](03_mf_similarity/08_SLIMElastic.md) | 일반 추천 | B3 행렬 분해·유사도 | B | ⬜ |
| 09 | [ADMMSLIM](03_mf_similarity/09_ADMMSLIM.md) | 일반 추천 | B3 행렬 분해·유사도 | C | ⬜ |
| 10 | [FISM](03_mf_similarity/10_FISM.md) | 일반 추천 | B3 행렬 분해·유사도 | B | ⬜ |
| 11 | [NAIS](03_mf_similarity/11_NAIS.md) | 일반 추천 | B3 행렬 분해·유사도 | B | ⬜ |
| 12 | [NCEPLRec](03_mf_similarity/12_NCEPLRec.md) | 일반 추천 | B3 행렬 분해·유사도 | C | ⬜ |
| 13 | [ENMF](03_mf_similarity/13_ENMF.md) | 일반 추천 | B3 행렬 분해·유사도 | C | ⬜ |
| 14 | [LINE](03_mf_similarity/14_LINE.md) | 일반 추천 | B3 행렬 분해·유사도 | C | ⬜ |
| 15 | [FPMC](02_sequential_core/15_FPMC.md) | 순차 추천 | B2 순차 핵심 | B | ⬜ |
| 16 | [GRU4Rec](02_sequential_core/16_GRU4Rec.md) | 순차 추천 | B2 순차 핵심 | A | ⬜ |
| 17 | [NeuMF](04_neural_cf/17_NeuMF.md) | 일반 추천 | B4 신경망 CF | A | ⬜ |
| 18 | [DMF](04_neural_cf/18_DMF.md) | 일반 추천 | B4 신경망 CF | B | ⬜ |
| 19 | [ConvNCF](04_neural_cf/19_ConvNCF.md) | 일반 추천 | B4 신경망 CF | B | ⬜ |
| 20 | [NNCF](04_neural_cf/20_NNCF.md) | 일반 추천 | B4 신경망 CF | C | ⬜ |
| 21 | [SimpleX](04_neural_cf/21_SimpleX.md) | 일반 추천 | B4 신경망 CF | B | ⬜ |
| 22 | [CDAE](05_autoencoder_generative/22_CDAE.md) | 일반 추천 | B5 오토인코더·생성 | B | ⬜ |
| 23 | [MultiDAE](05_autoencoder_generative/23_MultiDAE.md) | 일반 추천 | B5 오토인코더·생성 | B | ⬜ |
| 24 | [MultiVAE](05_autoencoder_generative/24_MultiVAE.md) | 일반 추천 | B5 오토인코더·생성 | A | ⬜ |
| 25 | [MacridVAE](05_autoencoder_generative/25_MacridVAE.md) | 일반 추천 | B5 오토인코더·생성 | B | ⬜ |
| 26 | [RecVAE](05_autoencoder_generative/26_RecVAE.md) | 일반 추천 | B5 오토인코더·생성 | B | ⬜ |
| 27 | [RaCT](05_autoencoder_generative/27_RaCT.md) | 일반 추천 | B5 오토인코더·생성 | C | ⬜ |
| 28 | [DiffRec](05_autoencoder_generative/28_DiffRec.md) | 일반 추천 | B5 오토인코더·생성 | A | ⬜ |
| 29 | [LDiffRec](05_autoencoder_generative/29_LDiffRec.md) | 일반 추천 | B5 오토인코더·생성 | B | ⬜ |
| 30 | [GCMC](06_graph_cf/30_GCMC.md) | 일반 추천 | B6 그래프 CF | B | ⬜ |
| 31 | [SpectralCF](06_graph_cf/31_SpectralCF.md) | 일반 추천 | B6 그래프 CF | C | ⬜ |
| 32 | [NGCF](06_graph_cf/32_NGCF.md) | 일반 추천 | B6 그래프 CF | B | ⬜ |
| 33 | [LightGCN](06_graph_cf/33_LightGCN.md) | 일반 추천 | B6 그래프 CF | A | ⬜ |
| 34 | [DGCF](06_graph_cf/34_DGCF.md) | 일반 추천 | B6 그래프 CF | B | ⬜ |
| 35 | [SGL](06_graph_cf/35_SGL.md) | 일반 추천 | B6 그래프 CF | A | ⬜ |
| 36 | [NCL](06_graph_cf/36_NCL.md) | 일반 추천 | B6 그래프 CF | B | ⬜ |
| 37 | [NARM](07_sequential_extended/37_NARM.md) | 순차 추천 | B7 순차 확장 | B | ⬜ |
| 38 | [STAMP](07_sequential_extended/38_STAMP.md) | 순차 추천 | B7 순차 확장 | B | ⬜ |
| 39 | [Caser](07_sequential_extended/39_Caser.md) | 순차 추천 | B7 순차 확장 | B | ⬜ |
| 40 | [NextItNet](07_sequential_extended/40_NextItNet.md) | 순차 추천 | B7 순차 확장 | B | ⬜ |
| 41 | [RepeatNet](07_sequential_extended/41_RepeatNet.md) | 순차 추천 | B7 순차 확장 | B | ⬜ |
| 42 | [TransRec](07_sequential_extended/42_TransRec.md) | 순차 추천 | B7 순차 확장 | C | ⬜ |
| 43 | [FOSSIL](07_sequential_extended/43_FOSSIL.md) | 순차 추천 | B7 순차 확장 | C | ⬜ |
| 44 | [HRM](07_sequential_extended/44_HRM.md) | 순차 추천 | B7 순차 확장 | C | ⬜ |
| 45 | [NPE](07_sequential_extended/45_NPE.md) | 순차 추천 | B7 순차 확장 | C | ⬜ |
| 46 | [SHAN](07_sequential_extended/46_SHAN.md) | 순차 추천 | B7 순차 확장 | C | ⬜ |
| 47 | [HGN](07_sequential_extended/47_HGN.md) | 순차 추천 | B7 순차 확장 | C | ⬜ |
| 48 | [LR](11_ctr_basic/48_LR.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | C | ⬜ |
| 49 | [XGBOOST](11_ctr_basic/49_XGBOOST.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | C | ⬜ |
| 50 | [LIGHTGBM](11_ctr_basic/50_LIGHTGBM.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | C | ⬜ |
| 51 | [FM](11_ctr_basic/51_FM.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | A | ⬜ |
| 52 | [FFM](11_ctr_basic/52_FFM.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | B | ⬜ |
| 53 | [FwFM](11_ctr_basic/53_FwFM.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | C | ⬜ |
| 54 | [FNN](11_ctr_basic/54_FNN.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | C | ⬜ |
| 55 | [PNN](11_ctr_basic/55_PNN.md) | 컨텍스트(CTR) 추천 | B11 CTR 기초 | B | ⬜ |
| 56 | [WideDeep](12_ctr_deep/56_WideDeep.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | A | ⬜ |
| 57 | [NFM](12_ctr_deep/57_NFM.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | B | ⬜ |
| 58 | [AFM](12_ctr_deep/58_AFM.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | B | ⬜ |
| 59 | [DeepFM](12_ctr_deep/59_DeepFM.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | A | ⬜ |
| 60 | [DCN](12_ctr_deep/60_DCN.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | B | ⬜ |
| 61 | [DCN V2](12_ctr_deep/61_DCNV2.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | A | ⬜ |
| 62 | [xDeepFM](12_ctr_deep/62_xDeepFM.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | B | ⬜ |
| 63 | [AutoInt](12_ctr_deep/63_AutoInt.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | B | ⬜ |
| 64 | [DSSM](12_ctr_deep/64_DSSM.md) | 컨텍스트(CTR) 추천 | B12 CTR 딥러닝 | B | ⬜ |
| 65 | [BERT4Rec](08_transformer_ssl/65_BERT4Rec.md) | 순차 추천 | B8 트랜스포머·자기지도 | A | ⬜ |
| 66 | [S3Rec](08_transformer_ssl/66_S3Rec.md) | 순차 추천 | B8 트랜스포머·자기지도 | A | ⬜ |
| 67 | [LightSANs](08_transformer_ssl/67_LightSANs.md) | 순차 추천 | B8 트랜스포머·자기지도 | B | ⬜ |
| 68 | [CORE](08_transformer_ssl/68_CORE.md) | 순차 추천 | B8 트랜스포머·자기지도 | B | ⬜ |
| 69 | [FEARec](08_transformer_ssl/69_FEARec.md) | 순차 추천 | B8 트랜스포머·자기지도 | B | ⬜ |
| 70 | [SINE](08_transformer_ssl/70_SINE.md) | 순차 추천 | B8 트랜스포머·자기지도 | B | ⬜ |
| 71 | [SASRecCPR](08_transformer_ssl/71_SASRecCPR.md) | 순차 추천 | B8 트랜스포머·자기지도 | C | ⬜ |
| 72 | [GRU4RecCPR](08_transformer_ssl/72_GRU4RecCPR.md) | 순차 추천 | B8 트랜스포머·자기지도 | C | ⬜ |
| 73 | [SRGNN](09_graph_session/73_SRGNN.md) | 순차 추천 | B9 그래프 세션 | A | ⬜ |
| 74 | [GCSAN](09_graph_session/74_GCSAN.md) | 순차 추천 | B9 그래프 세션 | B | ⬜ |
| 75 | [GRU4RecF](10_feature_sequential/75_GRU4RecF.md) | 순차 추천 | B10 피처 결합 순차 | C | ⬜ |
| 76 | [SASRecF](10_feature_sequential/76_SASRecF.md) | 순차 추천 | B10 피처 결합 순차 | B | ⬜ |
| 77 | [FDSA](10_feature_sequential/77_FDSA.md) | 순차 추천 | B10 피처 결합 순차 | B | ⬜ |
| 78 | [GRU4RecKG](10_feature_sequential/78_GRU4RecKG.md) | 순차 추천 | B10 피처 결합 순차 | C | ⬜ |
| 79 | [KSR](10_feature_sequential/79_KSR.md) | 순차 추천 | B10 피처 결합 순차 | B | ⬜ |
| 80 | [DIN](13_interest_ctr/80_DIN.md) | 컨텍스트(CTR) 추천 | B13 관심사·최신 CTR | A | ⬜ |
| 81 | [DIEN](13_interest_ctr/81_DIEN.md) | 컨텍스트(CTR) 추천 | B13 관심사·최신 CTR | B | ⬜ |
| 82 | [FiGNN](13_interest_ctr/82_FiGNN.md) | 컨텍스트(CTR) 추천 | B13 관심사·최신 CTR | C | ⬜ |
| 83 | [KD_DAGFM](13_interest_ctr/83_KD_DAGFM.md) | 컨텍스트(CTR) 추천 | B13 관심사·최신 CTR | C | ⬜ |
| 84 | [EulerNet](13_interest_ctr/84_EulerNet.md) | 컨텍스트(CTR) 추천 | B13 관심사·최신 CTR | C | ⬜ |
| 85 | [CKE](14_knowledge_graph/85_CKE.md) | 지식 그래프 추천 | B14 지식 그래프 | B | ⬜ |
| 86 | [CFKG](14_knowledge_graph/86_CFKG.md) | 지식 그래프 추천 | B14 지식 그래프 | C | ⬜ |
| 87 | [KTUP](14_knowledge_graph/87_KTUP.md) | 지식 그래프 추천 | B14 지식 그래프 | C | ⬜ |
| 88 | [RippleNet](14_knowledge_graph/88_RippleNet.md) | 지식 그래프 추천 | B14 지식 그래프 | B | ⬜ |
| 89 | [MKR](14_knowledge_graph/89_MKR.md) | 지식 그래프 추천 | B14 지식 그래프 | C | ⬜ |
| 90 | [KGCN](14_knowledge_graph/90_KGCN.md) | 지식 그래프 추천 | B14 지식 그래프 | B | ⬜ |
| 91 | [KGNNLS](14_knowledge_graph/91_KGNNLS.md) | 지식 그래프 추천 | B14 지식 그래프 | C | ⬜ |
| 92 | [KGAT](14_knowledge_graph/92_KGAT.md) | 지식 그래프 추천 | B14 지식 그래프 | A | ⬜ |
| 93 | [KGIN](14_knowledge_graph/93_KGIN.md) | 지식 그래프 추천 | B14 지식 그래프 | B | ⬜ |
| 94 | [MCCLK](14_knowledge_graph/94_MCCLK.md) | 지식 그래프 추천 | B14 지식 그래프 | C | ⬜ |

## 실습 환경

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install recbole==1.2.1
```

- 평가: 일반·순차 추천은 Recall@10 · NDCG@10 · MRR@10 · Hit@10, CTR은 AUC · LogLoss

## 데이터셋

[RecBole 데이터셋 목록](https://recbole.io/dataset_list.html)에서 모델마다 1~3개를 골라 각 노트 머리말(🗂️ 추천 데이터)과 실험 표에 넣어 두었다.

- **공통 비교용**: 모든 모델이 `ml-1m`(MovieLens-1M)을 돌린다. [결과 비교표](results/README.md)는 이 데이터 기준이다. 지식 그래프 모델은 KG 링크 파일이 붙은 `ml-1m(+KG)`를 쓴다.
- **모델 특성용**: 원 논문 실험이나 데이터 성격에 맞는 데이터를 더한다. 깊이 A는 2개, B는 1개, C는 `ml-1m`만.
- 원래 크기가 큰 데이터(Netflix, YOOCHOOSE, LFM-1b, Criteo, Avazu)는 샘플링해서 쓴다. YOOCHOOSE는 세션 추천 논문들처럼 최근 1/64만 쓴다.
- RecBole의 `dataset=` 이름은 내려받은 atomic 파일 폴더 이름을 따른다. `ml-1m` 말고는 처음 쓸 때 이름을 확인한다.

| 데이터 | 도메인 | 성격 | 주로 쓰는 블록 |
|---|---|---|---|
| ml-1m | 영화 | 조밀한 평점·시간 정보. 공통 비교용 | 전체 |
| Amazon-Books | 쇼핑(도서) | 크고 희소한 암묵 피드백, KG 연결 가능 | 일반 CF · 그래프 · 생성 · 지식 그래프 |
| Amazon-Beauty | 쇼핑(뷰티) | 짧은 순차, 아이템 카테고리 피처 | 트랜스포머 · 피처 결합 순차 |
| Amazon-Electronics | 쇼핑(전자) | 사용자 행동 이력 + 카테고리 | 관심사 CTR (DIN·DIEN) |
| Yelp2018 · Yelp2022 | 지역 상점 | 희소한 리뷰 기반 상호작용 | 일반 CF · 그래프 · 자기지도 |
| Netflix | 영화 | 매우 큰 평점 행렬 (샘플링) | 유사도·선형 · 오토인코더 |
| Pinterest | 이미지 | 암묵 피드백 | 신경망 CF · FISM·NAIS |
| Gowalla | 체크인 | 그래프 추천 표준 벤치마크 | 그래프 CF |
| Foursquare | 체크인 | 위치 이동 순서 | CNN 순차 (Caser) |
| Ta Feng | 식료품 | 장바구니 거래 | FPMC |
| YOOCHOOSE · DIGINETICA | 쇼핑 세션 | 익명 세션 클릭 로그 | 세션 추천 (GRU4Rec·NARM·SR-GNN 등) |
| LFM-1b | 음악 | 대규모 청취 로그 (일부만) | 음악 순차·반복 소비 |
| Steam | 게임 | 긴 사용자 이력 | 트랜스포머 순차 |
| Criteo · Avazu | 광고 | 클릭 로그, 범주형 피처 다수 (샘플링) | CTR 전체 |
| iPinYou · AliEC | 광고 | 실제 광고 입찰·노출 로그 | PNN · DIN |
| MIND | 뉴스 | 노출·클릭 로그, 텍스트 | DSSM(투 타워) |
| Lastfm-track | 음악 | KG 연결 | 지식 그래프 |

### 설치 메모

- (설치하며 겪은 문제와 해결법을 여기에 남긴다)

## 참고

- [RecBole 모델 목록](https://recbole.io/docs/user_guide/model_intro.html)
- [RecBole 하이퍼파라미터 탐색 결과](https://recbole.io/hyperparameters/index.html)
