# 실험 결과 비교

[📚 전체 목차](../README.md)

모든 모델을 같은 설정(RecBole 기본 하이퍼파라미터, 같은 데이터 분할)으로 돌린 결과를 모은다. 튜닝한 결과는 비고에 따로 적는다.

## 일반 추천

| # | 모델 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 비고 |
|---|---|---|---|---|---|---|
| 01 | [Random](../01_baseline/01_Random.md) |  |  |  |  |  |
| 02 | [Pop](../01_baseline/02_Pop.md) |  |  |  |  |  |
| 03 | [ItemKNN](../01_baseline/03_ItemKNN.md) |  |  |  |  |  |
| 04 | [AsymKNN](../01_baseline/04_AsymKNN.md) |  |  |  |  |  |
| 05 | [BPR](../01_baseline/05_BPR.md) |  |  |  |  |  |
| 07 | [EASE](../03_mf_similarity/07_EASE.md) |  |  |  |  |  |
| 08 | [SLIMElastic](../03_mf_similarity/08_SLIMElastic.md) |  |  |  |  |  |
| 09 | [ADMMSLIM](../03_mf_similarity/09_ADMMSLIM.md) |  |  |  |  |  |
| 10 | [FISM](../03_mf_similarity/10_FISM.md) |  |  |  |  |  |
| 11 | [NAIS](../03_mf_similarity/11_NAIS.md) |  |  |  |  |  |
| 12 | [NCEPLRec](../03_mf_similarity/12_NCEPLRec.md) |  |  |  |  |  |
| 13 | [ENMF](../03_mf_similarity/13_ENMF.md) |  |  |  |  |  |
| 14 | [LINE](../03_mf_similarity/14_LINE.md) |  |  |  |  |  |
| 17 | [NeuMF](../04_neural_cf/17_NeuMF.md) |  |  |  |  |  |
| 18 | [DMF](../04_neural_cf/18_DMF.md) |  |  |  |  |  |
| 19 | [ConvNCF](../04_neural_cf/19_ConvNCF.md) |  |  |  |  |  |
| 20 | [NNCF](../04_neural_cf/20_NNCF.md) |  |  |  |  |  |
| 21 | [SimpleX](../04_neural_cf/21_SimpleX.md) |  |  |  |  |  |
| 22 | [CDAE](../05_autoencoder_generative/22_CDAE.md) |  |  |  |  |  |
| 23 | [MultiDAE](../05_autoencoder_generative/23_MultiDAE.md) |  |  |  |  |  |
| 24 | [MultiVAE](../05_autoencoder_generative/24_MultiVAE.md) |  |  |  |  |  |
| 25 | [MacridVAE](../05_autoencoder_generative/25_MacridVAE.md) |  |  |  |  |  |
| 26 | [RecVAE](../05_autoencoder_generative/26_RecVAE.md) |  |  |  |  |  |
| 27 | [RaCT](../05_autoencoder_generative/27_RaCT.md) |  |  |  |  |  |
| 28 | [DiffRec](../05_autoencoder_generative/28_DiffRec.md) |  |  |  |  |  |
| 29 | [LDiffRec](../05_autoencoder_generative/29_LDiffRec.md) |  |  |  |  |  |
| 30 | [GCMC](../06_graph_cf/30_GCMC.md) |  |  |  |  |  |
| 31 | [SpectralCF](../06_graph_cf/31_SpectralCF.md) |  |  |  |  |  |
| 32 | [NGCF](../06_graph_cf/32_NGCF.md) |  |  |  |  |  |
| 33 | [LightGCN](../06_graph_cf/33_LightGCN.md) |  |  |  |  |  |
| 34 | [DGCF](../06_graph_cf/34_DGCF.md) |  |  |  |  |  |
| 35 | [SGL](../06_graph_cf/35_SGL.md) |  |  |  |  |  |
| 36 | [NCL](../06_graph_cf/36_NCL.md) |  |  |  |  |  |

## 순차 추천

| # | 모델 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 비고 |
|---|---|---|---|---|---|---|
| 06 | [SASRec](../02_sequential_core/06_SASRec.md) |  |  |  |  |  |
| 15 | [FPMC](../02_sequential_core/15_FPMC.md) |  |  |  |  |  |
| 16 | [GRU4Rec](../02_sequential_core/16_GRU4Rec.md) |  |  |  |  |  |
| 37 | [NARM](../07_sequential_extended/37_NARM.md) |  |  |  |  |  |
| 38 | [STAMP](../07_sequential_extended/38_STAMP.md) |  |  |  |  |  |
| 39 | [Caser](../07_sequential_extended/39_Caser.md) |  |  |  |  |  |
| 40 | [NextItNet](../07_sequential_extended/40_NextItNet.md) |  |  |  |  |  |
| 41 | [RepeatNet](../07_sequential_extended/41_RepeatNet.md) |  |  |  |  |  |
| 42 | [TransRec](../07_sequential_extended/42_TransRec.md) |  |  |  |  |  |
| 43 | [FOSSIL](../07_sequential_extended/43_FOSSIL.md) |  |  |  |  |  |
| 44 | [HRM](../07_sequential_extended/44_HRM.md) |  |  |  |  |  |
| 45 | [NPE](../07_sequential_extended/45_NPE.md) |  |  |  |  |  |
| 46 | [SHAN](../07_sequential_extended/46_SHAN.md) |  |  |  |  |  |
| 47 | [HGN](../07_sequential_extended/47_HGN.md) |  |  |  |  |  |
| 65 | [BERT4Rec](../08_transformer_ssl/65_BERT4Rec.md) |  |  |  |  |  |
| 66 | [S3Rec](../08_transformer_ssl/66_S3Rec.md) |  |  |  |  |  |
| 67 | [LightSANs](../08_transformer_ssl/67_LightSANs.md) |  |  |  |  |  |
| 68 | [CORE](../08_transformer_ssl/68_CORE.md) |  |  |  |  |  |
| 69 | [FEARec](../08_transformer_ssl/69_FEARec.md) |  |  |  |  |  |
| 70 | [SINE](../08_transformer_ssl/70_SINE.md) |  |  |  |  |  |
| 71 | [SASRecCPR](../08_transformer_ssl/71_SASRecCPR.md) |  |  |  |  |  |
| 72 | [GRU4RecCPR](../08_transformer_ssl/72_GRU4RecCPR.md) |  |  |  |  |  |
| 73 | [SRGNN](../09_graph_session/73_SRGNN.md) |  |  |  |  |  |
| 74 | [GCSAN](../09_graph_session/74_GCSAN.md) |  |  |  |  |  |
| 75 | [GRU4RecF](../10_feature_sequential/75_GRU4RecF.md) |  |  |  |  |  |
| 76 | [SASRecF](../10_feature_sequential/76_SASRecF.md) |  |  |  |  |  |
| 77 | [FDSA](../10_feature_sequential/77_FDSA.md) |  |  |  |  |  |
| 78 | [GRU4RecKG](../10_feature_sequential/78_GRU4RecKG.md) |  |  |  |  |  |
| 79 | [KSR](../10_feature_sequential/79_KSR.md) |  |  |  |  |  |

## 컨텍스트(CTR) 추천

| # | 모델 | AUC | LogLoss | 학습 시간 | 비고 |
|---|---|---|---|---|---|
| 48 | [LR](../11_ctr_basic/48_LR.md) |  |  |  |  |
| 49 | [XGBOOST](../11_ctr_basic/49_XGBOOST.md) |  |  |  |  |
| 50 | [LIGHTGBM](../11_ctr_basic/50_LIGHTGBM.md) |  |  |  |  |
| 51 | [FM](../11_ctr_basic/51_FM.md) |  |  |  |  |
| 52 | [FFM](../11_ctr_basic/52_FFM.md) |  |  |  |  |
| 53 | [FwFM](../11_ctr_basic/53_FwFM.md) |  |  |  |  |
| 54 | [FNN](../11_ctr_basic/54_FNN.md) |  |  |  |  |
| 55 | [PNN](../11_ctr_basic/55_PNN.md) |  |  |  |  |
| 56 | [WideDeep](../12_ctr_deep/56_WideDeep.md) |  |  |  |  |
| 57 | [NFM](../12_ctr_deep/57_NFM.md) |  |  |  |  |
| 58 | [AFM](../12_ctr_deep/58_AFM.md) |  |  |  |  |
| 59 | [DeepFM](../12_ctr_deep/59_DeepFM.md) |  |  |  |  |
| 60 | [DCN](../12_ctr_deep/60_DCN.md) |  |  |  |  |
| 61 | [DCN V2](../12_ctr_deep/61_DCNV2.md) |  |  |  |  |
| 62 | [xDeepFM](../12_ctr_deep/62_xDeepFM.md) |  |  |  |  |
| 63 | [AutoInt](../12_ctr_deep/63_AutoInt.md) |  |  |  |  |
| 64 | [DSSM](../12_ctr_deep/64_DSSM.md) |  |  |  |  |
| 80 | [DIN](../13_interest_ctr/80_DIN.md) |  |  |  |  |
| 81 | [DIEN](../13_interest_ctr/81_DIEN.md) |  |  |  |  |
| 82 | [FiGNN](../13_interest_ctr/82_FiGNN.md) |  |  |  |  |
| 83 | [KD_DAGFM](../13_interest_ctr/83_KD_DAGFM.md) |  |  |  |  |
| 84 | [EulerNet](../13_interest_ctr/84_EulerNet.md) |  |  |  |  |

## 지식 그래프 추천

| # | 모델 | Recall@10 | NDCG@10 | MRR@10 | Hit@10 | 비고 |
|---|---|---|---|---|---|---|
| 85 | [CKE](../14_knowledge_graph/85_CKE.md) |  |  |  |  |  |
| 86 | [CFKG](../14_knowledge_graph/86_CFKG.md) |  |  |  |  |  |
| 87 | [KTUP](../14_knowledge_graph/87_KTUP.md) |  |  |  |  |  |
| 88 | [RippleNet](../14_knowledge_graph/88_RippleNet.md) |  |  |  |  |  |
| 89 | [MKR](../14_knowledge_graph/89_MKR.md) |  |  |  |  |  |
| 90 | [KGCN](../14_knowledge_graph/90_KGCN.md) |  |  |  |  |  |
| 91 | [KGNNLS](../14_knowledge_graph/91_KGNNLS.md) |  |  |  |  |  |
| 92 | [KGAT](../14_knowledge_graph/92_KGAT.md) |  |  |  |  |  |
| 93 | [KGIN](../14_knowledge_graph/93_KGIN.md) |  |  |  |  |  |
| 94 | [MCCLK](../14_knowledge_graph/94_MCCLK.md) |  |  |  |  |  |
