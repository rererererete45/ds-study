# 16장. 비전 및 멀티모달 트랜스포머

[← 15장. 자연어 처리와 챗봇을 위한 트랜스포머](15_transformers_for_nlp_and_chatbots.md) · [📚 전체 목차](../README.md) · [📖 용어집](glossary.md) · 17장. 트랜스포머 속도 높이기 (정리 예정) →

> 📘 **교재** 『핸즈온 머신러닝 with 사이킷런, 파이토치』 16장 · 노트북 `16_vision_and_multimodal_transformers.ipynb`
>
> 15장의 트랜스포머는 **텍스트 토큰**의 시퀀스를 다뤘다. 그런데 트랜스포머는 입력이 "벡터의 시퀀스"이기만 하면 무엇이든 처리할 수 있다. 16장은 이미지를 작은 **패치**로 잘라 토큰처럼 다루는 **비전 트랜스포머**(ViT)를 직접 만들고, 사전 훈련된 ViT·DeiT를 미세 튜닝하고, 레이블 없이 훈련한 DINO의 어텐션 맵으로 물체를 찾는다. 이어서 텍스트와 이미지를 함께 다루는 **멀티모달** 트랜스포머(CLIP, Perceiver, BLIP-2)를 써 보고, 마지막으로 상용 멀티모달 모델(Gemini)에 사진을 보여 주고 질문한다.
>
> - **16.1** **비전 트랜스포머**. 시각 어텐션 RNN과 DETR(개념) → **원본 ViT**를 직접 구현(패치 임베딩 + `[CLS]` + 위치 임베딩 + 인코더) → 사전 훈련된 **ViT**와 **DeiT**를 Oxford-IIIT Pet에 미세 튜닝 → PVT·Swin(개념) → **DINO** 어텐션 맵으로 비지도 분할
> - **16.2** **멀티모달 트랜스포머**. VideoBERT·ViLBERT(개념) → **CLIP**으로 제로샷 이미지 분류(파이프라인 + 직접 계산) → DALL·E(개념) → **Perceiver**의 푸리에 위치 인코딩 → Perceiver IO·Flamingo(개념) → **BLIP-2**로 이미지 캡션 생성
> - **16.3** **기타 멀티모달 모델**. 대표 모델 정리, **Gemini API**로 사진에 대해 질문하기
>
> ⬅️ **앞에서** [15장](15_transformers_for_nlp_and_chatbots.md)의 [트랜스포머 인코더 층](15_transformers_for_nlp_and_chatbots.md#1-7-인코더-층과-디코더-층)·[훈련되는 위치 임베딩](15_transformers_for_nlp_and_chatbots.md#1-2-위치-임베딩)·[BERT의 `[CLS]` 토큰](15_transformers_for_nlp_and_chatbots.md#3-1-bert의-구조와-사전-훈련)·[문장 임베딩과 코사인 유사도](15_transformers_for_nlp_and_chatbots.md#3-6-문장-임베딩-sentence-transformers)·[`generate()`](15_transformers_for_nlp_and_chatbots.md#4-2-gpt-2로-텍스트-생성하기)를 이미지에 그대로 옮겨 쓴다. 합성곱 층·전이 학습·데이터 증강은 [12장](12_deep_computer_vision_with_cnns.md), 허깅 페이스 `Trainer`·`pipeline`은 [14장 2-17·2-18절](14_nlp_with_rnns_and_attention.md#2-17-트레이너-api)에서 봤다.  
> ➡️ **다음으로** 이 장의 모델은 모두 크고 느리다(ViT-B 8,600만, BLIP-2 약 38억 파라미터). 17장은 트랜스포머를 **더 빠르고 가볍게** 만드는 기법(플래시 어텐션 같은 효율적인 어텐션, 상대 위치 인코딩 등)을 다룬다.

## 🎓 강의를 시작하며

> 🎤 **이 장을 한 문장으로** — 트랜스포머는 텍스트만의 것이 아니다. 이미지를 **패치로 잘라 단어처럼** 넣으면 비전도, 이미지와 텍스트를 **같은 공간에 놓으면** 멀티모달도 된다.

15장의 트랜스포머에 이미지를 넣으려면 질문 하나만 풀면 된다. "이미지의 '단어'는 뭐지?" 답: 16×16 픽셀 **조각(패치)**. 224×224 이미지는 패치 196개 = 단어 196개짜리 문장이다. 이게 ViT다.

이 장의 흐름:

1. **ViT** — 패치 + 위치 임베딩 + [CLS] 토큰 + 트랜스포머 인코더. CNN의 '가정'이 없어서 데이터가 많아야 한다
2. **데이터 효율·속도 개선** — DeiT(CNN 선생님에게 증류), Swin(윈도 안에서만 어텐션)
3. **레이블 없이 배우기** — DINO(자기 증류), 어텐션 맵만으로 물체 분할
4. **멀티모달** — CLIP(이미지와 캡션을 짝짓기 → 제로샷 분류), Perceiver, Flamingo, BLIP-2(이미지를 LLM에 연결)

> 🏁 **이 장이 끝나면** "CLIP은 훈련한 적 없는 클래스를 어떻게 분류하나?"를 그림 없이 설명할 수 있어야 한다.

---

## 이 장의 흐름

| 단계 | 절 | 핵심 질문 | 결과물 |
|---|---|---|---|
| 1. 패치 임베딩 | 16.1 | 이미지를 트랜스포머가 읽는 토큰 시퀀스로 바꾸려면 | `PatchEmbedding` — `Conv2d(kernel_size=16, stride=16)`로 224×224 이미지를 **패치 196개 × 768차원**으로. 패치를 잘라 선형 층에 넣은 것과 결과가 같음을 확인 |
| 2. ViT 조립 | 16.1 | 패치 시퀀스로 이미지를 분류하려면 | `ViT` — `[CLS]` 토큰 + 위치 임베딩 + 인코더 12층 + `[CLS]` 출력 → 1,000개 클래스. **86,567,656** 파라미터(ViT-B/16) |
| 3. ViT 미세 튜닝 | 16.1 | 사전 훈련된 ViT를 작은 데이터셋에 쓰려면 | `vit-base-patch16-224-in21k` + Oxford-IIIT Pet(37품종) `Trainer` 3 에포크 → 테스트 정확도 **91.44%** |
| 4. DeiT | 16.1 | 데이터가 적어도 잘 훈련되는 ViT는 | 증류 토큰을 쓴 `deit-base-distilled-patch16-224`, 같은 설정 → **94.41%** (검증 손실 1.15 → 0.20) |
| 5. PVT·Swin | 16.1 | 고해상도 이미지·밀집 예측에는 | 계층(피라미드) 구조, 공간 축소 어텐션, **이동 윈도 어텐션** — 계산량이 이미지 크기에 선형 (개념 + 계산량 비교) |
| 6. DINO | 16.1 | 레이블 없이 좋은 이미지 표현을 배우려면 | 자기 증류로 훈련한 `dino-vitb8`의 마지막 층 어텐션 `[12, 784]` → 헤드마다 고양이·리모컨·배경을 나눠 보는 **비지도 분할** |
| 7. CLIP 원리 | 16.2 | 이미지와 텍스트를 같은 공간에 놓으려면 | 이중 인코더 + **대조 학습**(대칭 교차 엔트로피). 보충 코드로 손실 구현 |
| 8. CLIP 제로샷 | 16.2 | 훈련 없이 레이블 이름만으로 분류하려면 | `zero-shot-image-classification` 파이프라인 — 곤충 3종 중 무당벌레 **99.73%**, 꽃 5종 중 민들레 68.3% |
| 9. CLIP 내부 계산 | 16.2 | 파이프라인의 확률은 어디서 나오나 | 정규화된 임베딩 `[1, 512]`·`[3, 512]` → 코사인 유사도 0.23 ~ 0.30 → **× 100**(온도) → 소프트맥스 → 같은 확률 |
| 10. Perceiver | 16.2 | 모달리티와 상관없이, 아주 긴 입력을 다루려면 | **푸리에 위치 인코딩** `[224, 224, 26]` + 작은 **잠재 배열**로 크로스 어텐션 (보충 코드: 픽셀 50,176개 → 잠재 256개) |
| 11. BLIP-2 | 16.2 | 이미지를 LLM에 보여 주려면 | 동결된 ViT + **Q-Former**(쿼리 32개) + 동결된 OPT-2.7B → "two cats laying on a couch" |
| 12. Gemini | 16.3 | 상용 멀티모달 모델은 | `google-genai` 클라이언트로 사진 업로드 + 질문 → `[cat, 2]` |

## 데이터

| 데이터 | 형태 | 사용 절 | 비고 |
|---|---|---|---|
| Oxford-IIIT Pet (`timm/oxford-iiit-pet`) | `{"image": PIL 이미지, "label": 품종 번호 0 ~ 36, ...}` | 16.1 | 반려동물 **37품종**(고양이 12, 개 25)의 사진. 훈련 3,680장 / 테스트 3,669장. 이미지마다 크기와 가로세로 비율이 다르고, RGB가 아닌 이미지도 섞여 있다 |
| COCO 2017 검증 이미지 `000000039769.jpg` | 640×480 RGB | 16.1, 16.2, 16.3 | 소파 위에 **고양이 두 마리**와 리모컨 두 개. DINO 분할, BLIP-2 캡션, Gemini 질문에 쓴다 |
| 무당벌레 사진 (`https://homl.info/ladybug`) | RGB | 16.2 | 민들레 위의 무당벌레. CLIP 제로샷 분류에 쓴다 |
| 사이킷런 샘플 이미지 `flower.jpg` | 427×640 RGB | 16.2 | Perceiver 절에서 출력만 하고 쓰지 않는다 (2-9절의 ⚠️ 참고) |
| 사전 훈련 모델 (허깅 페이스 허브) | | 16.1 ~ 16.2 | `google/vit-base-patch16-224-in21k`, `facebook/deit-base-distilled-patch16-224`, `facebook/dino-vitb8`, `openai/clip-vit-base-patch32`, `Salesforce/blip2-opt-2.7b`(가중치 파일 약 15GB) |
| Gemini API | | 16.3 | `gemini-2.5-flash`. 구글 AI 스튜디오에서 발급한 **API 키**가 필요하다 |

> 📌 이 장의 이미지 텐서는 **`[배치, 채널, 높이, 너비]`**(`[B, C, H, W]`, 채널 우선)이고, 패치로 자른 뒤에는 15장과 같은 **`[배치, 시퀀스 길이, 임베딩 차원]`**(`[B, L, E]`)이 된다. 패치 크기가 $P$이면 시퀀스 길이는 $L = (H/P) \times (W/P)$ 다. 코드 주석의 약자: `B` 배치, `C` 채널(또는 클래스 수), `H`·`W` 높이·너비, `L` 패치 수, `E` 임베딩 차원.

## 목차

- [0. 준비](#0-준비)
- [1. 비전 트랜스포머 (16.1)](#1-비전-트랜스포머-161)
   - [1-1. 시각 어텐션을 쓴 RNN과 DETR](#1-1-시각-어텐션을-쓴-rnn과-detr)
   - [1-2. 원본 ViT: 이미지는 16×16 단어의 모음](#1-2-원본-vit-이미지는-1616-단어의-모음)
   - [1-3. 패치 임베딩 구현](#1-3-패치-임베딩-구현)
   - [1-4. 보충 코드: 합성곱 = 패치 자르기 + 선형 층](#1-4-보충-코드-합성곱--패치-자르기--선형-층)
   - [1-5. ViT 구현](#1-5-vit-구현)
   - [1-6. 보충 코드: 파라미터 수와 입력 크기](#1-6-보충-코드-파라미터-수와-입력-크기)
   - [1-7. 사전 훈련된 ViT 미세 튜닝: 데이터와 전처리](#1-7-사전-훈련된-vit-미세-튜닝-데이터와-전처리)
   - [1-8. Trainer로 미세 튜닝하기](#1-8-trainer로-미세-튜닝하기)
   - [1-9. DeiT: 데이터 효율적인 이미지 트랜스포머](#1-9-deit-데이터-효율적인-이미지-트랜스포머)
   - [1-10. 사전 훈련된 DeiT 미세 튜닝](#1-10-사전-훈련된-deit-미세-튜닝)
   - [1-11. PVT와 Swin 트랜스포머](#1-11-pvt와-swin-트랜스포머)
   - [1-12. DINO: 자기 지도 시각 표현 학습](#1-12-dino-자기-지도-시각-표현-학습)
   - [1-13. DINO 어텐션 맵으로 비지도 분할하기](#1-13-dino-어텐션-맵으로-비지도-분할하기)
   - [1-14. 그 밖의 주요 비전 모델과 기법](#1-14-그-밖의-주요-비전-모델과-기법)
- [2. 멀티모달 트랜스포머 (16.2)](#2-멀티모달-트랜스포머-162)
   - [2-1. VideoBERT와 ViLBERT](#2-1-videobert와-vilbert)
   - [2-2. CLIP: 대조 사전 훈련](#2-2-clip-대조-사전-훈련)
   - [2-3. 보충 코드: CLIP의 대조 손실](#2-3-보충-코드-clip의-대조-손실)
   - [2-4. CLIP으로 제로샷 이미지 분류하기](#2-4-clip으로-제로샷-이미지-분류하기)
   - [2-5. CLIPModel로 직접 계산하기](#2-5-clipmodel로-직접-계산하기)
   - [2-6. 이미지와 텍스트를 따로 인코딩하기](#2-6-이미지와-텍스트를-따로-인코딩하기)
   - [2-7. DALL·E: 텍스트로 이미지 생성하기](#2-7-dalle-텍스트로-이미지-생성하기)
   - [2-8. Perceiver: 잠재 배열로 고해상도 입력 다루기](#2-8-perceiver-잠재-배열로-고해상도-입력-다루기)
   - [2-9. 푸리에 위치 인코딩](#2-9-푸리에-위치-인코딩)
   - [2-10. 보충 코드: Perceiver 블록](#2-10-보충-코드-perceiver-블록)
   - [2-11. Perceiver IO](#2-11-perceiver-io)
   - [2-12. Flamingo: 열린 시각 대화](#2-12-flamingo-열린-시각-대화)
   - [2-13. BLIP과 BLIP-2](#2-13-blip과-blip-2)
   - [2-14. BLIP-2로 이미지 캡션 생성하기](#2-14-blip-2로-이미지-캡션-생성하기)
- [3. 기타 멀티모달 모델 (16.3)](#3-기타-멀티모달-모델-163)
   - [3-1. 대표적인 멀티모달 모델](#3-1-대표적인-멀티모달-모델)
   - [3-2. Gemini API로 사진에 대해 질문하기](#3-2-gemini-api로-사진에-대해-질문하기)
- [4. 연습문제](#4-연습문제)
- [5. 정리](#5-정리)
   - [5-1. 전체 흐름](#5-1-전체-흐름)
   - [5-2. 개념과 코드 대응](#5-2-개념과-코드-대응)
   - [5-3. 방법 비교](#5-3-방법-비교)
   - [5-4. 주요 결과](#5-4-주요-결과)
   - [5-5. 자주 쓴 코드 패턴](#5-5-자주-쓴-코드-패턴)
   - [5-6. 핵심 원칙](#5-6-핵심-원칙)
   - [5-7. 일타 복습: 5분 셀프 테스트](#5-7-일타-복습-5분-셀프-테스트)

---

## 0. 준비

```python
import sys
assert sys.version_info >= (3, 10)

IS_COLAB = "google.colab" in sys.modules
IS_KAGGLE = "kaggle_secrets" in sys.modules

if IS_COLAB:
    %pip install -q torchmetrics

from packaging.version import Version
import torch
assert Version(torch.__version__) >= Version("2.6.0")

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

device
# 'cuda'

if device == "cpu":
    print("Neural nets can be very slow without a hardware accelerator.")
    if IS_COLAB:
        print("Go to Runtime > Change runtime and select a GPU hardware "
              "accelerator.")
    if IS_KAGGLE:
        print("Go to Settings > Accelerator and select GPU.")

import matplotlib.pyplot as plt
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
```

모든 장 노트북 맨 앞에 반복되는 환경 확인 + 그래프 글꼴 설정. 각 줄의 의미는 [1장 0. 준비](01_the_machine_learning_landscape.md#0-준비), 장치 선택은 [10장 1-5절](10_neural_nets_with_pytorch.md#1-5-하드웨어-가속) 참고.

- 14·15장과 달리 이 장에는 `evaluate_tm()`·`train()`·`del_vars()` 같은 도우미 함수가 없다. 직접 훈련 루프를 쓰지 않고 미세 튜닝은 모두 허깅 페이스 `Trainer`가 맡는다. 코랩에서 `torchmetrics`를 설치하지만 이 장에서는 쓰지 않는다
- GPU 메모리 정리 코드도 없다. 절마다 큰 모델(ViT·DeiT·DINO 각 약 340MB, BLIP-2는 float16으로 약 7.5GB)을 새로 올리므로, 메모리가 작은 GPU라면 [15장의 `del_vars()`](15_transformers_for_nlp_and_chatbots.md#0-준비)를 가져와 다 쓴 모델을 지우는 것이 좋다

> ⚠️ 노트북의 출력은 **코랩 GPU(`cuda`)** 에서 실행한 결과다. 이 노트의 **보충 코드**는 파이토치 2.15(CPU) + 트랜스포머스 5.23에서 직접 실행해 확인했다. 데이터셋·사전 훈련 모델을 내려받아야 하는 실험(미세 튜닝, DINO·CLIP·BLIP-2 추론, Gemini 호출)은 다시 돌리지 않고 **노트북에 저장된 출력**을 그대로 옮겼다. 파라미터 수와 텐서 형태는 내려받기 없이 설정(config)만으로 만든 같은 구조의 모델로 확인했다.

> ⚠️ 노트북은 트랜스포머스 4.x에서 실행됐다. **5.x에서는 두 셀이 그대로 돌지 않는다.** ① DINO 절의 `output_attentions=True`는 기본 어텐션 구현(`sdpa`)이 어텐션 가중치를 돌려주지 않아 `output.attentions`가 빈 튜플이 된다 → `attn_implementation="eager"`를 함께 넘긴다(1-13절). ② CLIP의 `get_image_features()`·`get_text_features()`가 텐서가 아니라 출력 객체를 돌려준다 → `.pooler_output`을 꺼내 쓴다(2-6절).

---

## 1. 비전 트랜스포머 (16.1)

> 🎯 **목적** — 합성곱 없이 트랜스포머만으로 이미지를 처리하는 비전 트랜스포머의 발전 과정을 따라가며, 원본 ViT를 직접 만들고 사전 훈련 모델을 미세 튜닝한다.

<!-- 일타 -->
> 💬 **이미지 = 패치 196개짜리 문장**
>
> ViT는 이미지를 16×16 패치로 자르고, 각 패치를 벡터 하나로 바꾼다(이건 사실 커널 16, 스트라이드 16인 합성곱과 똑같다). 맨 앞에 학습되는 [CLS] 토큰을 붙이고 위치 임베딩을 더해 트랜스포머 인코더에 넣는다. 분류는 [CLS] 출력 하나로 한다.
>
> - **귀납적 편향**: CNN은 "가까운 픽셀끼리 관련 있다", "패턴은 어디에 있든 같다"는 가정을 구조에 박아 두었다. ViT는 그 가정이 없어 **데이터가 적으면 CNN에 지고, 아주 많으면 이긴다**
> - **DeiT**: 데이터가 적어도 되게 하려고 CNN 선생님의 예측을 따라 하는 **증류 토큰**을 추가
> - **Swin**: 이미지 전체가 아니라 **작은 윈도 안에서만** 어텐션 → 비용이 이미지 크기에 선형. 윈도를 층마다 엇갈리게 옮겨 정보가 퍼지게 한다
> - **DINO**: 레이블 없이, 같은 이미지의 다른 크롭에 대해 학생이 선생(학생의 이동 평균)과 같은 출력을 내도록 배운다. 그 결과 어텐션 맵이 **물체 모양**을 그려 낸다


**개념·원리**

12장의 CNN은 이미지에 잘 맞는 두 가지 **귀납적 편향**(inductive bias, 모델 구조에 미리 넣어 둔 가정)을 갖고 있다.

| 편향 | CNN에서 | 트랜스포머에서 |
|---|---|---|
| **지역성** | 작은 커널이 가까운 픽셀끼리만 섞는다. 먼 영역은 층을 여러 번 거쳐야 연결된다 | 첫 층부터 **모든 패치가 모든 패치를 본다**. 가까운 패치가 중요하다는 것도 데이터로 배워야 한다 |
| **이동 등변성** | 같은 커널을 모든 위치에 적용 → 물체가 이동하면 특성 맵도 그만큼 이동 | 위치 임베딩을 더하므로 구조적으로 보장되지 않는다 |

이 편향 덕분에 CNN은 데이터가 적어도 잘 일반화된다. 트랜스포머는 이런 가정이 거의 없는 대신 **데이터가 충분히 많으면** 그 가정을 넘어서는 패턴까지 배운다. 이 절의 모델들은 이 트레이드오프를 다루는 다양한 방법이다.

| 모델 | 연도 | 핵심 | 이 노트 |
|---|---|---|---|
| 시각 어텐션 RNN | 2015 | CNN 특성 맵 위를 RNN 디코더가 어텐션으로 훑으며 캡션 생성 | 1-1 (개념) |
| DETR | 2020 | CNN 백본 + 트랜스포머 인코더–디코더로 객체 탐지 | 1-1 (개념) |
| **ViT** | 2020 | 패치 = 토큰, 순수 트랜스포머 인코더 | 1-2 ~ 1-8 |
| **DeiT** | 2020 | 증류 토큰 + 강한 증강 → ImageNet만으로 훈련 | 1-9 ~ 1-10 |
| PVT | 2021 | 피라미드 구조 + 공간 축소 어텐션 → 밀집 예측 | 1-11 (개념) |
| Swin | 2021 | 계층 구조 + 이동 윈도 어텐션 → 빠르고 범용적 | 1-11 (개념 + 보충 그래프) |
| **DINO** | 2021 | 레이블 없는 자기 증류 → 어텐션이 물체를 분할 | 1-12 ~ 1-13 |

### 1-1. 시각 어텐션을 쓴 RNN과 DETR

> 🎯 **목적** — ViT 이전에 어텐션이 컴퓨터 비전에 들어온 두 단계를 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 개념을 정리한 것이다.

**시각 어텐션을 쓴 RNN** — 2015년 논문 「Show, Attend and Tell」(쑤 등)은 14장의 인코더–디코더 + 어텐션을 **이미지 캡션 생성**에 적용했다.

```
이미지 → CNN → 특성 맵 14×14×512 (= 위치 196개의 벡터)
                    ↓ 키·값
RNN 디코더(LSTM) ─ 쿼리 → 어텐션 가중치(196개) → 가중 평균 → 다음 단어
   "A"  →  "bird"  →  "flying"  →  "over"  → ...
```

- 단어를 하나 만들 때마다 이미지의 **어느 위치를 볼지** 어텐션으로 고른다. "bird"를 만들 때는 새가 있는 위치에, "water"를 만들 때는 물이 있는 위치에 가중치가 몰린다 → 어텐션 가중치를 그리면 모델이 어디를 보고 단어를 골랐는지 **설명**할 수 있다
- 14장 [4절의 바다나우 어텐션](14_nlp_with_rnns_and_attention.md#4-2-바다나우-어텐션과-루옹-어텐션)에서 인코더의 "단어별 출력" 대신 CNN의 "위치별 특성 벡터"를 쓴 것과 같다

**DETR**(DEtection TRansformer, 2020, 페이스북) — CNN과 트랜스포머를 합친 **객체 탐지** 모델이다.

```
이미지 → CNN 백본(ResNet) → 특성 맵 → 펼치기 + 위치 인코딩
      → 트랜스포머 인코더 (특성 위치끼리 셀프 어텐션)
      → 트랜스포머 디코더 ← 학습되는 "객체 쿼리" N개 (예: 100개)
      → 쿼리마다 (클래스 또는 "물체 없음", 바운딩 박스)
```

| 특징 | 설명 |
|---|---|
| 객체 쿼리 | 디코더 입력은 문장이 아니라 **학습되는 벡터 N개**. 쿼리 하나가 물체 하나를 맡는다. 물체가 N개보다 적으면 나머지 쿼리는 "물체 없음"(∅)을 예측한다 |
| 집합 예측 손실 | 예측 N개와 정답 물체들을 **헝가리안 알고리즘**으로 일대일 매칭한 뒤 손실을 계산한다 → 같은 물체를 두 번 예측하면 벌을 받는다 |
| 없어진 것 | 12장 YOLO에 필요했던 **앵커 박스**와 [NMS](12_deep_computer_vision_with_cnns.md#10-1-슬라이딩-윈도와-nms) 후처리가 필요 없다. 디코더의 셀프 어텐션이 쿼리끼리 "누가 어떤 물체를 맡을지" 조율한다 |
| 한계 | 훈련이 느리고(수백 에포크), 작은 물체에 약하다 → 이후 Deformable DETR 등으로 개선 |

> 💡 DETR까지는 여전히 CNN이 이미지를 특성으로 바꾸는 일을 맡았다. 다음 절의 ViT는 **CNN을 완전히 빼고** 픽셀 패치를 바로 트랜스포머에 넣는다.

### 1-2. 원본 ViT: 이미지는 16×16 단어의 모음

> 🎯 **목적** — 2020년 구글의 비전 트랜스포머(ViT)가 이미지를 토큰 시퀀스로 바꾸는 방법과 전체 구조를 정리한다.

**개념·원리**

논문 제목은 「An Image is Worth 16x16 Words」(도소비츠키 등)다. 아이디어는 단순하다.

1. 224×224 이미지를 **16×16 패치**로 자른다 → 14 × 14 = **196개**
2. 패치마다 16×16×3 = 768개의 픽셀값을 펼쳐 **선형 투영**한다 → 패치 하나가 768차원 벡터(= 단어 임베딩)
3. 맨 앞에 BERT처럼 학습되는 **`[CLS]` 토큰**을 붙인다 → 197개
4. **훈련되는 위치 임베딩**(1차원, 197 × 768)을 더한다
5. 15장과 같은 **트랜스포머 인코더**를 통과시킨다
6. `[CLS]` 위치의 출력만 꺼내 **분류 헤드**(MLP)에 넣는다

```
이미지 [3, 224, 224]
  │ 16×16 패치 196개로 자르고 펼치기 → [196, 768]
  │ 선형 투영 (768 → 768)
  ▼
[CLS] p1 p2 p3 ... p196            ← 197개 토큰
  + 위치 임베딩 0, 1, 2, ..., 196
  ▼
트랜스포머 인코더 × 12 (멀티 헤드 셀프 어텐션 + MLP)
  ▼
[CLS] 출력 → 층 정규화 → 선형 층 → 클래스 확률
```

| 모델 | 층 | 은닉 차원 | MLP 차원 | 헤드 | 파라미터 |
|---|---|---|---|---|---|
| ViT-Base (ViT-B) | 12 | 768 | 3,072 | 12 | 약 8,600만 |
| ViT-Large (ViT-L) | 24 | 1,024 | 4,096 | 16 | 약 3억 700만 |
| ViT-Huge (ViT-H) | 32 | 1,280 | 5,120 | 16 | 약 6억 3,200만 |

이름 뒤의 숫자는 패치 크기다. ViT-B/16은 16×16 패치, ViT-B/32는 32×32 패치(2-4절의 CLIP 이미지 인코더)다. 구조가 BERT-base(15장 3-1절)와 **완전히 같다** — 입력만 단어에서 패치로 바뀌었다.

**결과와 교훈**

- ImageNet(130만 장)만으로 훈련하면 비슷한 크기의 ResNet보다 **조금 못하다.** 지역성·이동 등변성 편향이 없기 때문이다
- 구글 내부의 JFT-300M(3억 장)이나 ImageNet-21k(1,400만 장)로 **대규모 사전 훈련**한 뒤 미세 튜닝하면 당시 최고의 CNN을 넘어섰고, 훈련 계산량은 오히려 적었다
- 결론: **데이터가 충분하면 귀납적 편향보다 데이터에서 배우는 쪽이 낫다.** 이후 비전 모델의 흐름이 CNN에서 트랜스포머로 크게 옮겨 갔다

> 📌 패치 크기가 작을수록 토큰이 많아져 더 세밀하게 보지만, 셀프 어텐션의 계산량은 토큰 수의 **제곱**이다. 224×224 이미지에서 패치 32 → 49개, 16 → 196개, 14 → 256개, 8 → 784개(1-13절의 DINO). 어텐션 행렬 크기로는 패치 8이 패치 16보다 약 16배 크다.

### 1-3. 패치 임베딩 구현

> 🎯 **목적** — 이미지를 패치 임베딩 시퀀스로 바꾸는 층을 합성곱 층 하나로 구현한다.

**코드**

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=16):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, embed_dim,
                                kernel_size=patch_size, stride=patch_size)
    def forward(self, X):
        X = self.conv2d(X)  # shape [B=Batch, C=Channels, H=Height, W=Width]
        X = X.flatten(start_dim=2)  # shape [B, C, H * W]
        return X.transpose(1, 2)  # shape [B, H * W, C]
```

| 코드 | 의미 |
|---|---|
| `nn.Conv2d(in_channels, embed_dim, kernel_size=16, stride=16)` | **커널 크기 = 스트라이드**라서 커널이 겹치지 않고 패치마다 정확히 한 번씩 적용된다. 출력 채널 768개 = 임베딩 차원 |
| `self.conv2d(X)` | `[B, 3, 224, 224]` → `[B, 768, 14, 14]`. 출력의 각 위치 $(i, j)$가 패치 하나의 768차원 임베딩 |
| `X.flatten(start_dim=2)` | 공간 축 두 개를 하나로 → `[B, 768, 196]`. 패치 순서는 왼쪽 위에서 오른쪽으로, 위에서 아래로(행 우선) |
| `X.transpose(1, 2)` | `[B, 196, 768]` — 트랜스포머가 기대하는 `[B, L, E]` (`batch_first=True`) |

> 💡 합성곱 층은 각 패치(16×16×3 = 768개 값)와 필터(역시 768개 가중치)의 **점곱 + 편향**을 출력 채널마다 계산한다. 패치가 겹치지 않으니 이것은 "패치를 펼쳐 `nn.Linear(768, 768)`에 넣기"와 **정확히 같은 연산**이다. 합성곱 층으로 쓰면 패치를 자르고 펼치는 코드를 따로 쓸 필요가 없고 빠르다.

### 1-4. 보충 코드: 합성곱 = 패치 자르기 + 선형 층

> 🎯 **목적** — 1-3절의 💡를 직접 확인한다.

**보충 코드**

```python
# 보충 코드
torch.manual_seed(42)
patch_embed = PatchEmbedding(3, 768, 16)
X = torch.randn(4, 3, 224, 224)
out = patch_embed(X)
out.shape                                            # torch.Size([4, 196, 768])
sum(p.numel() for p in patch_embed.parameters())     # 590592  (= 768 × 768 + 768)

# 같은 연산을 패치 자르기 + 행렬 곱으로
patches = X.unfold(2, 16, 16).unfold(3, 16, 16)      # [4, 3, 14, 14, 16, 16]
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(4, 196, 3 * 16 * 16)  # [4, 196, 768]
W = patch_embed.conv2d.weight.reshape(768, -1)       # [768, 768] (필터마다 3×16×16을 펼침)
b = patch_embed.conv2d.bias
alt = patches @ W.T + b
torch.allclose(out, alt, atol=1e-5)                  # True  (최대 차이 3.2e-06)
```

| 코드 | 의미 |
|---|---|
| `X.unfold(2, 16, 16)` | 높이 축(2)을 크기 16, 간격 16으로 잘라 새 축을 만든다 → `[4, 3, 14, 224, 16]` |
| `.unfold(3, 16, 16)` | 너비 축도 → `[4, 3, 14, 14, 16, 16]` (배치, 채널, 패치 행, 패치 열, 패치 안 행, 패치 안 열) |
| `.permute(0, 2, 3, 1, 4, 5)` | 채널 축을 패치 축 뒤로 → `[4, 14, 14, 3, 16, 16]`. 합성곱 가중치 `[768, 3, 16, 16]`과 같은 순서(채널, 행, 열)로 펼치기 위해서 |
| `.reshape(4, 196, 768)` | 패치 196개 × 값 768개 |
| `conv2d.weight.reshape(768, -1)` | 필터 768개를 각각 768차원 벡터로 → 선형 층의 가중치 행렬 |

**결과 해석**

- 두 방법의 결과가 부동소수점 오차 안에서 같다 → 패치 임베딩은 "**패치별 선형 층**"이다
- 파라미터는 590,592개뿐이다. 이미지 크기와 상관없이 패치 크기와 임베딩 차원으로만 정해진다

### 1-5. ViT 구현

> 🎯 **목적** — 패치 임베딩, `[CLS]` 토큰, 위치 임베딩, 트랜스포머 인코더, 분류 헤드를 조립해 ViT-B/16을 만든다.

**코드**

```python
class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 ff_dim=3072, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        cls_init = torch.randn(1, 1, embed_dim) * 0.02
        self.cls_token = nn.Parameter(cls_init)  # shape [1, 1, E=embed_dim]
        num_patches = (img_size // patch_size) ** 2  # num_patches (noted L)
        pos_init = torch.randn(1, num_patches + 1, embed_dim) * 0.02
        self.pos_embed = nn.Parameter(pos_init)  # shape [1, 1 + L, E]
        self.dropout = nn.Dropout(p=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, X):
        Z = self.patch_embed(X)  # shape [B, L, E]
        cls_expd = self.cls_token.expand(Z.shape[0], -1, -1)  # shape [B, 1, E]
        Z = torch.cat((cls_expd, Z), dim=1)  # shape [B, 1 + L, E]
        Z = Z + self.pos_embed
        Z = self.dropout(Z)
        Z = self.encoder(Z)  # shape [B, 1 + L, E]
        Z = self.layer_norm(Z[:, 0])  # shape [B, E]
        logits = self.output(Z) # shape [B, C]
        return logits
```

| 코드 | 의미 |
|---|---|
| `nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)` | 학습되는 `[CLS]` 토큰 벡터 하나. 이미지 내용과 무관한 "빈 칸"으로 시작해, 인코더를 지나며 모든 패치의 정보를 어텐션으로 모아 **이미지 전체의 요약**이 된다 (BERT의 `[CLS]`와 같은 역할) |
| `num_patches = (img_size // patch_size) ** 2` | (224 // 16)² = 196 |
| `pos_embed` `[1, 197, 768]` | `[CLS]` 자리까지 포함한 **훈련되는 위치 임베딩**. 15장 [1-2절](15_transformers_for_nlp_and_chatbots.md#1-2-위치-임베딩)의 `PositionalEmbedding`과 같고, 앞에 배치 축 1을 둬서 그대로 더할 수 있다 |
| `nn.TransformerEncoderLayer(..., activation="gelu", batch_first=True)` | 15장의 인코더 층. ViT·BERT 관례대로 활성화 함수는 [GELU](11_training_deep_neural_networks.md) |
| `nn.TransformerEncoder(encoder_layer, num_layers=12)` | 같은 층을 12번 **복사**해 쌓는다 (15장 1-8절의 `deepcopy`) |
| `self.cls_token.expand(B, -1, -1)` | `[1, 1, E]` → `[B, 1, E]`. 메모리를 복사하지 않고 같은 벡터를 배치 크기만큼 **보이게만** 한다 (`-1` = 그 축은 그대로) |
| `torch.cat((cls_expd, Z), dim=1)` | 시퀀스 맨 앞에 `[CLS]`를 붙인다 → `[B, 197, E]` |
| `Z[:, 0]` | 인코더 출력에서 0번 위치(`[CLS]`)만 → `[B, E]`. 나머지 196개 패치 출력은 분류에 쓰지 않는다 |
| `self.layer_norm(...)` → `self.output(...)` | 층 정규화 후 선형 층으로 클래스 로짓 `[B, 1000]` |

```python
vit_model = ViT(
    img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768,
    depth=12, num_heads=12, ff_dim=3072, dropout=0.1)
batch = torch.randn(4, 3, 224, 224)
logits = vit_model(batch)  # shape [4, 1000]
logits.shape
# torch.Size([4, 1000])
```

> ⚠️ 파이토치 `nn.TransformerEncoderLayer`의 기본값은 `norm_first=False`, 즉 15장과 같은 **Post-LN**(서브층 → 잔차 연결 → 층 정규화)이다. 원본 ViT 논문은 **Pre-LN**(층 정규화 → 서브층 → 잔차 연결)을 쓰고, 마지막에 층 정규화를 한 번 더 한다. 이 코드의 마지막 `self.layer_norm`은 Pre-LN 구조의 그 마지막 정규화에 해당하므로, 논문과 똑같이 하려면 `nn.TransformerEncoderLayer(..., norm_first=True)`로 바꾼다. 허깅 페이스 `ViTLayer`도 `layernorm_before` → 어텐션 → 잔차, `layernorm_after` → MLP → 잔차 순서인 Pre-LN이다(트랜스포머스 5.23 소스로 확인). Pre-LN은 깊게 쌓아도 웜업 없이 훈련이 안정적이다.

> 📌 원본 ViT는 사전 훈련 때 분류 헤드로 은닉 층 하나짜리 MLP를 쓰고, 미세 튜닝 때 선형 층 하나로 바꿨다. 이 구현은 처음부터 선형 층 하나다.

### 1-6. 보충 코드: 파라미터 수와 입력 크기

> 🎯 **목적** — 직접 만든 ViT의 파라미터가 어디에 있는지 세어 보고, 입력 크기가 고정되는 이유를 확인한다.

**보충 코드**

```python
# 보충 코드
n = lambda m: sum(p.numel() for p in m.parameters())
vit = ViT()
n(vit)                                      # 86567656
n(vit.patch_embed), vit.cls_token.numel(), vit.pos_embed.numel()   # 590592, 768, 151296
n(vit.encoder), n(vit.encoder.layers[0])   # 85054464, 7087872
n(vit.layer_norm), n(vit.output)           # 1536, 769000
```

| 부품 | 파라미터 수 | 비중 |
|---|---|---|
| 패치 임베딩 (768 × 768 + 768) | 590,592 | 0.7% |
| `[CLS]` 토큰 | 768 | |
| 위치 임베딩 (197 × 768) | 151,296 | 0.2% |
| **인코더 12층** (층당 7,087,872) | **85,054,464** | **98.3%** |
| ├ 멀티 헤드 어텐션 (4 × 768 × 768 + 4 × 768) | 층당 2,362,368 | |
| ├ 피드포워드 (768 × 3,072 + 3,072 + 3,072 × 768 + 768) | 층당 4,722,432 | |
| └ 층 정규화 2개 | 층당 3,072 | |
| 마지막 층 정규화 | 1,536 | |
| 분류 헤드 (768 × 1,000 + 1,000) | 769,000 | 0.9% |
| **합계** | **86,567,656** | |

- ViT-B/16의 알려진 크기(약 8,660만)와 같다. 15장 작은 BERT는 파라미터의 89%가 토큰 임베딩이었지만, ViT는 98%가 **인코더**에 있다. 어휘 대신 패치 투영(59만 개)만 있기 때문이다
- 피드포워드 층이 어텐션의 두 배다 (15장과 같은 비율, $d_\text{ff} = 4d$)

```python
# 보충 코드 — 학습한 크기와 다른 이미지
vit.eval()
vit(torch.randn(1, 3, 256, 256))
# RuntimeError: The size of tensor a (257) must match the size of tensor b (197) at non-singleton dimension 1
```

- 256×256 이미지는 패치 256개 + `[CLS]` = 257개가 되는데 위치 임베딩은 197개뿐이다. 패치 임베딩(합성곱)은 어떤 크기든 처리하지만, **위치 임베딩이 입력 크기를 고정**한다
- 해결책: 위치 임베딩을 14×14 격자로 보고 새 격자 크기(16×16)로 **2D 보간**한다. 원본 ViT는 이렇게 고해상도(384×384)로 미세 튜닝해 성능을 높였다. 허깅 페이스 ViT는 `model(pixel_values, interpolate_pos_encoding=True)`로 지원한다

```python
# 보충 코드 (트랜스포머스 5.23, 내려받기 없이 config로만 생성)
from transformers import ViTConfig, ViTForImageClassification
hf_vit = ViTForImageClassification(ViTConfig(num_labels=37)).eval()
n(hf_vit), n(hf_vit.classifier)            # 85827109, 28453
hf_vit(pixel_values=torch.randn(1, 3, 384, 384),
       interpolate_pos_encoding=True).logits.shape     # torch.Size([1, 37])
```

- 허깅 페이스의 ViT-B/16(37클래스)은 85,827,109개 = 직접 만든 ViT 86,567,656 − 1,000클래스 헤드 769,000 + 37클래스 헤드 28,453. **구조가 같다**는 뜻이다. 다음 절에서는 이 구조에 사전 훈련된 가중치를 채워 쓴다

### 1-7. 사전 훈련된 ViT 미세 튜닝: 데이터와 전처리

> 🎯 **목적** — 처음부터 훈련하는 대신, 대규모 데이터로 사전 훈련된 ViT를 반려동물 품종 분류(37클래스)에 맞게 준비한다.

**코드** — 데이터셋

```python
from datasets import load_dataset

pets = load_dataset("timm/oxford-iiit-pet")

num_rows, num_cols = 2, 5
plt.figure(figsize=(num_cols * 2.5, num_rows * 2))
class_names = pets["train"].features["label"].names
for i in range(num_rows * num_cols):
    plt.subplot(num_rows, num_cols, i + 1)
    example = pets["train"][i]
    plt.imshow(example["image"])
    plt.title(class_names[example["label"]])
    plt.axis("off")
```

![Oxford-IIIT Pet 훈련 샘플](images/16/nb_pets_samples.png)

- 훈련 3,680장, 테스트 3,669장 (품종당 훈련 약 100장). 12장 [Flowers102](12_deep_computer_vision_with_cnns.md#8-1-데이터-적재)처럼 **작은** 데이터셋이라 처음부터 훈련하기는 어렵다
- `features["label"].names`에 품종 이름 37개가 들어 있다(`maine_coon`, `american_bulldog`, `pug` …). 이미지의 크기·비율이 제각각이다

**코드** — 사전 훈련 모델과 이미지 프로세서

```python
from transformers import ViTForImageClassification, AutoImageProcessor

model_id = "google/vit-base-patch16-224-in21k"
vit_model = ViTForImageClassification.from_pretrained(model_id, num_labels=37)
vit_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
```

```
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

| 코드 | 의미 |
|---|---|
| `"google/vit-base-patch16-224-in21k"` | ViT-B/16, 입력 224, **ImageNet-21k**(약 1,400만 장, 21,843클래스)로 사전 훈련. 내려받기 346MB |
| `ViTForImageClassification.from_pretrained(..., num_labels=37)` | ViT 본체는 사전 훈련 가중치로, 분류 헤드(`classifier`, 768 → 37)는 **새로 무작위 초기화** → 경고가 그 뜻이다. 14장 2-16절의 `BertForSequenceClassification`과 같은 방식 |
| `AutoImageProcessor.from_pretrained(model_id, use_fast=True)` | 이 모델이 사전 훈련 때 쓴 **전처리를 그대로** 재현하는 이미지 프로세서. 텍스트의 토크나이저에 해당한다. `use_fast=True`는 파이토치(토치비전) 기반의 빠른 구현 |

```python
vit_processor
```

```
ViTImageProcessorFast {
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5],
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {"height": 224, "width": 224},
  ...
}
```

| 설정 | 의미 |
|---|---|
| `do_resize`, `size` 224×224, `resample: 2` | 가로세로 비율과 상관없이 **224×224로 늘리거나 줄인다**(쌍선형 보간, PIL 코드 2). 자르지 않으므로 이미지가 찌그러질 수 있다 |
| `do_rescale`, `rescale_factor` = 1/255 | 0 ~ 255 정수 → 0 ~ 1 실수 |
| `do_normalize`, 평균 0.5, 표준 편차 0.5 | $(x - 0.5) / 0.5$ → **−1 ~ 1** 범위. 12장 토치비전 모델의 ImageNet 평균·표준 편차(0.485, 0.456, 0.406 …)와 다르다 → 모델마다 프로세서를 따로 불러야 하는 이유 |
| `data_format: channels_first` | 출력 `[3, 224, 224]` |

```python
num_rows, num_cols = 2, 5
plt.figure(figsize=(num_cols * 2.5, num_rows * 2))
for i in range(num_rows * num_cols):
    plt.subplot(num_rows, num_cols, i + 1)
    example = pets["train"][i]
    preprocessed = vit_processor(example["image"])["pixel_values"][0]
    plt.imshow((preprocessed.permute(1, 2, 0) + 1.) / 2.)
    plt.title(class_names[example["label"]])
    plt.axis("off")
```

![전처리 후 224×224로 바뀐 이미지](images/16/nb_pets_preprocessed.png)

- 모든 이미지가 정사각형(224×224)이 됐다. 원래 세로로 길거나 가로로 넓던 사진은 조금 찌그러졌다
- `permute(1, 2, 0)`: `[C, H, W]` → 맷플롯립이 기대하는 `[H, W, C]`. `(x + 1) / 2`: −1 ~ 1을 0 ~ 1로 되돌려 그린다

**코드** — 배치 만들기 함수

```python
def vit_collate_fn(batch):
    images = [example["image"] for example in batch]
    labels = [example["label"] for example in batch]
    inputs = vit_processor(images, return_tensors="pt", do_convert_rgb=True)
    inputs["labels"] = torch.tensor(labels)
    return inputs
```

| 코드 | 의미 |
|---|---|
| `vit_processor(images, return_tensors="pt")` | 이미지 리스트를 한꺼번에 전처리 → `{"pixel_values": [B, 3, 224, 224]}` |
| `do_convert_rgb=True` | 흑백(L)이나 투명도가 있는(RGBA) 이미지를 **RGB로 변환**. 데이터셋에 3채널이 아닌 이미지가 섞여 있어서, 없으면 채널 수가 맞지 않아 오류가 난다 |
| `inputs["labels"] = ...` | 모델에 `labels`를 함께 넘기면 `ViTForImageClassification`이 교차 엔트로피 손실을 직접 계산해 돌려준다 (`Trainer`가 이 손실로 훈련) |

> 💡 전처리를 데이터셋 전체에 미리(`map`) 하지 않고 **배치를 만들 때** 한다. 224×224×3 실수 텐서는 이미지당 약 600KB라 7,000장을 미리 변환해 두면 메모리를 많이 쓴다. 증강을 넣을 때도 이 방식이라야 에포크마다 다른 변환을 적용할 수 있다.

### 1-8. Trainer로 미세 튜닝하기

> 🎯 **목적** — 허깅 페이스 `Trainer`로 ViT 전체를 3 에포크 미세 튜닝하고 테스트 정확도를 잰다.

**코드**

```python
# extra code – to measure the model's accuracy
def compute_accuracy(logits_and_labels):
    logits, labels = logits_and_labels
    preds = torch.tensor(logits).argmax(dim=1)
    labels = torch.tensor(labels)
    accuracy = (preds == labels).float().mean()
    return {"accuracy": accuracy.item()}

from transformers import Trainer, TrainingArguments

args = TrainingArguments("my_pets_vit", per_device_train_batch_size=16,
                         eval_strategy="epoch", num_train_epochs=3,
                         remove_unused_columns=False,
                         report_to="none")  # to avoid requiring W&B or TB
trainer = Trainer(model=vit_model, args=args, data_collator=vit_collate_fn,
                  train_dataset=pets["train"], eval_dataset=pets["test"],
                  compute_metrics=compute_accuracy)  # extra code
train_output = trainer.train()
```

| 코드 | 의미 |
|---|---|
| `compute_accuracy(logits_and_labels)` | 평가 때 `Trainer`가 테스트 세트 전체의 로짓과 레이블을 **넘파이 배열**로 넘겨준다 → 텐서로 바꿔 `argmax` 정확도를 계산 |
| `TrainingArguments("my_pets_vit", ...)` | 첫 인수는 체크포인트를 저장할 출력 폴더 |
| `per_device_train_batch_size=16` | 3,680 / 16 = 에포크당 **230 스텝**, 3 에포크 690 스텝 |
| `eval_strategy="epoch"` | 에포크가 끝날 때마다 `eval_dataset`(테스트 세트)으로 평가 |
| `remove_unused_columns=False` | **필수.** `Trainer`는 기본적으로 모델의 `forward()` 인수 이름에 없는 열(`image`, `label`)을 데이터셋에서 지운다. 그러면 `vit_collate_fn`이 받을 `image`가 사라진다 |
| `report_to="none"` | 로그를 W&B·텐서보드로 보내지 않는다 |
| 옵티마이저 (기본값) | AdamW, 학습률 5e-5, 선형 감소 — 사전 훈련된 본체를 **작은 학습률로 전체 미세 튜닝**한다 |

**결과** (노트북 출력, 690 스텝 10분 52초)

| 에포크 | 훈련 손실 | 검증 손실 | 정확도 |
|---|---|---|---|
| 1 | (기록 없음) | 1.9747 | 90.05% |
| 2 | (기록 없음) | 1.3327 | 91.25% |
| 3 | 2.0079 | 1.1532 | **91.44%** |

**결과 해석**

- 37품종을 **3 에포크 만에 91.4%** 로 분류했다. 품종당 사진 100장으로 이 정도가 나오는 것은 ImageNet-21k로 배운 시각 특성 덕분이다(12장 [전이 학습](12_deep_computer_vision_with_cnns.md#8-사전-훈련된-모델을-사용한-전이-학습-128)과 같은 원리)
- 훈련 손실이 "No log"인 이유: `Trainer`는 기본적으로 **500 스텝마다** 훈련 손실을 기록한다(`logging_steps=500`). 에포크 1·2가 끝난 시점(230·460 스텝)에는 아직 기록이 없고, 3 에포크 칸의 2.0079는 처음 500 스텝의 평균이다
- 정확도는 91%인데 검증 손실이 1.15로 꽤 크다. 교차 엔트로피 1.15는 정답 클래스 확률의 기하 평균이 $e^{-1.15} \approx 0.32$라는 뜻이다 → 예측은 대부분 맞지만 **확신이 낮다.** 새로 초기화한 헤드가 작은 학습률로 690 스텝만 훈련돼 로짓의 크기가 아직 작기 때문으로 보인다. 손실이 계속 줄고 있으므로 더 훈련하면 나아질 것이다 (1-10절 그래프)

> 💡 같은 데이터로 12장처럼 CNN을 처음부터 훈련했다면 이 정확도에 한참 못 미쳤을 것이다. 비전 트랜스포머는 귀납적 편향이 약해 **작은 데이터로 처음부터 훈련하기에는 CNN보다 더 불리**하다. 그래서 실전에서는 거의 항상 사전 훈련된 ViT를 미세 튜닝한다.

### 1-9. DeiT: 데이터 효율적인 이미지 트랜스포머

> 🎯 **목적** — 거대한 비공개 데이터 없이 ImageNet만으로 ViT를 잘 훈련한 DeiT의 아이디어(강한 규제 + 증류 토큰)를 정리한다.

> ⚠️ 노트북에는 DeiT를 미세 튜닝하는 코드만 있다(1-10절). 아래 원리는 교재 절 제목에 맞춰 정리한 것이다.

**개념·원리**

2020년 페이스북의 **DeiT**(Data-efficient image Transformer, 투브롱 등)는 ViT와 **구조가 같은** 모델을 ImageNet-1k(130만 장)만으로, 8-GPU 서버 한 대에서 약 3일 훈련해 같은 크기의 CNN과 맞먹는 성능을 냈다. 비결은 두 가지다.

| 비결 | 내용 |
|---|---|
| **강한 증강과 규제** | RandAugment, Mixup(두 이미지를 섞고 레이블도 섞기), CutMix(한 이미지의 사각형 영역을 다른 이미지로 덮기), 랜덤 지우기, 확률적 깊이(층을 무작위로 건너뛰기), 반복 증강 등. 데이터가 부족한 만큼 **편향을 증강으로 채운다** |
| **증류 토큰** | 이미 잘 훈련된 **CNN 교사**(RegNetY-16GF)의 예측을 배우는 별도 토큰을 추가한다 |

**증류 토큰** — 15장 [3-7절의 DistilBERT](15_transformers_for_nlp_and_chatbots.md#3-7-bert-이후의-인코더-모델)는 큰 모델(교사)을 흉내 내는 작은 모델(학생)을 만드는 **지식 증류**였다. DeiT는 학생 ViT에 토큰을 하나 더 넣는다.

```
[CLS]  [DIST]  p1  p2  ...  p196        ← 198개 토큰 (위치 임베딩도 198개)
   │      │
 인코더 12층 (두 토큰 모두 셀프 어텐션으로 패치 정보를 모은다)
   │      │
 헤드 1  헤드 2
   │      └─ 손실 2: CNN 교사가 예측한 클래스(argmax)와의 교차 엔트로피 (하드 증류)
   └──────── 손실 1: 정답 레이블과의 교차 엔트로피
           전체 손실 = (손실 1 + 손실 2) / 2
추론: 두 헤드의 예측을 평균
```

| 포인트 | 설명 |
|---|---|
| 왜 CNN 교사인가 | CNN의 예측에는 지역성·이동 등변성 편향이 담겨 있다. 학생 ViT는 증류 토큰을 통해 **CNN의 귀납적 편향을 간접적으로 물려받는다.** 논문에서 CNN 교사가 같은 정확도의 트랜스포머 교사보다 나은 학생을 만들었다 |
| 하드 증류 | 교사의 확률 분포(소프트 레이블) 대신 교사가 고른 **클래스 하나**를 정답처럼 쓴다. 논문에서는 이쪽이 더 좋았다. 증강(크롭 등)으로 정답 물체가 잘려 나간 이미지라면 교사의 레이블이 원래 레이블보다 더 정확할 수 있다 |
| 두 토큰은 다른가 | 둘 다 같은 패치를 보지만 목표가 다르다. 훈련 초기에는 두 토큰의 출력이 꽤 다르다가 층이 깊어질수록 비슷해진다(완전히 같아지지는 않는다) |

**보충 코드** — 하드 증류 손실

```python
# 보충 코드
import torch.nn.functional as F

def deit_hard_distill_loss(cls_logits, dist_logits, labels, teacher_logits):
    teacher_labels = teacher_logits.argmax(dim=1)          # 교사가 고른 클래스
    return (0.5 * F.cross_entropy(cls_logits, labels)      # [CLS] 헤드 ↔ 정답
            + 0.5 * F.cross_entropy(dist_logits, teacher_labels))  # [DIST] 헤드 ↔ 교사

torch.manual_seed(42)
B, C = 4, 37
loss = deit_hard_distill_loss(torch.randn(B, C), torch.randn(B, C),
                              torch.randint(0, C, (B,)), torch.randn(B, C))
loss   # tensor(3.3154)  — 무작위 로짓이라 ln 37 ≈ 3.61 근처
```

**보충 코드** — 허깅 페이스 DeiT의 구조 (config로만 생성)

```python
# 보충 코드 (트랜스포머스 5.23)
from transformers import (DeiTConfig, DeiTForImageClassification,
                          DeiTForImageClassificationWithTeacher)
deit = DeiTForImageClassification(DeiTConfig(num_labels=37)).eval()
n(deit)                                              # 85828645  (= ViT 85,827,109 + 1,536)
deit.deit.embeddings.distillation_token.shape        # torch.Size([1, 1, 768])
deit.deit.embeddings.position_embeddings.shape       # torch.Size([1, 198, 768])

with torch.no_grad():
    out = deit(pixel_values=torch.randn(2, 3, 224, 224), output_hidden_states=True)
out.hidden_states[-1].shape                          # torch.Size([2, 198, 768])

deit_t = DeiTForImageClassificationWithTeacher(DeiTConfig(num_labels=37)).eval()
n(deit_t)                                            # 85857098  (헤드가 하나 더: + 28,453)
with torch.no_grad():
    out = deit_t(pixel_values=torch.randn(2, 3, 224, 224))
list(out.keys())                                     # ['logits', 'cls_logits', 'distillation_logits']
torch.allclose(out.logits, (out.cls_logits + out.distillation_logits) / 2)   # True
```

| 클래스 | 쓰는 출력 | 용도 |
|---|---|---|
| `DeiTForImageClassification` | `[CLS]`(0번) 출력만 → 헤드 하나 | **미세 튜닝**용 (1-10절). 증류 토큰은 모델 안에 있지만 분류에 쓰지 않는다 |
| `DeiTForImageClassificationWithTeacher` | `[CLS]` 헤드와 `[DIST]`(1번) 헤드의 **평균** | 증류로 훈련된 체크포인트로 **추론만** 할 때 |

- 파라미터 차이 1,536 = 증류 토큰 768 + 위치 임베딩 1개 768. 증류 토큰은 거의 공짜다

### 1-10. 사전 훈련된 DeiT 미세 튜닝

> 🎯 **목적** — 1-8절과 같은 설정으로 DeiT를 미세 튜닝해 ViT와 비교한다.

**코드**

```python
from transformers import DeiTForImageClassification, AutoImageProcessor

model_id = "facebook/deit-base-distilled-patch16-224"
deit_model = DeiTForImageClassification.from_pretrained(model_id, num_labels=37)
deit_processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)

def deit_collate_fn(batch):
    images = [example["image"] for example in batch]
    labels = [example["label"] for example in batch]
    inputs = deit_processor(images, return_tensors="pt", do_convert_rgb=True)
    inputs["labels"] = torch.tensor(labels)
    return inputs

args = TrainingArguments("my_pets_deit", per_device_train_batch_size=16,
                         eval_strategy="epoch", num_train_epochs=3,
                         remove_unused_columns=False,
                         report_to="none")  # to avoid requiring W&B or TB
trainer = Trainer(model=deit_model, args=args, data_collator=deit_collate_fn,
                  train_dataset=pets["train"], eval_dataset=pets["test"],
                  compute_metrics=compute_accuracy)  # extra code
train_output = trainer.train()
```

- 1-7 ~ 1-8절에서 **모델 이름, 모델 클래스, 프로세서**만 바뀌었다. 허깅 페이스의 `Auto…` 클래스와 `Trainer` 덕분에 모델을 바꿔 보는 비용이 매우 작다
- `deit-base-distilled-patch16-224`: DeiT-B(ViT-B와 같은 크기) + 증류 토큰, **ImageNet-1k**로 훈련. 여기서도 분류 헤드(768 → 37)는 새로 초기화된다
- 프로세서 설정이 ViT와 다르다(허브의 `preprocessor_config.json` 기준): 짧은 변을 256으로 줄인 뒤 가운데를 224×224로 **자르고**(`do_center_crop`), ImageNet 평균(0.485, 0.456, 0.406)·표준 편차(0.229, 0.224, 0.225)로 정규화한다. 그래서 `vit_collate_fn`을 재사용하지 않고 `deit_processor`를 쓰는 함수를 새로 만들었다

**결과** (노트북 출력, 690 스텝 11분 36초)

| 에포크 | 훈련 손실 | 검증 손실 | 정확도 |
|---|---|---|---|
| 1 | (기록 없음) | 0.2697 | 91.58% |
| 2 | (기록 없음) | 0.2205 | 93.79% |
| 3 | 0.3658 | 0.2014 | **94.41%** |

![ViT와 DeiT 미세 튜닝 비교](images/16/extra_vit_vs_deit.png)

**결과 해석** (노트북 출력값으로 그린 그래프)

| | ViT (`in21k`) | DeiT (`distilled`) |
|---|---|---|
| 사전 훈련 데이터 | ImageNet-21k (1,400만 장) | ImageNet-1k (130만 장) + CNN 교사 증류 |
| 파라미터 (37클래스) | 85,827,109 | 85,828,645 |
| 1 에포크 후 정확도 | 90.05% | 91.58% |
| 3 에포크 후 정확도 | 91.44% | **94.41%** |
| 3 에포크 후 검증 손실 | 1.153 | **0.201** |

- 사전 훈련 데이터가 10배 적은 DeiT가 **3%포인트 더 정확**하고, 손실은 1/6 수준이다. 정답 클래스 확률의 기하 평균이 $e^{-0.20} \approx 0.82$로 예측에 확신이 있다
- 가능한 이유: ① DeiT 체크포인트는 ImageNet-1k **분류로 끝까지 훈련된** 모델이다. ImageNet-1k에는 개 품종이 100개 이상, 고양이도 여러 품종이 있어 품종을 구별하는 특성이 이미 잘 다듬어져 있다. ② `in21k` 체크포인트는 2만여 개의 넓은 클래스로 사전 훈련됐고 분류 헤드가 제거된 상태라, 새 작업에 맞추려면 더 오래 훈련해야 한다. ③ 증강·증류로 훈련된 DeiT의 특성이 작은 데이터셋에 더 잘 일반화된다
- 두 실험 모두 3 에포크뿐이라 아직 수렴하지 않았다. 특히 ViT는 손실이 빠르게 줄고 있어 차이가 줄어들 여지가 있다 → **같은 예산에서 더 빨리 좋은 결과**를 낸 쪽이 DeiT라고 보는 것이 정확하다

### 1-11. PVT와 Swin 트랜스포머

> 🎯 **목적** — 분류만 하던 ViT를 객체 탐지·분할 같은 밀집 예측과 고해상도 이미지에 쓸 수 있게 만든 계층형 비전 트랜스포머를 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 개념을 정리한 것이다.

**ViT의 두 가지 문제**

| 문제 | 원인 | 영향 |
|---|---|---|
| **단일 해상도** | 모든 층이 같은 14×14 격자(1/16 해상도)의 토큰을 다룬다 | 12장의 분할·탐지 모델(FCN, YOLO 등)은 1/4, 1/8, 1/16, 1/32 해상도의 **여러 크기 특성 맵**(특성 피라미드)이 필요하다. 작은 물체는 고해상도 특성 맵에서 찾아야 한다 |
| **제곱 비용** | 전역 셀프 어텐션은 토큰 수 $N$에 대해 $O(N^2)$ | 패치를 4×4로 작게 자르면 224×224에서도 토큰 3,136개, 고해상도 이미지에서는 감당할 수 없다 |

**PVT**(Pyramid Vision Transformer, 2021, 왕 등)

- CNN처럼 **4단계 피라미드** 구조: 단계마다 패치 임베딩으로 해상도를 줄이고 채널을 늘린다 → 1/4, 1/8, 1/16, 1/32 해상도의 특성 맵 4개. 12장 CNN 백본(ResNet 등)을 그대로 대체할 수 있다
- **공간 축소 어텐션**(SRA): 쿼리는 모든 토큰에서 만들지만, **키·값은 공간 해상도를 $R$배 줄인** 특성 맵에서 만든다 → 비용이 $O(N^2)$에서 $O(N^2/R^2)$으로. 여전히 전역 어텐션(모든 위치가 전체 이미지를 본다)이지만 키·값이 요약된다

**Swin 트랜스포머**(Shifted WINdow, 2021, 마이크로소프트, 류 등)

```
단계 1: 4×4 패치 → 56×56 토큰, C = 96      ┐
   [윈도 어텐션] → [이동 윈도 어텐션] (× n)   │ 각 단계 끝에서 패치 병합:
단계 2: 28×28, 2C                            │ 이웃한 2×2 토큰을 이어 붙여(4C)
단계 3: 14×14, 4C                            │ 선형 층으로 2C로 → 해상도 1/2, 채널 ×2
단계 4: 7×7, 8C                              ┘
```

| 구성 | 설명 |
|---|---|
| **윈도 어텐션**(W-MSA) | 특성 맵을 겹치지 않는 $M \times M$ **윈도**(기본 7×7 = 토큰 49개)로 나누고, **윈도 안에서만** 셀프 어텐션. 윈도 하나의 비용은 고정이고 윈도 수는 이미지 넓이에 비례 → 전체 비용이 이미지 크기에 **선형** |
| **이동 윈도**(SW-MSA) | 윈도 안에서만 보면 윈도 경계 너머와 정보를 주고받지 못한다. 그래서 다음 층에서는 윈도를 $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$만큼 **비켜서** 나눈다 → 이전 층의 서로 다른 윈도에 있던 토큰이 같은 윈도에 들어간다. 두 층을 번갈아 쌓으면 정보가 점점 멀리 퍼진다 (CNN의 수용장이 커지는 것과 비슷) |
| **패치 병합** | CNN의 풀링처럼 단계마다 해상도를 절반으로 → PVT처럼 **특성 피라미드** |
| 상대 위치 편향 | 절대 위치 임베딩 대신, 윈도 안 두 토큰의 **상대 위치**에 따라 학습되는 편향을 어텐션 점수에 더한다 |

Swin 논문의 계산량(층 하나, $h \times w$ 토큰, 채널 $C$):

$$
\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C, \qquad \Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC
$$

첫 항($4hwC^2$)은 쿼리·키·값·출력 투영이라 같고, 둘째 항(어텐션 행렬)에서 $(hw)^2$이 $M^2 hw$로 바뀐다.

**보충 코드** — 전역 어텐션과 윈도 어텐션의 계산량

```python
# 보충 코드
def msa(h, w, C):  return 4*h*w*C**2 + 2*(h*w)**2*C
def wmsa(h, w, C, M=7):  return 4*h*w*C**2 + 2*M**2*h*w*C

for hw in (56, 112, 224):
    print(hw, f"{msa(hw, hw, 96):.3e}", f"{wmsa(hw, hw, 96):.3e}",
          round(msa(hw, hw, 96) / wmsa(hw, hw, 96), 1))
```

```
56 2.004e+09 1.451e+08 13.8
112 3.067e+10 5.804e+08 52.8
224 4.852e+11 2.322e+09 209.0
```

![전역 어텐션과 윈도 어텐션의 계산량](images/16/extra_swin_attention_cost.png)

- Swin-T의 첫 단계(56×56 토큰, $C$ = 96)에서 이미 윈도 어텐션이 **약 14배** 싸다. 토큰 격자가 224×224(고해상도 입력)면 **209배**
- 윈도 어텐션의 곡선은 $hw$에 선형이라, 로그 축에서 전역 어텐션보다 완만하게 오른다

> 📌 Swin은 분류(ImageNet)뿐 아니라 COCO 탐지·ADE20K 분할에서 당시 최고 성능을 내며 "범용 비전 백본"으로 자리 잡았다. 2022년의 ConvNeXt([12장 5-11절](12_deep_computer_vision_with_cnns.md#5-11-그-외-주목할-구조))는 거꾸로 Swin의 설계(큰 커널, 단계 구성, 층 정규화, GELU 등)를 CNN에 옮겨 CNN도 같은 성능을 낼 수 있음을 보였다.

### 1-12. DINO: 자기 지도 시각 표현 학습

> 🎯 **목적** — 레이블 없이 ViT를 훈련하는 DINO의 자기 증류 방식과, 그 결과 어텐션에서 저절로 나타나는 성질을 정리한다.

> ⚠️ 노트북에는 사전 훈련된 DINO를 쓰는 코드만 있다(1-13절). 아래 훈련 원리는 교재 절 제목에 맞춰 정리한 것이다.

**개념·원리**

15장의 BERT·GPT는 레이블 없는 텍스트로 스스로 문제를 만들어 사전 훈련했다(자기 지도 학습). 2021년 페이스북의 **DINO**(self-DIstillation with NO labels, 카롱 등)는 이미지로 같은 일을 한다. 핵심은 "**같은 이미지의 다른 조각을 보고 같은 결론을 내라**"이다.

```
원본 이미지 ─ 무작위 크롭·증강 ─┬─ 전역 뷰 2개 (224×224, 원본의 40 ~ 100% 영역)
                               └─ 지역 뷰 여러 개 (96×96, 원본의 5 ~ 40% 영역)

학생 ViT ← 모든 뷰 입력      → 소프트맥스 (온도 τs = 0.1)  ─┐
교사 ViT ← 전역 뷰만 입력    → 센터링 → 소프트맥스 (τt = 0.04, 더 날카롭게) ─┤
                                                     손실: 교차 엔트로피(교사 ‖ 학생)
                                                     (같은 뷰끼리는 비교하지 않음)
교사 가중치 = 학생 가중치의 지수 이동 평균(EMA)  ← 역전파 없음
```

| 요소 | 역할 |
|---|---|
| **학생과 교사** | 구조가 같은 ViT 두 개. 학생만 경사 하강법으로 훈련한다 |
| **교사 = 학생의 EMA** | $\theta_t \leftarrow \lambda\theta_t + (1-\lambda)\theta_s$ ($\lambda$는 0.996 → 1). 교사는 학생의 "느리게 움직이는 평균"이라 더 안정적이고 성능도 조금 낫다. 11장의 [모멘텀](11_training_deep_neural_networks.md)처럼 과거 모델을 평균 낸 것 |
| **다중 크롭**(multi-crop) | 학생은 작은 지역 뷰만 보고도, 교사가 전체(전역 뷰)를 보고 낸 결론을 맞혀야 한다 → "**부분에서 전체로**"의 대응을 배운다. 고양이 귀 조각만 봐도 "고양이 이미지"라는 표현을 내도록 |
| **붕괴 방지** | 학생과 교사가 모든 입력에 같은 출력을 내면 손실이 0이 되는 쉬운 해(붕괴)가 있다. ① **센터링**: 교사 출력에서 배치 평균을 빼 한 차원이 항상 이기지 못하게 하고 ② **샤프닝**: 교사 소프트맥스 온도를 낮춰 균등 분포로 가지 못하게 한다. 둘이 서로를 견제한다 |
| 레이블·대조 쌍 불필요 | 2-2절의 CLIP 같은 대조 학습은 "다른 샘플"(음성 쌍)이 필요하지만 DINO는 필요 없다 |

**저절로 나타나는 성질**

- 사람이 분할 레이블을 준 적이 없는데, 마지막 층의 **`[CLS]` 어텐션이 물체의 윤곽을 따라간다** → 다음 절에서 그대로 **비지도 분할**에 쓴다. 지도 학습으로 훈련한 ViT의 어텐션은 이만큼 깔끔하지 않다
- `[CLS]` 출력이 좋은 이미지 특성이라, 그 위에 **k-최근접 이웃**(1장)만 써도 ImageNet을 꽤 잘 분류한다(지도 학습 없이). 이미지 검색, 복사 이미지 탐지에도 쓴다
- 2023년의 **DINOv2**는 대규모로 선별한 데이터로 같은 방식을 키워, 미세 튜닝 없이 깊이 추정·분할·분류에 바로 쓰는 범용 시각 특성을 만들었다

### 1-13. DINO 어텐션 맵으로 비지도 분할하기

> 🎯 **목적** — 사전 훈련된 DINO ViT의 마지막 층 어텐션 가중치를 이미지 위에 겹쳐 그려, 레이블 없이 물체 영역을 찾는다.

**코드** — 이미지와 모델

```python
from PIL import Image
import urllib.request

image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(urllib.request.urlopen(image_url))
image
```

![COCO 고양이 이미지](images/16/nb_coco_cats.png)

```python
from transformers import AutoImageProcessor, AutoModel

model_id = "facebook/dino-vitb8"
model = AutoModel.from_pretrained(model_id, output_attentions=True)
processor = AutoImageProcessor.from_pretrained(model_id, do_convert_rgb=True)
```

```
Some weights of ViTModel were not initialized from the model checkpoint at facebook/dino-vitb8 and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
```

| 코드 | 의미 |
|---|---|
| `"facebook/dino-vitb8"` | DINO로 훈련한 **ViT-B/8**. 패치가 8×8이라 224×224 이미지가 28 × 28 = **784개 패치**가 된다(ViT-B/16의 4배) → 어텐션 맵이 더 세밀하다 |
| `AutoModel.from_pretrained(...)` | 헤드 없는 `ViTModel`. DINO 체크포인트에는 풀러(`[CLS]` → 선형 층 → tanh)가 없어서 새로 초기화된다는 경고가 나오지만, 풀러 출력을 쓰지 않으므로 상관없다 |
| `output_attentions=True` | 순전파 때 **모든 층의 어텐션 가중치**를 함께 돌려받는다 |
| 프로세서 | 224×224로 크기 조정(비율 무시) + ImageNet 평균·표준 편차로 정규화 |

> ⚠️ 트랜스포머스 5.x에서는 기본 어텐션 구현이 `sdpa`(파이토치의 `F.scaled_dot_product_attention`, 15장 1-3절의 📌)인데, 이 커널은 어텐션 가중치 행렬을 만들지 않는다. 그래서 `output_attentions=True`만 주면 `output.attentions`가 **빈 튜플**이 되고 아래 `output.attentions[-1]`에서 `IndexError`가 난다(보충 실험으로 확인). 가중치를 직접 계산하는 구현을 쓰도록 `AutoModel.from_pretrained(model_id, output_attentions=True, attn_implementation="eager")`로 바꾼다.

**코드** — 순전파와 어텐션 꺼내기

```python
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
  output = model(**inputs)

cls_token_output = output.last_hidden_state[:, 0]
cls_token_output.shape
# torch.Size([1, 768])

last_layer_attention_maps = output.attentions[-1]
# NOTE: this is the Patch-to-CLS attention, which seems to work better than CLS-to-Patch
cls_attn = last_layer_attention_maps[0, :, 1:, 0]
cls_attn.shape
# torch.Size([12, 784])
```

| 코드 | 형태 | 의미 |
|---|---|---|
| `output.last_hidden_state[:, 0]` | `[1, 768]` | `[CLS]` 출력 = DINO가 만든 **이미지 특성 벡터** (검색·k-NN 분류에 쓰는 것) |
| `output.attentions` | 층 12개의 튜플 | 층마다 `[배치, 헤드, 쿼리, 키]` = `[1, 12, 785, 785]` (785 = `[CLS]` 1 + 패치 784) |
| `[-1]` | `[1, 12, 785, 785]` | 마지막 층 |
| `[0, :, 1:, 0]` | `[12, 784]` | 배치 0, 모든 헤드, **쿼리 = 패치 784개**, **키 = `[CLS]`(0번)** → "각 패치가 `[CLS]`에 주는 어텐션 가중치" |

> 💡 어텐션 행렬에서 꺼내는 방향이 두 가지다. DINO 논문은 **`[CLS]` → 패치** 방향(`[0, :, 0, 1:]`, `[CLS]` 쿼리가 각 패치 키에 주는 가중치, 784개의 합이 1)을 그렸다. 노트북은 반대 방향인 **패치 → `[CLS]`**(`[0, :, 1:, 0]`, 패치마다 자기 어텐션 중 `[CLS]` 몫)을 쓰고, 이쪽이 더 잘 된다고 주석에 적었다. 둘 다 형태는 `[12, 784]`다. 패치 → `[CLS]` 값은 패치마다 **다른 소프트맥스**에서 나왔으므로 784개의 합이 1이 아니다.

**코드** — 헤드별 어텐션 맵 그리기

```python
import torchvision.transforms.functional as TF

num_heads, num_patches = cls_attn.shape
size = int(num_patches ** 0.5)
plt.figure(figsize=(12, 7))
for head_index in range(12):
    plt.subplot(3, 4, head_index + 1)
    attn = cls_attn[head_index].reshape(size, size)
    attn_map = attn.unsqueeze(0).unsqueeze(0)
    attn_resized = TF.resize(attn_map, image.size[::-1],
                             interpolation=TF.InterpolationMode.BILINEAR)[0, 0]
    plt.imshow(image)
    plt.imshow(attn_resized.numpy(), cmap='jet', alpha=0.5)
    plt.axis('off')

plt.show()
```

| 코드 | 의미 |
|---|---|
| `size = int(784 ** 0.5)` | 28. 패치 784개를 28×28 격자로 되돌린다 (1-3절의 `flatten`이 행 우선이었으므로 `reshape`로 원래 배치가 복원된다) |
| `attn.unsqueeze(0).unsqueeze(0)` | `[28, 28]` → `[1, 1, 28, 28]`. `TF.resize`가 기대하는 `[..., H, W]` 이미지 형태로 |
| `image.size[::-1]` | PIL의 `size`는 (너비, 높이) = (640, 480) → 뒤집어 (480, 640) |
| `TF.resize(..., BILINEAR)[0, 0]` | 28×28 어텐션 맵을 원본 크기로 **쌍선형 보간** → `[480, 640]` |
| `imshow(..., cmap='jet', alpha=0.5)` | 원본 위에 반투명하게 겹친다. 빨강 = 어텐션이 큼, 파랑 = 작음 |

![DINO 마지막 층의 헤드별 어텐션 맵](images/16/nb_dino_attention_maps.png)

**결과 해석**

- 12개 헤드가 **서로 다른 대상**에 주목한다. 어떤 헤드는 왼쪽 고양이의 몸통 무늬를, 어떤 헤드는 두 고양이 전체를(둘째 줄 오른쪽), 어떤 헤드는 **리모컨**을(둘째 줄 셋째 칸), 어떤 헤드는 **배경**(소파 윗부분, 첫째 줄 오른쪽)을 본다
- DINO는 고양이·리모컨·소파라는 레이블을 본 적이 없다. "부분 뷰에서 전체 이미지의 표현을 맞히기"를 배우는 과정에서, 의미 있는 물체 단위로 패치를 묶는 법을 **스스로** 배웠다
- 이 맵에 임곗값을 적용하면(예: 어텐션 상위 60%) 물체 마스크를 얻는다 → 레이블 없는 **비지도 분할**. 12장 [12절의 시맨틱 분할](12_deep_computer_vision_with_cnns.md#12-시맨틱-분할-1212)은 픽셀마다 레이블이 필요했다
- 맵이 조금 뭉툭한 것은 해상도 28×28을 640×480으로 늘렸기 때문이다. 또 프로세서가 4:3 이미지를 1:1로 찌그러뜨렸다가 다시 4:3으로 늘린 결과다

**보충 코드** — 어텐션 형태 확인 (config로만 만든 무작위 ViT-B/8)

```python
# 보충 코드 (트랜스포머스 5.23)
from transformers import ViTConfig, AutoModel
model = AutoModel.from_config(ViTConfig(patch_size=8), attn_implementation="eager").eval()
with torch.no_grad():
    out = model(pixel_values=torch.randn(1, 3, 224, 224), output_attentions=True)
len(out.attentions), out.attentions[-1].shape   # 12, torch.Size([1, 12, 785, 785])
out.attentions[-1][0].sum(dim=-1)[0, :3]        # tensor([1., 1., 1.])  ← 쿼리마다 키 축 합이 1
```

### 1-14. 그 밖의 주요 비전 모델과 기법

> 🎯 **목적** — 이 장에서 직접 다루지 않은 대표적인 비전 트랜스포머 계열을 짧게 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 대표 모델을 정리한 것이다.

| 모델·기법 | 연도·기관 | 핵심 아이디어 |
|---|---|---|
| **BEiT** | 2021, 마이크로소프트 | BERT의 MLM을 이미지에: 패치 일부를 가리고, 가린 패치의 **이산 시각 토큰**(2-7절 DALL·E의 dVAE 토큰)을 맞힌다 |
| **MAE**(Masked Autoencoder) | 2021, 페이스북 | 패치의 **75%** 를 가리고 보이는 25%만 인코더에 넣은 뒤, 가벼운 디코더가 가린 패치의 **픽셀**을 복원한다. 인코더가 25%만 처리해 훈련이 3배 이상 빠르다. 오토인코더는 18장 |
| **DINOv2** | 2023, 메타 | 1-12절의 DINO를 대규모 선별 데이터로 확장 → 미세 튜닝 없이 쓰는 범용 시각 특성 |
| **SAM**(Segment Anything) | 2023, 메타 | ViT 이미지 인코더 + 프롬프트(점, 박스, 텍스트) 인코더 + 마스크 디코더. 처음 보는 물체도 클릭 한 번으로 분할하는 **프롬프트 가능한 분할** 모델 |
| **하이브리드 모델** | | CNN 줄기(stem)로 저수준 특성을 뽑고 트랜스포머를 올리거나(원본 ViT 논문의 하이브리드, DETR), 합성곱과 어텐션을 섞은 층(CoAtNet 등)을 쓴다 |
| **대형 ViT** | 2023 | ViT-22B처럼 파라미터 수백억 규모까지 키운 이미지 인코더. 멀티모달 모델의 "눈"으로 쓰인다 (2절) |

> 💡 MAE·BEiT·DINO는 모두 **레이블 없는 이미지로 사전 훈련**한다. 15장에서 텍스트가 걸어간 길(지도 학습 → 자기 지도 사전 훈련 + 미세 튜닝)을 비전도 똑같이 따라갔다.

---

## 2. 멀티모달 트랜스포머 (16.2)

> 🎯 **목적** — 텍스트·이미지·비디오·오디오처럼 **여러 모달리티**를 함께 다루는 트랜스포머들이 모달리티를 어떻게 섞는지 정리하고, CLIP·Perceiver·BLIP-2를 직접 써 본다.

<!-- 일타 -->
> 💬 **이미지와 글을 같은 지도 위에**
>
> CLIP은 이미지 인코더와 텍스트 인코더를 따로 두고, **짝이 맞는 (사진, 캡션)은 가깝게, 안 맞는 쌍은 멀게** 임베딩을 학습한다(대조 학습). 그러면 "a photo of a cat"이라는 문장 벡터와 고양이 사진 벡터가 같은 동네에 산다. 분류하고 싶은 클래스 이름으로 문장을 만들고, 사진과 가장 가까운 문장을 고르면 끝 — **한 번도 그 클래스로 훈련하지 않고도**(제로샷).
>
> - **핵심 한 줄**: 배치가 클수록 오답 후보가 많아 더 정교하게 배운다
> - **Perceiver·Q-Former·Resampler의 공통점**: 거대한 입력(픽셀 수만 개, 이미지 특성)을 **학습되는 쿼리 몇십 개**가 크로스 어텐션으로 요약한다. 그 요약을 LLM에 넣으면 이미지에 대해 대화하는 모델(BLIP-2, Flamingo)이 된다
> - **헷갈림 주의**: CLIP의 확률은 **후보 레이블끼리의 상대 비교**다. 후보 목록에 정답이 없으면 엉뚱한 레이블에도 높은 확률이 나온다


**개념·원리**

**모달리티**(modality)는 데이터의 종류(텍스트, 이미지, 오디오, 비디오, 로봇 센서 등)다. 트랜스포머는 입력이 벡터 시퀀스이기만 하면 되므로, 모달리티마다 **토큰으로 바꾸는 방법**만 정하면 같은 구조로 함께 처리할 수 있다. 문제는 "서로 다른 모달리티의 토큰을 **어디서, 어떻게 만나게 하는가**"다.

| 결합 방식 | 구조 | 이 절의 모델 |
|---|---|---|
| **단일 스트림** | 두 모달리티의 토큰을 **이어 붙여** 트랜스포머 하나에 넣는다 | VideoBERT, DALL·E |
| **이중 스트림 + 공동 어텐션** | 모달리티마다 트랜스포머를 두고, 중간에 서로의 토큰을 **크로스 어텐션**으로 본다 | ViLBERT |
| **이중 인코더** | 각자 따로 인코딩해 벡터 하나씩 만들고, **임베딩 공간에서만** 비교한다 | CLIP |
| **잠재 배열** | 어떤 모달리티든 크로스 어텐션으로 작은 잠재 배열에 모은다 | Perceiver, Perceiver IO |
| **동결된 LLM에 연결** | 사전 훈련된 이미지 인코더와 LLM은 그대로 두고, 둘을 잇는 작은 **다리**만 훈련한다 | Flamingo, BLIP-2 |
| **처음부터 멀티모달** | 모든 모달리티를 토큰으로 바꿔 거대한 모델 하나로 처음부터 함께 훈련 | Gemini 등 (3절) |

### 2-1. VideoBERT와 ViLBERT

> 🎯 **목적** — BERT를 텍스트 + 비디오, 텍스트 + 이미지로 확장한 초기 멀티모달 트랜스포머 두 개를 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 개념을 정리한 것이다.

**VideoBERT**(2019, 구글, 쑨 등) — 텍스트와 비디오를 **같은 시퀀스**에 넣은 BERT

```
요리 동영상 (유튜브)
 ├ 음성 → 자동 음성 인식(ASR) → 텍스트 토큰 (WordPiece)
 └ 1.5초 클립 → 사전 훈련된 비디오 CNN 특성 → 계층적 k-평균 군집 → "시각 단어" 번호
   (군집 중심 20,736개 = 어휘)

[CLS] 텍스트 토큰들 [>] 시각 토큰들 [SEP]   → BERT (MLM: 텍스트·시각 토큰을 가리고 맞히기)
```

- 핵심 트릭은 연속적인 비디오 특성을 **k-평균**(8장)으로 군집해 **이산 토큰**으로 만든 것이다. 그러면 비디오도 단어처럼 어휘가 생기고, BERT의 MLM을 그대로 쓸 수 있다
- 응용: "now let me show you how to [MASK] the [MASK]" 같은 템플릿으로 동작(동사·명사)을 **제로샷 분류**하고, 텍스트(조리법)로 다음 장면의 시각 토큰을 예측한다

**ViLBERT**(Vision-and-Language BERT, 2019, 루 등) — 이미지와 텍스트를 **두 스트림**으로

```
이미지 → 객체 탐지기(Faster R-CNN) → 영역 특성 10 ~ 36개 + 위치 ─→ 이미지 스트림(트랜스포머)
텍스트 → WordPiece 토큰 ───────────────────────────────────→ 텍스트 스트림(BERT)
                         ⇅ 공동 어텐션(co-attention) 층
     이미지 쪽 쿼리 ↔ 텍스트 쪽 키·값,  텍스트 쪽 쿼리 ↔ 이미지 쪽 키·값
```

| 사전 훈련 과제 (Conceptual Captions, 이미지–캡션 약 330만 쌍) | 내용 |
|---|---|
| 마스크드 멀티모달 학습 | 단어와 이미지 영역의 15%를 가리고 맞힌다 (가린 영역은 탐지기가 본 클래스 분포를 맞힌다) |
| 멀티모달 정렬 예측 | 캡션이 이 이미지를 설명하는 것인지 이진 분류 (BERT의 NSP와 비슷) |

- 공동 어텐션은 15장 [디코더의 크로스 어텐션](15_transformers_for_nlp_and_chatbots.md#1-7-인코더-층과-디코더-층)을 **양방향**으로 한 것이다. 모달리티마다 다른 크기·깊이의 스트림을 쓸 수 있다
- 시각 질의응답(VQA), 시각 상식 추론, 지시 표현(“왼쪽의 빨간 컵”) 찾기, 캡션으로 이미지 검색 등에 미세 튜닝했다

### 2-2. CLIP: 대조 사전 훈련

> 🎯 **목적** — 이미지와 텍스트를 같은 임베딩 공간에 놓도록 훈련한 CLIP의 구조와 대조 학습 원리를 이해한다.

**개념·원리**

2021년 OpenAI의 **CLIP**(Contrastive Language–Image Pre-training, 래드퍼드 등)은 인터넷에서 모은 **이미지–텍스트 쌍 4억 개**로 훈련한 **이중 인코더**다.

```
이미지 ─→ 이미지 인코더 (ViT-B/32 등) ─→ 선형 투영 → 정규화 → 이미지 임베딩 I (512차원)
텍스트 ─→ 텍스트 인코더 (트랜스포머) ──→ 선형 투영 → 정규화 → 텍스트 임베딩 T (512차원)

배치 N쌍의 유사도 행렬 (N × N):     T₁     T₂     T₃    ...
                              I₁  [ I₁·T₁  I₁·T₂  I₁·T₃ ... ]   ← 대각선(짝이 맞는 쌍)은 크게
                              I₂  [ I₂·T₁  I₂·T₂  I₂·T₃ ... ]      나머지(짝이 아닌 쌍)는 작게
                              ...
```

**대조 손실** — 유사도 행렬의 각 **행**을 "이 이미지의 캡션은 N개 중 어느 것인가"를 맞히는 N-클래스 분류로, 각 **열**을 "이 캡션의 이미지는 어느 것인가"로 본다. 두 방향의 교차 엔트로피를 평균한다.

$$
\mathcal{L} = \frac{1}{2}\left[\text{CE}\left(\frac{IT^\top}{\tau}, \text{diag}\right) + \text{CE}\left(\frac{TI^\top}{\tau}, \text{diag}\right)\right]
$$

($I$, $T$는 정규화된 이미지·텍스트 임베딩 행렬 `[N, 512]`, diag는 정답이 대각선 위치라는 뜻)

| 요소 | 설명 |
|---|---|
| 정규화된 임베딩 | 길이를 1로 맞추므로 점곱 = **코사인 유사도**(−1 ~ 1). 15장 [3-6절의 문장 임베딩](15_transformers_for_nlp_and_chatbots.md#3-6-문장-임베딩-sentence-transformers)과 같은 비교 방식 |
| 온도 $\tau$ | 코사인 유사도는 범위가 좁아 그대로 소프트맥스에 넣으면 확률이 거의 균등하다. $1/\tau$를 곱해 점수 차이를 키운다. CLIP은 $\tau$를 **학습되는 파라미터**(`logit_scale` = $\ln(1/\tau)$)로 두고 0.07에서 시작했으며, 훈련이 불안정해지지 않도록 $1/\tau$가 100을 넘지 않게 잘랐다 |
| 큰 배치 | 배치 크기 **32,768**. 한 배치 안의 다른 샘플이 모두 음성(오답) 예시가 되므로, 배치가 클수록 어려운 문제가 되어 좋은 표현을 배운다 |
| 무엇을 배우나 | "이 사진과 이 문장이 짝인가"를 맞히려면 사진 속 물체·장면·스타일·글자를 언어와 연결해야 한다 → **언어로 기술할 수 있는 시각 개념**을 폭넓게 배운다 |

**제로샷 분류** — 클래스 이름을 문장으로 만들어 텍스트 인코더에 넣고, 이미지 임베딩과 가장 가까운 문장을 고른다. 분류기를 훈련하지 않고 **클래스 목록만 바꿔** 새 분류기를 만들 수 있다(2-4절). CLIP의 가장 큰 모델은 ImageNet 훈련 이미지를 한 장도 보지 않고 ImageNet 정확도 76.2%를 내, 지도 학습으로 훈련한 원조 ResNet-50과 맞먹었다.

> 💡 "cat" 대신 "A photo of a cat."처럼 **프롬프트 템플릿**에 넣으면 정확도가 오른다. 인터넷 캡션에 단어 하나만 있는 경우는 드물고, "cat"이라는 단어 하나는 동물인지 리눅스 명령어인지 모호하기 때문이다. 여러 템플릿("a drawing of a …", "a close-up photo of a …")의 텍스트 임베딩을 평균하는 **프롬프트 앙상블**도 쓴다.

> 📌 CLIP의 인코더는 이후 수많은 모델의 부품이 됐다. 이미지 검색, 생성 이미지의 순위 매기기(2-7절 DALL·E), 텍스트로 이미지 생성 모델을 조종하는 텍스트 인코더(스테이블 디퓨전, 18장), 멀티모달 LLM의 이미지 인코더(LLaVA 등, 3절).

### 2-3. 보충 코드: CLIP의 대조 손실

> 🎯 **목적** — 2-2절의 대칭 교차 엔트로피 손실을 직접 구현하고, 짝이 맞을 때와 무작위일 때 값을 비교한다.

**보충 코드**

```python
# 보충 코드
import math

def clip_loss(img_emb, txt_emb, logit_scale):
    img_emb = F.normalize(img_emb, dim=1)               # 길이 1로
    txt_emb = F.normalize(txt_emb, dim=1)
    logits = logit_scale * img_emb @ txt_emb.T          # [N, N] 코사인 유사도 × 1/τ
    labels = torch.arange(len(img_emb))                 # 정답 = 대각선 (i번 이미지 ↔ i번 텍스트)
    return (F.cross_entropy(logits, labels)             # 행 방향: 이미지 → 텍스트
            + F.cross_entropy(logits.T, labels)) / 2    # 열 방향: 텍스트 → 이미지

torch.manual_seed(42)
N, D = 8, 512
img, txt = torch.randn(N, D), torch.randn(N, D)
clip_loss(img, txt, 1 / 0.07)                            # 2.4569   (ln 8 = 2.0794)
clip_loss(img, img + 0.1 * torch.randn(N, D), 1 / 0.07)  # 5.0e-06  (짝이 맞는 쌍)
```

| 코드 | 의미 |
|---|---|
| `F.normalize(x, dim=1)` | 행마다 $x / \lVert x \rVert$ |
| `logit_scale * img @ txt.T` | 이 장의 CLIP은 `logit_scale`을 로그로 저장하고 `.exp()`해서 곱한다(2-5절). 여기서는 곱할 값 $1/\tau$를 바로 넘겼다 |
| `torch.arange(N)` | i번째 이미지의 정답 텍스트는 i번째. 배치 안의 나머지 N − 1개가 **오답 후보** |
| `logits.T` | 같은 행렬을 전치하면 텍스트 기준 분류 문제 |

**결과 해석**

- 무작위 임베딩: 512차원 무작위 벡터끼리의 코사인 유사도는 거의 0(±0.04)이라 로짓이 ±0.6 정도로 고르다 → 손실이 균등 추측의 값 $\ln N = \ln 8 \approx 2.08$ 근처(조금 위)
- 짝이 맞는 쌍(텍스트 = 이미지 + 작은 잡음): 대각선 유사도 ≈ 1, 나머지 ≈ 0 → 로짓 차이가 약 14 → 손실이 거의 0
- 배치 크기 $N$이 커질수록 무작위일 때의 손실 $\ln N$이 커지고, 정답을 고르기가 어려워진다 → CLIP이 큰 배치를 쓴 이유

### 2-4. CLIP으로 제로샷 이미지 분류하기

> 🎯 **목적** — 허깅 페이스 파이프라인으로, 훈련 없이 레이블 이름 목록만으로 이미지 분류기를 만든다.

**코드** — 곤충 분류기

```python
from transformers import pipeline

model_id = "openai/clip-vit-base-patch32"
clip_pipeline = pipeline(task="zero-shot-image-classification", model=model_id,
                         device_map="auto", dtype="auto")
candidate_labels = ["cricket", "ladybug", "spider"]
image_url = "https://homl.info/ladybug"  # a photo of a ladybug on a dandelion
results = clip_pipeline(image_url, candidate_labels=candidate_labels,
                        hypothesis_template="This is a photo of a {}.")
results
```

```
[{'score': 0.9972853660583496, 'label': 'ladybug'},
 {'score': 0.0016511697322130203, 'label': 'spider'},
 {'score': 0.0010634352220222354, 'label': 'cricket'}]
```

| 코드 | 의미 |
|---|---|
| `"openai/clip-vit-base-patch32"` | 이미지 인코더 **ViT-B/32**(224×224 → 패치 7×7 = 49개 + `[CLS]`), 텍스트 인코더 12층 트랜스포머(최대 77 토큰), 공동 임베딩 512차원. 내려받기 약 605MB |
| `task="zero-shot-image-classification"` | 이미지 + 후보 레이블 → 레이블별 확률 |
| `device_map="auto", dtype="auto"` | 15장 [4-2절](15_transformers_for_nlp_and_chatbots.md#4-2-gpt-2로-텍스트-생성하기)과 같다. 가능한 장치(GPU)에 올리고, 저장된 정밀도 그대로 |
| `clip_pipeline(image_url, ...)` | 이미지 자리에 **URL**을 바로 줘도 내려받아 연다 |
| `hypothesis_template="This is a photo of a {}."` | 레이블마다 `{}`에 끼워 "This is a photo of a ladybug." 같은 캡션을 만든다 (2-2절의 💡) |
| 출력 | 점수 내림차순. 세 점수의 합 = 1 (후보 레이블끼리의 소프트맥스) |

**코드** — 같은 사진으로 꽃 분류기

```python
candidate_labels2 = ["dandelion", "lily", "popy", "rose", "sunflower"]
results2 = clip_pipeline(image_url, candidate_labels=candidate_labels2,
                         hypothesis_template="This is a photo of a {}.")
results2
```

```
[{'score': 0.6832454204559326, 'label': 'dandelion'},
 {'score': 0.28150027990341187, 'label': 'popy'},
 {'score': 0.020853344351053238, 'label': 'sunflower'},
 {'score': 0.012900755740702152, 'label': 'rose'},
 {'score': 0.0015001024585217237, 'label': 'lily'}]
```

**결과 해석**

- 같은 모델, 같은 사진인데 **후보 목록만 바꾸자** 곤충 분류기에서 꽃 분류기가 됐다. 사진의 주인공(무당벌레)이 후보에 없으니 배경의 꽃(민들레)을 골랐다(68.3%)
- 곤충(99.7%)보다 확신이 낮다. 사진 속 민들레가 작게 나와 있고, 빨간 무당벌레가 빨간 꽃인 양귀비(poppy)를 연상시켰을 수 있다(28.2%)
- ⚠️ 노트북의 `"popy"`는 `"poppy"`(양귀비)의 오타다. CLIP의 BPE 토크나이저는 오타도 조각 토큰으로 처리하므로 오류는 나지 않지만, 정확한 단어를 써야 결과를 믿을 수 있다
- 확률은 **후보끼리의 상대 비교**다. 후보에 정답이 없어도 합이 1이 되도록 무언가를 고른다 → "해당 없음"을 판단하지 못한다. 필요하면 "something else" 같은 후보를 넣거나 유사도의 절대값을 함께 본다

### 2-5. CLIPModel로 직접 계산하기

> 🎯 **목적** — 파이프라인이 내부에서 하는 계산(임베딩 → 코사인 유사도 → 온도 → 소프트맥스)을 직접 따라 한다.

**코드**

```python
import PIL
import urllib.request
from transformers import CLIPProcessor, CLIPModel

clip_processor = CLIPProcessor.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id)
image = PIL.Image.open(urllib.request.urlopen(image_url)).convert("RGB")
captions = [f"This is a photo of a {label}." for label in candidate_labels]
inputs = clip_processor(text=captions, images=[image], return_tensors="pt",
                        padding=True)
with torch.no_grad():
    outputs = clip_model(**inputs)

text_features = outputs.text_embeds    # shape [3, 512]  # 3 captions
image_features = outputs.image_embeds  # shape [1, 512]  # 1 image (ladybug)
```

| 코드 | 의미 |
|---|---|
| `CLIPProcessor` | **토크나이저 + 이미지 프로세서**를 묶은 것. `text=`와 `images=`를 함께 받아 `input_ids`·`attention_mask`·`pixel_values`를 한 번에 만든다 |
| `.convert("RGB")` | 1-7절의 `do_convert_rgb=True`와 같은 목적 |
| `padding=True` | 캡션 3개의 길이를 맞춘다 (14장 2-10절) |
| `outputs.text_embeds` `[3, 512]` | 캡션마다 텍스트 인코더 → 투영 → **정규화**된 임베딩 |
| `outputs.image_embeds` `[1, 512]` | 이미지 인코더(`[CLS]` 출력) → 투영 → 정규화된 임베딩 |

```python
image_features.norm(dim=1)     # tensor([1.])
text_features.norm(dim=1)      # tensor([1.0000, 1.0000, 1.0000])

similarities = image_features @ text_features.T  # shape [1, 3]
similarities
# tensor([[0.2337, 0.3021, 0.2381]])
```

- 이미 길이가 1이므로 점곱이 곧 **코사인 유사도**다. 순서는 cricket, ladybug, spider
- 무당벌레 캡션이 가장 크지만(0.30), 차이가 0.07 정도로 작다. 코사인 유사도의 절대값도 0.2 ~ 0.3에 몰려 있다. CLIP의 이미지 임베딩과 텍스트 임베딩은 공간에서 서로 다른 영역에 모여 있어(**모달리티 간극**), 짝이 맞아도 코사인이 1에 가깝지 않다. 중요한 것은 **상대적인 순서**다

```python
temperature = clip_model.logit_scale.detach().exp()
rescaled_similarities = similarities * temperature
probabilities = torch.nn.functional.softmax(rescaled_similarities , dim=1)
probabilities
# tensor([[0.0011, 0.9973, 0.0017]])

temperature
# tensor(100.0000)
```

| 코드 | 의미 |
|---|---|
| `clip_model.logit_scale` | 학습된 스칼라 파라미터 $\ln(1/\tau)$ |
| `.detach().exp()` | 계산 그래프에서 떼고 지수 → **100** ($\tau$ = 0.01). 훈련 중 커지다가 상한 100에 걸린 값이다 |
| `similarities * temperature` | 23.37, 30.21, 23.81 → 소프트맥스 → 파이프라인과 **같은 확률** (0.0011, 0.9973, 0.0017) |

> ⚠️ 노트북 설명에는 `logit_scale`이 "ln(0.07)로 초기화됐다"고 적혀 있지만, 정확히는 **$\ln(1/0.07) \approx 2.659$** 다($\tau$ = 0.07이고, 곱하는 값은 $1/\tau \approx 14.3$). 트랜스포머스 `CLIPConfig`의 기본값 `logit_scale_init_value=2.6592`, 무작위 초기화한 `CLIPModel`의 `logit_scale.exp()` = 14.28로 확인했다. 상한은 $\ln 100$이다.

**보충 코드** — 온도에 따라 확률이 어떻게 바뀌나

```python
# 보충 코드
sims = torch.tensor([[0.2337, 0.3021, 0.2381]])
for T in (1, 1 / 0.07, 100):
    print(T, (sims * T).softmax(dim=1))
```

```
1 tensor([[0.3252, 0.3482, 0.3266]])
14.285714285714286 tensor([[0.2118, 0.5628, 0.2255]])
100 tensor([[0.0011, 0.9973, 0.0017]])
```

![CLIP 온도와 소프트맥스 확률](images/16/extra_clip_temperature.png)

- 온도 없이(× 1) 소프트맥스하면 세 확률이 거의 같다(0.33 근처). 초기값(× 14.3)이면 무당벌레 56%, 학습된 값(× 100)이면 99.7%
- 같은 유사도라도 곱하는 값에 따라 **확신의 정도**가 크게 달라진다. 14장 [온도 샘플링](14_nlp_with_rnns_and_attention.md)의 온도와 같은 역할이다(여기서는 온도로 **나누는** 대신 $1/\tau$를 곱한다)
- 파이프라인 확률을 해석할 때 주의: 99.7%는 "후보 3개 중에서 상대적으로" 그렇다는 뜻이고, 내부의 유사도 차이는 0.07에 불과하다

**보충 코드** — CLIP ViT-B/32의 크기 (config로만 생성)

```python
# 보충 코드 (트랜스포머스 5.23)
from transformers import CLIPConfig, CLIPModel
clip = CLIPModel(CLIPConfig())       # 기본 설정 = ViT-B/32 구조
n(clip), n(clip.vision_model), n(clip.text_model)          # 151277313, 87456000, 63165952
n(clip.visual_projection) + n(clip.text_projection)        # 655360 (768×512 + 512×512)
```

- 전체 약 1억 5천만 = 이미지 인코더(ViT-B/32, 8,700만) + 텍스트 인코더(6,300만) + 투영 층 두 개 + `logit_scale` 1개

### 2-6. 이미지와 텍스트를 따로 인코딩하기

> 🎯 **목적** — 이미지와 텍스트를 **별도로** 인코딩하는 방법을 익힌다. 이미지 검색처럼 한쪽을 미리 계산해 두는 응용에 필요하다.

**코드**

```python
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPModel
from PIL import Image
import urllib.request

model_id = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
clip_image_processor = CLIPImageProcessor.from_pretrained(model_id)
clip_model = CLIPModel.from_pretrained(model_id)

image = Image.open(urllib.request.urlopen(image_url)).convert("RGB")
image_inputs = clip_image_processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = clip_model.get_image_features(**image_inputs)  # [1, 512]
    image_features /= image_features.norm(dim=1, keepdim=True)

captions = [f"This is a photo of a {label}." for label in candidate_labels]
text_inputs = clip_tokenizer(captions, padding=True, return_tensors="pt")
with torch.no_grad():
    text_features = clip_model.get_text_features(**text_inputs)  # [3, 512]
    text_features /= text_features.norm(dim=1, keepdim=True)

similarities = image_features @ text_features.T
temperature = clip_model.logit_scale.detach().exp()
rescaled_similarities = similarities * temperature
probabilities = torch.nn.functional.softmax(rescaled_similarities , dim=1)
probabilities
# tensor([[0.0011, 0.9973, 0.0017]])
```

| 코드 | 의미 |
|---|---|
| `CLIPTokenizer` / `CLIPImageProcessor` | 프로세서를 쓰지 않고 토크나이저와 이미지 프로세서를 따로 |
| `get_image_features(**image_inputs)` | 이미지 인코더 + 투영만 실행 (텍스트 인코더는 돌지 않는다) |
| `get_text_features(**text_inputs)` | 텍스트 인코더 + 투영만 |
| `x /= x.norm(dim=1, keepdim=True)` | 이 함수들은 **정규화 전**의 투영 벡터를 돌려주므로 직접 정규화한다. `keepdim=True`로 `[1, 1]`·`[3, 1]`을 유지해 브로드캐스트 |

- 2-5절과 같은 확률이 나온다 — 계산이 같기 때문이다
- **이미지 검색 시스템**: 이미지 수백만 장의 임베딩을 미리 계산해 벡터 데이터베이스에 넣어 두고, 사용자가 문장을 입력하면 텍스트 임베딩 하나만 계산해 가장 가까운 이미지를 찾는다. 반대로 이미지로 비슷한 이미지나 캡션을 찾을 수도 있다

> ⚠️ 트랜스포머스 5.x에서는 `get_image_features()`·`get_text_features()`가 텐서가 아니라 `BaseModelOutputWithPooling` 객체를 돌려준다. 그래서 위 코드의 `image_features /= image_features.norm(...)`에서 `AttributeError`가 난다(보충 실험으로 확인). 투영된 임베딩은 **`.pooler_output`** 에 들어 있다: `clip_model.get_image_features(**image_inputs).pooler_output` (형태 `[1, 512]`). 이것을 정규화하면 `clip_model(...)`의 `image_embeds`와 같다(`allclose` → `True`).

> 📌 `clip_model(**inputs)`의 출력에는 `logits_per_image`(`[이미지 수, 텍스트 수]`)와 `logits_per_text`(전치)도 있다. `logits_per_image`는 `logit_scale.exp() * image_embeds @ text_embeds.T`, 즉 2-5절의 `rescaled_similarities`와 같다. 소프트맥스만 하면 확률이 된다.

### 2-7. DALL·E: 텍스트로 이미지 생성하기

> 🎯 **목적** — 텍스트 프롬프트로 이미지를 생성한 DALL·E의 원리를 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 개념을 정리한 것이다.

**DALL·E**(2021, OpenAI, 라메시 등) — 이미지를 **토큰**으로 바꿔, GPT처럼 텍스트 다음에 이미지 토큰을 이어 쓰게 했다.

```
단계 1 (이미지 토크나이저): 이산 VAE(dVAE)
   256×256 이미지 → 32×32 격자의 토큰 1,024개 (어휘 8,192개)  → 복원도 가능

단계 2 (생성 모델): 디코더 전용 트랜스포머 (120억 파라미터)
   [텍스트 BPE 토큰 최대 256개] [이미지 토큰 1,024개]
   → 15장의 GPT처럼 다음 토큰 예측으로 훈련 (이미지–캡션 2억 5천만 쌍)

생성: 텍스트 → 이미지 토큰 1,024개를 하나씩 샘플링 → dVAE 디코더로 이미지 복원
      여러 장 생성 → CLIP으로 프롬프트와 가장 잘 맞는 이미지를 골라 순위 매김
```

- VideoBERT(2-1절)가 비디오 특성을 k-평균으로 토큰화했듯이, DALL·E는 **학습되는 오토인코더**로 이미지를 토큰화했다. 이미지 픽셀(256 × 256 × 3 ≈ 20만 개)을 직접 생성하는 대신 토큰 1,024개만 생성하면 된다
- "아보카도 모양의 안락의자"처럼 세상에 없는 조합도 그럴듯하게 그려 큰 화제가 됐다

| 후속 모델 | 연도 | 방식 |
|---|---|---|
| DALL·E 2 | 2022 | **CLIP 임베딩 공간**을 거친다: 텍스트 → CLIP 텍스트 임베딩 → (사전 모델) → CLIP 이미지 임베딩 → **확산 모델** 디코더 → 이미지. 해상도와 품질이 크게 올랐다 |
| DALL·E 3 | 2023 | 훈련 이미지마다 **아주 자세한 합성 캡션**을 만들어 다시 훈련해 프롬프트를 훨씬 정확히 따른다. ChatGPT와 연결 |

> 📌 지금의 텍스트–이미지 생성 모델(DALL·E 2·3, 스테이블 디퓨전, Imagen 등)은 대부분 **확산 모델**이다. 확산 모델은 18장에서 다룬다.

### 2-8. Perceiver: 잠재 배열로 고해상도 입력 다루기

> 🎯 **목적** — 입력이 아무리 길어도 계산량이 폭발하지 않도록, 작은 잠재 배열이 입력을 크로스 어텐션으로 읽게 한 Perceiver의 구조를 이해한다.

> ⚠️ 노트북에는 Perceiver의 푸리에 위치 인코딩 코드만 있다(2-9절). 구조는 교재 절 제목에 맞춰 정리하고, 블록 하나는 2-10절에서 보충 코드로 만든다.

**개념·원리**

ViT는 패치로 잘라 토큰 수를 줄였지만, 이것은 **이미지 전용** 요령이다. 오디오(1초에 48,000개 샘플), 비디오, 3D 점구름처럼 입력이 아주 길고 모달리티마다 구조가 다르면 매번 다른 요령이 필요하다. 2021년 딥마인드의 **Perceiver**(제글 등)는 하나의 구조로 모든 모달리티를 다루려 했다.

```
입력 배열 (바이트 배열) M × C                잠재 배열 N × D (학습되는 파라미터, N ≪ M)
예: 224×224 픽셀 = 50,176 × (RGB 3 + 위치 26)    예: 512 × 1024
        │ 키·값                                    │ 쿼리
        └──────────────→ 크로스 어텐션 ←────────────┘     비용 O(M·N)
                               ▼
                    잠재 트랜스포머 (셀프 어텐션 × 여러 층)   비용 O(N²) — M과 무관
                               ▼
                    (크로스 어텐션 → 잠재 트랜스포머)를 몇 번 반복 (가중치 공유 가능)
                               ▼
                    잠재 배열 평균 → 분류 헤드
```

| 요소 | 설명 |
|---|---|
| **잠재 배열** | 입력과 무관한 **학습되는 벡터 N개**(예: 512개). DETR의 객체 쿼리(1-1절)처럼 "입력에서 무엇을 읽어 올지" 묻는 쿼리 역할 |
| **비대칭 크로스 어텐션** | 쿼리는 잠재 배열(N개), 키·값은 입력(M개) → 어텐션 행렬 $N \times M$. 입력 길이에 **선형**. 전체 셀프 어텐션이라면 $M \times M$ |
| **잠재 트랜스포머** | 이후의 깊은 처리는 작은 잠재 배열에서만 → 층을 깊게 쌓아도 $M$과 상관없다 |
| **반복 읽기** | 크로스 어텐션을 한 번만 하면 정보가 병목에 걸리므로, 중간중간 입력을 **다시 읽는다**. 순환 신경망처럼 가중치를 공유해 파라미터를 아낄 수 있다 |
| **모달리티 비종속** | 입력 구조에 대한 가정은 **위치 인코딩**에만 들어간다. 이미지면 2D 좌표, 비디오면 3D(시간 포함), 오디오면 1D. 그래서 위치 정보를 잘 표현하는 방법이 중요하다 → 2-9절 |

- ImageNet에서 합성곱 없이 **픽셀 50,176개를 그대로** 입력으로 받아 ResNet-50·ViT와 비슷한 정확도를 냈다. 같은 구조로 오디오 + 비디오(AudioSet), 점구름(ModelNet40)도 처리했다

> 💡 Perceiver는 입력 크기가 아무리 커도 잠재 배열 크기 $N$에 맞춰 **정보를 압축**한다. 이 "작은 학습되는 쿼리 집합이 큰 입력을 크로스 어텐션으로 요약한다"는 아이디어는 뒤의 Flamingo(Perceiver Resampler, 2-12절)와 BLIP-2(Q-Former, 2-13절)에서 **이미지 → 고정 개수의 토큰** 변환기로 다시 쓰인다.

### 2-9. 푸리에 위치 인코딩

> 🎯 **목적** — Perceiver가 쓰는 푸리에 특성 위치 인코딩을 구현하고, 2D 이미지 좌표에서 어떤 모양이 되는지 그려 본다.

**개념·원리**

각 위치의 좌표 $x_d$(이미지라면 $d$ = 2: $x$, $y$)를 $[-1, 1]$로 정규화한 뒤, 주파수 대역 $K$개로 사인·코사인 특성을 만든다.

$$
\text{PE}(x_d) = \left[\, x_d,\; \sin(\pi f_1 x_d),\, \cos(\pi f_1 x_d),\; \dots,\; \sin(\pi f_K x_d),\, \cos(\pi f_K x_d) \,\right]
$$

- 주파수 $f_k$는 1부터 $\mu/2$까지 **같은 간격**으로 $K$개. $\mu$는 그 축의 해상도(샘플 수)다. $\mu/2$는 $\mu$개의 샘플로 표현할 수 있는 가장 높은 주파수(나이퀴스트 주파수)다
- 차원마다 $2K + 1$개 → 전체 $d(2K + 1)$개. 15장 [7절의 사인·코사인 인코딩](15_transformers_for_nlp_and_chatbots.md#7-추가-자료-고정-위치-인코딩)을 좌표축마다 적용한 것과 비슷하다. 다만 주파수가 기하급수가 아니라 **선형** 간격이고, 원래 좌표 $x_d$도 함께 넣는다
- Perceiver는 이 인코딩을 입력 특성에 **더하지 않고 이어 붙인다** → 픽셀 하나 = RGB 3개 + 위치 26개 = 29차원

**코드**

```python
class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_bands, max_resolution):
        super().__init__()
        self.num_bands = num_bands  # number of frequency bands (K in the paper)
        self.max_resolution = max_resolution  # max coord value (μ in paper)
        frequencies = torch.linspace(1.0, max_resolution / 2, steps=num_bands)
        self.register_buffer("frequencies", frequencies)

    def forward(self, X):
        out = [X]  # X shape is [..., d] (e.g., d=2 for images, d=3 for videos)
        for freq in self.frequencies:
            angles = torch.pi * freq * X
            out += [angles.sin(), angles.cos()]
        return torch.cat(out, dim=-1)  # output shape: [..., d * (2K + 1)]
```

| 코드 | 의미 |
|---|---|
| `torch.linspace(1.0, max_resolution / 2, steps=num_bands)` | $f_k$ = 1 ~ $\mu/2$를 같은 간격으로 $K$개 |
| `self.register_buffer("frequencies", ...)` | 15장 7절과 같이 **버퍼**: 훈련되지 않지만 `state_dict`에 저장되고 `.to(device)`로 함께 옮겨진다 |
| `out = [X]` | 원래 좌표도 결과에 포함 |
| `angles = torch.pi * freq * X` | `X`의 마지막 축(좌표 $d$개)에 모두 같은 주파수 → `[..., d]` |
| `out += [angles.sin(), angles.cos()]` | 주파수마다 사인 $d$개, 코사인 $d$개 |
| `torch.cat(out, dim=-1)` | `[..., d + 2Kd]` = `[..., d(2K + 1)]` |

```python
H, W = 224, 224
num_bands = 6
coords_y = torch.linspace(-1, 1, H)
coords_x = torch.linspace(-1, 1, W)
grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
pos = torch.stack([grid_x, grid_y], dim=-1)  # shape [H, W, 2]

X = torch.rand(10, 224 * 224, 3)
fourier_pos_enc = FourierPositionalEncoding(num_bands=num_bands, max_resolution=H)
pos_encodings = fourier_pos_enc(pos)
```

| 코드 | 의미 |
|---|---|
| `torch.linspace(-1, 1, H)` | 픽셀 행 224개의 $y$ 좌표를 −1 ~ 1로 |
| `torch.meshgrid(..., indexing="ij")` | 모든 (행, 열) 조합의 좌표 격자. `"ij"`는 첫 축 = 행(y), 둘째 축 = 열(x) |
| `torch.stack([grid_x, grid_y], dim=-1)` | 픽셀마다 $(x, y)$ → `[224, 224, 2]` |
| `pos_encodings` | `[224, 224, 26]` (= 2 × (2 × 6 + 1)) |

**보충 코드** — 형태와 주파수

```python
# 보충 코드
pos_encodings.shape                       # torch.Size([224, 224, 26])
fourier_pos_enc.frequencies               # tensor([  1.0000,  23.2000,  45.4000,  67.6000,  89.8000, 112.0000])
sum(p.numel() for p in fourier_pos_enc.parameters())   # 0  (버퍼만 있다)
dx = 2 / (W - 1)                          # 이웃한 픽셀 사이의 좌표 간격 0.00897
fourier_pos_enc.frequencies * dx          # 픽셀 하나 이동할 때 위상 변화(π 단위)
# tensor([0.0090, 0.2081, 0.4072, 0.6063, 0.8054, 1.0045])
```

```python
titles = ["x", "y"]
for f in range(1, 6+1):
    titles += [f"$sin(πf_{f}x)$", f"$sin(πf_{f}y)$", f"$cos(πf_{f}x)$", f"$cos(πf_{f}y)$"]

plt.figure(figsize=(6, 8))
for dim in range(26):
    plt.subplot(8, 4, dim + 1 + (2 if dim > 1 else 0))
    plt.imshow(pos_encodings[..., dim])
    plt.title(titles[dim], fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

![푸리에 위치 인코딩의 26개 채널](images/16/nb_fourier_positional_encoding.png)

**결과 해석**

- 맨 위: 원래 좌표 $x$(왼쪽 → 오른쪽으로 증가), $y$(위 → 아래로 증가). 채널 순서는 `[x, y, sin(πf₁x), sin(πf₁y), cos(πf₁x), cos(πf₁y), …]`이다 (`angles`가 `[x, y]` 두 개를 한꺼번에 담기 때문). `subplot` 번호에 2를 더한 것은 첫 줄 오른쪽 두 칸을 비워 줄을 맞추기 위해서다
- $f_1 = 1$: 이미지 전체에 걸쳐 반 주기 → 아주 느린 변화. 이 채널만으로도 대략적인 위치를 안다
- $f_2 = 23.2$: 촘촘한 줄무늬. 좌표의 세밀한 차이를 구별한다 → 15장처럼 **여러 규모의 주파수**를 함께 써서 가까운 위치와 먼 위치를 모두 구별한다
- $f_3$ ~ $f_6$: 그림이 줄무늬가 아니라 얼룩이나 느린 띠처럼 보인다. 보충 코드처럼 이 주파수들은 픽셀 하나 사이에 위상이 $0.4\pi$ ~ $1.0\pi$나 바뀐다. 특히 $f_6 = 112 = \mu/2$는 위상 변화가 $1.0045\pi$로 나이퀴스트 한계($\pi$)를 살짝 넘어 **에일리어싱**이 일어난다(이웃 픽셀마다 부호가 바뀌면서 아주 느린 맥놀이가 생긴다). 그림을 작게 줄여 그린 탓에 생기는 모아레도 섞여 있다. 값 자체는 정확하지만, 이 해상도로는 이 주파수의 사인파를 제대로 **볼** 수 없다

> ⚠️ 노트북은 `X = torch.rand(10, 224 * 224, 3)`(픽셀 특성 10장 분량)을 만들고 다음 셀에서 사이킷런의 `flower.jpg`를 그리기만 한 뒤, 둘 다 쓰지 않고 BLIP-2로 넘어간다. Perceiver 예제가 아직 미완성으로 보인다. 다음 절의 보충 코드에서 `X`와 `pos_encodings`를 실제로 Perceiver 블록에 넣어 본다.

![사이킷런 샘플 이미지](images/16/nb_flower_sample.png)

### 2-10. 보충 코드: Perceiver 블록

> 🎯 **목적** — 푸리에 위치 인코딩을 이어 붙인 픽셀 입력을, 잠재 배열이 크로스 어텐션으로 읽는 Perceiver 블록 하나를 만들어 형태와 비용을 확인한다.

**보충 코드**

```python
# 보충 코드
pe_flat = pos_encodings.reshape(H * W, -1).expand(10, -1, -1)   # [10, 50176, 26]
inputs = torch.cat([X, pe_flat], dim=-1)                         # [10, 50176, 29]  RGB 3 + 위치 26

class PerceiverBlock(nn.Module):
    def __init__(self, num_latents=256, latent_dim=512, input_dim=29, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=1, kdim=input_dim,
                                                vdim=input_dim, batch_first=True)
        self.self_attn = nn.TransformerEncoderLayer(latent_dim, num_heads, 2048,
                                                    batch_first=True, norm_first=True)

    def forward(self, X):
        lat = self.latents.expand(X.size(0), -1, -1)                 # [B, 256, 512]
        lat = lat + self.cross_attn(lat, X, X, need_weights=False)[0]  # 쿼리 = 잠재, 키·값 = 입력
        return self.self_attn(lat)                                     # 잠재끼리 셀프 어텐션

block = PerceiverBlock().eval()
with torch.no_grad():
    out = block(inputs[:2])
out.shape                                          # torch.Size([2, 256, 512])
sum(p.numel() for p in block.parameters())         # 3839488
```

| 코드 | 의미 |
|---|---|
| `pos_encodings.reshape(H * W, -1)` | `[224, 224, 26]` → `[50176, 26]`. 픽셀을 한 줄로 펼친다 (Perceiver에게 입력은 그냥 긴 배열이다) |
| `.expand(10, -1, -1)` | 모든 이미지에 같은 위치 인코딩 (복사 없이) |
| `torch.cat([X, pe_flat], dim=-1)` | 픽셀 특성 뒤에 위치를 **이어 붙인다** → 29차원 |
| `nn.MultiheadAttention(512, 1, kdim=29, vdim=29)` | 쿼리(잠재)는 512차원, 키·값(입력)은 29차원. `kdim`·`vdim`으로 차원이 다른 키·값을 받는다. 원 논문처럼 크로스 어텐션은 헤드 1개 |
| `lat + self.cross_attn(...)[0]` | 잔차 연결 (원 논문은 앞뒤에 층 정규화와 MLP도 둔다 — 여기서는 생략) |
| `nn.TransformerEncoderLayer(..., norm_first=True)` | 잠재 배열의 셀프 어텐션 층 (1-5절의 Pre-LN) |

```python
# 보충 코드 — 어텐션 행렬 크기 비교 (이미지 한 장, 헤드 하나)
M = 224 * 224
M * M          # 2,517,630,976  전체 픽셀 셀프 어텐션 (float32로 약 10GB)
256 * M        # 12,845,056     잠재 256개의 크로스 어텐션 (196배 작다)
256 * 256      # 65,536         잠재 셀프 어텐션
```

**결과 해석**

- 픽셀 50,176개짜리 입력이 **256 × 512의 잠재 배열**로 요약됐다. 이후 층은 몇 개를 쌓든 256개 토큰만 처리한다
- 파라미터 수(3,839,488)에 입력 길이 $M$이 들어가지 않는다 → 같은 블록에 더 큰 이미지나 긴 오디오를 넣어도 된다 (계산량만 $M$에 선형으로 는다)
- 픽셀끼리 셀프 어텐션을 하면 어텐션 행렬 하나가 약 25억 원소(헤드 하나만 float32로 약 10GB)다. ViT가 패치로 자르고, Swin이 윈도로 나누고, Perceiver가 잠재 배열을 쓰는 것은 모두 이 **제곱 비용**을 피하는 서로 다른 방법이다

### 2-11. Perceiver IO

> 🎯 **목적** — 출력 쪽에도 쿼리를 둬서 분류뿐 아니라 임의 형태의 출력을 내게 한 Perceiver IO를 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 개념을 정리한 것이다.

Perceiver는 잠재 배열을 평균 내 **분류**만 할 수 있었다. 2021년의 **Perceiver IO**는 출력에도 크로스 어텐션을 하나 더 둔다.

```
입력 (M개) ──크로스 어텐션──→ 잠재 (N개) ──셀프 어텐션 × L──→ 잠재 (N개)
     인코딩                         처리                         │ 키·값
                                                                 ▼
출력 쿼리 (O개) ─────────────────────── 크로스 어텐션 (쿼리 = 출력 쿼리) ──→ 출력 (O개)
     디코딩
```

| 출력 쿼리 | 작업 |
|---|---|
| 학습되는 벡터 1개 | 분류 (Perceiver와 같음) |
| 입력 위치마다 위치 인코딩 | 픽셀마다 출력: **광학 흐름**(두 프레임 사이 픽셀의 이동), 분할 |
| 토큰 위치마다 위치 인코딩 | 위치마다 출력: **마스크드 언어 모델** — 토크나이저 없이 **UTF-8 바이트**를 그대로 입력·출력해 BERT 수준의 성능 |
| 작업 임베딩 + 위치 | 여러 작업, 여러 모달리티의 출력(오디오 + 비디오 + 레이블 동시 복원) |

- 입력 길이 $M$에도, 출력 길이 $O$에도 비용이 **선형**이고 깊은 처리는 크기 $N$의 잠재 공간에서만 한다
- "출력 쿼리로 원하는 출력의 모양을 지정한다"는 점에서 DETR(1-1절)의 객체 쿼리와 같은 발상이다

### 2-12. Flamingo: 열린 시각 대화

> 🎯 **목적** — 동결된 대형 언어 모델에 이미지를 보여 주는 방법을 처음으로 크게 성공시킨 Flamingo의 구조를 정리한다.

> ⚠️ 노트북에는 이 소절의 코드가 없다. 아래는 교재 절 제목에 맞춰 개념을 정리한 것이다.

2022년 딥마인드의 **Flamingo**(알레이락 등)는 이미지와 텍스트가 **섞인** 입력(예: "<이미지 1> 이건 고양이다. <이미지 2> 이건 강아지다. <이미지 3> 이건")을 받아 텍스트를 생성한다.

```
이미지 → 동결된 비전 인코더 (NFNet, CLIP 방식으로 사전 훈련)
       → Perceiver Resampler: 학습되는 잠재 쿼리 64개가 이미지 특성을 크로스 어텐션으로 요약
       → 이미지마다 시각 토큰 64개 (이미지 크기·프레임 수와 무관)

동결된 LLM (Chinchilla 700억) 의 층 사이사이에 새 층을 끼운다:
   [게이트 크로스 어텐션 층] ← 쿼리 = 텍스트, 키·값 = 시각 토큰
   [동결된 LM 층]
   [게이트 크로스 어텐션 층]
   [동결된 LM 층] ...
   게이트: 출력 × tanh(α), α는 0으로 초기화
```

| 설계 | 이유 |
|---|---|
| **LLM과 비전 인코더 동결** | 이미 가진 언어 능력·시각 능력을 잃지 않고, 새 층만 훈련하므로 효율적이다 |
| **Perceiver Resampler** | 2-8절의 잠재 배열 아이디어. 이미지(또는 비디오)를 **고정 개수의 토큰**으로 → LLM 쪽 계산이 일정 |
| **tanh 게이트, α = 0 초기화** | 훈련 시작 시점에 새 층의 출력이 0 → 모델이 원래 LLM과 **완전히 같게** 동작한다. 훈련하면서 시각 정보를 조금씩 섞는다. 무작위 층을 끼워 넣어 LLM이 망가지는 것을 막는다 |
| 이미지별 마스킹 | 텍스트 토큰은 바로 앞에 나온 이미지의 시각 토큰만 본다 → 이미지 여러 장이 섞인 대화를 처리 |
| 훈련 데이터 | 이미지·텍스트가 섞인 웹 페이지 4,300만 개 + 이미지–텍스트 쌍 + 비디오–텍스트 쌍 |

- 결과: 15장 [4-4절의 퓨샷 프롬프트](15_transformers_for_nlp_and_chatbots.md#4-4-퓨샷-프롬프트)처럼 **예시 몇 개(이미지 + 답)를 보여 주는 것만으로** 시각 질의응답·캡션 생성 같은 여러 벤치마크에서 각 작업에 미세 튜닝된 모델과 경쟁했다. 이미지에 대해 자유롭게 대화할 수도 있다(**열린 시각 대화**)
- Flamingo 이후 "**동결된 비전 인코더 + 작은 연결부 + 동결된(또는 가볍게 튜닝한) LLM**"이 멀티모달 LLM의 표준 설계가 됐다

### 2-13. BLIP과 BLIP-2

> 🎯 **목적** — 잡음 많은 웹 데이터를 스스로 정제한 BLIP과, 작은 Q-Former로 동결된 이미지 인코더와 LLM을 연결한 BLIP-2를 정리한다.

> ⚠️ 노트북에는 사전 훈련된 BLIP-2를 쓰는 코드만 있다(2-14절). 아래 원리는 교재 절 제목에 맞춰 정리한 것이다.

**BLIP**(Bootstrapping Language-Image Pre-training, 2022, 세일즈포스, 리 등)

| 아이디어 | 내용 |
|---|---|
| **인코더–디코더 혼합 모델**(MED) | 하나의 모델이 세 가지로 동작: ① 이미지·텍스트를 따로 인코딩해 대조 학습(**ITC**, CLIP과 같음) ② 크로스 어텐션으로 섞어 "짝이 맞나" 이진 분류(**ITM**) ③ 이미지를 보고 캡션을 생성(**LM**) |
| **CapFilt** (부트스트래핑) | 웹 이미지의 alt 텍스트는 잡음이 많다. 깨끗한 데이터로 미세 튜닝한 **캡셔너**가 웹 이미지에 새 캡션을 만들고, **필터**(ITM)가 이미지와 맞지 않는 캡션(원래 것이든 생성한 것이든)을 버린다 → 정제된 데이터로 다시 사전 훈련 |

**BLIP-2**(2023) — 이미지 인코더와 LLM을 **모두 동결**하고, 둘 사이를 잇는 가벼운 **Q-Former**(Querying Transformer, 약 1억 8,800만 파라미터)만 훈련한다.

```
이미지 → 동결된 ViT-g (EVA-CLIP, 14×14 패치 → 토큰 257개 × 1408차원)
                     │ 키·값 (크로스 어텐션, 두 블록마다)
학습되는 쿼리 32개 → Q-Former (BERT-base 크기, 12층, 768차원) → 쿼리 출력 32 × 768
                     → 선형 층 (768 → LLM 임베딩 차원, OPT-2.7B면 2560)
                     → 32개의 "소프트 시각 토큰"
동결된 LLM (OPT 또는 Flan-T5) ← [시각 토큰 32개] + [텍스트 프롬프트] → 텍스트 생성
```

| 단계 | 훈련 방법 | 목적 |
|---|---|---|
| 1. 표현 학습 | 동결된 이미지 인코더 + Q-Former. BLIP의 세 과제(ITC, ITM, 이미지 기반 텍스트 생성)를 **어텐션 마스크만 바꿔** 함께 학습 | 쿼리 32개가 이미지에서 **텍스트와 관련 있는 정보**만 뽑도록 |
| 2. 생성 학습 | Q-Former 출력 → 선형 층 → 동결된 LLM 입력 앞에 붙여 다음 토큰 예측 | 쿼리 출력을 LLM이 이해하는 "언어"로 맞추기 |

- 훈련하는 파라미터가 Flamingo보다 수십 배 적은데도, 제로샷 시각 질의응답에서 Flamingo(800억)를 앞섰다
- 쿼리 32개는 Flamingo의 Perceiver Resampler(64개), Perceiver의 잠재 배열과 같은 **"학습되는 쿼리 + 크로스 어텐션으로 요약"** 이다 (2-8절의 💡)

**보충 코드** — BLIP-2 기본 설정 (config로만 확인)

```python
# 보충 코드 (트랜스포머스 5.23)
from transformers import Blip2Config
cfg = Blip2Config()
cfg.num_query_tokens                                                    # 32
cfg.qformer_config.hidden_size, cfg.qformer_config.num_hidden_layers    # 768, 12
cfg.qformer_config.cross_attention_frequency                            # 2  (두 층마다 크로스 어텐션)
cfg.qformer_config.encoder_hidden_size                                  # 1408 (이미지 특성 차원)
v = cfg.vision_config
v.hidden_size, v.num_hidden_layers, v.image_size, v.patch_size          # 1408, 39, 224, 14
```

- 이미지 인코더 ViT-g: 224 / 14 = 16 → 패치 256개 + `[CLS]` = 257개 토큰. Q-Former는 이 257개를 32개로 줄인다
- `Salesforce/blip2-opt-2.7b`의 허브 설정도 `num_query_tokens` = 32, LLM은 OPT(은닉 차원 2,560)다

### 2-14. BLIP-2로 이미지 캡션 생성하기

> 🎯 **목적** — 사전 훈련된 BLIP-2(OPT-2.7B)로 고양이 사진의 캡션을 생성한다.

**코드**

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

model_id = "Salesforce/blip2-opt-2.7b"
blip2_processor = Blip2Processor.from_pretrained(model_id)
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    model_id, device_map=device, dtype=torch.float16)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats
image = Image.open(urllib.request.urlopen(image_url))
inputs = blip2_processor(images=image, return_tensors="pt")
inputs = inputs.to(device, dtype=torch.float16)
with torch.no_grad():
    generated_ids = blip2_model.generate(**inputs)

generated_text = blip2_processor.batch_decode(generated_ids)
generated_text
```

```
['<image><image><image> ... <image></s>two cats laying on a couch\n']   ← <image> 32개
```

| 코드 | 의미 |
|---|---|
| `Blip2Processor` | 이미지 프로세서 + OPT 토크나이저 (CLIP의 프로세서와 같은 구성) |
| `Blip2ForConditionalGeneration` | ViT-g + Q-Former + 선형 투영 + OPT-2.7B 전체, 약 38억 파라미터. 가중치 파일이 10.0GB + 4.98GB(float32) |
| `dtype=torch.float16` | 반정밀도로 올려 GPU 메모리를 절반(약 7.5GB)으로. 혼합 정밀도·양자화는 부록 B |
| `device_map=device` | 모델 전체를 그 장치에 |
| `blip2_processor(images=image)` | 텍스트 없이 이미지만 → **프롬프트 없는 캡션 생성** |
| `inputs.to(device, dtype=torch.float16)` | `BatchFeature.to()`는 **실수 텐서**(`pixel_values`)만 float16으로 바꾸고, 정수 텐서(`input_ids`)는 그대로 둔다 |
| `blip2_model.generate(**inputs)` | 15장의 `generate()`와 같다. 기본 설정은 탐욕적 디코딩 |
| `batch_decode(generated_ids)` | 생성된 토큰 ID → 문자열 |

**결과 해석**

- 출력 앞의 **`<image>` 32개**는 2-13절의 쿼리 32개가 들어간 자리다. 프로세서가 입력 토큰 앞에 이미지 자리 표시 토큰을 `num_query_tokens`(32)개 넣고, 모델은 그 자리의 임베딩을 Q-Former 출력(투영된 시각 토큰)으로 바꿔 LLM에 넣는다. 그다음 `</s>`는 OPT의 시작 토큰(BOS)이다
- 생성된 부분은 "two cats laying on a couch" — 사진(1-13절)을 정확히 설명한다. LLM(OPT)은 동결된 채이고, 이미지를 "읽을" 수 있게 된 것은 Q-Former 덕분이다

```python
generated_text = blip2_processor.batch_decode(generated_ids,
                                              skip_special_tokens=True)
generated_text
# ['two cats laying on a couch\n']
```

- `skip_special_tokens=True`로 `<image>`·`</s>` 같은 특수 토큰을 빼면 캡션만 남는다 (15장과 같은 옵션)
- 텍스트 프롬프트를 함께 주면 질의응답도 된다: `blip2_processor(images=image, text="Question: how many cats are there? Answer:", ...)` (BLIP-2 모델 카드의 형식)

---

## 3. 기타 멀티모달 모델 (16.3)

> 🎯 **목적** — 이 장에서 다룬 것 외의 대표적인 멀티모달 모델을 정리하고, 상용 멀티모달 모델을 API로 써 본다.

<!-- 일타 -->
> 💬 **이제는 모든 감각을 한 모델에**
>
> 현재의 멀티모달 모델(GPT-4o, Gemini 등)은 텍스트·이미지·음성을 한 모델이 함께 다룬다. 원리는 이 장에서 본 것의 조합이다. 각 모달리티를 토큰(벡터)으로 바꾸고, 하나의 트랜스포머가 그 토큰들을 함께 본다.


### 3-1. 대표적인 멀티모달 모델

> 🎯 **목적** — 이미지·텍스트·음성 등 여러 입력을 함께 다루는 대표 모델들을 정리한다.

> ⚠️ 노트북에는 Gemini API 코드만 있다(3-2절). 아래 표는 교재 절 제목에 맞춰 대표적인 모델을 정리한 것이다.

| 모델 | 연도·기관 | 모달리티 | 핵심 |
|---|---|---|---|
| **LLaVA** | 2023, 위스콘신대·MS | 이미지 → 텍스트 | CLIP ViT-L/14 + **선형 층(이후 MLP) 하나** + 공개 LLM(Vicuna). GPT-4로 만든 이미지 대화 데이터로 **시각 지시 튜닝** → 단순한 연결부로도 BLIP-2·Flamingo 계열과 비슷한 대화 능력. 오픈 멀티모달 LLM의 기본형 |
| **PaliGemma** | 2024, 구글 | 이미지 → 텍스트 | SigLIP(CLIP의 손실을 시그모이드로 바꾼 이미지 인코더) + Gemma LLM. 미세 튜닝하기 쉬운 작은 모델 |
| **GPT-4V / GPT-4o** | 2023 / 2024, OpenAI | 텍스트·이미지·오디오 | GPT-4o는 텍스트·이미지·오디오를 **하나의 모델**로 입력·출력하는 상용 모델 |
| **Gemini** | 2023 ~, 구글 딥마인드 | 텍스트·이미지·오디오·비디오·코드 | 처음부터 여러 모달리티를 함께 훈련한 **네이티브 멀티모달** 모델. 아주 긴 문맥(긴 동영상, 문서 수백 쪽) → 3-2절 |
| **Claude**(3 이후) | 2024 ~, Anthropic | 텍스트·이미지 → 텍스트 | 이미지·문서·차트를 읽고 추론하는 상용 모델 |
| **Whisper** | 2022, OpenAI | 오디오 → 텍스트 | 인코더–디코더 트랜스포머(15장 6절). 68만 시간의 다국어 음성으로 훈련한 음성 인식·번역 |
| **ImageBind** | 2023, 메타 | 6가지(이미지·텍스트·오디오·깊이·열화상·움직임) | **이미지를 중심으로** 각 모달리티를 CLIP처럼 대조 학습 → 직접 짝지어 훈련한 적 없는 모달리티끼리(예: 오디오 ↔ 텍스트)도 같은 공간에서 비교 |
| **RT-2** | 2023, 구글 딥마인드 | 이미지·텍스트 → 로봇 행동 | 비전–언어 모델이 로봇 팔의 **행동을 텍스트 토큰처럼** 출력하도록 미세 튜닝(비전–언어–행동, VLA 모델) |

> 💡 모달리티를 다루는 방법이 점점 하나로 모이고 있다. **무엇이든 토큰(또는 임베딩 시퀀스)으로 바꾼 뒤 트랜스포머에 넣는다.** 이미지는 패치(ViT)나 이산 토큰(DALL·E), 오디오는 스펙트로그램 패치나 오디오 코덱 토큰, 로봇 행동은 이산화한 숫자 토큰이 된다.

### 3-2. Gemini API로 사진에 대해 질문하기

> 🎯 **목적** — 구글의 상용 멀티모달 모델 Gemini에 사진을 업로드하고 질문한다.

**코드** — API 키 불러오기

```python
if IS_COLAB:
    from google.colab import userdata

    gemini_api_key = userdata.get('gemini_api_key')
else:
    # if you are not running this notebook in Colab, then you can
    # save your access token in a file named gemini_api_key in this
    # notebook's directory (keep it safe!), and load the secret using this code:
    gemini_api_key = open("gemini_api_key.secret").read().strip()

    # Alternatively, you can simply hard-code your access token like this:
    # gemini_api_key = "..."  # WARNING: hard-coding secrets is very insecure
```

- 코랩에서는 왼쪽의 🔑(보안 비밀) 메뉴에 `gemini_api_key`라는 이름으로 키를 저장해 두고 `userdata.get()`으로 읽는다. 15장 [Mistral의 액세스 토큰](15_transformers_for_nlp_and_chatbots.md#4-5-더-큰-모델-mistral-7b)과 같은 방식이다
- ⚠️ 주석에는 파일 이름이 `gemini_api_key`라고 돼 있지만 코드는 **`gemini_api_key.secret`** 파일을 연다. 코드 쪽 이름으로 만든다. 키를 코드에 직접 적으면 노트북을 공유할 때 키가 유출된다

**코드** — 사진 업로드와 질문

```python
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # two cats
image = Image.open(urllib.request.urlopen(image_url))
image.save("my_cats_photo.jpg")

from google import genai

gemini_client = genai.Client(api_key=gemini_api_key)
cats_photo = gemini_client.files.upload(file="my_cats_photo.jpg")
question = "What animal and how many? Format: [animal, number]"
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",  # or "gemini-2.5-pro"
    contents=[cats_photo, question])
print(response.text)
# [cat, 2]
```

| 코드 | 의미 |
|---|---|
| `from google import genai` | 구글 생성형 AI SDK(`google-genai` 패키지). 코랩에는 설치돼 있고, 다른 환경은 `pip install google-genai` |
| `genai.Client(api_key=...)` | API 클라이언트 |
| `gemini_client.files.upload(file=...)` | 이미지를 구글 서버(Files API)에 올리고 파일 참조를 받는다 |
| `generate_content(model, contents=[이미지, 질문])` | 이미지와 텍스트를 **한 리스트에 섞어** 보낸다. 순서가 곧 프롬프트의 순서 |
| `"gemini-2.5-flash"` | 빠르고 저렴한 모델. 더 어려운 추론에는 `"gemini-2.5-pro"` |
| `response.text` | 생성된 답 |

**결과 해석**

- 고양이 2마리를 정확히 세고, 요청한 `[animal, number]` **형식**까지 지켰다. BLIP-2(2-14절)는 캡션만 만들었지만, 지시를 따르도록 정렬된(15장 5절) 대형 멀티모달 모델은 이미지에 대한 자유로운 질문과 출력 형식 지시를 함께 처리한다
- 모델 가중치를 내려받지 않고 API로 호출하므로 GPU가 필요 없다. 대신 **사진이 외부 서버로 전송**되고, 사용량에 따라 비용·요청 한도가 있다. 민감한 이미지라면 BLIP-2·LLaVA 같은 로컬 모델이 나을 수 있다

---

## 4. 연습문제

노트북의 연습문제 풀이 부분은 "Work in progress"(준비 중)로 비어 있다. 풀이가 공개되면 이 절에 추가한다. 대신 이 장의 핵심을 스스로 점검할 수 있도록 **직접 만든 확인 문제**를 정리해 둔다(교재 연습문제가 아니다).

| 번호 | 질문 | 답 요약 | 관련 절 |
|---|---|---|---|
| 1 | CNN에 있고 ViT에 없는 귀납적 편향 두 가지. 그 결과 ViT는 언제 유리하고 언제 불리한가 | 지역성과 이동 등변성. 데이터가 적으면 CNN이 유리하고, 대규모 데이터로 사전 훈련하면 ViT가 같거나 더 낫다 | 1, 1-2 |
| 2 | 224×224 이미지를 16×16 패치로 자르면 토큰은 몇 개인가. 패치를 8×8로 바꾸면 어텐션 비용은 어떻게 되나 | 196개(+ `[CLS]` = 197). 8×8이면 784개 → 토큰 수 4배, 어텐션 행렬 약 16배 | 1-2, 1-13 |
| 3 | 패치 임베딩을 `Conv2d(kernel_size=16, stride=16)`로 구현할 수 있는 이유 | 커널이 겹치지 않으므로 출력 채널마다 "패치 768개 값과 필터의 점곱 + 편향" = 펼친 패치에 선형 층을 적용한 것과 같다 | 1-3, 1-4 |
| 4 | ViT에서 `[CLS]` 토큰의 역할. 분류에 나머지 패치 출력을 쓰지 않아도 되는 이유 | 학습되는 빈 토큰이 인코더의 셀프 어텐션으로 모든 패치의 정보를 모아 이미지 요약이 된다 (평균 풀링을 대신 써도 된다) | 1-5 |
| 5 | 224로 훈련한 ViT에 384×384 이미지를 넣으면 무엇이 문제이고 어떻게 해결하나 | 패치 수(576)가 위치 임베딩 개수(196)와 맞지 않는다. 위치 임베딩을 2D 격자로 보고 보간한다 (`interpolate_pos_encoding=True`) | 1-6 |
| 6 | 사전 훈련 ViT를 미세 튜닝할 때 모델별 이미지 프로세서를 써야 하는 이유 | 사전 훈련 때의 크기 조정·자르기·정규화(평균·표준 편차)와 같아야 한다. ViT-in21k는 평균 0.5, DeiT·DINO는 ImageNet 평균을 쓴다 | 1-7, 1-10 |
| 7 | `Trainer`에 `remove_unused_columns=False`가 필요했던 이유 | 모델 `forward()`의 인수에 없는 `image` 열을 `Trainer`가 지우면 콜레이트 함수가 이미지를 받지 못한다 | 1-8 |
| 8 | DeiT의 증류 토큰은 무엇을 배우며, 교사가 CNN인 이유는 | CNN 교사가 예측한 클래스를 맞히도록 훈련된다. CNN의 귀납적 편향을 학생 ViT에 간접적으로 전해 적은 데이터로도 잘 훈련된다 | 1-9 |
| 9 | Swin의 윈도 어텐션이 이미지 크기에 선형인 이유와, 이동 윈도가 필요한 이유 | 고정 크기 윈도($M^2$ 토큰) 안에서만 어텐션하므로 비용이 $M^2hw$. 윈도끼리 정보를 교환하려면 다음 층에서 윈도를 비켜 나눠야 한다 | 1-11 |
| 10 | DINO에서 붕괴(모든 입력에 같은 출력)를 막는 장치와 교사를 만드는 방법 | 교사 출력의 센터링 + 낮은 온도로 샤프닝. 교사는 학생 가중치의 지수 이동 평균 | 1-12 |
| 11 | CLIP으로 제로샷 분류하는 절차 | 레이블마다 "This is a photo of a {label}." 캡션 → 텍스트 임베딩, 이미지 임베딩과 코사인 유사도 → × exp(`logit_scale`) → 소프트맥스 | 2-2, 2-4, 2-5 |
| 12 | CLIP의 코사인 유사도가 0.23 / 0.30 / 0.24인데 확률이 0.1% / 99.7% / 0.2%가 되는 이유 | 학습된 온도 100을 곱해 소프트맥스하기 때문. 확률은 후보끼리의 상대 비교라 후보에 정답이 없어도 무언가를 고른다 | 2-5 |
| 13 | CLIP 대조 손실에서 배치 크기가 중요한 이유 | 배치의 나머지 $N-1$개가 오답 후보다. 클수록 어려운 분류 문제가 되어 더 세밀한 표현을 배운다 (무작위일 때 손실 ≈ $\ln N$) | 2-2, 2-3 |
| 14 | Perceiver가 픽셀 5만 개를 처리할 수 있는 이유와 위치 정보를 넣는 방법 | 작은 잠재 배열($N$개)이 입력을 크로스 어텐션으로 읽어 비용이 $O(MN)$. 깊은 처리는 잠재 배열에서만. 위치는 푸리에 특성을 입력에 이어 붙인다 | 2-8 ~ 2-10 |
| 15 | Perceiver Resampler·Q-Former·Perceiver 잠재 배열의 공통점 | 학습되는 쿼리 집합이 크로스 어텐션으로 큰 입력을 고정 개수의 토큰으로 요약한다 | 2-8, 2-12, 2-13 |
| 16 | Flamingo의 게이트 크로스 어텐션을 tanh(0)으로 초기화하는 이유 | 훈련 시작 때 새 층의 출력이 0이라 동결된 LLM이 원래대로 동작한다. 언어 능력을 해치지 않고 시각 정보를 서서히 섞는다 | 2-12 |
| 17 | BLIP-2의 디코딩 결과 앞에 `<image>`가 32개 붙은 이유 | Q-Former의 쿼리 32개가 LLM 입력의 이미지 자리 표시 토큰 32개를 대신하기 때문. `skip_special_tokens=True`로 지운다 | 2-13, 2-14 |

---

## 5. 정리

### 5-1. 전체 흐름

```
[16.1 비전 트랜스포머]  이미지 = 패치 토큰의 시퀀스
배경   : 시각 어텐션 RNN(캡션) → DETR(CNN + 트랜스포머, 객체 쿼리, 헝가리안 매칭, NMS 없음)
패치   : Conv2d(3, 768, kernel=16, stride=16) → flatten(2) → transpose  [B, 196, 768]
         = 패치 자르기(unfold) + Linear(768, 768)  (allclose True), 590,592 파라미터
ViT    : [CLS] + 패치 196 → + pos_embed [1, 197, 768] → TransformerEncoder × 12 (GELU)
         → LayerNorm([CLS]) → Linear(768, 1000)            86,567,656 파라미터 (인코더 98%)
         ⚠️ nn.TransformerEncoderLayer 기본은 Post-LN, 원본 ViT는 Pre-LN (norm_first=True)
         입력 크기 고정(197 ≠ 257) → interpolate_pos_encoding=True
미세 튜닝 : vit-base-patch16-224-in21k + Oxford-IIIT Pet(37품종, 3,680 / 3,669)
         AutoImageProcessor(224×224, 평균·표준 편차 0.5) + collate_fn(do_convert_rgb)
         Trainer(배치 16, 3 에포크 = 690 스텝, remove_unused_columns=False) → 91.44% (손실 1.15)
DeiT   : [CLS] + [DIST] + 패치, CNN 교사 하드 증류, 강한 증강 → ImageNet-1k만으로
         deit-base-distilled-patch16-224 같은 설정 → 94.41% (손실 0.20)
PVT·Swin : 피라미드(1/4 ~ 1/32) + 공간 축소 어텐션 / 윈도(7×7) + 이동 윈도 + 패치 병합
         W-MSA 비용 4hwC² + 2M²hwC → 56×56에서 14배, 224×224에서 209배 절약
DINO   : 학생·교사(EMA) 자기 증류, 다중 크롭, 센터링 + 샤프닝, 레이블 없음
         dino-vitb8: 28×28 = 784 패치, attentions[-1] [1, 12, 785, 785]
         [0, :, 1:, 0] → [12, 784] → 28×28 → 원본 크기로 보간 → 헤드별 물체·배경 분할
         ⚠️ 트랜스포머스 5.x: attn_implementation="eager" 필요

[16.2 멀티모달 트랜스포머]  모달리티를 어디서 만나게 하는가
VideoBERT : 비디오 특성 k-평균 → 시각 토큰, 텍스트와 이어 붙여 MLM (단일 스트림)
ViLBERT   : 텍스트·이미지 영역 두 스트림 + 공동 어텐션 (이중 스트림)
CLIP      : 이중 인코더, 4억 쌍, 대칭 교차 엔트로피, 배치 32,768, 온도 학습(0.07 → 상한 100)
            pipeline("zero-shot-image-classification") → 무당벌레 99.73%, 민들레 68.3%
            text_embeds [3, 512]·image_embeds [1, 512] (정규화) → 코사인 0.23 / 0.30 / 0.24
            × logit_scale.exp() = 100 → softmax → 0.0011 / 0.9973 / 0.0017
            get_image_features·get_text_features 따로 (⚠️ 5.x는 .pooler_output)
DALL·E    : dVAE(32×32 토큰, 어휘 8,192) + 120억 디코더 + CLIP 순위 → DALL·E 2·3은 확산
Perceiver : 잠재 배열(N) ← 크로스 어텐션 ← 입력(M), 비용 O(MN) + O(N²)
            푸리에 위치 인코딩 f = linspace(1, μ/2, K), [x, sin, cos ...] → [224, 224, 26]
            입력에 이어 붙임 → [10, 50176, 29] → 잠재 [B, 256, 512] (보충 코드)
Perceiver IO : 출력 쿼리 크로스 어텐션 → 임의 형태 출력 (광학 흐름, 바이트 MLM)
Flamingo  : 동결 비전 인코더 → Perceiver Resampler(64 토큰) → 동결 LLM + tanh 게이트 크로스 어텐션
BLIP-2    : 동결 ViT-g(257 × 1408) → Q-Former(쿼리 32 × 768) → 선형 → 동결 OPT-2.7B
            float16 generate → "<image>×32</s>two cats laying on a couch" → skip_special_tokens

[16.3 기타 멀티모달 모델]  LLaVA, PaliGemma, GPT-4o, Gemini, Claude, Whisper, ImageBind, RT-2
Gemini API : genai.Client → files.upload(사진) → generate_content([사진, 질문]) → "[cat, 2]"
```

### 5-2. 개념과 코드 대응

| 단계 | 핵심 코드 | 핵심 원칙 |
|---|---|---|
| 패치 임베딩 | `nn.Conv2d(C, E, kernel_size=P, stride=P)` → `.flatten(start_dim=2).transpose(1, 2)` | 겹치지 않는 합성곱 = 패치별 선형 층 |
| `[CLS]` 토큰 | `nn.Parameter(torch.randn(1, 1, E) * 0.02).expand(B, -1, -1)` → `torch.cat(..., dim=1)` | 이미지 요약 토큰, 복사 없이 확장 |
| 위치 임베딩 | `nn.Parameter(torch.randn(1, 1 + L, E) * 0.02)`, `Z + pos_embed` | 입력 크기를 고정한다 → 보간 |
| 인코더 | `nn.TransformerEncoder(nn.TransformerEncoderLayer(E, h, 4E, activation="gelu", batch_first=True), 12)` | 원본 ViT는 `norm_first=True` |
| 분류 | `layer_norm(Z[:, 0])` → `Linear(E, C)` | `[CLS]` 출력만 |
| 사전 훈련 비전 모델 | `ViTForImageClassification.from_pretrained(id, num_labels=37)` | 헤드만 새로 초기화 |
| 이미지 전처리 | `AutoImageProcessor.from_pretrained(id, use_fast=True)`, `processor(images, return_tensors="pt", do_convert_rgb=True)` | 모델마다 정규화가 다르다 |
| 이미지 배치 | `collate_fn`에서 전처리 + `inputs["labels"] = torch.tensor(labels)` | 배치를 만들 때 변환 |
| 미세 튜닝 | `Trainer(model, TrainingArguments(..., eval_strategy="epoch", remove_unused_columns=False, report_to="none"), data_collator, compute_metrics)` | 쓰지 않는 열을 지우지 않게 |
| 정확도 지표 | `torch.tensor(logits).argmax(dim=1) == torch.tensor(labels)` | `Trainer`는 넘파이 배열을 넘긴다 |
| 어텐션 가중치 | `AutoModel.from_pretrained(id, output_attentions=True, attn_implementation="eager")`, `output.attentions[-1][0, :, 1:, 0]` | `sdpa`는 가중치를 돌려주지 않는다 |
| 어텐션 맵 시각화 | `.reshape(28, 28)` → `TF.resize(..., image.size[::-1], BILINEAR)` → `imshow(alpha=0.5)` | PIL `size`는 (너비, 높이) |
| 제로샷 분류 | `pipeline("zero-shot-image-classification", model=id)(img, candidate_labels=..., hypothesis_template=...)` | 후보 목록이 곧 분류기 |
| CLIP 임베딩 | `CLIPModel(...)(**processor(text=..., images=..., padding=True))` → `image_embeds`, `text_embeds` | 정규화돼 있다 → 점곱 = 코사인 |
| CLIP 확률 | `(img @ txt.T * model.logit_scale.exp()).softmax(dim=1)` = `logits_per_image.softmax(1)` | 온도가 확신을 정한다 |
| 따로 인코딩 | `get_image_features(...)`, `get_text_features(...)` → `x / x.norm(dim=1, keepdim=True)` | 5.x는 `.pooler_output` |
| 대조 손실 | `(F.cross_entropy(L, arange(N)) + F.cross_entropy(L.T, arange(N))) / 2` | 대각선이 정답, 배치가 오답 후보 |
| 푸리에 위치 | `linspace(1, μ/2, K)`, `cat([X, sin(πfX), cos(πfX), ...], -1)`, `register_buffer` | 이어 붙인다, 파라미터 없음 |
| 좌표 격자 | `torch.meshgrid(ys, xs, indexing="ij")` → `torch.stack([gx, gy], -1)` | `[H, W, 2]` |
| 잠재 크로스 어텐션 | `nn.MultiheadAttention(D, 1, kdim=C, vdim=C, batch_first=True)(latents, X, X)` | 비용이 입력 길이에 선형 |
| 이미지 → 텍스트 생성 | `Blip2ForConditionalGeneration.from_pretrained(id, dtype=torch.float16)`, `generate(**inputs)`, `batch_decode(..., skip_special_tokens=True)` | 동결된 LLM + Q-Former |
| 상용 멀티모달 API | `genai.Client(api_key).files.upload(...)`, `models.generate_content(model, contents=[img, q])` | 키는 코드에 적지 않는다 |

### 5-3. 방법 비교

**CNN vs 비전 트랜스포머**

| | CNN (12장) | ViT | Swin / PVT |
|---|---|---|---|
| 기본 연산 | 지역 합성곱 | 전역 셀프 어텐션 | 윈도(지역) 어텐션 / 축소된 전역 어텐션 |
| 귀납적 편향 | 지역성 + 이동 등변성 (강함) | 거의 없음 | 지역성 일부 |
| 데이터가 적을 때 | 강함 | 약함 (사전 훈련·증류·증강 필요) | 중간 |
| 해상도 구조 | 피라미드 | 단일 (1/16) | 피라미드 |
| 비용 vs 이미지 크기 | 선형 | 제곱 | 선형 (Swin) |
| 주 용도 | 분류·탐지·분할 | 분류, 멀티모달 모델의 이미지 인코더 | 탐지·분할 백본 |

**이번 장의 이미지 분류 실험** (Oxford-IIIT Pet, 3 에포크)

| | ViT-B/16 `in21k` | DeiT-B 증류 |
|---|---|---|
| 추가 토큰 | `[CLS]` | `[CLS]` + `[DIST]` |
| 위치 임베딩 | 197개 | 198개 |
| 사전 훈련 | ImageNet-21k | ImageNet-1k + CNN 교사 |
| 전처리 | 224 크기 조정, 평균 0.5 | 256 조정 → 224 중앙 자르기, ImageNet 평균 |
| 정확도 / 검증 손실 | 91.44% / 1.153 | **94.41% / 0.201** |

**지도 학습 vs 자기 지도 학습 (비전)**

| | 지도 학습 (ViT, DeiT) | DINO (자기 증류) | MAE (마스크 복원) |
|---|---|---|---|
| 레이블 | 필요 | 불필요 | 불필요 |
| 학습 신호 | 정답 클래스 | 부분 뷰 ↔ 전체 뷰의 출력 일치 | 가린 패치의 픽셀 |
| 특징 | 분류 성능 | 어텐션이 물체를 분할, k-NN에 좋은 특성 | 효율적(25%만 인코딩), 미세 튜닝에 강함 |

**멀티모달 결합 방식**

| 방식 | 모델 | 장점 | 단점 |
|---|---|---|---|
| 단일 스트림 (토큰 이어 붙이기) | VideoBERT, DALL·E | 구조가 가장 단순 | 모든 토큰끼리 어텐션 → 비용 큼 |
| 이중 스트림 + 공동 어텐션 | ViLBERT | 모달리티마다 다른 구조 | 복잡 |
| 이중 인코더 + 대조 학습 | CLIP | 따로 인코딩 → 검색·제로샷에 빠름 | 세밀한 상호 작용(질의응답) 불가 |
| 잠재 배열 | Perceiver, Perceiver IO | 어떤 모달리티·길이든 같은 구조 | 위치 인코딩 설계에 의존 |
| 동결 모델 + 연결부 | Flamingo, BLIP-2, LLaVA | 적은 훈련으로 LLM의 언어 능력 활용 | 이미지 인코더·LLM의 한계를 그대로 물려받음 |
| 네이티브 멀티모달 | Gemini, GPT-4o | 모달리티 간 추론이 가장 자연스러움 | 훈련 비용이 막대, 주로 비공개 API |

**큰 입력의 제곱 비용을 피하는 방법**

| 방법 | 모델 | 토큰 / 어텐션 크기 (224×224 이미지) |
|---|---|---|
| 큰 패치로 자르기 | ViT-B/16 | 197 → 197 × 197 |
| 지역 윈도 | Swin | 윈도마다 49 × 49 |
| 키·값 축소 | PVT | $N \times N/R^2$ |
| 학습되는 쿼리로 요약 | Perceiver(잠재 256) · Flamingo(64) · BLIP-2(32) | 256 × 50,176 → 256 × 256 |
| (피해야 할 것) 픽셀 셀프 어텐션 | | 50,176 × 50,176 ≈ 25억 |

### 5-4. 주요 결과

| 실험 | 지표 | 값 |
|---|---|---|
| `PatchEmbedding(3, 768, 16)` | 출력 형태 / 파라미터 | `[4, 196, 768]` / 590,592 |
| 합성곱 vs 패치 자르기 + 행렬 곱 | 최대 차이 | 3.2e-06 (`allclose` True) |
| 직접 만든 ViT-B/16 (1,000클래스) | 파라미터 (인코더 비중) | 86,567,656 (98.3%) |
| 인코더 층 하나 (768, 12헤드, 3,072) | 파라미터 | 7,087,872 |
| 허깅 페이스 ViT / DeiT / DeiT+교사 헤드 (37클래스) | 파라미터 | 85,827,109 / 85,828,645 / 85,857,098 |
| ViT `in21k` 미세 튜닝 (3 에포크) | 검증 손실 / 정확도 | 1.975 → 1.333 → 1.153 / 90.05 → 91.25 → **91.44%** |
| DeiT 증류 미세 튜닝 (3 에포크) | 검증 손실 / 정확도 | 0.270 → 0.220 → 0.201 / 91.58 → 93.79 → **94.41%** |
| 미세 튜닝 시간 (코랩 GPU, 690 스텝) | ViT / DeiT | 10분 52초 / 11분 36초 |
| 윈도 vs 전역 어텐션 FLOPs ($C$ = 96, $M$ = 7) | 56² / 112² / 224² 토큰 | 13.8배 / 52.8배 / 209배 절약 |
| DINO ViT-B/8 | `[CLS]` 출력 / 마지막 층 어텐션 / 패치 → `[CLS]` | `[1, 768]` / `[1, 12, 785, 785]` / `[12, 784]` |
| CLIP 곤충 제로샷 | ladybug / spider / cricket | **99.73%** / 0.17% / 0.11% |
| CLIP 꽃 제로샷 | dandelion / popy / sunflower / rose / lily | **68.3%** / 28.2% / 2.1% / 1.3% / 0.15% |
| CLIP 코사인 유사도 | cricket / ladybug / spider | 0.2337 / 0.3021 / 0.2381 |
| CLIP 온도 | 초기값 / 학습 후 | 14.29 ($\ln(1/0.07)$ = 2.659) / **100** |
| 온도별 ladybug 확률 | × 1 / × 14.3 / × 100 | 34.8% / 56.3% / 99.7% |
| CLIP ViT-B/32 | 전체 / 이미지 / 텍스트 파라미터 | 151,277,313 / 87,456,000 / 63,165,952 |
| 대조 손실 ($N$ = 8) | 무작위 / 짝 맞음 | 2.457 ($\ln 8$ = 2.079) / 5.0e-06 |
| 푸리에 위치 인코딩 ($K$ = 6, $\mu$ = 224) | 형태 / 주파수 | `[224, 224, 26]` / 1, 23.2, 45.4, 67.6, 89.8, 112 |
| Perceiver 블록 (잠재 256 × 512) | 입력 → 출력 / 파라미터 | `[2, 50176, 29]` → `[2, 256, 512]` / 3,839,488 |
| BLIP-2 (OPT-2.7B, float16) | 캡션 / 이미지 토큰 수 | "two cats laying on a couch" / 32 |
| Gemini 2.5 Flash | "What animal and how many?" | `[cat, 2]` |

### 5-5. 자주 쓴 코드 패턴

| 패턴 | 용도 | 위치 |
|---|---|---|
| `X.flatten(start_dim=2).transpose(1, 2)` | `[B, C, H, W]` → `[B, H·W, C]` 토큰 시퀀스 | 1-3 |
| `X.unfold(2, P, P).unfold(3, P, P)` | 이미지를 패치로 자르기 | 1-4 |
| `self.cls_token.expand(B, -1, -1)` + `torch.cat(..., dim=1)` | 배치마다 특수 토큰 붙이기 | 1-5 |
| `Z[:, 0]` / `last_hidden_state[:, 0]` | `[CLS]` 출력 | 1-5, 1-13 |
| `sum(p.numel() for p in m.parameters())` | 파라미터 수 | 1-6, 1-9, 2-5, 2-10 |
| `XxxConfig(...)` → `XxxModel(config)` | 내려받기 없이 구조 확인 | 1-6, 1-9, 2-5, 2-13 |
| `from_pretrained(id, num_labels=k)` | 사전 훈련 본체 + 새 헤드 | 1-7, 1-10 |
| `processor(images, return_tensors="pt", do_convert_rgb=True)` | 이미지 배치 전처리 | 1-7, 1-10 |
| `(pixel_values.permute(1, 2, 0) + 1) / 2` | 정규화된 이미지를 다시 그리기 | 1-7 |
| `TrainingArguments(..., remove_unused_columns=False)` | 원본 열을 콜레이트 함수에 넘기기 | 1-8, 1-10 |
| `attn.unsqueeze(0).unsqueeze(0)` → `TF.resize(..., image.size[::-1])` | 저해상도 맵을 원본 크기로 | 1-13 |
| `plt.imshow(img); plt.imshow(map, cmap='jet', alpha=0.5)` | 히트맵 겹쳐 그리기 | 1-13 |
| `[f"This is a photo of a {label}." for label in labels]` | 제로샷 프롬프트 | 2-4 ~ 2-6 |
| `x / x.norm(dim=1, keepdim=True)` | L2 정규화 | 2-3, 2-6 |
| `(img_emb @ txt_emb.T * scale).softmax(dim=1)` | 유사도 → 확률 | 2-5, 2-6 |
| `torch.meshgrid(ys, xs, indexing="ij")` | 좌표 격자 | 2-9 |
| `inputs.to(device, dtype=torch.float16)` | 실수 입력만 반정밀도로 | 2-14 |
| `batch_decode(ids, skip_special_tokens=True)` | 생성 결과에서 특수 토큰 제거 | 2-14 |

### 5-6. 핵심 원칙

1. 트랜스포머는 **벡터의 시퀀스**만 있으면 된다. 이미지를 패치로 자르고 선형 투영하면(= 겹치지 않는 합성곱) 15장의 인코더를 그대로 쓸 수 있다.
2. 귀납적 편향이 약한 ViT는 **데이터를 많이 먹는다.** 대규모 사전 훈련(ViT), 증강·증류(DeiT), 자기 지도 학습(DINO·MAE)이 이를 메운다. 작은 데이터셋에서는 처음부터 훈련하지 말고 사전 훈련 모델을 미세 튜닝한다.
3. 셀프 어텐션의 **제곱 비용**이 비전·멀티모달 설계의 중심 문제다. 큰 패치, 지역 윈도(Swin), 키·값 축소(PVT), 작은 학습되는 쿼리 집합(Perceiver·Flamingo·BLIP-2)이 모두 이 비용을 줄이는 방법이다.
4. 밀집 예측(탐지·분할)에는 CNN처럼 여러 해상도의 **특성 피라미드**가 필요하다 — PVT·Swin이 이를 트랜스포머에 들여왔다.
5. 사전 훈련 모델은 **자기 전처리와 한 몸**이다. 크기 조정·자르기·정규화를 사전 훈련 때와 똑같이 하도록 모델의 이미지 프로세서를 쓴다.
6. 레이블 없이 훈련한 DINO의 어텐션이 물체를 분할하듯, 좋은 자기 지도 목표는 사람이 가르치지 않은 **구조를 스스로** 배우게 한다.
7. CLIP은 이미지와 텍스트를 **같은 임베딩 공간**에 놓는다. 그러면 클래스 이름 목록만으로 분류기를 만들고(제로샷), 문장으로 이미지를 검색할 수 있다. 확률은 후보끼리의 상대 비교이고, 학습된 온도가 확신의 정도를 정한다.
8. 멀티모달 모델의 핵심 설계는 **모달리티를 어디서 만나게 하느냐**다 — 토큰 이어 붙이기, 공동 어텐션, 임베딩 공간 비교, 잠재 배열, 동결된 LLM에 시각 토큰 주입, 처음부터 함께 훈련.
9. 이미 강한 모델(이미지 인코더, LLM)은 **동결하고 작은 연결부만** 훈련하는 것이 효율적이다. 새 층은 처음에 아무 영향이 없도록(tanh(0) 게이트) 초기화해 기존 능력을 지킨다.
10. 라이브러리 버전에 따라 같은 코드의 동작이 달라진다(`output_attentions`와 `sdpa`, `get_image_features`의 반환형). 결과가 비어 있거나 형이 다르면 **버전부터 확인**한다.

<!-- 일타 -->
### 5-7. 일타 복습: 5분 셀프 테스트

> 🧠 답을 보기 전에 **소리 내어** 먼저 답해 보자. 막히면 그 개념이 나온 절로 돌아가서 다시 읽고 오면 된다. 정답은 ▶를 눌러 펼친다.

**Q1. 224×224 이미지를 16×16 패치로 자르면 토큰은 몇 개인가?**

<details><summary>정답 보기</summary>

(224/16)² = 14 × 14 = **196개**. [CLS]를 더하면 197개.

</details>

**Q2. 데이터가 적을 때 ViT가 CNN보다 불리한 이유는?**

<details><summary>정답 보기</summary>

CNN은 지역성·이동 등변성이라는 **귀납적 편향**이 구조에 들어 있어 적은 데이터로도 이미지의 성질을 빨리 배운다. ViT는 그런 가정이 없어 데이터에서 직접 배워야 한다.

</details>

**Q3. Swin 트랜스포머의 어텐션 비용이 이미지 크기에 선형인 이유와, 이동 윈도가 필요한 이유는?**

<details><summary>정답 보기</summary>

고정 크기 윈도 안에서만 어텐션하므로 비용이 윈도 개수(= 이미지 면적)에 비례한다. 윈도 경계를 넘는 정보 교환이 막히므로 층마다 윈도를 이동시켜 이웃 윈도끼리 섞이게 한다.

</details>

**Q4. CLIP으로 제로샷 분류하는 절차는?**

<details><summary>정답 보기</summary>

레이블마다 "a photo of a {label}" 문장을 만들어 텍스트 임베딩을 구하고, 이미지 임베딩과의 코사인 유사도를 계산해(온도를 곱하고) 소프트맥스 → 가장 높은 레이블.

</details>

**Q5. DINO에서 교사 모델은 어떻게 만드나?**

<details><summary>정답 보기</summary>

별도로 훈련하지 않고, 학생 가중치의 **지수 이동 평균**으로 만든다. 붕괴를 막으려고 교사 출력에 센터링과 샤프닝을 쓴다.

</details>

**Q6. BLIP-2의 Q-Former가 하는 일은?**

<details><summary>정답 보기</summary>

학습되는 쿼리 32개가 이미지 인코더의 출력에 크로스 어텐션해 이미지를 고정 개수 토큰으로 **요약**하고, 그 토큰을 동결된 LLM의 입력으로 넘긴다.

</details>

---

> ➡️ **다음 장 — 17장. 트랜스포머 속도 높이기 (정리 예정)**  
> 이 장의 모델은 ViT-B만 해도 8,600만, BLIP-2는 약 38억 파라미터다. 17장은 어텐션 계산을 빠르게 하는 커널(플래시 어텐션), 상대 위치 인코딩(RoPE·ALiBi) 등으로 트랜스포머를 **더 빠르고 가볍게** 만드는 방법을 다룬다.

[← 15장. 자연어 처리와 챗봇을 위한 트랜스포머](15_transformers_for_nlp_and_chatbots.md) · [📚 전체 목차](../README.md) · [📖 용어집](glossary.md) · 17장. 트랜스포머 속도 높이기 (정리 예정) →
