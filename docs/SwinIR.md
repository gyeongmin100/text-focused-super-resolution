# SwinIR 이해와 프로젝트 적용 코드 설명

> 논문: [SwinIR: Image Restoration Using Swin Transformer](https://arxiv.org/abs/2108.10257) (Liang et al., ICCVW 2021)
> 이 문서는 SwinIR의 구조를 정리하고, 본 프로젝트 노트북(`notebook/v2_loss4_super_resolution.ipynb`)의 코드가 논문 내용과 어떻게 대응되는지 설명한다.
> 여기서 수동으로 정한 LossW 가중치·학습률은 이후 Optuna로 자동 탐색해 개선했다 — `notebook/v3_optuna_super_resolution.ipynb` 및 [메인 README의 Optuna 섹션](../README.md#optuna-하이퍼파라미터-탐색) 참고.

---

## 1. SwinIR이 풀려는 문제

이미지 복원(super-resolution, denoising 등)의 기존 주류는 CNN이었다. CNN의 한계:

- **convolution은 지역적(local)** — 같은 kernel을 이미지 전체에 동일 적용하므로, 멀리 떨어진 영역 간 관계(long-range dependency)를 직접 모델링하지 못함
- 넓은 문맥을 보려면 층을 깊게 쌓아야 해서 비효율적

반면 일반 Vision Transformer는 전역 attention이 가능하지만 계산량이 이미지 크기의 제곱에 비례해 고해상도 복원에 부적합하다. **Swin Transformer의 shifted window attention**은 이 절충점이다:

- attention을 M×M **윈도우 내부로 제한** → 계산량이 이미지 크기에 선형
- 층마다 윈도우를 **반칸씩 이동(shift)** → 윈도우 경계를 넘는 정보 교환 가능

SwinIR은 이 구조를 이미지 복원에 가져온 모델이다. 문서 이미지처럼 작은 글자가 조밀하게 반복되는 입력에서, 주변 글자의 획 패턴을 참조해 뭉개진 글자를 복원할 수 있다는 점이 본 프로젝트에 SwinIR을 선택한 이유다.

## 2. 전체 구조 — 3단계

```
LR 입력
  │
  ▼
① Shallow Feature Extraction      : 3×3 conv 1개 → 저수준 특징 (색·에지)
  │
  ▼
② Deep Feature Extraction         : RSTB × K  (+ 마지막 conv, 전체 residual)
  │    RSTB = [STL × L] + 3×3 conv + residual 연결
  │    STL  = (S)W-MSA + MLP  (LayerNorm, residual)
  ▼
③ HQ Image Reconstruction         : 업샘플링 (SR은 PixelShuffle 계열)
  │
  ▼
SR 출력
```

- **STL(Swin Transformer Layer)**: 윈도우 기반 multi-head self-attention(W-MSA)과 shifted 버전(SW-MSA)이 번갈아 나오는 기본 블록. 각 블록은 LayerNorm → attention → MLP 순서에 residual 연결.
- **RSTB(Residual Swin Transformer Block)**: STL을 L개 쌓고 끝에 conv 하나 + **블록 단위 residual**. 논문은 이 residual 덕에 저수준 특징이 얕은 층에서 깊은 층·출력까지 직접 전달돼 학습이 안정된다고 설명한다. 마지막 conv는 Transformer 특징에 convolution의 inductive bias(지역성·translational equivariance)를 섞는 역할.
- **Reconstruction**: shallow 특징(저주파)과 deep 특징(복원된 고주파)을 합쳐 업샘플. 저주파는 지름길로 통과시키고 Transformer는 고주파 디테일 복원에 집중하는 설계.

## 3. 본 프로젝트의 SwinIR 설정

노트북에서는 [공식 구현](https://github.com/JingyunLiang/SwinIR)의 `network_swinir.py`를 가져와 다음 설정으로 생성한다:

```python
SWINIR_CONFIG = dict(
    upscale=2,                      # x2 super-resolution
    in_chans=3,
    img_size=64,
    window_size=8,                  # 8×8 윈도우 attention
    img_range=1.,
    depths=[6, 6, 6, 6],            # RSTB 4개, 각 6 STL
    embed_dim=60,                   # 특징 차원
    num_heads=[6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffledirect', # 경량 업샘플러
    resi_connection='1conv',
)
```

이는 논문의 **lightweight SR 설정**이다 (classical SR: embed_dim 180, RSTB 6개). 차이점:

| | Classical | Lightweight (본 프로젝트) |
|---|---|---|
| embed_dim | 180 | **60** |
| RSTB 수 | 6 | **4** |
| 업샘플러 | PixelShuffle (다단) | **pixelshuffledirect** (conv 1개 + PixelShuffle 1회) |
| 파라미터 | ~11.8M | **~0.9M** |

GPU 메모리 제약(Colab)과 서비스 배포 시 CPU 추론을 고려해 경량 설정을 선택했다. `pixelshuffledirect`는 conv 한 번으로 채널을 `3×upscale²`로 늘린 뒤 PixelShuffle로 픽셀을 재배열하는 가장 가벼운 업샘플 방식이다.

## 4. 노트북 코드 흐름 설명

### 4-1. 데이터 준비 — 열화 생성 (`degrade_image_x2`)

원본(HR)을 ½로 축소한 뒤, 실제 저품질 문서의 특성을 흉내 내는 열화를 **확률적으로** 적용한다:

```python
lr = cv2.resize(hr, (w//2, h//2), interpolation=cv2.INTER_AREA)  # x2 다운샘플
# 20% 확률: Gaussian blur (sigma 0.6~1.4)
# 25% 확률: JPEG 재압축 (quality 50~85)
# 10% 확률: Gaussian noise (sigma 0.003~0.015)
```

고정 열화가 아닌 확률 조합이라 매 epoch 다른 열화가 생성돼 augmentation 효과를 겸한다.

### 4-2. Dataset — 텍스트 마스크와 crop 전략 (`SROIEDataset`)

- `_build_mask_from_box`: SROIE의 box 파일(텍스트 4점 좌표)을 `cv2.fillPoly`로 채워 **텍스트=1, 배경=0 마스크** 생성. 이 마스크가 텍스트 가중 loss의 핵심 입력이다.
- `_random_crop`: 영수증은 여백이 많아 무작위 crop이 자주 빈 배경만 뽑는다. 이를 막기 위해 **최대 10회 crop을 시도해 텍스트 비율 ≥ 0.005인 패치를 우선 선택**하고, 없으면 후보 중 텍스트 비율 최대 패치를 사용한다.
- 반환: `(LR 96×96, HR 192×192, mask 192×192, id)`

### 4-3. 텍스트 가중 손실 (`compute_losses`)

```python
LossW(loss_4) = 2.0  × L1_text          # 텍스트 영역 L1
              + 0.05 × L1_background    # 배경 L1 (약하게)
              + 0.1  × (1 − SSIM_text)  # 텍스트 영역 구조 유사도
              + 0.02 × (1 − SSIM_global)
```

구현 디테일:

- `masked_l1`: `|pred − target| × mask`를 마스크 픽셀 수로 정규화 — 텍스트 면적이 이미지마다 달라도 스케일이 일정.
- `masked_bbox_ssim_loss`: 텍스트 SSIM을 `pred × mask`로 계산하면 검은 배경이 대량 포함돼 SSIM이 왜곡된다. 대신 **마스크의 bounding box를 crop해서 SSIM을 계산**한다 (SSIM window 11보다 작은 crop은 건너뜀).
- 비교 기준인 `loss_1`은 전체 균등 `F.l1_loss` 하나다.

### 4-4. 학습 루프 (`train_one_loss`)

- scratch: `Adam(lr=1e-4)`, fine-tune: `Adam(lr=2e-5)` — 사전학습 가중치 파괴를 막기 위해 fine-tune은 낮은 학습률
- `CosineAnnealingLR(T_max=num_epochs, eta_min=1e-6)`, batch 32, 50 epochs
- fine-tune은 공식 x2 lightweight 사전학습 가중치(`002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`)에서 시작
- 매 epoch validation에서 **PSNR/SSIM을 global과 text 영역으로 분리 측정**하고, `PSNR Text` 최고와 `SSIM Text` 최고 시점의 가중치를 각각 저장 → `weights/`의 `*_psnr_*` / `*_ssim_*` 파일이 이것

### 4-5. 전체 이미지 추론 — 슬라이딩 윈도우 (`sliding_window_sr`)

학습은 96×96 패치 단위이므로 임의 크기 이미지는 패치로 잘라 복원 후 이어붙인다:

1. 입력을 96의 배수 크기로 resize
2. 96×96 패치를 stride 96으로 순회하며 추론 (경계 패치는 끝에서 역방향으로 맞춰 잘림 방지)
3. 각 패치 출력을 HR 좌표(×2)에 누적하고 겹침 횟수로 나눠 평균
4. 최종적으로 원본×2 크기로 resize

이 함수가 그대로 [또렷 서비스](https://ddoreot.site) 백엔드의 추론 로직이 되었다. (서비스 가중치는 현재 Optuna 탐색 모델 `weights/optuna/best_swinir_optuna_psnr.pt` 사용)

### 4-6. 평가

- `evaluate_metrics`: test 347장에 대해 PSNR/SSIM을 global·text 분리 계산
- OCR 평가: EasyOCR로 복원 이미지의 텍스트를 인식해 정답과 비교, CER(문자 오류율)/WER(단어 오류율) 산출 — 결과는 [메인 README](../README.md#실험-결과) 참고

## 5. 정리 — 논문 ↔ 코드 대응표

| 논문 개념 | 노트북 코드 |
|---|---|
| Shallow/Deep/Reconstruction 3단 구조 | `network_swinir.py`의 `SwinIR` 클래스 (공식 구현) |
| Lightweight SR 설정 | `SWINIR_CONFIG` (embed_dim 60, depths [6,6,6,6], pixelshuffledirect) |
| 사전학습 → 파인튜닝 | x2 lightweight 공식 가중치 로드 후 `lr=2e-5`로 추가 학습 |
| 픽셀 loss (논문은 L1) | `loss_1` = `F.l1_loss` |
| (본 프로젝트 확장) 텍스트 가중 loss | `compute_losses`의 `loss_4` = LossW |
| 임의 크기 테스트 이미지 처리 | `sliding_window_sr` |
