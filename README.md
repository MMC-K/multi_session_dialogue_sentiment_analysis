# 복합대화2 멀티 세션 대화 감성 분석 모델 학습 및 추론 코드

이 문서는 `config_regression.json`을 사용하여 멀티 모달 감정 분석 모델인 CENET 모델을 학습하는 방법을 설명합니다.
`멀티 모달 대화 모델을 위한 패션 지식 대화 데이터 셋` ([링크](https://github.com/MMC-K/multimodal_fashion_dialog_dataset))과 같은
커스텀 데이터셋을 MOSEI 형식으로 준비하여 학습하는 과정을 다룹니다.

## 환경 설정

### 패키지 설치

```bash
cd MMSA_master/MMSA
pip install .
```

### 필요 패키지

- Python 3.7+
- PyTorch 1.8+
- transformers
- numpy
- pandas

## 데이터셋 준비

### 데이터셋 구조 (MOSEI 형식)

커스텀 데이터셋은 MOSEI와 동일한 구조로 구성해야 합니다:

```
dataset_root_dir/
└── YOUR_DATASET/
    ├── Raw/
    │   └── video_id/
    │       ├── clip_id1.mp4
    │       ├── clip_id2.mp4
    │       └── ...
    └── label.csv
```

### label.csv 형식

`label.csv` 파일은 다음 컬럼을 포함해야 합니다:

- `video_id`: 비디오 ID
- `clip_id`: 클립 ID
- `text`: 텍스트 내용
- `label`: 감정 레이블 (regression 값, 예: -3.0 ~ 3.0)
- `label_T`: 텍스트 모달리티 레이블 (선택사항)
- `label_A`: 오디오 모달리티 레이블 (선택사항)
- `label_V`: 비디오 모달리티 레이블 (선택사항)
- `annotation`: 어노테이션 문자열
- `mode`: 데이터 분할 (`train`, `valid`, `test`)

## 데이터셋 빌드 (JSON → Raw + label.csv)

원본이 JSON 형태의 대화 데이터라면 `Feature_extraction/build_dataset.py`로 MOSEI와 동일한 디렉토리 구조(`Raw/`, `label.csv`)를 생성할 수 있습니다.  

### 실행 예시

```bash
cd Feature_extraction
python build_dataset.py \
  --json /data/source/data.json \
  --out /data/YourDataset \
  --fps 25 \
  --sec 1.0 \
  --image_policy reuse_last \
  --verbosity 1
```

- `--json`: 대화/주석이 담긴 JSON 또는 JSONL 경로
- `--out`: 결과를 저장할 루트 디렉토리 (`Raw/`, `label.csv`가 생성됨)
- `--fps`, `--sec`: 생성할 더미 비디오의 FPS·길이
- `--image_policy`: `reuse_last`(이전 이미지 재사용) 또는 `blank`(검은 화면)
- `--root_key`: JSON 최상단이 `{ "data": [...] }` 형태일 때 사용

성공 시 `YourDataset/Raw/<video_id>/<clip_id>.mp4` 와 학습용 `label.csv`가 생성됩니다. 이후 단계(Feature 추출)에서 이 디렉토리를 그대로 사용하면 됩니다.

## Feature 추출 (JSON → MOSEI Pkl)

멀티모달 feature를 직접 추출해야 한다면 `Feature_extraction` 디렉토리에 포함된 스크립트들을 순서대로 실행하여 MOSEI 형식의 pkl 또는 manifest를 만들 수 있습니다.

### 1. JSON → label.csv 변환

대화 형태의 JSON을 사용한다면 `dialoguelabel.py`로 `label.csv`의 `text` 열을 원하는 컨텍스트로 치환할 수 있습니다.

```bash
python dialoguelabel.py
```

생성된 `label_context.csv`를 `label.csv`로 교체하거나, 이후 단계에서 사용할 디렉토리로 복사합니다.

### 2. MMSA-FET 추출 (샤드 생성)

1. 추출용 작업 디렉토리를 준비합니다 (`label.csv`, `Raw/` 등 필요한 자원이 있어야 합니다).
2. 환경 변수를 설정하여 사용할 설정 파일 및 출력 파일명을 정합니다.
   - `CFG_PATH` : 사용할 MMSA-FET 설정 (기본값 `MMSA-FET/config_text_video.json`)
   - `OUT` : 출력 파일 프리픽스 (기본값 `ourdata_Vt_unaligned.pkl`)
   - `ASSET_ROOT` : 비디오/오디오 등 원본 파일이 있는 경로 (필요 시)

예시(Linux/macOS):

```bash
cd Feature_extraction
export CFG_PATH=../MMSA-FET/config_text_video.json
export OUT=../Processed/ourdata_unaligned.pkl
export ASSET_ROOT=/data/Raw
python run_shard.py
```

스크립트가 완료되면 `OUT` 경로를 기준으로 다수의 `*.pkl.partXXXXX.pkl` 파일과 `manifest_text.txt`가 생성됩니다. 각 part는 이미 패딩이 적용된 numpy 배열 형태입니다.

### 3. (선택) 텍스트 전용 샤드 후처리

텍스트 feature만 사용하거나, 오디오/비디오를 0으로 채운 샤드를 별도로 만들고 싶다면 `featurepadding.py`를 이용해 변환 manifest를 생성할 수 있습니다.

```bash
python Feature_extraction/featurepadding.py \
  --manifest ../Processed/manifest_text.txt \
  --out_dir ../Processed/padded \
  --suffix _padded
```

새로 생성된 manifest(`manifest_ftonly.txt` 등)와 변환된 pkl을 `config_regression.json`에서 사용할 수 있습니다.


### 4. 샤드 검증 (선택)

생성된 manifest가 정상인지 확인하려면 `Feature_extraction/test.py`를 사용합니다.

```bash
cd Feature_extraction
python test.py 
```

모든 샤드가 정상적으로 로드되면 다음 단계를 진행할 수 있습니다.

## config_regression.json 설정

### 1. 데이터셋 루트 디렉토리 설정

```json
{
  "datasetCommonParams": {
    "dataset_root_dir": "root_directory",
    ...
  }
}
```

### 2. 새로운 데이터셋 추가

`config_regression.json`의 `datasetCommonParams` 섹션에 새로운 데이터셋 정보를 추가합니다.

#### 단일 pkl 파일 사용

```json
{
  "datasetCommonParams": {
    "dataset_root_dir": "root_directory",
    "your_dataset": {
      "aligned": {
        "featurePath": "YOUR_DATASET/Processed/aligned_50.pkl",
        "seq_lens": [50, 50, 50],  // [text, audio, video] 시퀀스 길이
        "feature_dims": [768, 74, 35],  // [text, audio, video] feature 차원
        "train_samples": 1000, 
        "num_classes": 3,
        "language": "en", 
        "KeyEval": "Loss",
        "missing_rate": [0.2, 0.2, 0.2],
        "missing_seed": [1111, 1111, 1111]
      }
    },
    ...
  }
}
```

#### Manifest 파일 사용 (대용량 데이터셋)

pkl 파일이 너무 큰 경우, 여러 pkl 파일을 manifest 텍스트 파일로 관리할 수 있습니다. `featurePath`를 `.txt` 파일로 설정하고, 각 줄에 pkl 파일명을 작성합니다:

```json
{
  "datasetCommonParams": {
    "dataset_root_dir": "root_directory",
    "your_dataset": {
      "unaligned": {
        "featurePath": "YOUR_DATASET/Processed/manifest_path.txt",
        "seq_lens": [50, 500, 375],
        "feature_dims": [768, 74, 35],
        "train_samples": 1000,
        "num_classes": 3,
        "language": "en",
        "KeyEval": "Loss",
        "missing_rate": [0.2, 0.2, 0.2],
        "missing_seed": [1111, 1111, 1111]
      }
    }
  }
}
```

**Manifest 파일 예시 (`manifest_path.txt`):**
```
part_000.pkl
part_001.pkl
part_002.pkl
```

manifest 파일과 pkl 파일들은 같은 디렉토리에 있어야 하며, 모든 pkl 파일의 `feature_dims`와 `seq_lens`는 동일해야 합니다.

**주요 파라미터 설명:**

- `seq_lens`: 각 모달리티의 시퀀스 길이
  - Text: BERT 토큰 길이 (일반적으로 50)
  - Audio: 오디오 feature 프레임 수
  - Video: 비디오 feature 프레임 수
- `feature_dims`: 각 모달리티의 feature 차원
  - Text: BERT hidden size (768)
  - Audio: 오디오 feature 차원 
  - Video: 비디오 feature 차원
- `train_samples`: 학습 데이터 샘플 수

### 3. CENET 모델 설정

`config_regression.json`의 `cenet` 섹션에 새로운 데이터셋의 하이퍼파라미터를 추가합니다:

```json
{
  "cenet": {
    "commonParams": {
      "need_data_aligned": false,
      "need_model_aligned": false,
      "need_normalized": false,
      "use_bert": true,
      "use_finetune": true,
      "early_stop": 8
    },
    "datasetParams": {
      "your_dataset": {
        "pretrained": "bert-base-multilingual-cased", 
        "learning_rate": 1e-5,
        "weight_decay": 0.0001,
        "max_grad_norm": 2,
        "adam_epsilon": 1e-8,
        "batch_size": 64
      },
      ...
    }
  }
}
```

## 학습 실행

### 기본 학습 명령어

```bash
python -m MMSA -m cenet -d your_dataset -c config_regression.json -s 1111 -s 1112 -s 1113
```

### 명령어 옵션

- `-m, --model`: 모델 이름 (`cenet`)
- `-d, --dataset`: 데이터셋 이름 (`your_dataset`)
- `-c, --config`: 설정 파일 경로 (`config_regression.json`)
- `-s, --seeds`: 랜덤 시드 (여러 번 지정 가능)
- `-g, --gpu-ids`: 사용할 GPU ID (예: `-g 0`)
- `--model-save-dir`: 모델 저장 디렉토리
- `--res-save-dir`: 결과 저장 디렉토리
- `--log-dir`: 로그 파일 저장 디렉토리
- `-n, --num-workers`: 데이터 로더 워커 수 (기본값: 8)
- `-v, --verbose`: 출력 상세도 


## 하이퍼파라미터 조정

### config_regression.json에서 수정

```json
{
  "cenet": {
    "datasetParams": {
      "your_dataset": {
        "learning_rate": 2e-5,  
        "batch_size": 32,     
        "weight_decay": 0.0001,
        "max_grad_norm": 2,
        "adam_epsilon": 1e-8
      }
    }
  }
}
```


## 결과 확인

### 모델 저장 위치

- 기본 위치: `~/MMSA/saved_models/cenet-{dataset_name}.pth`
- 커스텀 위치: `{model-save-dir}/cenet-{dataset_name}.pth`

### 결과 파일 위치

- 기본 위치: `~/MMSA/results/normal/{dataset_name}.csv`
- 커스텀 위치: `{res-save-dir}/normal/{dataset_name}.csv`

### 로그 파일 위치

- 기본 위치: `~/MMSA/logs/cenet-{dataset_name}.log`
- 커스텀 위치: `{log-dir}/cenet-{dataset_name}.log`

### 결과 메트릭

CENET는 데이터셋에 따라 다음 메트릭을 계산합니다:

#### MOSEI/MOSI 형식 데이터셋

- **Mult_acc_3**: 3-class Multiclass Accuracy (-1 ~ 1 범위)
- **MAE**: Mean Absolute Error
- **Corr**: Pearson Correlation
- **Non0_acc_2**: 2-class Accuracy (0 제외, > 0 vs < 0)
- **Non0_F1_score**: F1 Score (0 제외)


# `멀티 모달 대화 모델을 위한 패션 지식 대화 데이터 셋`에서 성능 결과
아래는 멀티 모달 대화 내 감성 분석 수행 결과의 성능이다.

| Mult_acc_3 |   MAE   |  Corr  | Non0_acc_2 | Non0_F1_score |
|-----------:|:-------:|:------:|:----------:|:-------------:|
|   0.9912   | 0.0133  | 0.9772 |   0.9907   |    0.9907     |

<멀티 모달 대화 내 감성 분석 결과>

- 부정/비부정(Has0) 지표를 제외하고 종합적으로 매우 높은 정확도와 일관성을 보인다.  
- 모델의 전체 3클래스 감성 분류 정확도(Mult_acc_3)가 매우 높게 나타났으며, MAE와 Corr를 통해 실제 감정 레이블과 높은 일관성을 보임을 알 수 있다.  
- 중립을 제외한 긍정/부정(Non0) 분류에서는 정확도와 F1 점수가 높게 나타났다. 이는 중립이 제거되었을 때 모델이 명확한 감성들을 보다 안정적으로 구분할 수 있음을 나타낸다.

  
# Acknowledgement
본 연구는 정부(과학기술정보통신부)의 재원으로 지원을 받아 수행된 연구입니다. (No. RS-2022-II220320, 상황인지 및 사용자 이해를 통한 인공지능 기반 1:1 복합대화 기술 개발)
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
