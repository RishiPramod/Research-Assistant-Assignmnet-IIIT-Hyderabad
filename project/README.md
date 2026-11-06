# FastSpeech 2 Text-to-Speech Synthesis: English (Svarah) & Telugu (IndicTTS)

This repository contains the complete implementation, preprocessing, forced alignment, training, and variance adaptor analysis pipeline for **FastSpeech 2** trained independently on **English (`Svarah`)** and **Telugu (`IndicTTS Telugu`)**, adapted from [`ming024/FastSpeech2`](https://github.com/ming024/FastSpeech2) and the original paper (*FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, Ren et al., ICLR 2021).

---

## 1. Repository Structure

```text
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── README.md
└── requirements.txt
```

---

## 2. Environment Setup

```bash
# 1. Create conda environment (recommended for Montreal Forced Aligner + PyTorch)
conda create -n fastspeech2 python=3.10 -y
conda activate fastspeech2

# 2. Install Montreal Forced Aligner (MFA) via conda-forge
conda install -c conda-forge montreal-forced-aligner -y

# 3. Install Python dependencies
pip install -r requirements.txt
```

---

## 3. End-to-End Pipeline: English (`Svarah`)

All English commands are executed from `project/English/`:

```bash
cd English
```

### Step 3.1: Data Selection, Subsampling (5h Train + 1h Val) & Alignment Preparation
Place the Svarah dataset inside `English/data/Svarah/` (or pass `--download` to automatically fetch `ai4bharat/Svarah` from HuggingFace `datasets`). This step filters utterances to `[1.0s, 15.0s]`, removes duplicates, deterministically selects **5.0 hours for training** and **1.0 hour for validation** (`seed=42`, zero overlap), resamples audio to `22050 Hz` 16-bit PCM, writes `.lab` transcriptions, and expands the CMUdict lexicon with `g2p_en` for Indian-English OOV words:

```bash
python preprocess.py --config config.yaml --prepare_align --download
```

### Step 3.2: Run Montreal Forced Aligner (MFA)
Generate phoneme-level `.TextGrid` alignments:

```bash
# Option A: Align using MFA pretrained English acoustic model
mfa model download acoustic english_us_arpa
mfa align ./raw_data/Svarah ./lexicon/librispeech-lexicon.txt english_us_arpa ./preprocessed_data/Svarah/TextGrid --clean

# Option B: Flat-start train MFA directly on Svarah (as in Footnote 3 of the FastSpeech 2 paper)
mfa train ./raw_data/Svarah ./lexicon/librispeech-lexicon.txt ./preprocessed_data/Svarah/TextGrid --clean
```

### Step 3.3: Feature Extraction (`preprocess.py`)
Extracts ground-truth phoneme durations from `.TextGrid`, pitch ($F_0$) contours via PyWorld (`dio` + `stonemask`), frame-level STFT $L_2$ energy, 80-bin Mel-Spectrograms, global normalization statistics (`stats.json`), and non-overlapping `train.txt` (5h) / `val.txt` (1h):

```bash
python preprocess.py --config config.yaml
```

### Step 3.4: Model Training (`train.py`)
Trains the English FastSpeech 2 model, saves checkpoints to `English/checkpoints/`, and automatically plots `Results/English/training_loss.png` and `Results/English/validation_loss.png`:

```bash
python train.py --config config.yaml
```

### Step 3.5: Inference & Variance Adaptor Plots (`inference.py`)
Synthesizes all five English test sentences (`E1`, `E2a`, `E2b`, `E3`, `E4`) into `Results/English/test_generated_audios/` and saves Duration, Pitch ($F_0$), and Energy plots to `Results/English/variance_adaptor_plots/{duration,pitch,energy}/`:

```bash
python inference.py --config config.yaml
```

---

## 4. End-to-End Pipeline: Telugu (`IndicTTS Telugu`)

All Telugu commands are executed from `project/Telugu/`:

```bash
cd ../Telugu
```

### Step 4.1: Data Selection, Subsampling (5h Train + 1h Val) & Telugu Lexicon Creation
Place the IndicTTS Telugu dataset inside `Telugu/data/IndicTTS_Telugu/` (supports IITM Festival `txt.done.data` + `wav/`, CSV/TSV, JSONL, or pass `--download`). This step normalizes Telugu Unicode (NFC), expands numbers/ordinals/times into Telugu words (`text/telugu_numbers.py`), deterministically selects **5.0 hours for training** and **1.0 hour for validation** (`seed=42`, zero overlap), resamples audio to `22050 Hz`, and builds `lexicon/telugu_lexicon.txt` using the Akshara-to-Phoneme G2P engine (`text/telugu_phonemes.py`):

```bash
python preprocess.py --config config.yaml --prepare_align
```

### Step 4.2: Run Montreal Forced Aligner (MFA) on Telugu
Following Footnote 3 of the FastSpeech 2 paper, train and align MFA directly on the paired Telugu corpus and generated pronunciation lexicon:

```bash
mfa train ./raw_data/IndicTTS_Telugu ./lexicon/telugu_lexicon.txt ./preprocessed_data/IndicTTS_Telugu/TextGrid --clean
```

### Step 4.3: Feature Extraction (`preprocess.py`)
Extracts Telugu phoneme durations, PyWorld $F_0$, STFT energy, 80-bin Mel-Spectrograms, `stats.json`, `train.txt` (5h), and `val.txt` (1h):

```bash
python preprocess.py --config config.yaml
```

### Step 4.4: Model Training (`train.py`)
Trains the Telugu FastSpeech 2 model, saves checkpoints to `Telugu/checkpoints/`, and automatically generates `Results/Telugu/training_loss.png` and `Results/Telugu/validation_loss.png`:

```bash
python train.py --config config.yaml
```

### Step 4.5: Inference & Variance Adaptor Plots (`inference.py`)
Synthesizes all five Telugu test sentences (`T1`, `T2a`, `T2b`, `T3`, `T4`) into `Results/Telugu/test_generated_audios/` and saves Duration, Pitch ($F_0$), and Energy plots into `Results/Telugu/variance_adaptor_plots/{duration,pitch,energy}/`:

```bash
python inference.py --config config.yaml
```

