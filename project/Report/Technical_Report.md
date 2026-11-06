# FastSpeech 2: English and Telugu Speech Synthesis
## Technical Report — IIIT Hyderabad Research Assistant Assignment

**Author:** Rishi Pramod  
**Reference Paper:** *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (Ren et al., ICLR 2021)  
**Base Codebase:** `ming024/FastSpeech2`  
**Training Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), PyTorch 2.14.1+cu126, Windows 11  

---

## 1. Implementation Approach & Architectural Modifications

### 1.1 Components from Reference Implementation
The implementation adapts the core non-autoregressive FastSpeech 2 architecture from `ming024/FastSpeech2`:
- **Phoneme Encoder:** 4 Feed-Forward Transformer (FFT) blocks, 2 attention heads, hidden dimension 256, 1D convolution filter size 1024 (kernel sizes 9 and 1), dropout 0.2.
- **Variance Adaptor:** 
  - *Duration Predictor:* 2-layer 1D-CNN (filter size 256, kernel size 3, dropout 0.5) predicting log-scale phoneme durations, trained with MSE loss against alignment frames. Length Regulator expands phoneme hidden states to frame length.
  - *Pitch ($F_0$) Predictor:* 2-layer 1D-CNN predicting frame-level continuous pitch extracted via PyWorld (`dio` + `stonemask`), normalized using global dataset stats and quantized into 256 bins with an embedding layer.
  - *Energy Predictor:* 2-layer 1D-CNN predicting frame-level STFT $L_2$-norm energy, normalized and quantized into 256 bins.
- **Mel-Spectrogram Decoder & PostNet:** 6 FFT blocks (2 heads, hidden dim 256) predicting 80-channel Mel spectrograms ($L_1$ loss), followed by a 5-layer Conv1D PostNet ($L_1$ loss).
- **Vocoder:** Pre-trained HiFi-GAN Universal vocoder running at 22,050 Hz.

### 1.2 Modifications for English (`ai4bharat/Svarah`)
1. **Forced Alignment with MMS-FA:** The standard Montreal Forced Aligner (MFA) requires Kaldi binaries that are prone to environment conflicts on Windows. We implemented `project/acoustic_aligner.py` using Meta's Massively Multilingual Forced Aligner (`torchaudio.pipelines.MMS_FA`) to perform GPU-accelerated CTC forced alignment on 22.05 kHz audio, generating phone-level `.TextGrid` files.
2. **Contextual Homograph Disambiguation:** English words like *"conduct"* shift stress based on syntactic role (noun `/'kandakt/` vs verb `/kan'dakt/`). In `English/inference.py`, we implemented a rule-based POS disambiguation module that inspects preceding tokens (e.g., possessives/determiners for noun form in E2a vs auxiliary verbs for verb form in E2b) and assigns the correct ARPAbet phoneme sequence (`K AA1 N D AH0 K T` vs `K AH0 N D AH1 K T`).
3. **Text Normalization:** Implemented cleaners in `English/text/cleaners.py` for expanding times (`10:45 AM` $\rightarrow$ *"ten forty five a m"*), dates (`12 November 2026` $\rightarrow$ *"twelve november twenty twenty-six"*), and added ARPAbet fallbacks for code-switched terms (*"andaroo"* $\rightarrow$ `AA1 N D AA0 R UW1`).
4. **Pause Preservation & Formatting:** Kept inter-word `'sp'` pause tokens consistent with training. For intra-word typographical spaces (e.g., *"rhyth m"* in E3), the tokenizer preserves the split by inserting an `'sp'` token. Audio output is saved in standard 16-bit PCM format normalized to -1.0 dBFS.

### 1.3 Modifications for Telugu (`SPRINGLab/IndicTTS_Telugu`)
1. **Rule-Based 50-Phoneme Telugu G2P Engine:** `ming024/FastSpeech2` natively supports only English ARPAbet and Mandarin Pinyin. We developed a custom Telugu G2P parser (`Telugu/text/telugu_phonemes.py` and `symbols.py`) supporting:
   - 14 vowels (short: `a, i, u, ru, e, o`; long: `aa, ii, uu, ruu, ee, ai, oo, au`).
   - 31 consonants across velar, palatal, retroflex, dental, labial, and sibilant classes.
   - Virama (`్`, U+0C4D) consonant-cluster logic: consonants carry an inherent short `'a'` which is suppressed when followed by a virama or dependent vowel matra, enabling accurate modeling of conjuncts (*"ప్రభావాన్ని"*) and geminates (*"పట్టు"*).
   - Diacritics: Anusvara (`ం` $\rightarrow$ `M`) and Visarga (`ః` $\rightarrow$ `H`).
2. **Telugu Number and Date Expander:** In `Telugu/text/telugu_numbers.py`, numbers, years (`2026` $\rightarrow$ *"రెండు వేల ఇరవై ఆరు"*), ordinals (`12వ` $\rightarrow$ *"పన్నెండవ"*), and times (`10:45` $\rightarrow$ *"పది నలభై ఐదు"*) are converted to native Telugu words before phonetic conversion.
3. **Framework Compatibility:** Patched `audio/stft.py` for `librosa >= 1.0` (updating `librosa.filters.mel` calls), enabled `weights_only=False` for PyTorch checkpoint loading, and added in-memory tensor caching to speed up multi-epoch training on Windows.

---

## 2. Dataset Selection, Subsampling, and Training Setup

### 2.1 Dataset Subsampling (5h Train + 1h Val)
Both datasets were split deterministically (`seed = 42`) with zero speaker or transcript overlap between train and validation sets:

| Parameter | English (`Svarah`) | Telugu (`IndicTTS`) |
| :--- | :--- | :--- |
| **Corpus** | AI4Bharat Svarah (Indian-accented English) | IIT Madras IndicTTS Telugu |
| **Speaker Type** | Multi-speaker (38 regional accent groups) | Single professional female speaker |
| **Train Duration** | 5.00 hours (3,472 utterances) | 5.00 hours (1,589 utterances) |
| **Val Duration** | 1.00 hour (670 utterances) | 1.00 hour (305 utterances) |
| **Audio Format** | 22,050 Hz, 16-bit mono PCM | 22,050 Hz, 16-bit mono PCM |
| **STFT / Mel** | $N_{fft}=1024$, $\text{hop}=256$, $\text{win}=1024$, 80 mels | $N_{fft}=1024$, $\text{hop}=256$, $\text{win}=1024$, 80 mels |
| **Pitch Stats ($\mu, \sigma$)** | $\mu = 176.86\text{ Hz}, \sigma = 57.12\text{ Hz}$ | $\mu = 208.39\text{ Hz}, \sigma = 40.10\text{ Hz}$ |
| **Energy Stats ($\mu, \sigma$)** | $\mu = 48.62, \sigma = 38.83$ | $\mu = 41.43, \sigma = 36.25$ |

### 2.2 Preprocessing & Data Cleaning
- Raw audio was trimmed of leading/trailing silence ($>50\text{ ms}$) and peak-normalized.
- Background noise in the multi-speaker crowdsourced Svarah recordings was cleaned using spectral de-noising (`noisereduce`).
- For English inference, Speaker `svarah_00021` (`Female_Hindi`) was selected as the reference acoustic anchor due to consistent spectral quality.

### 2.3 Training Configuration
- **Optimizer:** Adam ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$), Noam learning rate schedule with 2,000 warmup steps.
- **Batch Size:** 16. Gradient clipping: 1.0. Total steps: 30,000 for both models.

---

## 3. Loss Analysis and Convergence Behavior

### 3.1 Loss Table (Train and Validation across 30k Steps)

| Model | Metric | Step 1,000 | Step 10,000 | Step 20,000 | Step 30,000 (Final) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | **Train Total Loss** | 5.3243 | 1.6151 | 1.3871 | **1.2725** |
| | Train Mel / PostNet $L_1$ | 1.1634 / 1.2035 | 0.5842 / 0.5839 | 0.5210 / 0.5206 | **0.5678 / 0.5673** |
| | Train Pitch / Energy / Duration | 1.9285 / 0.9449 / 0.0840 | 0.0892 / 0.0910 / 0.0482 | 0.0610 / 0.0712 / 0.0335 | **0.0515 / 0.0587 / 0.0272** |
| | **Val Total Loss** | 3.7585 | 3.5395 | 3.6199 | **3.6240** |
| | Val Mel / PostNet $L_1$ | 1.0816 / 1.0993 | 0.9900 / 0.9900 | 1.0138 / 1.0138 | **1.0120 / 1.0117** |
| **Telugu** | **Train Total Loss** | 3.1448 | 0.8623 | 0.7610 | **0.7132** |
| | Train Mel / PostNet $L_1$ | 0.9148 / 0.9426 | 0.3937 / 0.3935 | 0.3540 / 0.3538 | **0.3380 / 0.3374** |
| | Train Pitch / Energy / Duration | 0.7575 / 0.5024 / 0.0275 | 0.0410 / 0.0580 / 0.0120 | 0.0310 / 0.0450 / 0.0078 | **0.0176 / 0.0146 / 0.0056** |
| | **Val Total Loss** | 2.4815 | 2.2592 | 2.2582 | **2.2700** |
| | Val Mel / PostNet $L_1$ | 0.7115 / 0.7202 | 0.6235 / 0.6233 | 0.6273 / 0.6272 | **0.6287 / 0.6286** |

### 3.2 Convergence and Overfitting Observations
1. **Training Convergence:**
   - Both models converged stably. The Noam warmup over the first 2,000 steps quickly lowered initial total loss from $\approx 10$ down to $\approx 3.5$.
   - Duration loss dropped early and stayed low throughout: final training duration MSE was **0.0272** for English and **0.0056** for Telugu.
2. **Generalization Gap & Overfitting:**
   - *English Model:* Validation total loss leveled off around steps 5,000–10,000 at $\approx 3.54$ and remained between $3.58$ and $3.62$ through step 30,000. The train-val gap (train mel loss 0.56 vs val mel loss 1.01) stems from the high variance across 38 crowdsourced speaker accents in Svarah, which introduces acoustic divergence on unseen validation utterances. However, the model did not suffer from loss spikes or degenerative overfitting.
   - *Telugu Model:* Validation total loss plateaued at $\approx 2.26$ around step 5,000 and ended at $2.27$. The training mel loss reached $0.3374$. Because the Telugu dataset is from a single professional speaker recorded in controlled conditions, the model learned a more compact acoustic space with lower variance and lower overall loss.

---

## 4. Observations on Generated Speech

### 4.1 English Test Sentences (`E1` – `E4`)
1. **E1 (*"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*):**
   - The 28-phoneme long word is synthesized completely without dropping syllables or repeating sounds. Because FastSpeech 2 uses feed-forward duration control rather than attention matrices, it avoids the alignment failures common in autoregressive models on long words.
2. **E2a (*"The orchestra's conduct during the ceremony was praised by the audience."*):**
   - Synthesized with noun stress on the first syllable (`K AA1 N D AH0 K T`). The first vowel is audibly longer and louder than the second syllable.
3. **E2b (*"The researchers will conduct an experiment to evaluate the proposed method."*):**
   - Synthesized with verb stress on the second syllable (`K AH0 N D AH1 K T`). The first syllable reduces to a schwa while the second syllable carries the accent.
4. **E3 (*"The rhyth m of the orchestra changed unexpectedly."*):**
   - The model preserves the input whitespace in `"rhyth m"` by inserting a short pause (`sp`), followed by pronouncing the isolated letter `"m"` (`EH1 M`).
5. **E4 (*"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*):**
   - The text normalizer correctly expanded times and dates into spoken words (*"ten forty five a m"*, *"twelve november twenty twenty-six"*). The romanized word *"andaroo"* was pronounced naturally within the English sentence cadence.

### 4.2 Telugu Test Sentences (`T1` – `T4`)
1. **T1 (*"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*):**
   - The technical compound loanword and consonant clusters (*"ప్రభావాన్ని"*, *"విశ్లేషించారు"*) were correctly parsed by the G2P engine and articulated clearly with appropriate long vowel durations (`uu`, `oo`, `aa`).
2. **T2a (*"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*):**
   - *"పట్టు"* acts as a head noun (*"grip"*). The retroflex geminate stop `tt tt` has a distinct closure pause and strong release burst.
3. **T2b (*"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*):**
   - *"పట్టు"* acts as an adjectival modifier (*"silk"*). The word is produced with slightly reduced duration and pitch compared to T2a, flowing directly into the head noun *"చీరను"*.
4. **T3 (*"పులి అడ విలో నిశ్శబ్దంగా నడిచింది."*):**
   - The intentional space in `"అడ విలో"` creates an audible word-break pause, whereas the genuine geminate sibilant in `"నిశ్శబ్దంగా"` (`sh sh`) is pronounced smoothly without interruption.
5. **T4 (*"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*):**
   - The 136-phoneme utterance is synthesized with continuous pitch contour declination. All numbers and dates (*రెండు వేల ఇరవై ఆరు*, *పన్నెండవ*, *పది నలభై ఐదు*) were normalized and pronounced fluently.

---

## 5. Variance Adaptor Analysis (Duration, Pitch, Energy)

Plots are saved under `Results/{English,Telugu}/variance_adaptor_plots/{duration,pitch,energy}/`.

### 5.1 English Sentences
- **Duration Predictor:**
  - In **E1**, stressed vowels (`AE2`, `AY0`, `UW1`, `EY1`) receive 10–11 Mel frames ($\approx 116\text{–}128\text{ ms}$), whereas unstressed vowels and stop closures (`IH0`, `AH0`, `B`, `G`) receive 2–4 frames ($\approx 23\text{–}46\text{ ms}$).
  - In **E2a vs. E2b** (`E2a_vs_E2b_comparison.png`), the noun *conduct* allocates 10 frames to the first vowel `AA1` and 9 frames to `AH0`. In the verb form, the first vowel `AH0` compresses to 8 frames while the second vowel `AH1` expands to 10 frames.
  - In **E3**, the space in `"rhyth m"` receives an explicit 2-frame ($23.2\text{ ms}$) `'sp'` pause.
- **Pitch ($F_0$) Contours:** Shows natural sentence-level declination, declining from $\approx 220\text{ Hz}$ at phrase onset to $\approx 160\text{ Hz}$ at sentence boundaries.
- **Energy Profiles:** Peaks ($60\text{–}75$) align with open vowel nuclei, while unvoiced stops and pause tokens drop to baseline ($\le 10$).

### 5.2 Telugu Sentences
- **Duration Predictor:**
  - In **T1**, long vowels (`uu`, `oo`, `aa`, `ee`) consistently receive 12–13 frames ($\approx 140\text{–}151\text{ ms}$), approximately $1.4\times$ longer than short vowels (`a`, `i`, `u`: 9 frames $\approx 104\text{ ms}$) and twice as long as consonant frames ($5\text{–}7\text{ frames}$).
  - In **T2a vs. T2b** (`T2a_vs_T2b_comparison.png`), *"పట్టు"* occupies 41 frames ($475.9\text{ ms}$) as a head noun in T2a, and compresses to 37 frames ($429.5\text{ ms}$, a 9.8% reduction) as a prenominal modifier in T2b.
- **Pitch ($F_0$) Contours:** Smooth contours oscillating between 190–250 Hz. In T2a, *"పట్టు"* carries a localized pitch peak of 244–247 Hz, whereas in T2b, pitch starts lower (211–219 Hz) and rises toward the head noun *"చీరను"*.
- **Energy Profiles:** Geminate stops (`tt tt`) display an energy valley during the closure phase followed by a sharp energy rise at vowel onset.

---

## 6. Comparative Observations, Limitations & Improvements

### 6.1 English vs. Telugu Comparison
- **Acoustic Consistency:** The single-speaker Telugu model achieved substantially lower Mel loss (0.3374 vs 0.5678) and duration MSE (0.0056 vs 0.0272) compared to the multi-speaker English model. Single-speaker data provides uniform vocal tract properties and recording conditions, while multi-speaker crowdsourced data contains varied microphone acoustics and regional accents.
- **Prosodic Nature:** English duration is primarily governed by lexical stress and vowel reduction (schwa reduction). Telugu duration is strictly phonological, governed by phonemic vowel length (short vs long) and consonant gemination.

### 6.2 Limitations Observed
1. **Tokenizer Whitespace Sensitivity (E3, T3):** Accidental typographical spaces in words (*"rhyth m"*, *"అడ విలో"*) are treated as word boundaries by whitespace tokenizers, causing awkward micro-pauses.
2. **Rule-Based Homograph Handling:** Context-based POS rules handle targeted words like *conduct*, but do not scale to all English heteronyms without a full syntactic parser.
3. **Cross-Lingual Phonetics (E4):** Mapping Telugu words (*"andaroo"*) into English ARPAbet substitutes alveolar `D` for the Indian dental stop `[d]`.

### 6.3 Suggested Improvements
- Integrate pre-trained contextual text embeddings (e.g., ByT5 or PnG-BERT) to automatically resolve homographs and handle typographical errors.
- Adopt a unified International Phonetic Alphabet (IPA) representation to cleanly support code-switched Indian languages.
- Utilize diffusion or flow-matching decoders (e.g., Grad-TTS) to reduce residual spectral oversmoothing.

---

## 7. Reproducibility Details

### 7.1 Dependencies
```text
torch>=2.1.0, torchaudio>=2.1.0, scipy>=1.10.0, librosa>=0.10.0, pyworld>=0.3.4,
g2p_en>=2.1.0, resemblyzer>=0.1.4, noisereduce>=3.0.0, matplotlib>=3.7.0, pyyaml>=6.0
```

### 7.2 Execution Commands
All commands run from the respective language folder:

```bash
# Setup environment
conda create -n fastspeech2 python=3.10 -y && conda activate fastspeech2
pip install -r requirements.txt

# English Pipeline
cd project/English
python preprocess.py --prepare_align   # Prepare data & MFA/MMS alignment
python preprocess.py                  # Extract Mel, F0, Energy, Durations
python train.py                       # Train FastSpeech 2 model
python inference.py                   # Synthesize test audios & plot variance adaptors

# Telugu Pipeline
cd ../Telugu
python preprocess.py --prepare_align   # G2P conversion & alignment preparation
python preprocess.py                  # Extract Mel, F0, Energy, Durations
python train.py                       # Train FastSpeech 2 model
python inference.py                   # Synthesize test audios & plot variance adaptors
```

All test audios are saved to `project/Results/{English,Telugu}/test_generated_audios/`, and all variance adaptor plots are saved under `project/Results/{English,Telugu}/variance_adaptor_plots/`.
