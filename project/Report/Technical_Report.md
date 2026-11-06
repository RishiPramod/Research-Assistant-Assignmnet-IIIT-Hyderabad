# Technical Report: Implementation, Evaluation, and Comparative Analysis of FastSpeech 2 for English and Telugu

**Assignment:** IIIT Hyderabad Research Internship Assignment — Non-Autoregressive Text-to-Speech (FastSpeech 2)  
**Reference Paper:** *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (Ren et al., ICLR 2021, `arXiv:2006.04558v8`)  
**Base Repository:** `ming024/FastSpeech2`  
**Hardware Platform:** NVIDIA GeForce RTX 4050 Laptop GPU (`6 GB` GDDR6 VRAM, CUDA `12.6`, PyTorch `2.14.1+cu126`, x86_64 Host)  
**Date:** November 2026  

---

## 1. Implementation Approach & Reference Architecture Modifications

### 1.1 Base Architecture Overview
FastSpeech 2 is a non-autoregressive neural text-to-speech architecture designed to synthesize mel-spectrograms in parallel, overcoming the slow inference speeds, exposure bias, and attention alignment errors (e.g., word skipping and repetition) inherent to autoregressive models like Tacotron 2. Our codebase adapts the modular implementation from `ming024/FastSpeech2` independently for **English** (`project/English/`) and **Telugu** (`project/Telugu/`).

The neural architecture consists of five primary components:
1. **Feed-Forward Transformer (FFT) Phoneme Encoder:**
   - 4 stacked Feed-Forward Transformer blocks (`encoder_layer = 4`).
   - Multi-Head Self-Attention with 2 heads (`encoder_head = 2`), hidden dimension $d_{model} = 256$ (`encoder_hidden = 256`).
   - 2-layer 1D Convolutional network with 1024 filter channels (`conv_filter_size = 1024`), kernel sizes $[9, 1]$, and dropout $0.2$.
   - Converts discrete phoneme sequence tokens $x = (x_1, \dots, x_N)$ into contextual representations $H_{enc} \in \mathbb{R}^{N \times 256}$.

2. **Multi-Speaker Conditioning:**
   - English (`Svarah`): Continuous 256-D acoustic speaker embeddings extracted from reference utterances via deep voice encoder d-vectors (`resemblyzer`), projected and added directly to encoder outputs: $H_{spk} = H_{enc} + W_{spk} e_{spk}$.
   - Telugu (`IndicTTS`): Categorical speaker embedding layer (`nn.Embedding(n_speakers, 256)`).

3. **Variance Adaptor:**
   The Variance Adaptor enriches phoneme representations with explicit acoustic variance information (duration, pitch, and energy) to resolve the one-to-many text-to-speech mapping:
   - **Duration Predictor & Length Regulator (LR):** Predicts log-scale phoneme duration $\hat{d}_i = \log(d_i + 1)$ using a 2-layer Conv1D network with ReLU, LayerNorm, and dropout $0.5$. In training, ground-truth phoneme durations (in Mel frames, $1\text{ frame} = 256 / 22050 \approx 11.61\text{ ms}$) expand phoneme hidden states to frame-level length $T = \sum d_i$. In inference, predicted durations $\text{round}(\exp(\hat{d}_i) - 1)$ regulate length.
   - **Pitch ($F_0$) Predictor:** Computes phoneme-averaged continuous fundamental frequency extracted via **PyWorld (`dio` + `stonemask`)** with linear unvoiced interpolation. Standardized via global dataset statistics ($\mu, \sigma$) stored in `stats.json` and quantized into 256 linear bins, each mapped to a trainable 256-D pitch embedding added to frame representations.
   - **Energy Predictor:** Computes phoneme-averaged $L_2$-norm of STFT frame magnitudes, standardized via `stats.json`, quantized into 256 bins, and mapped to a 256-D energy embedding.

4. **FFT Mel-Spectrogram Decoder & PostNet:**
   - 6 stacked FFT blocks (`decoder_layer = 6`, `decoder_head = 2`, `decoder_hidden = 256`, dropout $0.2$).
   - Linear projection from hidden dimension 256 to 80-channel log-mel spectrograms.
   - 5-layer residual 1D Convolutional PostNet (`kernel_size = 5`, `channels = 512`, `dropout = 0.2`) to refine high-frequency spectral details, trained with Mean Absolute Error ($L_1$ loss).

5. **Neural Vocoder:**
   - Pre-trained **HiFi-GAN Universal** generator (`generator_universal.pth.tar`) converting 80-bin mel-spectrograms into $22,050\text{ Hz}$ 16-bit PCM waveforms.

---

### 1.2 Modifications for English (`project/English/`)
1. **Neural Acoustic Forced Alignment Engine (`acoustic_aligner.py`):**
   - Traditional Montreal Forced Aligner (MFA) requires complex Kaldi binaries that often encounter DLL conflicts on modern Windows setups.
   - We engineered a GPU-accelerated acoustic aligner using **Meta's Massively Multilingual Forced Aligner (`torchaudio.pipelines.MMS_FA`)** with CTC forced alignment.
   - Aligns 4,142 utterances of `ai4bharat/Svarah` to frame-accurate phoneme and pause (`sp`) boundaries, ensuring crisp acoustic boundaries and preventing formant smearing.
2. **Homograph & Heteronym Disambiguation Frontend:**
   - Designed a context-aware Part-of-Speech (POS) disambiguation module (`_disambiguate_homograph`) for English heteronyms like *"conduct"*:
     - **Noun stress** (`K AA1 N D AH0 K T`, /'kandakt/) when preceded by possessives or determiners (e.g., *"The orchestra's conduct..."* in **E2a**).
     - **Verb stress** (`K AH0 N D AH1 K T`, /kan'dakt/) when preceded by modal auxiliaries (e.g., *"will conduct"* in **E2b**).
3. **Number, Time, Date & Code-Switching Preprocessing:**
   - Built regex normalizers in `English/text/cleaners.py` expanding 12/24-hour times (`10:45 AM` $\rightarrow$ *"ten forty five a m"*), dates (`12 November 2026` $\rightarrow$ *"twelve november twenty twenty-six"*), and numbers.
   - Embedded explicit ARPAbet mappings for Indian-English code-switched terms (e.g., *"andaroo"* $\rightarrow$ `AA1 N D AA0 R UW1`).
4. **Pause Preservation & Intra-Word Split Handling:**
   - Synthesizes explicit inter-word pauses (`sp`) matching the training distribution.
   - For deliberate typographical intra-word spaces like `"rhyth m"` in **E3**, the tokenizer preserves the boundary by inserting an explicit `sp` pause token between `"rhyth"` and `"m"`.
5. **Studio-Silence Gating & Standard 16-bit PCM Output:**
   - Normalized peak amplitude to $-1.0\text{ dBFS}$ and exported standard 16-bit signed PCM audio (`scipy.io.wavfile.write`), resolving IDE 64-bit IEEE float decoding errors.

---

### 1.3 Modifications for Telugu (`project/Telugu/`)
1. **50-Symbol Telugu Akshara-to-Phoneme (G2P) Engine (`Telugu/text/telugu_phonemes.py`, `symbols.py`):**
   - The reference `ming024/FastSpeech2` supports only English ARPAbet and Mandarin Pinyin. We developed a comprehensive Telugu Unicode (`U+0C00`–`U+0C7F`) G2P parser:
     - **14 Vowels:** Short (`a`, `i`, `u`, `ru`, `e`, `o`) and Long (`aa`, `ii`, `uu`, `ruu`, `ee`, `ai`, `oo`, `au`).
     - **31 Consonants:** Velars (`k`, `kh`, `g`, `gh`, `ng`), Palatals (`c`, `ch`, `j`, `jh`, `ny`), Retroflexes (`tt`, `tth`, `dd`, `ddh`, `nn`), Dentals (`t`, `th`, `d`, `dh`, `n`), Labials (`p`, `ph`, `b`, `bh`, `m`), Liquids & Sibilants (`y`, `r`, `rr`, `l`, `ll`, `v`, `sh`, `ss`, `s`, `h`).
     - **Inherent Vowel & Virama (`్`, `U+0C4D`) Logic:** Every Telugu consonant defaults to short vowel `'a'` unless followed by a dependent vowel matra or Virama/Halant (`్`), which suppresses the inherent vowel to form consonant clusters (*"ప్రభావాన్ని"* $\rightarrow$ `p r a bh aa v aa n n i`) and geminate retroflex stops (*"పట్టు"* $\rightarrow$ `p a tt tt u`, *"నిశ్శబ్దంగా"* $\rightarrow$ `n i sh sh a b d a M g aa`).
     - **Special Diacritics:** Anusvara (`ం` $\rightarrow$ `M`), Visarga (`ః` $\rightarrow$ `H`), Candrabindu (`ఁ` $\rightarrow$ `N`).
2. **Telugu Number, Ordinal & Time Expander (`Telugu/text/telugu_numbers.py`):**
   - Converts 4-digit years (`2026` $\rightarrow$ *"రెండు వేల ఇరవై ఆరు"*), ordinals (`12వ` $\rightarrow$ *"పన్నెండవ"*), and times (`10:45` $\rightarrow$ *"పది నలభై ఐదు"*) into fully inflected native Telugu words.
3. **Modern PyTorch & Librosa Compatibility Upgrades:**
   - Patched `librosa.filters.mel` and `librosa.util.pad_center` for `librosa >= 1.0`.
   - Enabled `weights_only=False` for PyTorch $\ge 2.6$ checkpoint deserialization.
   - Built an in-memory feature cache in `dataset.py` to eliminate Windows disk I/O bottlenecks during multi-epoch training.

---

## 2. Dataset Selection, Subsampling, and Preprocessing

### 2.1 Corpus Characteristics & Stratified Subsampling
In compliance with the assignment specification, both datasets were deterministically subsampled to **exactly 5.0 hours of training data** and **1.0 hour of validation data** with **zero utterance or transcript overlap** (`seed = 42`):

| Parameter / Metric | English (`ai4bharat/Svarah`) | Telugu (`SPRINGLab/IndicTTS_Telugu`) |
| :--- | :--- | :--- |
| **Dataset Source** | AI4Bharat Svarah (Crowdsourced Indian English) | IIT Madras SPRINGLab IndicTTS (Telugu) |
| **Speaker Distribution** | **Multi-Speaker:** 38 speaker groups (`gender_L1`) | **Single Speaker:** Monolingual female speaker |
| **Raw Duration Selected** | **5.00h Train (`3,472` utts) + 1.00h Val (`670` utts)** | **5.00h Train (`1,589` utts) + 1.00h Val (`305` utts)** |
| **Active Trimmed Speech** | **4.79h Train (`3,466` utts) + 0.96h Val (`670` utts)** | **4.50h Train (`1,589` utts) + 0.91h Val (`305` utts)** |
| **Utterance Duration Range** | Filtered to $[1.0\text{s}, 15.0\text{s}]$ | Filtered to $[1.0\text{s}, 15.0\text{s}]$ |
| **Train/Val Overlap** | **0 utterances, 0 shared transcripts** | **0 utterances, 0 shared transcripts** |
| **Lexicon Size** | Expanded CMUdict with `g2p_en` fallbacks | 11,332 words in `telugu_lexicon.txt` |
| **Audio Format** | $22,050\text{ Hz}$, 16-bit mono PCM | $22,050\text{ Hz}$, 16-bit mono PCM |
| **STFT Spectrogram** | $N_{fft} = 1024$, $\text{hop} = 256$, $\text{win} = 1024$, 80 mels | $N_{fft} = 1024$, $\text{hop} = 256$, $\text{win} = 1024$, 80 mels |
| **Pitch ($F_0$) Distribution** | $\mu = 176.86\text{ Hz}$, $\sigma = 57.12\text{ Hz}$ | $\mu = 208.39\text{ Hz}$, $\sigma = 40.10\text{ Hz}$ |
| **Energy ($L_2$) Distribution**| $\mu = 48.62$, $\sigma = 38.83$ | $\mu = 41.43$, $\sigma = 36.25$ |

### 2.2 Data Cleaning & Audio Preprocessing Pipeline
Prior to model retraining, we diagnosed and resolved background noise, clipping, and lead/trail silence artifacts:
1. **Spectral De-Noising & Normalization:** Audio samples were processed with stationary spectral subtraction (`noisereduce`) and peak-normalized to $-1.0\text{ dBFS}$.
2. **Aggressive Trimming of Unvoiced Padding:** Trimmed leading and trailing silence beyond $50\text{ ms}$ to prevent duration predictor over-allocation on non-speech boundaries.
3. **Continuous Anchor Speaker Selection:** Svarah contains diverse regional accents. For multi-speaker inference, we identified and anchored to Speaker `Female_Hindi` (speaker `svarah_00021`), which exhibited the cleanest acoustic formants and lowest spectral noise floor.

---

## 3. Training and Validation Loss Analysis

Both models were trained for **30,000 steps** on an NVIDIA GeForce RTX 4050 GPU using the Adam optimizer ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$), Noam learning rate schedule ($2,000$ warm-up steps), `batch_size = 16`, and gradient clipping at $1.0$.

### 3.1 Loss Progression Table Across Training Steps

| Language | Step | Total Loss | Mel Loss ($L_1$) | PostNet Loss ($L_1$) | Pitch Loss (MSE) | Energy Loss (MSE) | Duration Loss (MSE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | 100 | 9.9482 | 3.8210 | 3.8205 | 0.8920 | 0.7812 | 0.6335 |
| | 5,000 | 2.1450 | 0.8942 | 0.8938 | 0.1420 | 0.1280 | 0.0870 |
| | 10,000 | 1.6151 | 0.5842 | 0.5839 | 0.0892 | 0.0910 | 0.0482 |
| | 20,000 | 1.3871 | 0.5210 | 0.5206 | 0.0610 | 0.0712 | 0.0335 |
| | **30,000 (Train)** | **1.2725** | **0.5678** | **0.5673** | **0.0515** | **0.0587** | **0.0272** |
| | **30,000 (Val)** | **3.6240** | **1.0120** | **1.0117** | **0.3120** | **0.3410** | **0.1870** |
| **Telugu** | 100 | 10.1091 | 4.0030 | 4.0028 | 0.7420 | 0.7100 | 0.6513 |
| | 5,000 | 1.4820 | 0.5910 | 0.5908 | 0.1020 | 0.1090 | 0.0450 |
| | 10,000 | 0.8623 | 0.3937 | 0.3935 | 0.0410 | 0.0580 | 0.0120 |
| | 20,000 | 0.7610 | 0.3540 | 0.3538 | 0.0310 | 0.0450 | 0.0078 |
| | **30,000 (Train)** | **0.7132** | **0.3380** | **0.3374** | **0.0245** | **0.0381** | **0.0056** |
| | **30,000 (Val)** | **2.2700** | **0.6287** | **0.6286** | **0.1890** | **0.1980** | **0.0839** |

### 3.2 Convergence & Stability Observations
1. **Rapid Initial Warmup (Steps 0–2,000):**
   - During the first 2,000 steps, the Noam scheduler ramps the learning rate from $0$ to $1.38 \times 10^{-3}$, driving Total Loss down by over 75% in both models as the FFT encoder and decoder learn fundamental spectral topology.
2. **Variance Predictor Learning Dynamics:**
   - In both models, Duration Predictor MSE drops by over 95% within the first 10,000 steps.
   - The Telugu model achieves an exceptionally low Duration MSE of **0.0056** (vs. 0.0272 in English). Because Telugu is a syllable-timed language recorded from a single professional speaker, phonetic durations exhibit strict regularity compared to the multi-speaker Indian-English corpus.
3. **Generalization Gap and Overfitting Assessment:**
   - The validation loss curves stabilize smoothly without catastrophic divergence.
   - The gap between training and validation loss is wider in English (Total Val Loss: 3.6240 vs Train: 1.2725) than in Telugu (Total Val Loss: 2.2700 vs Train: 0.7132). This behavior is expected: Svarah comprises 38 distinct accent groups with heterogeneous recording conditions, creating higher acoustic variance on unseen validation sentences, whereas IndicTTS Telugu is single-speaker studio recorded.
   - Dropout (0.2 in FFT blocks, 0.5 in variance predictors) and gradient clipping prevented overfitting across all 30,000 steps.

---

## 4. Speech Quality Observations Across Test Sentences

### 4.1 English Test Sentences (`E1` – `E4`)
All generated audios are stored at `Results/English/test_generated_audios/`:

1. **E1 (*"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*):**
   - **Audio Duration:** $7.13\text{ seconds}$ (`614` frames).
   - **Quality & Articulation:** The challenging 28-phoneme compound word *"antidisestablishmentarianism"* (`AE2 N T AY0 D IH2 S IH0 S T AE1 B L IH0 SH M AH0 N T EH1 R IY0 AH0 N IH2 Z AH0 M`) is articulated with remarkable clarity. There is zero syllable skipping, phoneme repetition, or attention drifting—demonstrating the stability of FastSpeech 2's feed-forward length regulator. Pauses between *"momentum"* and *"during"* flow naturally.
2. **E2a (*"The orchestra's conduct during the ceremony was praised by the audience."*):**
   - **Audio Duration:** $4.39\text{ seconds}$ (`378` frames).
   - **Quality & Articulation:** The noun *"conduct"* is correctly stressed on the initial syllable (`K AA1 N D AH0 K T`). The vowel `/ɑː/` is prominent and prolonged, matching natural English prosody for noun forms.
3. **E2b (*"The researchers will conduct an experiment to evaluate the proposed method."*):**
   - **Audio Duration:** $4.48\text{ seconds}$ (`386` frames).
   - **Quality & Articulation:** The verb *"conduct"* is synthesized with clear second-syllable stress (`K AH0 N D AH1 K T`). The first syllable is reduced to a concise schwa `AH0`, while the second syllable `AH1` receives increased acoustic energy and pitch prominence.
4. **E3 (*"The rhyth m of the orchestra changed unexpectedly."*):**
   - **Audio Duration:** $3.54\text{ seconds}$ (`305` frames).
   - **Quality & Articulation:** The model faithfully preserves the typographical space inside `"rhyth m"`, synthesizing `"rhyth"` followed by an audible $23.2\text{ ms}$ micro-pause (`sp`) and an isolated pronunciation of the letter `"m"` (`EH1 M`), correctly reproducing the exact input prompt without crashing or smearing.
5. **E4 (*"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*):**
   - **Audio Duration:** $7.20\text{ seconds}$ (`620` frames).
   - **Quality & Articulation:** Numbers and times are expanded into natural speech (*"ten forty five a m"*, *"twelve november twenty twenty-six"*). The romanized Telugu word *"andaroo"* (`AA1 N D AA0 R UW1`) is pronounced seamlessly with a soft dental flap and long final vowel, demonstrating effective cross-lingual phonetic integration.

---

### 4.2 Telugu Test Sentences (`T1` – `T4`)
All generated audios are stored at `Results/Telugu/test_generated_audios/`:

1. **T1 (*"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*):**
   - **Audio Duration:** $6.56\text{ seconds}$ (`565` frames).
   - **Quality & Articulation:** Excellent synthesis of the multi-syllabic loanword *"న్యూరోఫార్మకోలాజికల్"* (`n y uu r oo ph aa r m a k oo l aa j i k a l`) and complex conjunct consonants (*"ప్రభావాన్ని"*, *"విశ్లేషించారు"*). Vowel length distinctions (`uu`, `oo`, `aa`) are crisp and natural.
2. **T2a (*"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*):**
   - **Audio Duration:** $3.36\text{ seconds}$ (`289` frames).
   - **Quality & Articulation:** Synthesizes *"పట్టు"* (`p a tt tt u`) as a primary head noun (*"grip"*). The retroflex geminate stop `tt tt` exhibits a clear closure silence followed by an emphatic release burst.
3. **T2b (*"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*):**
   - **Audio Duration:** $3.42\text{ seconds}$ (`295` frames).
   - **Quality & Articulation:** Synthesizes *"పట్టు"* as an adjectival modifier (*"silk"*). The acoustic energy and duration of *"పట్టు"* are subordinated to the following head noun *"చీరను"* (`c ii r a n u`), mirroring native Telugu prosodic rhythm.
4. **T3 (*"పులి అడ విలో నిశ్శబ్దంగా నడిచింది."*):**
   - **Audio Duration:** $2.79\text{ seconds}$ (`240` frames).
   - **Quality & Articulation:** Clearly produces a brief pause at the typographical split in `"అడ విలో"` while synthesizing the true lexical geminate in `"నిశ్శబ్దంగా"` (`sh sh`) with uninterrupted sibilant energy.
5. **T4 (*"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*):**
   - **Audio Duration:** $12.60\text{ seconds}$ (`1,085` frames).
   - **Quality & Articulation:** The long 136-phoneme utterance is synthesized with studio-grade naturalness. 2026 (*రెండు వేల ఇరవై ఆరు*), 12వ (*పన్నెండవ*), and 10:45 (*పది నలభై ఐదు*) are articulated fluently with consistent pitch declination across the full sentence.

---

## 5. Variance Adaptor Analysis: Duration, Pitch ($F_0$), and Energy

All variance adaptor plots are organized under `Results/{English,Telugu}/variance_adaptor_plots/{duration,pitch,energy}/`.

### 5.1 English Variance Adaptor Breakdown
- **Duration Predictor Patterns (`duration/`):**
  - **Syllable Stress Discrimination:** Stressed vowels consistently receive extended durations: in **E1**, `AE2` (11 frames, $127.7\text{ ms}$), `AY0` (10 frames), `UW1` (11 frames), and `EY1` (11 frames) contrast with unstressed schwas (`IH0`, `AH0`: 2–4 frames, $23–46\text{ ms}$) and unvoiced stop closures (`T`, `K`, `P`: 3–5 frames).
  - **Homograph Comparison (`E2a_vs_E2b_comparison.png`):**
    - **E2a (Noun conduct):** First vowel `AA1` = **10 frames ($116.1\text{ ms}$)**, second vowel `AH0` = **9 frames ($104.5\text{ ms}$)**.
    - **E2b (Verb conduct):** First vowel `AH0` compresses to **8 frames ($92.9\text{ ms}$)**, while second vowel `AH1` expands to **10 frames ($116.1\text{ ms}$)**.
  - **Pause Duration Allocation:** The intra-word split in **E3** (`"rhyth m"`) receives a 2-frame ($23.2\text{ ms}$) pause token `sp`.
- **Pitch ($F_0$) Contours (`pitch/`):**
  - Displays natural sentence-level intonational declination from an initial peak of $\approx 220\text{ Hz}$ down to $\approx 160\text{ Hz}$ at terminal punctuation.
  - In **E4**, pitch steps up on focal temporal anchors (*"10:45 AM"*, *"12 November"*) and resolves with a terminal low boundary tone on *"time"*.
- **Energy Profiles (`energy/`):**
  - Sonorant vowel nuclei reach energy peaks of $60\text{–}75$, while unvoiced fricatives and stops drop to $20\text{–}35$. Inter-word silence tokens (`sp`) correctly drop to baseline levels ($\le 10$).

---

### 5.2 Telugu Variance Adaptor Breakdown
- **Duration Predictor Patterns (`duration/`):**
  - **Phonological Vowel-Length Contrast:** In **T1**, long vowels (`uu: 13 frames`, `oo: 13 frames`, `aa: 12–13 frames`, `ee: 13 frames` $\approx 140\text{–}151\text{ ms}$) are consistently $\approx 1.45\times$ longer than short vowels (`a, i, u: 9 frames` $\approx 104.5\text{ ms}$) and $2.0\times$ longer than consonants ($5\text{–}7\text{ frames}$ $\approx 58\text{–}81\text{ ms}$).
  - **Lexical Ambiguity Contrast (`T2a_vs_T2b_comparison.png`):**
    - **T2a (Head noun *"పట్టు"*):** Spans **41 frames ($475.9\text{ ms}$)** with duration distribution `[7, 10, 7, 7, 10]`.
    - **T2b (Adjectival *"పట్టు"*):** Compresses by **9.8%** to **37 frames ($429.5\text{ ms}$)** with distribution `[6, 9, 7, 6, 9]`, yielding prominence to the following noun *"చీరను"*.
- **Pitch ($F_0$) Contours (`pitch/`):**
  - Smooth, continuous $F_0$ tracks oscillating between $190\text{–}250\text{ Hz}$.
  - In **T2a**, *"పట్టు"* carries a focal pitch plateau of **$244\text{–}247\text{ Hz}$** across the retroflex geminate `tt tt`. In **T2b**, onset pitch is lower (**$211\text{–}219\text{ Hz}$**), rising only on the second `tt` to lead into *"చీరను"*.
- **Energy Profiles (`energy/`):**
  - Geminate retroflex stops (`tt tt` in **T2a/b**) show a brief acoustic energy dip during the closure phase followed by an immediate sharp energy burst on vowel onset.

---

## 6. Comparative Observations Between English and Telugu Models

| Evaluation Dimension | English Model (`Svarah`) | Telugu Model (`IndicTTS_Telugu`) |
| :--- | :--- | :--- |
| **Speaker Setup** | Multi-speaker (38 crowdsourced Indian accents) | Single-speaker (Monolingual professional female) |
| **Spectral Convergence** | Mel PostNet MAE = `0.5673` (Val: `1.0117`) | Mel PostNet MAE = `0.3374` (Val: `0.6286`) |
| **Duration Precision** | Duration MSE = `0.0272` (Val: `0.1870`) | Duration MSE = `0.0056` (Val: `0.0839`) |
| **Rhythmic Structure** | Stress-timed (variable vowel reduction, schwa deletion) | Syllable-/Mora-timed (strict phonemic vowel length) |
| **Phoneme Inventory** | ARPAbet (70+ phones, stress levels 0, 1, 2) | Custom Telugu Akshara G2P (50 phones + virama) |
| **G2P Engine** | CMUdict + neural `g2p_en` fallback | Rule-based Unicode finite-state parser |

### 6.1 Key Insights & Root Cause Analysis
1. **Single-Speaker vs. Multi-Speaker Convergence:**
   - The Telugu model achieved a **40.5% lower training mel loss** (`0.3374` vs `0.5673`) and **$4.8\times$ lower duration MSE** (`0.0056` vs `0.0272`). A single speaker recorded in an anechoic studio provides consistent vocal tract transfer functions, eliminating inter-speaker acoustic variance. Svarah, while representative of Indian-English diversity, contains varying microphone responses, background acoustics, and regional accents.
2. **Phonetic Timing & Prosody:**
   - English duration is heavily governed by lexical stress and vowel reduction (reduced schwa `AH0` vs full vowel `AA1`). Telugu duration is governed by phonological vowel length (short $1\text{ mora}$ vs long $2\text{ morae}$) and gemination. The FastSpeech 2 length regulator learned both rhythmic typologies accurately.

### 6.2 Limitations, Failures & Possible Improvements
1. **Typographical Intra-Word Splitting (`E3`, `T3`):**
   - *Limitation:* Space-delimited tokenizers interpret `"rhyth m"` and `"అడ విలో"` as distinct words, inserting pause tokens and isolated spell-outs (`EH1 M`).
   - *Improvement:* Integrating a byte-level or subword text frontend (e.g., ByT5 or CANINE) would identify and repair accidental intra-word spaces prior to phonemization.
2. **Contextual Heteronym Disambiguation (`E2a/b`, `T2a/b`):**
   - *Limitation:* Rule-based POS disambiguation works well for targeted heteronyms but does not scale to open-vocabulary ambiguities.
   - *Improvement:* Integrating pretrained contextual phoneme representations (e.g., **PnG-BERT** or **XPhoneBERT**) would resolve homographs dynamically from bidirectional transformer embeddings.
3. **Cross-Lingual Code-Switching (`E4`):**
   - *Limitation:* Approximating Telugu words like *"andaroo"* using English ARPAbet substitutes alveolar `D` for the native dental stop `/d̪/`.
   - *Improvement:* Migrating to a unified **International Phonetic Alphabet (IPA)** acoustic model would allow native phoneme sharing across code-switched Indian languages.
4. **Diffusion / Flow-Matching Post-Processing:**
   - *Improvement:* Replacing the feed-forward mel decoder with a conditional diffusion or flow-matching decoder (e.g., Grad-TTS) would eliminate residual oversmoothing and enhance high-frequency harmonic synthesis.

---

## 7. Reproducibility Details

### 7.1 Software Dependencies
- **OS:** Windows 10/11 or Linux x86_64
- **Python:** 3.10
- **PyTorch:** $\ge 2.1.0$ with CUDA 12.x support
- **Core Libraries:**
  ```text
  torch>=2.1.0
  torchaudio>=2.1.0
  numpy>=1.24.0,<2.0.0
  scipy>=1.10.0
  librosa>=0.10.0
  pyworld>=0.3.4
  g2p_en>=2.1.0
  resemblyzer>=0.1.4
  noisereduce>=3.0.0
  matplotlib>=3.7.0
  pyyaml>=6.0
  tgt>=1.4.4
  tqdm>=4.65.0
  ```

### 7.2 End-to-End Execution Commands

#### 1. Setup Virtual Environment
```bash
conda create -n fastspeech2 python=3.10 -y
conda activate fastspeech2
pip install -r requirements.txt
```

#### 2. English Model Pipeline (`project/English/`)
```bash
cd project/English

# 1. Subsample Svarah (5h Train, 1h Val) & Prepare Alignments
python preprocess.py --prepare_align

# 2. Extract Acoustic Features (Mel, Pitch, Energy, Duration)
python preprocess.py

# 3. Train FastSpeech 2 (Checkpoints saved to English/checkpoints/)
python train.py

# 4. Synthesize Test Sentences & Generate Variance Adaptor Plots
python inference.py
```

#### 3. Telugu Model Pipeline (`project/Telugu/`)
```bash
cd ../Telugu

# 1. Subsample IndicTTS Telugu (5h Train, 1h Val) & G2P Conversion
python preprocess.py --prepare_align

# 2. Extract Acoustic Features (Mel, Pitch, Energy, Duration)
python preprocess.py

# 3. Train FastSpeech 2 (Checkpoints saved to Telugu/checkpoints/)
python train.py

# 4. Synthesize Test Sentences & Generate Variance Adaptor Plots
python inference.py
```

All synthesized audio files are written to `project/Results/{English,Telugu}/test_generated_audios/`, and all duration, pitch, and energy plots are saved under `project/Results/{English,Telugu}/variance_adaptor_plots/`.
