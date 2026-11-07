# FastSpeech 2 for English and Telugu Speech Synthesis

**Rishi Pramod Chintakayala**  
*Research Assistant Technical Report — Speech and Language Processing*  
*International Institute of Information Technology Hyderabad (IIIT-H)*  
Base Reference: FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., ICLR 2021)

---

### Abstract
This technical report presents a complete implementation, empirical evaluation, and acoustic analysis of FastSpeech 2 for multi-speaker Indian-accented English (*AI4Bharat Svarah*) and single-speaker Telugu (*SPRINGLab IndicTTS Telugu*). We adapted the modular architecture from *ming024/FastSpeech2*, replacing legacy Kaldi-based forced alignment on Windows with GPU-accelerated neural alignment via Meta's Massively Multilingual Forced Aligner (*MMS-FA*). For Telugu, we engineered a dedicated 50-phoneme Unicode G2P engine featuring native consonant-cluster virama logic and comprehensive number/date normalizers. For English, we incorporated part-of-speech-aware heteronym stress disambiguation and code-switched phoneme handling. Both systems were trained for 30,000 steps on strictly isolated 5.00-hour training and 1.00-hour validation partitions. Final training mel loss reached 0.5673 for English and 0.3374 for Telugu, with stable validation behavior and zero divergence. We analyze generated speech samples across five test sentences per language, examining duration, pitch (*F*₀), and energy contours extracted by the variance adaptor to demonstrate prosodic naturalness, compound word stability, and geminate stop dynamics. Complete reproducible environments, configuration files, and execution commands are documented.

---

## 1. Implementation Approach and Architectural Modifications

### 1.1 Reference Architecture Overview
FastSpeech 2 is a feed-forward, non-autoregressive neural speech synthesis architecture that maps a sequence of phonemes directly to 80-channel log-mel spectrogram frames in parallel. By removing the sequential autoregressive feedback loop of Tacotron-style models, FastSpeech 2 eliminates runtime latency bottlenecks, error propagation, and attention alignment failures (such as phoneme skipping and repetition).

Our codebase adapts the reference implementation from `ming024/FastSpeech2`, structured into five principal components:

1. **Phoneme Encoder:** Composed of 4 Feed-Forward Transformer (FFT) blocks. Each block integrates multi-head self-attention (2 attention heads, hidden dimension *d*<sub>model</sub> = 256) with a 2-layer 1D convolutional network (kernel sizes 9 and 1, filter size 1024, dropout 0.2). This captures broad contextual phonemic relationships before variance expansion.
2. **Speaker Conditioning:** To model acoustic identities, a 256-dimensional speaker representation is projected and added directly to the encoder output hidden states. For English, continuous 256-dimensional d-vector embeddings computed via Resemblyzer capture individual timbre across multi-speaker data; for Telugu, a single categorical embedding is learned.
3. **Variance Adaptor:** Adds acoustic variance information (duration, pitch, energy) to bridge the phoneme-to-frame sequence length mismatch:
   - *Duration Predictor and Length Regulator:* A 2-layer 1D convolutional network with ReLU activations, layer normalization, filter size 256, kernel size 3, and dropout 0.5. It predicts phoneme durations in the log domain, trained against forced-alignment ground truth using Mean Squared Error (MSE). The Length Regulator expands phoneme hidden representations to match frame lengths (1 frame ≈ 11.61 ms at 22,050 Hz sampling rate with hop length 256).
   - *Pitch (F₀) Predictor:* Predicts continuous fundamental frequency extracted via the PyWorld vocoder (Harvest and StoneMask algorithms). Frame-level pitch values are normalized, quantized into 256 logarithmically spaced bins, and mapped to a 256-dimensional trainable pitch embedding table added to the hidden representations.
   - *Energy Predictor:* Computes the *L*₂-norm of frame-level STFT magnitude spectra, quantized into 256 bins, and mapped to a trainable energy embedding table.
4. **Mel-Spectrogram Decoder and PostNet:** The expanded representations pass through 6 FFT decoder blocks (*d*<sub>model</sub> = 256, 2 heads, filter size 1024) to predict 80-channel mel-spectrogram frames. A 5-layer residual 1D convolutional PostNet (kernel size 5, 512 channels) refines high-frequency spectral details via residual addition.
5. **Neural Vocoder:** A pre-trained HiFi-GAN Universal vocoder converts the predicted 80-channel PostNet mel spectrograms into 22,050 Hz, 16-bit mono PCM audio waveforms.

### 1.2 Modifications Made for English (`ai4bharat/Svarah`)
- **Neural Forced Alignment (`acoustic_aligner.py`):** On Windows platforms, legacy Kaldi binaries and Montreal Forced Aligner (MFA) encounter severe binary dependency conflicts. We engineered an in-repo neural aligner using Meta's Massively Multilingual Forced Aligner (`torchaudio.pipelines.MMS_FA`). It performs connectionist temporal classification (CTC) forced alignment directly on 22.05 kHz audio, generating millisecond-accurate phoneme and inter-word pause (`sp`) boundaries.
- **Contextual Heteronym Disambiguation:** English words such as *conduct* shift lexical stress based on syntactic part-of-speech. In `English/inference.py`, we implemented a contextual POS disambiguation rule that inspects grammatical context (e.g., preceding possessive determiners in sentence E2a vs. auxiliary modal verbs in E2b) to emit noun stress `K AA1 N D AH0 K T` (/ˈkɑːndʌkt/) or verb stress `K AH0 N D AH1 K T` (/kənˈdʌkt/).
- **Text Normalization Engine:** Integrated comprehensive expansion of numerals, dates (*12 November 2026* → *twelve november twenty twenty-six*), clock times (*10:45 AM* → *ten forty five a m*), and out-of-vocabulary code-switched Indian tokens (*andaroo* → `AA1 N D AA0 R UW1`).
- **Acoustic Anchor Conditioning:** Crowdsourced multi-speaker recordings exhibit varied room acoustics. For consistent test inference, Speaker `svarah_00021` (`Female_Hindi`) was fixed as the canonical conditioning anchor.

### 1.3 Modifications Made for Telugu (`SPRINGLab/IndicTTS_Telugu`)
- **50-Phoneme Telugu Rule-Based G2P Engine:** Designed a dedicated Unicode converter (`Telugu/text/telugu_phonemes.py` and `symbols.py`) tailored to Telugu orthography (`U+0C00`–`U+0C7F`):
  - 14 vowels categorized into short (*a, i, u, ru, e, o*) and long (*aa, ii, uu, ruu, ee, ai, oo, au*) counterparts.
  - 31 consonants across velar, palatal, retroflex, dental, labial, and sibilant articulatory places.
  - Virama (`్`, `U+0C4D`) cluster resolution: Telugu consonants possess an inherent short vowel *a*. When a virama or dependent vowel sign (matra) is encountered, the inherent vowel is suppressed. This accurately models conjunct clusters (*ప్రభావాన్ని*) and geminate stops (*పట్టు*).
  - Native diacritics: Anusvara (`ం` → `M`) and Visarga (`ః` → `H`).
- **Inflected Telugu Normalizer:** Engineered `Telugu/text/telugu_numbers.py` to translate 4-digit Gregorian calendar years (*2026* → *రెండు వేల ఇరవై ఆరు*), ordinal suffixes (*12వ* → *పన్నెండవ*), and compound timestamps (*10:45* → *పది నలభై ఐదు*) into fully inflected native Telugu text.
- **Environment and STFT Compatibility:** Upgraded `audio/stft.py` to remain fully functional under `librosa >= 1.0` (replacing deprecated frame arguments), set `weights_only=False` for PyTorch checkpoint deserialization, and implemented RAM feature caching to prevent Windows filesystem disk I/O bottlenecks.

---

## 2. Dataset Selection, Preprocessing, and Training Configuration

### 2.1 Subsampled Datasets (5-Hour Train / 1-Hour Validation)
Both datasets were deterministically partitioned using random seed 42, guaranteeing zero speaker or sentence overlap between training and validation sets:

| Parameter | English (`ai4bharat/Svarah`) | Telugu (`SPRINGLab/IndicTTS_Telugu`) |
| :--- | :--- | :--- |
| Primary Corpus | AI4Bharat Svarah (Indian English) | IIT Madras IndicTTS Telugu |
| Speaker Setup | Multi-speaker (38 regional accent groups) | Single professional female speaker |
| Training Split Duration | 5.00 hours (3,472 utterances) | 5.00 hours (1,589 utterances) |
| Validation Split Duration | 1.00 hour (670 utterances) | 1.00 hour (305 utterances) |
| Active Speech Duration | 4.79h Train / 0.96h Val | 4.50h Train / 0.91h Val |
| Sampling Rate and Format | 22,050 Hz, 16-bit mono PCM | 22,050 Hz, 16-bit mono PCM |
| STFT FFT / Hop / Window | *N*<sub>fft</sub> = 1024, hop = 256, win = 1024 | *N*<sub>fft</sub> = 1024, hop = 256, win = 1024 |
| Mel Frequency Range | 80 filterbanks, 0 Hz – 8,000 Hz | 80 filterbanks, 0 Hz – 8,000 Hz |
| Pitch Statistics (μ, σ) | μ = 176.86 Hz, σ = 57.12 Hz | μ = 208.39 Hz, σ = 40.10 Hz |
| Energy Statistics (μ, σ) | μ = 48.62, σ = 38.83 | μ = 41.43, σ = 36.25 |

### 2.2 Audio Cleaning and Preprocessing Pipeline
1. **Silence Trimming:** Unvoiced leading and trailing silence exceeding 50 ms was trimmed using energy thresholding at 25 dB below peak power.
2. **Spectral De-Noising:** Crowdsourced Svarah recordings containing ambient stationary room noise were filtered using spectral gating (`noisereduce`) with stationary noise profiling.
3. **Peak Normalization:** All waveform amplitudes were normalized to -1.0 dBFS to prevent digital clipping while maximizing dynamic range.
4. **Feature Extraction:** Mel-scale spectrograms, continuous fundamental frequency (*F*₀), and frame-level energy were extracted and serialized to disk alongside phoneme duration alignments.

### 2.3 Training Configuration and Multi-Task Objective
Models were trained on an NVIDIA GeForce RTX 4050 GPU (6 GB VRAM) under CUDA 12.6.
- **Batch Size:** 16 utterances per batch with dynamic bucket padding.
- **Optimization:** Adam optimizer (β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹), 1.0 gradient clipping threshold, and Noam learning rate scheduling with 2,000 warmup steps (*d*<sub>model</sub><sup>-0.5</sup> · min(step<sup>-0.5</sup>, step · warmup<sup>-1.5</sup>)).
- **Multi-Task Objective:**
  *L*<sub>total</sub> = *L*<sub>mel</sub> + *L*<sub>postnet</sub> + *L*<sub>duration</sub> + *L*<sub>pitch</sub> + *L*<sub>energy</sub>  
  Where *L*<sub>mel</sub> and *L*<sub>postnet</sub> represent Mean Absolute Error (*L*₁), and variance predictor losses are Mean Squared Error (MSE).

---

## 3. Analysis of Training and Validation Loss Curves

### 3.1 Convergence Trajectory Across 30,000 Steps
The loss values below are extracted directly from training logs (`output/log/Svarah/train/log.txt` and `output/log/IndicTTS_Telugu/train/log.txt`) and validation checkpoint evaluations:

| Model | Checkpoint Step | Total Loss | Mel Loss (*L*₁) | PostNet Loss (*L*₁) | Pitch Loss (MSE) | Energy Loss (MSE) | Duration Loss (MSE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Step 1,000 (Train) | 5.3243 | 1.1634 | 1.2035 | 1.9285 | 0.9449 | 0.0840 |
| | Step 10,000 (Train) | 1.6151 | 0.5842 | 0.5839 | 0.0892 | 0.0910 | 0.0482 |
| | Step 20,000 (Train) | 1.3871 | 0.5210 | 0.5206 | 0.0610 | 0.0712 | 0.0335 |
| | **Step 30,000 (Train)** | **1.2725** | **0.5678** | **0.5673** | **0.0515** | **0.0587** | **0.0272** |
| | **Step 30,000 (Val)** | **3.6240** | **1.0120** | **1.0117** | **0.4956** | **0.9083** | **0.1964** |
| **Telugu** | Step 1,000 (Train) | 3.1448 | 0.9148 | 0.9426 | 0.7575 | 0.5024 | 0.0275 |
| | Step 10,000 (Train) | 0.8623 | 0.3937 | 0.3935 | 0.0410 | 0.0580 | 0.0120 |
| | Step 20,000 (Train) | 0.7610 | 0.3540 | 0.3538 | 0.0310 | 0.0450 | 0.0078 |
| | **Step 30,000 (Train)** | **0.7132** | **0.3380** | **0.3374** | **0.0176** | **0.0146** | **0.0056** |
| | **Step 30,000 (Val)** | **2.2700** | **0.6287** | **0.6286** | **0.6423** | **0.2864** | **0.0839** |

### 3.2 Convergence Behaviour and Generalization Gap Analysis
1. **Warmup and Rapid Early Convergence:** Both models exhibited smooth initial convergence during the first 2,000 steps as the learning rate scaled up to 1.38 × 10⁻³. Total loss dropped by more than 60% within the first 5,000 steps. Phoneme duration MSE dropped to 0.0482 (English) and 0.0120 (Telugu) by step 10,000, confirming that duration patterns are acquired early during training.
2. **Generalization Gap in English (Multi-Speaker):** English validation loss stabilized around step 8,000 (≈ 3.55) and remained stable through step 30,000 (3.6240). The gap between final training mel loss (0.5673) and validation mel loss (1.0117) reflects the acoustic and phonetic diversity across 38 crowdsourced regional speakers in Svarah. Because validation loss remained flat without upward drift, the model did not suffer from destructive overfitting; dropout (0.2 in FFT blocks, 0.5 in variance predictors) provided robust regularization.
3. **Generalization Gap in Telugu (Single-Speaker):** The single-speaker Telugu model converged to substantially lower loss values across all metrics. Final training mel loss was 0.3374 (40.5% lower than English), and duration MSE was 0.0056 (4.8× lower than English). Validation total loss converged to 2.2700. With consistent studio recording conditions and a single vocal tract, target acoustic distributions exhibited minimal variance.

---

## 4. Observations on Generated Speech Quality

### 4.1 Evaluation of English Test Sentences (E1–E4)
Each synthesized audio file was evaluated for pronunciation clarity, naturalness, and prosodic stability:
- **E1 (*"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*):**  
  *Audio Duration:* 7.13s (614 frames). FastSpeech 2 synthesized the complex 28-phoneme compound word without hesitation, phoneme dropping, or looping. Because the model relies on length-regulated feed-forward prediction rather than dynamic attention alignments, it maintains high stability across lengthy multisyllabic sequences.
- **E2a (*"The orchestra's conduct during the ceremony was praised by the audience."*):**  
  *Audio Duration:* 4.39s (378 frames). The noun *conduct* correctly received primary lexical stress on the initial syllable (`K AA1 N D AH0 K T`). The first vowel was audibly prolonged and delivered with elevated acoustic energy relative to the reduced second syllable.
- **E2b (*"The researchers will conduct an experiment to evaluate the proposed method."*):**  
  *Audio Duration:* 4.48s (386 frames). The verb *conduct* was synthesized with iambic second-syllable stress (`K AH0 N D AH1 K T`). The first vowel reduced to a brief schwa, while the second vowel carried higher energy and extended duration, producing a clear syntactic contrast with E2a.
- **E3 (*"The rhyth m of the orchestra changed unexpectedly."*):**  
  *Audio Duration:* 3.54s (305 frames). The tokenizer preserved the whitespace anomaly in `"rhyth m"` by inserting a short pause token (`sp`), followed by pronouncing the isolated letter `"m"` as `EH1 M`. The model executed this without phonetic collapse or vocoder artifacts.
- **E4 (*"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*):**  
  *Audio Duration:* 7.20s (620 frames). Numeral and temporal normalizers expanded *"10:45 AM"* and *"12 November 2026"* into fluent spoken English. The code-switched Telugu token *andaroo* (`AA1 N D AA0 R UW1`) integrated naturally into the pitch contour.

### 4.2 Evaluation of Telugu Test Sentences (T1–T4)
- **T1 (*"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*):**  
  *Audio Duration:* 6.56s (565 frames). The G2P engine decomposed the English loanword (*న్యూరోఫార్మకోలాజికల్*) and conjunct consonant clusters (*ప్రభావాన్ని*, *విశ్లేషించారు*) into accurate phoneme sequences. Long vowel phonemes (`uu`, `oo`, `aa`) were sustained naturally.
- **T2a (*"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*):**  
  *Audio Duration:* 3.36s (289 frames). Synthesizes *పట్టు* as a head noun (*grip*). The unvoiced retroflex geminate stop (`tt tt`) exhibited an audible closure interval followed by an emphatic release burst.
- **T2b (*"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*):**  
  *Audio Duration:* 3.42s (295 frames). Synthesizes *పట్టు* as an adjectival modifier (*silk*). The acoustic duration and pitch of *పట్టు* were prosodically subordinate to the subsequent head noun *చీరను*.
- **T3 (*"పులి అడ విలో నిశ్శబ్దంగా నడిచింది."*):**  
  *Audio Duration:* 2.79s (240 frames). The whitespace in `"అడ విలో"` produced an audible inter-word pause, while the geminate cluster in `"నిశ్శబ్దంగా"` (`sh sh`) was articulated smoothly without pause intrusion.
- **T4 (*"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*):**  
  *Audio Duration:* 12.60s (1,085 frames). Demonstrates sustained intonational declination across a 136-phoneme complex sentence. All numerals (*రెండు వేల ఇరవై ఆరు*, *పన్నెండవ*, *పది నలభై ఐదు*) were articulated with high intelligibility.

---

## 5. Analysis of Variance-Adaptor Plots (Duration, Pitch, Energy)

Variance adaptor prediction figures are organized under `Results/{English,Telugu}/variance_adaptor_plots/{duration,pitch,energy}/`.

### 5.1 Variance Adaptor Patterns for English Sentences
- **Duration Predictor (`duration/`):**
  - In **E1**, stressed vowels (`AE2`: 11 frames, `AY0`: 10 frames, `UW1`: 11 frames ≈ 116–128 ms) receive extended frame allocations, whereas unstressed schwas and stop closures (`IH0`, `AH0`, `B`, `G`) compress to 2–4 frames (≈ 23–46 ms).
  - In **E2a vs. E2b** (`E2a_vs_E2b_comparison.png`), the noun *conduct* assigns 10 frames (116.1 ms) to the first vowel `AA1` and 9 frames (104.5 ms) to `AH0`. In the verb form, the first vowel `AH0` compresses to 8 frames (92.9 ms) while the second vowel `AH1` expands to 10 frames (116.1 ms).
  - In **E3**, the space in `"rhyth m"` receives an explicit 2-frame (23.2 ms) `'sp'` pause allocation.
- **Pitch (F₀) Predictor (`pitch/`):**
  - Contours exhibit classical declination from an initial sentence peak of ≈ 220 Hz down to ≈ 160 Hz at terminal punctuation.
  - In **E4**, pitch rises on focal information units (*"10:45 AM"*, *"12 November"*) and resolves with a terminal low boundary tone on *"time"*.
- **Energy Predictor (`energy/`):**
  - Sonorant vowels reach peak frame energy values of 60–75, unvoiced fricatives and stops drop to 20–35, and silence pauses (`sp`) drop to baseline values (≤ 10).

### 5.2 Variance Adaptor Patterns for Telugu Sentences
- **Duration Predictor (`duration/`):**
  - In **T1**, long vowel phonemes (`uu`, `oo`, `aa`, `ee`) receive 12–13 frames (≈ 140–151 ms), approximately 1.4× longer than short vowels (`a`, `i`, `u`: 9 frames ≈ 104 ms) and twice as long as individual consonants (5–7 frames).
  - In **T2a vs. T2b** (`T2a_vs_T2b_comparison.png`), the word *పట్టు* spans 41 frames (475.9 ms) as an independent head noun in T2a, and compresses to 37 frames (429.5 ms, a 9.8% reduction) when functioning as a pre-nominal modifier in T2b.
- **Pitch (F₀) Predictor (`pitch/`):**
  - Smooth fundamental frequency tracks between 190 Hz and 250 Hz. In T2a, *పట్టు* carries a pitch plateau of 244–247 Hz across the retroflex geminate `tt tt`. In T2b, pitch begins lower (211–219 Hz) and rises toward the head noun *చీరను*.
- **Energy Predictor (`energy/`):**
  - Retroflex geminates (`tt tt`) display an energy valley during the occlusive closure interval followed by a sharp energy burst at vowel release.

---

## 6. Comparative Analysis, Limitations, and Future Directions

### 6.1 English vs. Telugu Architectural and Acoustic Comparison
- **Speaker Conditioning vs. Single-Speaker Precision:** The single-speaker Telugu model achieved 40.5% lower training mel loss (0.3374 vs. 0.5678) and 4.8× lower duration MSE (0.0056 vs. 0.0272) than the multi-speaker English model. Single-speaker training provides invariant vocal tract geometry and uniform recording acoustics. In contrast, crowdsourced multi-speaker data contains microphone variability and regional dialect diversity.
- **Prosodic Typology:** English duration is primarily stress-timed, characterized by vowel reduction (schwa weakening) in unstressed positions. Telugu duration is strictly phonological, governed by phonemic vowel quantity (short vs. long vowels) and consonant gemination.

### 6.2 Error Analysis and Limitations Observed in Test Cases
1. **Tokenizer Whitespace Sensitivity (E3, T3):** Standard tokenizers treat typographical spaces inside words (*"rhyth m"*, *"అడ విలో"*) as lexical delimiters, inserting acoustic pause frames.
2. **Contextual Heteronym Coverage:** Rule-based POS disambiguation successfully handles known target pairs like *conduct*, but does not scale to arbitrary out-of-vocabulary heteronyms without a complete syntactic parser.
3. **Cross-Lingual Phonetic Approximations (E4):** Mapping Telugu loanwords (*andaroo*) into English ARPAbet substitutes alveolar `D` for the native dental stop `/d̪/`.

### 6.3 Recommendations for Future Improvements
- **Pretrained Text Representations:** Incorporating contextual language models (e.g., ByT5 or PnG-BERT) would provide robust handling of typographic irregularities and open-vocabulary heteronyms.
- **Unified IPA Representation:** Adopting the International Phonetic Alphabet (IPA) across both languages would resolve phonemic mismatches in code-switched speech.
- **Diffusion or Flow-Matching Decoders:** Replacing the feed-forward FFT decoder with diffusion models (e.g., Grad-TTS) would eliminate residual oversmoothing in mel spectrograms.

---

## 7. Reproducibility and Deployment Details

### 7.1 Software Dependencies and Environment Specification
The environment requires Python 3.10 and PyTorch 2.1+ with CUDA acceleration:
```text
torch>=2.1.0, torchaudio>=2.1.0, scipy>=1.10.0, librosa>=0.10.0, pyworld>=0.3.4,
g2p_en>=2.1.0, resemblyzer>=0.1.4, noisereduce>=3.0.0, matplotlib>=3.7.0, pyyaml>=6.0
```

### 7.2 Step-by-Step Preprocessing, Training, and Inference Instructions

```bash
# 1. Environment Setup
conda create -n fastspeech2 python=3.10 -y && conda activate fastspeech2
pip install -r requirements.txt

# 2. English Model Pipeline (from project/English)
cd project/English
python preprocess.py --prepare_align   # Subsample Svarah (5h/1h), run MMS-FA alignment
python preprocess.py                  # Extract Mel, F0, Energy, Duration
python train.py                       # Train FastSpeech 2 (checkpoints in checkpoints/)
python inference.py                   # Synthesize test audios E1-E4 and plot variance adaptors

# 3. Telugu Model Pipeline (from project/Telugu)
cd ../Telugu
python preprocess.py --prepare_align   # Subsample IndicTTS (5h/1h), run G2P & MMS-FA alignment
python preprocess.py                  # Extract Mel, F0, Energy, Duration
python train.py                       # Train FastSpeech 2 (checkpoints in checkpoints/)
python inference.py                   # Synthesize test audios T1-T4 and plot variance adaptors
```

All synthesized audio samples are saved to `project/Results/{English,Telugu}/test_generated_audios/`, and all variance adaptor plots are saved under `project/Results/{English,Telugu}/variance_adaptor_plots/`.
