# FastSpeech 2 for English and Telugu Speech Synthesis

<div class="author-header">
<strong>Rishi Pramod Chintakayala</strong><br>
<em>Research Assistant Technical Report — Speech and Language Processing</em><br>
<em>International Institute of Information Technology Hyderabad (IIIT-H)</em><br>
Base Reference: FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., ICLR 2021)
</div>

---

### Abstract
This report presents our work on building and evaluating FastSpeech 2 models for Indian-accented English and Telugu. FastSpeech 2 is a non-autoregressive text-to-speech architecture that synthesizes speech in parallel by predicting acoustic features directly from phoneme sequences. For English, we utilized the multi-speaker AI4Bharat Svarah corpus, incorporating neural forced alignment, part-of-speech-aware heteronym stress handling, and comprehensive text normalization. For Telugu, we used the single-speaker IndicTTS corpus and developed a rule-based 50-phoneme grapheme-to-phoneme engine with support for native consonant clusters, virama suppression, and inflected numeral expansion. Both models were trained for 30,000 steps on strictly separated 5-hour training and 1-hour validation sets. In this report, we analyze the training and validation loss curves, evaluate the quality and naturalness of five test sentences for each language, study the predicted variance adaptor contours (duration, pitch, and energy), and discuss key differences and areas for future improvement.

---

## 1. Implementation Approach and Architectural Modifications

### 1.1 Reference Architecture Overview
FastSpeech 2 is a feed-forward neural text-to-speech architecture that generates mel spectrograms in parallel, avoiding the sequential generation bottleneck and phoneme-skipping issues common in autoregressive systems. Our work builds upon the modular implementation of FastSpeech 2, which consists of five main components:

1. **Phoneme Encoder:** Uses 4 Feed-Forward Transformer (FFT) blocks. Each block combines multi-head self-attention (2 heads, hidden dimension 256) with a 2-layer 1D convolutional network to produce contextual phoneme representations.
2. **Speaker Conditioning:** Adds a 256-dimensional speaker embedding to the encoder representations before the variance adaptor. For English, continuous speaker embeddings capture individual vocal characteristics across multiple speakers; for Telugu, a single categorical embedding is used.
3. **Variance Adaptor:** Bridges the temporal mismatch between the phoneme sequence and mel-spectrogram frame length:
   - **Duration Predictor and Length Regulator:** A 2-layer convolutional network predicting log phoneme durations trained against forced-alignment labels. The Length Regulator expands the phoneme representations to match frame lengths (where one frame represents approximately 11.6 ms at 22,050 Hz).
   - **Pitch (F₀) Predictor:** Predicts continuous fundamental frequency extracted from the speech waveform. Predicted values are quantized into 256 bins and mapped to a trainable pitch embedding table.
   - **Energy Predictor:** Predicts frame-level spectral energy, quantized into 256 bins and added as an embedding to the representations.
4. **Mel-Spectrogram Decoder and PostNet:** A 6-block FFT decoder predicts 80-channel log-mel spectrogram frames, followed by a 5-layer convolutional PostNet that refines high-frequency details using residual connections.
5. **Neural Vocoder:** A pre-trained HiFi-GAN vocoder converts the predicted 80-channel mel spectrograms into 22,050 Hz, 16-bit mono audio waveforms.

### 1.2 Modifications for English
- **Neural Forced Alignment:** Instead of relying on legacy forced alignment tools that require external system binaries, we integrated a neural forced aligner based on Meta's Massively Multilingual Speech (MMS) model. This performs alignment directly on the 22.05 kHz audio to obtain millisecond-accurate phoneme and pause boundaries.
- **Contextual Heteronym Disambiguation:** English words such as *conduct* shift stress depending on their grammatical role. We implemented a contextual rule that checks whether the word functions as a noun (e.g., following a possessive word) or as a verb (e.g., following an auxiliary modal), assigning either initial-syllable noun stress or second-syllable verb stress.
- **Text Normalization:** We added rules to expand calendar dates, clock times, quantities, and abbreviations into full spoken English words, and mapped borrowed Indian words into appropriate phoneme sequences.
- **Reference Acoustic Anchor:** Because the English dataset contains recordings from multiple speakers, we selected a consistent female speaker with neutral pronunciation as the reference anchor during evaluation to keep voice quality uniform.

### 1.3 Modifications for Telugu
- **Rule-Based 50-Phoneme G2P Engine:** Telugu uses an abugida script where consonants have an inherent vowel unless modified. We developed a Unicode-based phonetic converter covering:
  - 14 vowels (distinguishing short vowels from their long counterparts).
  - 31 consonants organized by articulatory place.
  - Native virama suppression logic: when a consonant is followed by a virama or a dependent vowel sign, the inherent vowel is suppressed. This correctly handles conjunct consonant clusters and geminate stops.
  - Diacritics representing nasalized sounds and aspiration.
- **Telugu Number and Date Normalization:** We built a dedicated normalizer that converts four-digit years, ordinal numbers, and timestamps into fully inflected Telugu word forms.
- **Audio Processing Upgrades:** We updated the spectral feature extraction routines for modern library compatibility and enabled in-memory feature caching to speed up data loading during training.

---

## 2. Dataset Selection, Preprocessing, and Training Configuration

### 2.1 Subsampled Datasets (5-Hour Train / 1-Hour Validation)
Both datasets were split with a fixed random seed (seed 42), ensuring zero sentence or speaker overlap between the training and validation sets:

| Parameter | English (AI4Bharat Svarah) | Telugu (IndicTTS Telugu) |
| :--- | :--- | :--- |
| Primary Corpus | AI4Bharat Svarah (Indian English) | IIT Madras IndicTTS Telugu |
| Speaker Setup | Multi-speaker (38 regional accent groups) | Single professional female speaker |
| Training Duration | 5.00 hours (3,472 utterances) | 5.00 hours (1,589 utterances) |
| Validation Duration | 1.00 hour (670 utterances) | 1.00 hour (305 utterances) |
| Active Speech Duration | 4.79 hours Train / 0.96 hours Val | 4.50 hours Train / 0.91 hours Val |
| Sampling Rate and Format | 22,050 Hz, 16-bit mono PCM | 22,050 Hz, 16-bit mono PCM |
| STFT Configuration | FFT = 1024, Hop = 256, Window = 1024 | FFT = 1024, Hop = 256, Window = 1024 |
| Mel Filterbanks | 80 channels, 0 Hz to 8,000 Hz | 80 channels, 0 Hz to 8,000 Hz |
| Pitch Statistics (μ, σ) | μ = 176.86 Hz, σ = 57.12 Hz | μ = 208.39 Hz, σ = 40.10 Hz |
| Energy Statistics (μ, σ) | μ = 48.62, σ = 38.83 | μ = 41.43, σ = 36.25 |

### 2.2 Audio Cleaning and Preprocessing Pipeline
1. **Silence Trimming:** Leading and trailing unvoiced silence exceeding 50 ms was trimmed using an energy threshold of 25 dB below peak power.
2. **Noise Reduction:** Multi-speaker crowdsourced English recordings containing stationary background noise were filtered using spectral noise suppression.
3. **Peak Normalization:** Waveforms were normalized to -1.0 dBFS to prevent digital clipping while maximizing dynamic range.
4. **Feature Extraction:** 80-channel mel spectrograms, fundamental frequency (F₀) contours, and frame-level energy were extracted and saved alongside phoneme duration alignments.

### 2.3 Training Configuration and Multi-Task Objective
Both models were trained using the Adam optimizer with a batch size of 16 and a learning rate schedule with 2,000 warmup steps. Gradient clipping was applied to maintain numerical stability.

The training uses a multi-task objective that combines acoustic reconstruction with variance prediction:
*L*<sub>total</sub> = *L*<sub>mel</sub> + *L*<sub>postnet</sub> + *L*<sub>duration</sub> + *L*<sub>pitch</sub> + *L*<sub>energy</sub>

The mel-spectrogram reconstruction loss (*L*<sub>mel</sub>) and PostNet refinement loss (*L*<sub>postnet</sub>) are measured using Mean Absolute Error (L₁ norm) against ground-truth mel spectrograms. The duration, pitch, and energy predictors are trained using Mean Squared Error (MSE) against their aligned targets.

---

## 3. Analysis of Training and Validation Loss Curves

### 3.1 Loss Table and Convergence Trajectory Across 30,000 Steps
Table 2 reports the progression of training and validation loss values across 30,000 steps for both the English and Telugu models:

| Model | Checkpoint Step | Total Loss | Mel Loss (L₁) | PostNet Loss (L₁) | Pitch Loss (MSE) | Energy Loss (MSE) | Duration Loss (MSE) |
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

<div class="figure-full">
  <img src="../../Results/English/training_loss.png" alt="English Training Loss">
  <div class="caption">Figure 1a: English FastSpeech 2 training loss curves across 30,000 steps, showing Total Loss, Mel MAE, PostNet MAE, Duration MSE, Pitch MSE, and Energy MSE.</div>
</div>

<div class="figure-full">
  <img src="../../Results/English/validation_loss.png" alt="English Validation Loss">
  <div class="caption">Figure 1b: English FastSpeech 2 validation loss curves evaluated across validation checkpoints, demonstrating stable convergence without overfitting.</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/training_loss.png" alt="Telugu Training Loss">
  <div class="caption">Figure 1c: Telugu FastSpeech 2 training loss curves across 30,000 steps, showing rapid initial convergence and steady refinement.</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/validation_loss.png" alt="Telugu Validation Loss">
  <div class="caption">Figure 1d: Telugu FastSpeech 2 validation loss curves evaluated across validation checkpoints, showing smooth generalization on clean single-speaker speech.</div>
</div>

### 3.2 Convergence Behaviour and Generalization Analysis
1. **Warmup and Convergence:** Both models converged smoothly during training. As the learning rate reached its peak after the 2,000 warmup steps, total loss dropped by more than 60% within the first 5,000 steps. Duration prediction loss decreased early and stabilized below 0.03 for English and 0.006 for Telugu by step 30,000.
2. **Generalization in the English Model:** English validation loss stabilized around step 8,000 (at approximately 3.55) and remained stable near 3.62 through step 30,000. The difference between training mel loss (0.5673) and validation mel loss (1.0117) is due to the natural variation across 38 speakers and diverse recording environments in Svarah. Because the validation loss remained flat and did not climb upward, the model avoided destructive overfitting.
3. **Generalization in the Telugu Model:** Because the Telugu dataset consists of clean, single-speaker studio recordings, the model achieved lower loss values across all components. Final training mel loss was 0.3374 and duration MSE was 0.0056. Validation total loss settled smoothly at 2.2700, showing consistent generalization.

---

## 4. Observations on Generated Speech Quality

### 4.1 Evaluation of English Test Sentences (E1–E4)

**Sentence E1:** *"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*  
The model synthesized the complex 28-phoneme compound word clearly without hesitation, phoneme skipping, or looping. Because FastSpeech 2 uses feed-forward duration regulation instead of attention-based alignments, it maintains high stability on long, multisyllabic words.

**Sentence E2a:** *"The orchestra's conduct during the ceremony was praised by the audience."*  
The noun *conduct* correctly received primary stress on the first syllable. The first vowel was audibly prolonged and had higher acoustic prominence relative to the reduced second syllable.

**Sentence E2b:** *"The researchers will conduct an experiment to evaluate the proposed method."*  
The verb *conduct* was synthesized with stress on the second syllable. The first vowel reduced to a brief schwa, while the second vowel carried higher pitch and duration, producing a clear audible contrast with E2a.

**Sentence E3:** *"The rhyth m of the orchestra changed unexpectedly."*  
The tokenizer preserved the whitespace in `"rhyth m"` by inserting a short pause, followed by pronouncing the isolated letter `"m"`. The model synthesized this prompt cleanly without audio artifacts.

**Sentence E4:** *"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*  
The text normalizer correctly expanded the clock time and calendar date into spoken words. The borrowed Telugu word *andaroo* blended smoothly into the English sentence intonation.

### 4.2 Evaluation of Telugu Test Sentences (T1–T4)

**Sentence T1:** *"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*  
The phonetic engine broke down the technical loanword and complex consonant clusters into clean phonemes. Long vowel sounds were held for the correct length, matching natural Telugu pronunciation.

**Sentence T2a:** *"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*  
Synthesizes *పట్టు* as a noun meaning *grip*. The retroflex double consonant produced a clear closure silence followed by an emphatic release.

**Sentence T2b:** *"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*  
Synthesizes *పట్టు* as an adjective modifying *silk saree*. The duration and pitch of *పట్టు* were slightly reduced, keeping the main focus on the following noun *చీరను*.

**Sentence T3:** *"పులి అడ విలో నిశ్శబ్దంగా నడిచింది."*  
The artificial space in `"అడ విలో"` resulted in a brief inter-word pause, while the double consonant in `"నిశ్శబ్దంగా"` was articulated smoothly without unintended breaks.

**Sentence T4:** *"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*  
Demonstrates stable pitch declination across a long 136-phoneme sentence. The year, date ordinal, and time were all articulated clearly in spoken Telugu.

---

## 5. Analysis of Variance-Adaptor Plots (Duration, Pitch, Energy)

The variance adaptor predicts the duration, pitch, and energy for each phoneme, shaping the expressive rhythm and melody of the speech. Below, we analyze these predicted features alongside representative plots from our test sentences.

### 5.1 Duration Patterns
In English sentence E1, stressed vowels receive between 10 and 11 frames (approximately 116 to 128 ms), while unstressed vowels and stop consonants compress to 2 to 4 frames (approximately 23 to 46 ms). When comparing E2a and E2b (Figure 2a), the noun form of *conduct* assigns 10 frames to the first syllable and 9 frames to the second, whereas the verb form compresses the first syllable to 8 frames and lengthens the second syllable to 10 frames.

In Telugu, phonemic vowel length directly governs duration. In sentence T1, long vowels receive 12 to 13 frames (approximately 140 to 151 ms), which is about 1.4 times longer than short vowels (9 frames). In sentences T2a and T2b (Figure 2b), *పట్టు* occupies 41 frames as an independent noun in T2a, and compresses to 37 frames (a 9.8% reduction) when acting as an adjective in T2b.

<div class="figure-full">
  <img src="../../Results/English/variance_adaptor_plots/duration/E2a_vs_E2b_comparison.png" alt="English Duration Comparison">
  <div class="caption">Figure 2a: English phoneme-level duration comparison showing syntactic stress shift on <em>conduct</em> between noun (E2a) and verb (E2b).</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/variance_adaptor_plots/duration/T2a_vs_T2b_comparison.png" alt="Telugu Duration Comparison">
  <div class="caption">Figure 2b: Telugu phoneme-level duration comparison showing contextual duration compression on <em>పట్టు</em> between head noun (T2a) and pre-nominal modifier (T2b).</div>
</div>

### 5.2 Pitch (F₀) Contours
Pitch curves show a natural downward declination across sentences, starting near 220 Hz and resolving near 160 Hz at sentence boundaries. In English sentence E4 (Figure 3a), noticeable pitch peaks appear on focal information units such as the clock time and date before resolving at the end of the sentence. In Telugu (Figure 3b), pitch contours remain smooth and melodic between 190 Hz and 250 Hz, with a pitch plateau over retroflex geminate consonants.

<div class="figure-full">
  <img src="../../Results/English/variance_adaptor_plots/pitch/E4_pitch.png" alt="English Pitch Contour">
  <div class="caption">Figure 3a: English predicted fundamental frequency (F₀) contour and aligned spectrogram for sentence E4.</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/variance_adaptor_plots/pitch/T2a_pitch.png" alt="Telugu Pitch Contour">
  <div class="caption">Figure 3b: Telugu predicted fundamental frequency (F₀) contour and aligned spectrogram for sentence T2a.</div>
</div>

### 5.3 Energy Profiles
Energy contours closely track speech sonority. Vowel centers reach peak frame energy values between 60 and 75, while unvoiced consonants and stop closures drop to 20 to 35. Inter-word pause tokens drop to baseline energy values below 10. In Telugu geminate stops, the energy curve exhibits a distinct valley during the closure phase followed by a sharp energy burst at vowel release.

<div class="figure-full">
  <img src="../../Results/English/variance_adaptor_plots/energy/E4_energy.png" alt="English Energy Profile">
  <div class="caption">Figure 4a: English predicted frame-level energy profile and spectrogram for sentence E4.</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/variance_adaptor_plots/energy/T2a_energy.png" alt="Telugu Energy Profile">
  <div class="caption">Figure 4b: Telugu predicted frame-level energy profile and spectrogram for sentence T2a.</div>
</div>

---

## 6. Comparative Observations, Limitations, and Future Directions

### 6.1 English vs. Telugu Comparison
- **Speaker Variance vs. Single-Speaker Precision:** The Telugu model achieved a lower training mel loss (0.3374 vs. 0.5678) and significantly lower duration error (0.0056 vs. 0.0272) compared to the English model. Training on a single speaker provides consistent vocal tract characteristics and clean studio acoustics. In contrast, crowdsourced multi-speaker data contains microphone differences and regional accents.
- **Rhythmic Differences:** English rhythm is primarily stress-timed, where unstressed syllables reduce to short schwas. Telugu rhythm is syllable-timed and phonological, with distinct duration governed by short versus long vowels and geminate consonants.

### 6.2 Limitations Observed in Test Cases
1. **Whitespace Sensitivity (E3, T3):** Standard tokenizers treat accidental spaces inside words as word boundaries, introducing unintended brief pauses.
2. **Contextual Heteronym Scope:** While rule-based grammatical checks successfully disambiguate common words like *conduct*, they do not cover all possible English heteronyms without a full syntactic parser.
3. **Cross-Lingual Phonetics (E4):** Mapping borrowed Telugu words into the English phoneme alphabet substitutes English alveolar sounds for native dental stops.

### 6.3 Future Improvements
- **Contextual Language Model Embeddings:** Incorporating pretrained text embeddings (such as ByT5) would help the model handle typographical errors and ambiguous words more effectively.
- **Unified Phonetic Alphabet:** Using the International Phonetic Alphabet (IPA) across both languages would improve pronunciation for code-switched Indian speech.
- **Advanced Decoder Architectures:** Integrating diffusion-based or flow-matching decoders could reduce slight spectral oversmoothing and enhance voice clarity.

---

## 7. Reproducibility Details

### 7.1 Software Dependencies
The system runs on Python 3.10 and PyTorch 2.1+ with GPU acceleration:
```text
torch>=2.1.0, torchaudio>=2.1.0, scipy>=1.10.0, librosa>=0.10.0, pyworld>=0.3.4,
g2p_en>=2.1.0, resemblyzer>=0.1.4, noisereduce>=3.0.0, matplotlib>=3.7.0, pyyaml>=6.0
```

### 7.2 Execution Commands

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

All synthesized test audio files and generated variance adaptor plots are stored in the results directory under their respective language folders.
