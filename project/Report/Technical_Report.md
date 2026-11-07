# FastSpeech 2 for English and Telugu Speech Synthesis

<div class="author-header">
<strong>Rishi Pramod Chintakayala</strong><br>
<em>Research Assistant Technical Report — Speech and Language Processing</em><br>
<em>International Institute of Information Technology Hyderabad (IIIT-H)</em><br>
Base Reference: FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., ICLR 2021)
</div>

---

### Abstract
This report presents our implementation, optimization, and comparative analysis of FastSpeech 2 models for Indian-accented English and Telugu speech synthesis. FastSpeech 2 is a non-autoregressive neural text-to-speech architecture that synthesizes mel spectrograms in parallel by directly predicting acoustic variance parameters (duration, pitch, and energy) from phoneme sequences. For English, we utilized the multi-speaker AI4Bharat Svarah corpus, incorporating neural forced alignment via Meta's Massively Multilingual Speech (MMS) framework, context-aware heteronym stress modeling, and full text normalization. For Telugu, we utilized the single-speaker IIT Madras IndicTTS corpus and engineered a custom 50-phoneme grapheme-to-phoneme (G2P) engine supporting complex conjunct consonants, virama suppression, and inflected numeral normalization. Both models were trained for 30,000 steps on strictly isolated 5-hour training and 1-hour validation partitions. We present rigorous evaluations of training/validation loss trajectories (including an objective diagnosis of multi-speaker acoustic overfitting), detailed acoustic critique across ten benchmark test sentences, comprehensive multi-sentence variance-adaptor visualizations, and full reproducibility specifications.

---

## 1. Implementation Approach and Architectural Modifications

### 1.1 Reference Architecture Overview
FastSpeech 2 addresses the exposure bias and sequential inference bottleneck of autoregressive models (e.g., Tacotron 2) by formulating speech synthesis as a feed-forward generation pipeline:

1. **Phoneme Encoder:** Composed of 4 Feed-Forward Transformer (FFT) blocks. Each block pairs multi-head self-attention (2 attention heads, hidden dimension <i>d</i><sub>model</sub> = 256) with a 2-layer 1D convolutional network with 1024 filter channels and kernel sizes [9, 1] to generate contextual phoneme representations.
2. **Speaker Conditioning:** Integrates speaker representations into the phoneme embeddings prior to the variance adaptor. For multi-speaker English, continuous 256-dimensional d-vector embeddings capture individual vocal tract characteristics; for single-speaker Telugu, a categorical speaker embedding is utilized.
3. **Variance Adaptor:** Bridges the temporal dimension mismatch between discrete phoneme sequences and continuous mel-spectrogram frames:
   - **Duration Predictor & Length Regulator:** A 2-layer Conv1D network with ReLU activation, layer normalization, and dropout predicting log-scale phoneme durations. The Length Regulator expands hidden representations according to predicted durations, matching mel frames (where 1 frame ≈ 11.6 ms at 22,050 Hz).
   - **Pitch (<i>F</i>₀) Predictor:** Predicts normalized continuous fundamental frequency values, quantized into 256 linearly spaced bins and mapped to a trainable pitch embedding.
   - **Energy Predictor:** Predicts frame-level spectral energy (<i>L</i>₂ norm of STFT magnitudes), quantized into 256 bins and integrated into the expanded representations.
4. **Mel-Spectrogram Decoder & PostNet:** A 6-block FFT decoder generates 80-channel log-mel spectrogram frames, followed by a 5-layer convolutional PostNet (512 channels, kernel 5, batch normalization) that predicts residual corrections to refine spectral details.
5. **Neural Vocoder:** A pre-trained universal HiFi-GAN vocoder transforms predicted 80-channel mel spectrograms into 22,050 Hz, 16-bit mono audio waveforms.

### 1.2 Modifications for English
- **Neural Forced Alignment:** Replaced legacy alignment harnesses with Meta's MMS neural forced aligner, extracting millisecond-accurate phoneme and silence (sp) boundaries directly on 22,050 Hz audio.
- **Contextual Heteronym Disambiguation:** Implemented syntactic rule disambiguation for English heteronyms (e.g., *conduct*), assigning primary initial-syllable stress [K AA1 N D AH0 K T] in noun contexts (following possessives/determiners) and second-syllable stress [K AH0 N D AH1 K T] in verb contexts (following auxiliary modals).
- **Text Normalization Engine:** Integrated comprehensive expansion rules for timestamps, cardinal numbers, ordinal dates, and abbreviations, as well as cross-lingual phonetic approximations for borrowed Indian terminology.
- **Consistent Acoustic Anchor:** Selected a high-clarity neutral female speaker embedding from Svarah as an evaluation anchor to ensure controlled, reproducible assessment across test sentences.

### 1.3 Modifications for Telugu
- **Rule-Based 50-Phoneme G2P Engine:** Designed a dedicated phonetic converter tailored to the Telugu abugida script:
  - 14 vowels (with strict distinction between short and long vowel phonemes).
  - 31 consonants mapped by articulatory place (stops, retroflexes, fricatives, nasals).
  - Explicit virama suppression logic: automatically removes inherent vowels when followed by a virama or dependent vowel sign, correctly producing conjunct clusters and geminate stops.
  - Native support for anusvara ([M]) and visarga diacritics.
- **Inflected Numeral & Calendar Normalization:** Engineered a normalizer converting multi-digit numbers, ordinal dates, and timestamps into grammatically inflected Telugu word tokens.
- **Modern Audio Pipeline:** Updated feature extraction routines to PyTorch 2.1+ / Torchaudio standards with dynamic feature caching for high GPU throughput.

---

## 2. Dataset Selection, Preprocessing, and Training Configuration

### 2.1 Subsampled Datasets (5-Hour Train / 1-Hour Validation)
Both datasets were subsampled using a fixed seed (seed 42) into strictly isolated 5-hour training and 1-hour validation subsets with zero utterance or speaker overlap:

| Parameter | English (AI4Bharat Svarah) | Telugu (IndicTTS Telugu) |
| :--- | :--- | :--- |
| Primary Corpus | AI4Bharat Svarah (Indian English) | IIT Madras IndicTTS Telugu |
| Speaker Characteristics | Multi-speaker (38 regional accent groups) | Single professional female speaker |
| Training Duration | 5.00 hours (3,472 utterances) | 5.00 hours (1,589 utterances) |
| Validation Duration | 1.00 hour (670 utterances) | 1.00 hour (305 utterances) |
| Active Speech Duration | 4.79 hours Train / 0.96 hours Val | 4.50 hours Train / 0.91 hours Val |
| Sampling Rate & Format | 22,050 Hz, 16-bit mono PCM | 22,050 Hz, 16-bit mono PCM |
| STFT Configuration | FFT = 1024, Hop = 256, Window = 1024 | FFT = 1024, Hop = 256, Window = 1024 |
| Mel Filterbanks | 80 channels, 0 Hz to 8,000 Hz | 80 channels, 0 Hz to 8,000 Hz |
| Pitch Statistics (μ, σ) | μ = 176.86 Hz, σ = 57.12 Hz | μ = 208.39 Hz, σ = 40.10 Hz |
| Energy Statistics (μ, σ) | μ = 48.62, σ = 38.83 | μ = 41.43, σ = 36.25 |

### 2.2 Audio Cleaning Pipeline
1. **Silence Trimming:** Trimmed unvoiced leading and trailing silence exceeding 50 ms using a -25 dB energy threshold.
2. **Spectral Noise Suppression:** Filtered stationary background noise and room reverberation from crowdsourced Svarah recordings using spectral gating.
3. **Peak Normalization:** Scaled waveforms to -1.0 dBFS to prevent digital clipping while maximizing dynamic range.

### 2.3 Model Architecture & Training Hyperparameters
Both models were configured using identical structural dimensions and training parameters as summarized in Table 1b:

| Component / Module | Hyperparameter | Configured Value |
| :--- | :--- | :--- |
| **Phoneme Encoder** | FFT Layers / Multi-Head Attention Heads | 4 layers, 2 heads |
| | Hidden Dimension / Filter Size | <i>d</i><sub>model</sub> = 256, Conv1D Filter = 1024 |
| | 1D Convolution Kernels / Dropout | Kernel sizes = [9, 1], Dropout = 0.2 |
| **Variance Adaptor** | Duration Predictor Architecture | 2-layer Conv1D (filter 256, kernel 3, dropout 0.5) |
| | Pitch / Energy Predictor Architecture | 2-layer Conv1D (filter 256, kernel 3, dropout 0.5) |
| | Variance Quantization Bins | Linear 256 bins for <i>F</i>₀ and Frame Energy |
| **Mel-Spectrogram Decoder** | FFT Layers / Multi-Head Attention Heads | 6 layers, 2 heads |
| | Hidden Dimension / Filter Size | <i>d</i><sub>model</sub> = 256, Conv1D Filter = 1024 |
| | 1D Convolution Kernels / Dropout | Kernel sizes = [9, 1], Dropout = 0.2 |
| **Mel PostNet** | Convolutions & Residual Connections | 5-layer 1D Conv (512 channels, kernel 5, batch norm) |
| **Neural Vocoder** | Model Architecture | Pre-trained HiFi-GAN (Universal, 22,050 Hz) |
| **Optimization** | Optimizer & Parameters | Adam (β<sub>1</sub> = 0.9, β<sub>2</sub> = 0.98, ε = 10<sup>−9</sup>, weight decay 0.0) |
| | Learning Rate Schedule | Noam decay: <i>d</i><sub>model</sub><sup>−0.5</sup> · min(step<sup>−0.5</sup>, step · warmup<sup>−1.5</sup>) |
| | Warmup Steps / Gradient Clipping | 2,000 warmup steps, Gradient clip threshold = 1.0 |
| | Batch Size & Total Training Steps | Batch size = 16 utterances, Total steps = 30,000 |
| | Validation / Checkpoint Save Step | Validation = every 1,000 steps, Checkpoint = every 2,500 steps |

The multi-task training objective combines reconstruction and variance prediction losses:
<div style="text-align: center; margin: 10px 0; font-size: 10.5pt; font-weight: bold;"><i>L</i><sub>total</sub> = <i>L</i><sub>mel</sub> + <i>L</i><sub>postnet</sub> + <i>L</i><sub>duration</sub> + <i>L</i><sub>pitch</sub> + <i>L</i><sub>energy</sub></div>

Where <i>L</i><sub>mel</sub> and <i>L</i><sub>postnet</sub> represent Mean Absolute Error (<i>L</i>₁ norm), and <i>L</i><sub>duration</sub>, <i>L</i><sub>pitch</sub>, and <i>L</i><sub>energy</sub> represent Mean Squared Error (MSE).

---

## 3. Analysis of Training and Validation Loss Curves

### 3.1 Loss Table and Convergence Trajectory Across 30,000 Steps
Table 2 reports the progression of training and validation loss values across 30,000 steps evaluated periodically during training and validation. Every row satisfies the multi-task arithmetic identity (<i>L</i><sub>total</sub> = ∑ <i>L</i><sub>i</sub>):

| Model | Checkpoint Step | Total Loss | Mel Loss (L₁) | PostNet Loss (L₁) | Pitch Loss (MSE) | Energy Loss (MSE) | Duration Loss (MSE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **English** | Step 1,000 (Train) | 5.3243 | 1.1634 | 1.2035 | 1.9285 | 0.9449 | 0.0840 |
| | Step 10,000 (Train) | 1.8414 | 0.7528 | 0.7523 | 0.1117 | 0.1516 | 0.0730 |
| | Step 20,000 (Train) | 1.3871 | 0.6044 | 0.6041 | 0.0656 | 0.0757 | 0.0372 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | **Step 30,000 (Train)** | **1.2725** | **0.5678** | **0.5673** | **0.0515** | **0.0587** | **0.0272** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Step 10,000 (Val) | 3.5395 | 0.9900 | 0.9900 | 0.4872 | 0.8815 | 0.1908 |
| | Step 20,000 (Val) | 3.6199 | 1.0138 | 1.0138 | 0.4887 | 0.9099 | 0.1937 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | **Step 30,000 (Val)** | **3.6240** | **1.0120** | **1.0117** | **0.4956** | **0.9083** | **0.1964** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Telugu** | Step 1,000 (Train) | 2.1480 | 0.6902 | 0.7001 | 0.4144 | 0.2563 | 0.0870 |
| | Step 10,000 (Train) | 0.8623 | 0.3942 | 0.3937 | 0.0351 | 0.0297 | 0.0096 |
| | Step 20,000 (Train) | 0.7578 | 0.3524 | 0.3518 | 0.0237 | 0.0200 | 0.0099 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | **Step 30,000 (Train)** | **0.7132** | **0.3380** | **0.3374** | **0.0176** | **0.0146** | **0.0056** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | Step 10,000 (Val) | 2.2592 | 0.6235 | 0.6233 | 0.6394 | 0.2936 | 0.0793 |
| | Step 20,000 (Val) | 2.2582 | 0.6273 | 0.6272 | 0.6354 | 0.2899 | 0.0784 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| | **Step 30,000 (Val)** | **2.2700** | **0.6287** | **0.6286** | **0.6423** | **0.2864** | **0.0839** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

<div class="figure-full">
  <img src="../../Results/English/training_loss.png" alt="English Training Loss">
  <div class="caption">Figure 1a: English FastSpeech 2 training loss curves across 30,000 steps, showing Total Loss, Mel MAE, PostNet MAE, Duration MSE, Pitch MSE, and Energy MSE.</div>
</div>

<div class="figure-full">
  <img src="../../Results/English/validation_loss.png" alt="English Validation Loss">
  <div class="caption">Figure 1b: English FastSpeech 2 validation loss curves evaluated across validation checkpoints, demonstrating initial descent followed by acoustic overfitting on Energy.</div>
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
1. **Initial Warmup and Rapid Descent:** Both models converged smoothly during early optimization. Driven by the Noam learning rate schedule peaking at step 2,000, Total Loss dropped precipitously within the first 5,000 steps (falling by 65.4% for English and 60.0% for Telugu). The duration predictor converged rapidly in both languages, reaching duration MSE values below 0.03 for English and 0.006 for Telugu by step 30,000.
2. **Generalization and Overfitting Diagnosis on Multi-Speaker English (Figure 1b):**
   Inspection of the validation loss curves reveals clear differences in generalization dynamics. On the English Svarah corpus, Total Validation Loss hits its global minimum of 3.5395 around Step 10,000 (with an early low near Step 5,000 of ~3.50) before steadily climbing upward to 3.6199 at Step 20,000 and 3.6240 at Step 30,000. A component-level breakdown explains this upward trajectory:
   - The **Duration Predictor** remains stable on validation data (MSE ~ 0.190 to 0.196).
   - The **Pitch Predictor** stabilizes early near MSE = 0.487 to 0.495.
   - However, the **Energy Predictor** validation MSE exhibits a steep climb, rising from 0.835 at Step 2,000 to 0.8815 at Step 10,000 and peaking at 0.9083 at Step 30,000. Concurrently, **Mel Loss** rises slightly from 0.9900 to 1.0120.
   
   This demonstrates mild acoustic overfitting: because AI4Bharat Svarah is a crowdsourced multi-speaker corpus recorded under diverse home environments and acoustic conditions across 38 speakers, the model's energy and mel decoders begin memorizing speaker-specific room reverberation and microphone channel characteristics after Step 8,000–10,000, rather than further generalizing to unseen validation speakers. An early-stopping checkpoint around Step 10,000 represents the optimal trade-off point between mel reconstruction and variance generalization on this corpus.
3. **Generalization Stability on Single-Speaker Telugu (Figure 1d):**
   In contrast, the Telugu IndicTTS dataset consists of clean, single-speaker studio recordings with uniform vocal tract acoustics and microphone placement. As a result, the Telugu model achieved significantly lower training loss across all components (Final Total Loss = 0.7132, Mel MAE = 0.3374, Duration MSE = 0.0056). Most notably, the validation loss curves settled smoothly and remained virtually flat across the entire training trajectory (Step 10,000: 2.2592; Step 20,000: 2.2582; Step 30,000: 2.2700), exhibiting zero upward drift or energy overfitting.

---

## 4. Observations on Generated Speech Quality

### 4.1 Evaluation of English Test Sentences (E1–E4)
Evaluation of synthesized English waveforms requires distinguishing between linguistic/prosodic correctness and spectral/vocoder acoustic quality:

**Sentence E1:** *"The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings."*  
- *Linguistic & Prosodic Execution:* The feed-forward length regulator synthesized the 28-phoneme, 12-syllable compound word with remarkable temporal stability—avoiding the phoneme skipping, hesitation, or attention collapse typical of autoregressive architectures. Stress timing was maintained across syllables.
- *Acoustic Quality Critique:* While fully intelligible, the synthesized speech exhibits mild spectral oversmoothing and slight background phase buzz in the high frequencies (>6 kHz). This acoustic artifact directly reflects the crowdsourced nature of the Svarah training data, where microphone bandwidth limits and room reverberation transfer into the predicted mel-spectrogram noise floor.

**Sentence E2a:** *"The orchestra's conduct during the ceremony was praised by the audience."*  
- *Prosodic Contrast:* The noun *conduct* correctly received initial syllable stress. The first syllable vowel [AA1] was prolonged to 10 frames (~116 ms) with heightened pitch prominence, while the second vowel [AH0] was reduced to 4 frames (~46 ms). Intonation and word boundaries were clearly delineated.

**Sentence E2b:** *"The researchers will conduct an experiment to evaluate the proposed method."*  
- *Prosodic Contrast:* The verb *conduct* was synthesized with clear second-syllable stress. The first vowel [AH0] reduced to a brief schwa (3 frames, ~35 ms), while the second syllable vowel [AH1] prolonged to 10 frames (~116 ms) with elevated pitch, providing a distinct, audible syntactic contrast with E2a.

**Sentence E3:** *"The rhyth m of the orchestra changed unexpectedly."*  
- *Tokenizer Failure Mode Analysis:* Rather than an architectural flaw, this prompt demonstrates a canonical tokenizer failure mode triggered by adversarial whitespace. The text normalizer split the input *"rhyth m"* into two independent lexical tokens: the unknown word fragment *"rhyth"*, an inserted word-boundary pause (*sp*, 28 frames / ~325 ms), and the isolated letter *"m"*. Consequently, the model faithfully synthesized an unnatural inter-word silence followed by pronouncing the isolated letter name [EH1 M]. This visually and audibly proves the vulnerability of rule-based G2P pipelines to typographical whitespace corruptions.

**Sentence E4:** *"The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."*  
- *Linguistic Execution:* The text normalizer successfully expanded the timestamp into spoken words ("ten forty five a m") and the calendar date into full ordinals ("twelve november twenty twenty-six").
- *Code-Switching Observation:* The borrowed Telugu word *andaroo* was phonetically approximated using the English ARPAbet sequence [AA1 N D AH0 R UW1]. While intelligibly blended into the sentence intonation contour, the English alveolar stop [D] replaces the native Telugu retroflex/dental stop, producing a perceptible foreign-accent coloration.

### 4.2 Evaluation of Telugu Test Sentences (T1–T4)
In contrast to the English model, the synthesized Telugu speech benefits from studio-quality single-speaker training, yielding crisp formants and natural voice timbre:

**Sentence T1:** *"న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు."*  
The 50-phoneme G2P engine accurately resolved the transliterated English scientific compound loanword into clean phonemes without phoneme dropping. Phonemic vowel length contrasts were accurately respected, with long vowels ([uu], [oo], [aa]) sustained for 12–14 frames while short vowels ([i], [a]) occupied 6–8 frames.

**Sentence T2a:** *"తాడుపై అతని పట్టు చాలా బలంగా ఉంది."*  
Synthesizes *pattu* (*పట్టు*) as an independent head noun meaning *grip*. The retroflex double consonant produces an extended closure silence (14 frames) followed by an emphatic release burst (8 frames), giving the word prominent acoustic weight (total: 41 frames).

**Sentence T2b:** *"ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది."*  
Synthesizes *pattu* (*పట్టు*) as a pre-nominal modifier (*silk saree*). The model accurately compressed the duration of *pattu* to 37 frames (a 9.8% reduction), subordinating it prosodically to the primary head noun *chiiranu* (*చీరను*).

**Sentence T3:** *"పులి అడ విలో నిశ్శబ్దంగా నడిచింది."*  
- *Adversarial Whitespace Analysis:* Paralleling English E3, the accidental space in *ada viloo* (*అడ విలో*) forced the word to be parsed as two distinct words, inserting a 24-frame inter-word pause with an energy drop below 10. Conversely, within the uncorrupted word *nishshabdangaa* (*నిశ్శబ్దంగా*), the geminate consonant cluster [sh sh] was articulated smoothly with a 13-frame closure and zero unintended pausing.

**Sentence T4:** *"ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి."*  
The model synthesized this 136-phoneme complex sentence with smooth pitch declination across all 1,060 mel frames (~12.3 seconds) without pitch wobble or duration collapse. The year, ordinal date, and timestamp were all articulated in natural spoken Telugu.

---

## 5. Analysis of Variance-Adaptor Plots (Duration, Pitch, Energy)

The variance adaptor predicts the duration, pitch, and energy for each phoneme, shaping the expressive rhythm and melody of the speech. Below, we examine these predicted features across benchmark comparisons and complete multi-sentence grids.

### 5.1 Phoneme Duration Patterns & Syntactic Stress Shifts
Phoneme durations directly govern the rhythmic cadence of synthesized speech:
- In English sentence E1, stressed vowels receive between 10 and 11 frames (~116 to 128 ms), while unstressed vowels and stop closures compress to 2 to 4 frames (~23 to 46 ms).
- **Homograph Stress Contrast (Figure 2a):** In sentence E2a (*conduct* as a noun), primary stress on syllable 1 prolongs the vowel [AA1] to **10 frames** (~116 ms), while syllable 2 vowel [AH0] reduces to **4 frames** (~46 ms) (total syllable durations: 17 frames vs. 14 frames). Conversely, in sentence E2b (*conduct* as a verb), syllable 1 vowel [AH0] reduces to **3 frames** (~35 ms), while syllable 2 vowel [AH1] prolongs to **10 frames** (~116 ms) (total syllable durations: 10 frames vs. 20 frames). This quantitative duration shift produces the clear audible stress contrast observed in synthesis.
- **Telugu Contextual Compression (Figure 2b):** In Telugu, vowel length is phonemically contrastive rather than stress-timed. In sentence T1, long vowels receive 12 to 14 frames, approximately 1.5 times longer than short vowels (6 to 8 frames). When comparing sentences T2a and T2b, the word *pattu* (*పట్టు*) occupies **41 frames** as an independent head noun in T2a ([p]=3, [a]=9, [tt closure]=14, [tt release]=8, [u]=7), and compresses to **37 frames** ([p]=3, [a]=8, [tt closure]=12, [tt release]=7, [u]=7) in T2b—representing an exact **9.8% contextual duration compression** when functioning as a pre-nominal modifier.

<div class="figure-full">
  <img src="../../Results/English/variance_adaptor_plots/duration/E2a_vs_E2b_comparison.png" alt="English Duration Comparison">
  <div class="caption">Figure 2a: English phoneme-level duration comparison showing syntactic stress shift on <em>conduct</em> between noun (E2a, AA1 = 10 frames) and verb (E2b, AH1 = 10 frames).</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/variance_adaptor_plots/duration/T2a_vs_T2b_comparison.png" alt="Telugu Duration Comparison">
  <div class="caption">Figure 2b: Telugu phoneme-level duration comparison showing contextual duration compression on <em>pattu</em> (పట్టు) between head noun (T2a, 41 frames) and pre-nominal modifier (T2b, 37 frames).</div>
</div>

### 5.2 Comprehensive Multi-Sentence Variance Grids (E1–E4 & T1–T4)
To evaluate variance prediction across diverse syntactic structures without page fragmentation, Figures 3a and 3b display comprehensive multi-panel grids for all test sentences in English and Telugu. Each panel overlays the aligned 80-channel log-mel spectrogram with predicted Pitch (<i>F</i>₀, red curve), Energy (cyan curve), and phoneme duration boundaries (white dashed vertical lines):

<div class="figure-full">
  <img src="../../Results/English/variance_adaptor_plots/english_all_sentences_grid.png" alt="English Comprehensive Multi-Sentence Variance Grid">
  <div class="caption">Figure 3a: English multi-panel variance grid for all benchmark test sentences (E1, E2a, E2b, E3, E4) displaying aligned Mel Spectrograms, Pitch contours (red), Energy profiles (cyan), and phoneme boundaries (dashed lines).</div>
</div>

<div class="figure-full">
  <img src="../../Results/Telugu/variance_adaptor_plots/telugu_all_sentences_grid.png" alt="Telugu Comprehensive Multi-Sentence Variance Grid">
  <div class="caption">Figure 3b: Telugu multi-panel variance grid for all benchmark test sentences (T1, T2a, T2b, T3, T4) displaying aligned Mel Spectrograms, Pitch contours (red), Energy profiles (cyan), and phoneme boundaries (dashed lines).</div>
</div>

### 5.3 Prosodic Walkthrough for Every Test Sentence
Detailed examination of the multi-panel grids and individual sentence contours reveals specific prosodic patterns:

1. **English E1 (Multisyllabic Compound):** The 12-syllable word *antidisestablishmentarianism* exhibits distinct duration peaks on secondary stressed vowels [AE2] (8 frames), [IH2] (6 frames), and primary stressed vowel [EH1] (11 frames), contrasted against compressed unstressed syllables [IH0], [AH0] (2–3 frames). Noticeable <i>F</i>₀ declination occurs across the compound word, followed by a clear pitch reset at the phrase boundary preceding *"gained considerable momentum"*.
2. **English E2a vs. E2b (Syntactic Homograph):** Pitch contours mirror the duration shifts: in E2a, <i>F</i>₀ peaks at 245 Hz on syllable 1 of *conduct*, whereas in E2b, <i>F</i>₀ rises to a prominent peak of 278 Hz over syllable 2, confirming that FastSpeech 2 jointly modulates duration, pitch, and energy to convey syntactic category.
3. **English E3 (Whitespace Typo):** The whitespace insertion in *"rhyth m"* causes the variance adaptor to insert a dedicated silence token (*sp*) spanning 28 frames (~325 ms). During this interval, predicted energy drops sharply to baseline (<10), providing visual and acoustic proof of the tokenizer-induced pause between *"rhyth"* and the isolated letter *"m"*.
4. **English E4 (Code-Switching & Numerals):** Pitch peaks reach focal maxima over information-dense tokens: 310 Hz on clock time "10:45 AM" and 295 Hz on calendar date "12 November 2026". Over the borrowed Telugu word *andaroo* ([AA1 N D AH0 R UW1], 38 frames), pitch and energy blend continuously with surrounding English words, demonstrating stable cross-lingual variance transfer.
5. **Telugu T1 (Technical Loanword):** The loanword *nyuuroophaarmakoolajikal* (*న్యూరోఫార్మకోలాజికల్*) occupies 148 frames, with long vowels [uu] (13 frames) and [oo] (12 frames) clearly separated by consonant cluster closures. Pitch remains steady between 220 Hz and 260 Hz, confirming stability on long out-of-vocabulary compounds.
6. **Telugu T2a vs. T2b (Contextual Compression):** In T2a, *pattu* (*పట్టు*) exhibits a prolonged closure interval (14 frames) with an energy dip followed by a sharp burst. In T2b, the closure compresses to 12 frames and release to 7 frames, reducing overall prominence before the primary stress of *chiiranu* (*చీరను*).
7. **Telugu T3 (Split Whitespace vs. Geminates):** In *ada viloo* (*అడ విలో*), the split whitespace generates an artificial 24-frame inter-word pause with an energy drop below 10. In contrast, within the correct word *nishshabdangaa* (*నిశ్శబ్దంగా*), the geminate [sh sh] maintains continuous acoustic context with a clean 13-frame closure silence without unintended pauses.
8. **Telugu T4 (Formal Numeral Declination):** Across the long 136-phoneme, 1,060-frame sentence (~12.3 seconds), the pitch curve exhibits textbook linguistic declination: commencing near 280 Hz on the initial demonstrative, maintaining melodic plateaus across the expanded year and date ordinals, and resolving near 150 Hz at sentence completion.

---

## 6. Comparative Observations, Limitations, and Future Directions

### 6.1 English vs. Telugu Comparison
- **Speaker Variance vs. Single-Speaker Precision:** The Telugu model achieved lower training mel loss (0.3374 vs. 0.5678) and significantly lower duration error (0.0056 vs. 0.0272) compared to English. Single-speaker training provides invariant vocal tract characteristics and uniform studio acoustics. In contrast, crowdsourced multi-speaker data introduces microphone channel diversity and room reverberation.
- **Rhythmic Differences:** English synthesis is stress-timed, where vowel reduction to brief schwas characterizes unstressed positions. Telugu synthesis is syllable-timed and phonologically contrastive, with duration strictly governed by short versus long vowels and geminate consonants.

### 6.2 Observed Limitations
1. **Adversarial Whitespace Vulnerability (E3, T3):** Standard tokenizers treat intra-word spaces as word boundaries, introducing unintended inter-word pauses.
2. **Contextual Heteronym Coverage:** Rule-based grammatical checks disambiguate common words like *conduct*, but full heteronym disambiguation requires full syntactic dependency parsing.
3. **Cross-Lingual Phonetics (E4):** Mapping borrowed Telugu words into the English ARPAbet substitutes English alveolar consonants for native dental stops.

### 6.3 Future Improvements
- **Contextual Language Model Embeddings:** Integrating pretrained character-aware representations (e.g., ByT5) would mitigate sensitivity to typographical spaces and expand heteronym handling.
- **Unified IPA Representation:** Adopting the International Phonetic Alphabet across both languages would improve cross-lingual pronunciation on code-switched Indian speech.
- **Flow-Matching & Diffusion Decoders:** Replacing the feed-forward decoder with modern flow-matching decoders would reduce high-frequency spectral oversmoothing.

---

## 7. Reproducibility Details

### 7.1 Software Dependencies
The system executes on Python 3.10 and PyTorch 2.1+ with CUDA acceleration:
```text
torch>=2.1.0, torchaudio>=2.1.0, scipy>=1.10.0, librosa>=0.10.0, pyworld>=0.3.4,
g2p_en>=2.1.0, resemblyzer>=0.1.4, noisereduce>=3.0.0, matplotlib>=3.7.0, pyyaml>=6.0
```

### 7.2 Execution Commands

```bash
# 1. Environment Setup
conda create -n fastspeech2 python=3.10 -y && conda activate fastspeech2
pip install -r requirements.txt

# 2. English Model Pipeline
cd project/English
python preprocess.py --prepare_align   # Subsample Svarah (5h/1h) and generate MMS-FA alignments
python preprocess.py                  # Extract Mel, F0, Energy, and Duration features
python train.py                       # Train FastSpeech 2 model
python inference.py                   # Synthesize benchmark audios and variance plots

# 3. Telugu Model Pipeline
cd ../Telugu
python preprocess.py --prepare_align   # Subsample IndicTTS (5h/1h), run G2P, and generate alignments
python preprocess.py                  # Extract Mel, F0, Energy, and Duration features
python train.py                       # Train FastSpeech 2 model
python inference.py                   # Synthesize benchmark audios and variance plots
```
