# FastSpeech 2 for English and Telugu Speech Synthesis

**Project Report — IIIT Hyderabad Research Assistant Assignment**  
**Author:** Rishi Pramod  
**Base implementation:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)  
**Reference paper:** *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, Ren et al., ICLR 2021

## 1. Project Overview

In this project, I worked on text-to-speech generation for English and Telugu using FastSpeech 2. I trained a separate model for each language and used a pre-trained HiFi-GAN vocoder to convert the predicted mel spectrograms into audio.

The work included preparing the datasets, processing text, obtaining duration labels, training the models, and testing speech generation. I also studied duration, pitch, and energy to understand how the models handled pronunciation and pauses.

The English model used multi-speaker recordings, while the Telugu model used single-speaker recordings. Each model was trained for 30,000 steps. Both models showed lower training loss over time, but validation loss stopped improving much earlier.

> **About the results:** The numerical values below are retained from the original experiment report. The source code, raw logs, generated audio, and plots were not included with this report. Conflicting values and details that still need verification are identified below.

## 2. Project Objectives

The main objectives were to:

- Adapt the FastSpeech 2 pipeline for English and Telugu.
- Convert text, numbers, dates, and times into suitable model inputs.
- Train both models with a limited amount of speech data.
- Test difficult words, context-dependent pronunciation, and unusual spaces.
- Compare training and validation losses.
- Study the duration, pitch, and energy outputs.

## 3. Model and Workflow

### 3.1 Workflow

During training, the pipeline prepares text and audio, extracts acoustic features and duration labels, and uses these targets to train FastSpeech 2.

During inference, the pipeline converts an input sentence into phonemes. FastSpeech 2 predicts their duration, pitch, and energy, then generates a mel spectrogram. HiFi-GAN converts this spectrogram into a speech waveform.

### 3.2 Main Components

| Component | Purpose |
| --- | --- |
| Text processor | Cleans the sentence and converts it into phoneme tokens. |
| Encoder | Builds a representation of the input phonemes. |
| Duration predictor | Estimates how many acoustic frames each phoneme should occupy. |
| Pitch predictor | Estimates the pitch pattern. |
| Energy predictor | Estimates the energy pattern. |
| Length regulator | Expands the phoneme sequence using its durations. |
| Decoder | Predicts an 80-channel mel spectrogram. |
| PostNet | Refines the predicted mel spectrogram. |
| HiFi-GAN vocoder | Converts the mel spectrogram into audio. |

The PostNet is part of the chosen reference implementation. It should not be described as a component of the original FastSpeech 2 paper.

### 3.3 Reported Model Settings

| Setting | Value |
| --- | --- |
| Encoder blocks | 4 |
| Decoder blocks | 6 |
| Attention heads | 2 |
| Hidden dimension | 256 |
| FFT convolution filter size | 1,024 |
| FFT convolution kernel sizes | 9 and 1 |
| Transformer dropout | 0.2 |
| Variance predictor layers | 2 convolution layers |
| Variance predictor filter size | 256 |
| Variance predictor kernel size | 3 |
| Variance predictor dropout | 0.5 |
| Pitch and energy bins | 256 each |
| Mel channels | 80 |
| PostNet layers | 5 convolution layers |

These are the settings recorded in the draft. The saved configuration files should be used to confirm the exact settings for each training run, including whether pitch and energy were represented at the frame level or phoneme level.

## 4. Datasets and Preprocessing

### 4.1 Dataset Summary

| Setting | English | Telugu |
| --- | --- | --- |
| Dataset | `ai4bharat/Svarah` | `SPRINGLab/IndicTTS_Telugu` |
| Speech type | Indian-accented English, multi-speaker | Telugu, single-speaker |
| Reported training duration | 5.00 hours | 5.00 hours |
| Reported training utterances | 3,472 | 1,589 |
| Reported validation duration | 1.00 hour | 1.00 hour |
| Reported validation utterances | 670 | 305 |
| TTS audio sample rate | 22,050 Hz | 22,050 Hz |
| Audio format | Mono, 16-bit PCM | Mono, 16-bit PCM |
| Split seed | 42 | 42 |

The draft records 38 English speaker entries. Speaker identity and accent category are different labels, so these entries should not automatically be described as 38 distinct accents. The final mapping should be checked in the speaker metadata.

The Telugu training and validation sets share the same speaker because the dataset is single-speaker. Therefore, this is not a speaker-independent split. For both languages, the manifests should be checked for duplicate audio and repeated transcripts. The English speaker overlap also needs to be recorded explicitly.

### 4.2 Preprocessing Steps

The reported preprocessing steps were:

1. Prepare the audio and its matching transcript.
2. Convert audio to the required TTS format.
3. Trim leading and trailing silence and apply peak normalization.
4. Apply spectral noise reduction to the English recordings.
5. Normalize text and convert it into phoneme tokens.
6. Obtain alignment information for duration targets.
7. Extract mel spectrograms, pitch, and energy.
8. Prepare training and validation metadata.

Noise reduction can also change speech details. Its effect should be checked by listening to recordings before and after processing.

### 4.3 Acoustic Features

| Feature setting | Value |
| --- | --- |
| FFT size | 1,024 |
| Window length | 1,024 samples |
| Hop length | 256 samples |
| Mel channels | 80 |
| Pitch extraction | PyWorld `dio` and `stonemask` |
| Energy extraction | L2 norm of the STFT magnitude |

The draft reports the following normalization statistics:

| Statistic | English | Telugu |
| --- | ---: | ---: |
| Pitch mean (Hz) | 176.86 | 208.39 |
| Pitch standard deviation (Hz) | 57.12 | 40.10 |
| Energy mean | 48.62 | 41.43 |
| Energy standard deviation | 38.83 | 36.25 |

Energy values depend on the feature extraction method and are not sound levels in decibels. Normalization statistics should be fitted on the training set and reused for validation and inference. The original report does not establish whether this restriction was followed.

## 5. Implementation Changes

### 5.1 English Text Processing

I added text processing rules for times, dates, and selected words. For example, `10:45 AM` was expanded into “ten forty five a m.” The romanized Telugu word `andaroo` was assigned an approximate English phoneme sequence for a code-switching test.

I also added rules for the word *conduct*, which has different stress patterns depending on its use:

| Usage | Example | Target phonemes |
| --- | --- | --- |
| Noun | The orchestra's conduct | `K AA1 N D AH0 K T` |
| Verb | The researchers will conduct | `K AH0 N D AH1 K T` |

These rules use nearby words to select a pronunciation. They cover the selected examples but do not provide a general solution for all words with multiple pronunciations. Correct phoneme input also does not guarantee that the generated audio expresses the intended stress clearly.

The tokenizer retained word boundaries using `sp` pause tokens. An accidental space inside a word, such as `rhyth m`, could therefore affect pronunciation or introduce a pause.

### 5.2 Alignment

The project draft describes a custom alignment script, `project/acoustic_aligner.py`, using TorchAudio's MMS forced alignment pipeline as an alternative to MFA on Windows.

MMS-FA aligns normalized text tokens with audio. Its default token timestamps are not automatically phoneme boundaries. A phoneme-based FastSpeech 2 pipeline therefore needs a documented conversion from those timestamps to phoneme duration labels. This conversion should be checked in the custom script.

The aligner input must also use the sample rate required by its model. The 22,050 Hz TTS sample rate should not be assumed to be the alignment model's input rate. These requirements are described in the [TorchAudio alignment tutorial](https://docs.pytorch.org/audio/2.3.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html).

### 5.3 Telugu Text Processing

I added a rule-based Telugu grapheme-to-phoneme converter to process Telugu characters. The implementation described in the draft handles vowels, consonants, dependent vowel signs, virama, anusvara, and visarga.

A consonant normally carries an inherent vowel. The parser changes or removes that vowel when a dependent vowel sign or virama follows it. This is needed for words with consonant clusters and doubled consonants, such as `ప్రభావాన్ని` and `పట్టు`.

The original claim of exactly 50 phonemes is not retained because the listed categories do not establish that total. The exact inventory should be taken from `symbols.py`.

### 5.4 Telugu Number and Date Processing

The Telugu normalizer expands numerical expressions before phoneme conversion.

| Input | Reported expansion |
| --- | --- |
| `2026` | రెండు వేల ఇరవై ఆరు |
| `12వ` | పన్నెండవ |
| `10:45` | పది నలభై ఐదు |

These examples show the intended normalization behavior. Additional tests are needed for other number formats and sentence contexts.

### 5.5 Environment Changes

The draft describes changes to mel-filter calls, checkpoint loading, and in-memory tensor caching for the Windows environment. The exact package versions and code changes should be recorded from the working environment. The draft's version labels are not sufficient to reproduce the setup reliably.

## 6. Training Setup

| Setting | Reported value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU memory | 6 GB |
| Operating system | Windows 11 |
| Optimizer | Adam |
| Adam beta values | 0.9 and 0.98 |
| Adam epsilon | 1e-9 |
| Learning rate schedule | Noam |
| Warmup steps | 2,000 |
| Batch size | 16 |
| Gradient clipping | 1.0 |
| Training steps | 30,000 per model |

The reported losses include mel and PostNet L1 losses, along with pitch, energy, and log-duration mean squared errors. The implementation should be checked for the weights and averaging method used to calculate the total loss.

## 7. Training Results

### 7.1 Reported Total Loss

| Model | Step | Training loss | Validation loss |
| --- | ---: | ---: | ---: |
| English | 1,000 | 5.3243 | 3.7585 |
| English | 10,000 | 1.6151 | 3.5395 |
| English | 20,000 | 1.3871 | 3.6199 |
| English | 30,000 | 1.2725 | 3.6240 |
| Telugu | 1,000 | 3.1448 | 2.4815 |
| Telugu | 10,000 | 0.8623 | 2.2592 |
| Telugu | 20,000 | 0.7610 | 2.2582 |
| Telugu | 30,000 | 0.7132 | 2.2700 |

### 7.2 Final Component Losses

| Metric at step 30,000 | English | Telugu |
| --- | ---: | ---: |
| Training mel loss | 0.5678 | 0.3380 |
| Training PostNet mel loss | 0.5673 | 0.3374 |
| Training pitch loss | 0.0515 | 0.0176 |
| Training energy loss | 0.0587 | 0.0146 |
| Training duration loss | 0.0272 | 0.0056 |
| Validation mel loss | 1.0120 | 0.6287 |
| Validation PostNet mel loss | 1.0117 | 0.6286 |

**Loss consistency:** Some reported totals do not match the sum of their listed components. For example, the final Telugu components sum to **0.7132**, which matches its total. The final English components sum to **1.2725**, which also matches. However, the original English components at step 10,000 sum to **1.3965**, while its reported total is **1.6151**. Other intermediate rows also contain differences. These values should be checked against the raw logs rather than corrected by guessing.

### 7.3 Interpretation

Training loss decreased for both models. Validation loss improved at first and then remained nearly unchanged or increased slightly. This shows a gap between fitting the training data and performing well on validation data, and it may indicate overfitting.

Among the checkpoints listed, English had its lowest validation loss at step 10,000. Telugu had its lowest validation loss at step 20,000. The final checkpoint was therefore not the best listed checkpoint for either language.

Telugu had lower reported losses than English. Consistent single-speaker recordings may have contributed to this result. However, the two datasets differ in speakers, language, recording conditions, and feature distributions. Their losses alone cannot establish which model produces better speech.

## 8. Speech Generation Tests

The following sentences were used in the reported tests. They cover several text-processing challenges. The descriptions explain what each test examines; they do not claim verified listening results.

### 8.1 English Sentences

| ID | Sentence | Purpose |
| --- | --- | --- |
| E1 | The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings. | Check a long word and sentence completeness. |
| E2a | The orchestra's conduct during the ceremony was praised by the audience. | Check noun stress in “conduct.” |
| E2b | The researchers will conduct an experiment to evaluate the proposed method. | Check verb stress in “conduct.” |
| E3 | The rhyth m of the orchestra changed unexpectedly. | Check the effect of a space inside a word. |
| E4 | The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time. | Check dates, times, and a romanized Telugu word. |

### 8.2 Telugu Sentences

| ID | Sentence | Purpose |
| --- | --- | --- |
| T1 | న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. | Check a technical word, long vowels, and consonant clusters. |
| T2a | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. | Check “పట్టు” in the sense of grip. |
| T2b | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. | Check “పట్టు” in the sense of silk. |
| T3 | పులి అడ విలో నిశ్శబ్దంగా నడిచింది. | Check an unusual word break and doubled consonants. |
| T4 | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. | Check number, date, and time expansion in a longer sentence. |

The Telugu word `పట్టు` has different meanings in T2a and T2b, but this does not require a different phoneme sequence. Differences in duration or pitch can come from sentence context and do not prove that the model understands the word's meaning.

The original report describes clear pronunciation and natural speech for these examples. Those statements need confirmation from the saved audio and a documented listening evaluation. No listener scores, word error rates, or formal intelligibility results were provided.

## 9. Duration, Pitch, and Energy Analysis

### 9.1 Duration

With a hop length of 256 samples and a sample rate of 22,050 Hz, one mel frame represents approximately **11.61 ms**.

The draft reports the following examples from the predicted durations:

| Example | Reported duration | Approximate time |
| --- | ---: | ---: |
| First vowel in noun “conduct” | 10 frames | 116.1 ms |
| First vowel in verb “conduct” | 8 frames | 92.9 ms |
| Pause in `rhyth m` | 2 frames | 23.2 ms |
| `పట్టు` in T2a | 41 frames | 476.0 ms |
| `పట్టు` in T2b | 37 frames | 429.6 ms |

The reported Telugu difference is four frames, or about 46.4 ms. This is a reduction of approximately 9.8% relative to T2a. These calculations are consistent with the stated frame counts, but the counts themselves still need confirmation from the saved outputs.

### 9.2 Pitch and Energy

The draft describes English pitch values around 160–220 Hz and Telugu values around 190–250 Hz in selected examples. It also describes higher energy near vowels and lower energy near pauses or consonant closures.

These observations should be checked against the plots. The analysis should state whether each plot shows model predictions or measurements extracted from the generated waveform. Normalized pitch values must be converted back to Hz before they are labelled as frequencies.

Duration, pitch, and energy plots help explain model behavior, but they do not replace listening tests.

## 10. Limitations and Next Steps

| Limitation | Next step |
| --- | --- |
| Validation loss stops improving while training loss falls. | Compare earlier checkpoints using validation results and listening tests. |
| English recordings contain speaker and recording variation. | Review the speaker mapping, sample balance, and recording quality. |
| Alignment-to-phoneme conversion is not fully documented. | Inspect duration labels and manually check a small set of alignments. |
| Rule-based pronunciation handles only selected cases. | Add more test cases before extending the rules. |
| Spaces inside words can change pronunciation. | Test input correction without changing valid word boundaries. |
| English phonemes only approximate Telugu code-switched words. | Review the pronunciation mapping with Telugu speakers. |
| Audio quality claims lack formal evaluation. | Conduct listening tests for clarity, pronunciation, and naturalness. |
| Some loss values and environment details conflict. | Regenerate tables from logs and save exact environment versions. |

The first priority is to verify the data, alignments, speaker selection, and checkpoint choice. These checks will make later model improvements easier to evaluate.

## 11. Reproducibility and Output Files

The draft refers to the following project files and output locations:

| Path | Reported purpose |
| --- | --- |
| `project/acoustic_aligner.py` | Custom alignment processing |
| `project/English/inference.py` | English speech generation |
| `project/English/text/cleaners.py` | English text normalization |
| `project/Telugu/text/telugu_phonemes.py` | Telugu phoneme conversion |
| `project/Telugu/text/telugu_numbers.py` | Telugu number and date expansion |
| `project/Results/English/test_generated_audios/` | Generated English audio |
| `project/Results/Telugu/test_generated_audios/` | Generated Telugu audio |
| `project/Results/English/variance_adaptor_plots/` | English duration, pitch, and energy plots |
| `project/Results/Telugu/variance_adaptor_plots/` | Telugu duration, pitch, and energy plots |

The original execution commands depend on custom scripts and have not been verified against the repository. They should not be treated as tested setup instructions. Before sharing runnable commands, confirm the script arguments, configuration paths, checkpoint paths, and working directory.

For a reproducible release, include:

- The exact dependency versions from the working environment.
- Training and validation manifests, with post-preprocessing counts and durations.
- Speaker IDs and sample counts for the English model.
- Model, preprocessing, and training configuration files.
- Alignment settings and the method used to obtain phoneme durations.
- Raw training logs and the selected checkpoint step.
- Inference commands, speaker selection, audio samples, and analysis plots.

## 12. Conclusion

This project explored FastSpeech 2 for English and Telugu speech synthesis. I worked on language-specific text processing, dataset preparation, model training, and analysis of duration, pitch, and energy.

Both models showed decreasing training loss, while validation improvement slowed earlier. The main lesson is that lower training loss alone is not enough to judge speech quality. Reliable alignment, suitable speaker data, checkpoint selection, and listening tests are also needed to evaluate the final system.

## References

1. Ren et al. [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), ICLR 2021.
2. [ming024/FastSpeech2 reference implementation](https://github.com/ming024/FastSpeech2).
3. [TorchAudio: Forced alignment for multilingual data](https://docs.pytorch.org/audio/2.3.0/tutorials/forced_alignment_for_multilingual_data_tutorial.html).
4. [AI4Bharat Svarah dataset](https://huggingface.co/datasets/ai4bharat/Svarah).
5. [SPRINGLab IndicTTS Telugu dataset](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu).
