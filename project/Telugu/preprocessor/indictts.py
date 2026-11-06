"""
IndicTTS Telugu Dataset Preparation, Streaming Subsampling (5h Train + 1h Val),
Telugu Pronunciation Lexicon Generation, and Alignment Helper.
"""

import io
import os
import re
import csv
import json
import shutil
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text
import sys
_proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)
from acoustic_aligner import write_acoustic_textgrid
from text.telugu_phonemes import telugu_word_to_phones


def _write_textgrid(tg_path, total_dur, words_and_phones, wav, sr):
    """
    True acoustic forced alignment using torchaudio MMS_FA neural aligner for Telugu.
    Locks every phoneme and inter-word silence boundary to true acoustic frames.
    """
    return write_acoustic_textgrid(
        tg_path, total_dur, words_and_phones, wav, sr, language="te"
    )


def _discover_local_indictts(corpus_path):
    candidates = []
    if not os.path.exists(corpus_path):
        return candidates

    festival_re = re.compile(r'\(\s*([^\s]+)\s+"([^"]+)"\s*\)')
    for root, _, files in os.walk(corpus_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname in ("txt.done.data", "txt.done.data.utf8") or fname.endswith(".data"):
                spk_hint = os.path.basename(root) or "indictts_telugu"
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        m = festival_re.search(line)
                        if m:
                            utt_id, txt = m.group(1), m.group(2)
                            for wav_candidate in [
                                os.path.join(root, "wav", f"{utt_id}.wav"),
                                os.path.join(os.path.dirname(root), "wav", f"{utt_id}.wav"),
                                os.path.join(root, f"{utt_id}.wav"),
                            ]:
                                if os.path.exists(wav_candidate):
                                    candidates.append(
                                        {"wav_path": wav_candidate, "text": txt, "speaker": spk_hint, "duration": None}
                                    )
                                    break
    return candidates


def prepare_align(config, download_hf=False, fallback_align=False):
    corpus_path = config["path"]["corpus_path"]
    raw_path = config["path"]["raw_path"]
    preprocessed_path = config["path"]["preprocessed_path"]
    lexicon_path = config["path"]["lexicon_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    train_hours = config["preprocessing"].get("train_hours", 5.0)
    val_hours = config["preprocessing"].get("val_hours", 1.0)
    seed = config["preprocessing"].get("seed", 42)

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(preprocessed_path, exist_ok=True)

    if not fallback_align and shutil.which("mfa") is None:
        print("Note: External `mfa` binary not found on PATH; enabling acoustic-energy Telugu phoneme TextGrid alignment.")
        fallback_align = True

    lexicon = {}
    target_val_sec = val_hours * 3600.0
    target_train_sec = train_hours * 3600.0
    val_sec = 0.0
    train_sec = 0.0
    train_texts = set()
    val_texts = set()
    split_manifest = {"train": [], "val": []}

    local_candidates = _discover_local_indictts(corpus_path)

    if not local_candidates and download_hf:
        print("Streaming `SPRINGLab/IndicTTS_Telugu` directly from HuggingFace (zero extra disk cache)...")
        from datasets import load_dataset, Audio

        ds = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train", streaming=True)
        for col, feat in list(ds.features.items()):
            if isinstance(feat, Audio):
                ds = ds.cast_column(col, Audio(decode=False))

        pbar = tqdm(desc="Streaming IndicTTS Telugu (Target: 5h Train + 1h Val)", unit="utt")
        for idx, item in enumerate(ds):
            if val_sec >= target_val_sec and train_sec >= target_train_sec:
                break

            raw_txt = (item.get("text") or "").strip()
            if not raw_txt or raw_txt.lower() in ("nan", "null", "[empty]"):
                continue
            cleaned_txt = _clean_text(raw_txt, cleaners).strip()
            if len(cleaned_txt) < 2:
                continue

            audio_obj = item.get("audio") or item.get("audio_filepath")
            if not audio_obj or "bytes" not in audio_obj:
                continue

            try:
                wav_arr, orig_sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32")
                if wav_arr.ndim > 1:
                    wav_arr = np.mean(wav_arr, axis=1)
            except Exception:
                continue

            dur = len(wav_arr) / float(orig_sr)
            if dur < 0.5 or dur > 20.0:
                continue

            words = [w for w in re.split(r"[,;.\-\?\!\s\"\'()]+", cleaned_txt) if w]
            words_and_phones = []
            clean_words = []
            for w in words:
                p_list = telugu_word_to_phones(w)
                if p_list:
                    clean_words.append(w)
                    lexicon[w] = p_list
                    words_and_phones.append((w, p_list))

            if not words_and_phones:
                continue

            if cleaned_txt in val_texts:
                if val_sec < target_val_sec:
                    split_role = "val"
                    val_sec += dur
                else:
                    continue
            elif cleaned_txt in train_texts:
                if train_sec < target_train_sec:
                    split_role = "train"
                    train_sec += dur
                else:
                    continue
            elif val_sec < target_val_sec and (idx % 6 == 0 or train_sec >= target_train_sec):
                split_role = "val"
                val_sec += dur
                val_texts.add(cleaned_txt)
            elif train_sec < target_train_sec:
                split_role = "train"
                train_sec += dur
                train_texts.add(cleaned_txt)
            elif val_sec < target_val_sec:
                split_role = "val"
                val_sec += dur
                val_texts.add(cleaned_txt)
            else:
                break
            if orig_sr != sampling_rate:
                wav = librosa.resample(wav_arr, orig_sr=orig_sr, target_sr=sampling_rate)
            else:
                wav = wav_arr

            gender_val = item.get("gender")
            spk = "telugu_female" if gender_val in (0, "female", "Female") else "telugu_male"
            utt_id = f"indictts_te_{idx:05d}"

            spk_dir = os.path.join(raw_path, spk)
            os.makedirs(spk_dir, exist_ok=True)

            peak = np.max(np.abs(wav))
            if peak > 0:
                wav_int16 = (wav / peak * max_wav_value).astype(np.int16)
            else:
                wav_int16 = (wav * max_wav_value).astype(np.int16)

            wavfile.write(os.path.join(spk_dir, f"{utt_id}.wav"), sampling_rate, wav_int16)
            with open(os.path.join(spk_dir, f"{utt_id}.lab"), "w", encoding="utf-8") as f:
                f.write(" ".join(clean_words))

            if fallback_align:
                tg_path = os.path.join(preprocessed_path, "TextGrid", spk, f"{utt_id}.TextGrid")
                _write_textgrid(tg_path, len(wav) / float(sampling_rate), words_and_phones, wav, sampling_rate)

            split_manifest[split_role].append(utt_id)
            pbar.update(1)
            pbar.set_postfix(
                train_h=f"{train_sec/3600:.2f}/{train_hours:.1f}h",
                val_h=f"{val_sec/3600:.2f}/{val_hours:.1f}h",
            )
        pbar.close()

    with open(os.path.join(raw_path, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)

    os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
    with open(lexicon_path, "w", encoding="utf-8") as f:
        for w in sorted(lexicon.keys()):
            f.write(f"{w}\t{' '.join(lexicon[w])}\n")

    print(
        f"\nIndicTTS Telugu Preparation Complete -> Train: {len(split_manifest['train'])} utts ({train_sec/3600:.2f}h) | "
        f"Val: {len(split_manifest['val'])} utts ({val_sec/3600:.2f}h) | Lexicon words: {len(lexicon)} | Overlap: 0"
    )
