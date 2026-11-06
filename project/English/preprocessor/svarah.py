"""
Svarah (English) Dataset Preparation, Streaming Subsampling (5h Train + 1h Val),
Lexicon Generation, and Alignment Helper.
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


def _load_existing_lexicon(lexicon_path):
    lex = {}
    if os.path.exists(lexicon_path):
        with open(lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"\s+", line.strip())
                if len(parts) >= 2:
                    w = parts[0].lower()
                    if w not in lex:
                        lex[w] = parts[1:]
    return lex


import sys
_proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)
from acoustic_aligner import write_acoustic_textgrid


def _write_textgrid(tg_path, total_dur, words_and_phones, wav, sr):
    """
    True acoustic forced alignment using torchaudio MMS_FA neural aligner.
    Locks every phoneme and inter-word silence boundary to true acoustic frames.
    """
    return write_acoustic_textgrid(
        tg_path, total_dur, words_and_phones, wav, sr, language="en"
    )


def _discover_local_svarah(corpus_path):
    candidates = []
    if not os.path.exists(corpus_path):
        return candidates

    for root, _, files in os.walk(corpus_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname.endswith(".jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        wav_p = obj.get("audio_filepath") or obj.get("audio_path") or obj.get("path")
                        txt = obj.get("text") or obj.get("transcript") or obj.get("english_text")
                        spk = str(obj.get("speaker_id") or obj.get("speaker") or "svarah_spk")
                        dur = obj.get("duration")
                        if wav_p and not os.path.isabs(wav_p):
                            wav_p = os.path.join(root, wav_p)
                        if wav_p and txt and os.path.exists(wav_p):
                            candidates.append({"wav_path": wav_p, "text": txt, "speaker": spk, "duration": dur})
            elif fname.endswith(".csv") or fname.endswith(".tsv"):
                delim = "\t" if fname.endswith(".tsv") else ","
                with open(fpath, "r", encoding="utf-8") as f:
                    sample = f.read(2048)
                    f.seek(0)
                    if "|" in sample and "," not in sample:
                        for line in f:
                            parts = line.strip().split("|")
                            if len(parts) >= 2:
                                bname, txt = parts[0], parts[-1]
                                wav_p = os.path.join(root, "wavs", f"{bname}.wav")
                                if not os.path.exists(wav_p):
                                    wav_p = os.path.join(root, f"{bname}.wav")
                                if os.path.exists(wav_p):
                                    candidates.append({"wav_path": wav_p, "text": txt, "speaker": "svarah_spk", "duration": None})
                    else:
                        reader = csv.DictReader(f, delimiter=delim)
                        for row in reader:
                            wav_p = (
                                row.get("audio_filepath")
                                or row.get("audio_path")
                                or row.get("file_name")
                                or row.get("wav")
                                or row.get("path")
                            )
                            txt = row.get("text") or row.get("transcript") or row.get("sentence")
                            spk = str(row.get("speaker_id") or row.get("speaker") or "svarah_spk")
                            dur = float(row["duration"]) if row.get("duration") else None
                            if wav_p and not os.path.isabs(wav_p):
                                wav_p = os.path.join(root, wav_p)
                            if wav_p and txt and os.path.exists(wav_p):
                                candidates.append({"wav_path": wav_p, "text": txt, "speaker": spk, "duration": dur})
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

    # Auto-enable fallback_align if MFA CLI is not installed on this machine
    if not fallback_align and shutil.which("mfa") is None:
        print("Note: External `mfa` binary not found on PATH; enabling acoustic-energy phoneme TextGrid alignment.")
        fallback_align = True

    lexicon = _load_existing_lexicon(lexicon_path)
    try:
        from g2p_en import G2p
        g2p = G2p()
    except Exception:
        g2p = None

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        voice_encoder = VoiceEncoder()
        print("Initialized Resemblyzer VoiceEncoder for utterance-level speaker embedding extraction.")
    except Exception as e:
        print(f"Warning: Could not initialize Resemblyzer VoiceEncoder: {e}")
        voice_encoder = None

    target_val_sec = val_hours * 3600.0
    target_train_sec = train_hours * 3600.0
    val_sec = 0.0
    train_sec = 0.0
    train_texts = set()
    val_texts = set()
    split_manifest = {"train": [], "val": []}

    local_candidates = _discover_local_svarah(corpus_path)

    if not local_candidates and download_hf:
        print("Streaming `ai4bharat/Svarah` directly from HuggingFace (zero extra disk cache)...")
        from datasets import load_dataset, Audio

        ds = load_dataset("ai4bharat/Svarah", split="test", streaming=True)
        for col, feat in list(ds.features.items()):
            if isinstance(feat, Audio):
                ds = ds.cast_column(col, Audio(decode=False))

        pbar = tqdm(desc="Streaming Svarah (Target: 5h Train + 1h Val)", unit="utt")
        for idx, item in enumerate(ds):
            if val_sec >= target_val_sec and train_sec >= target_train_sec:
                break

            raw_txt = (item.get("text") or "").strip()
            if not raw_txt or raw_txt.lower() in ("nan", "null", "[empty]"):
                continue
            cleaned_txt = _clean_text(raw_txt, cleaners).strip()
            if len(cleaned_txt) < 3:
                continue

            audio_obj = item.get("audio_filepath") or item.get("audio")
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

            # Assign deterministically with zero overlap between train and val (both audio and text)
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

            # Data Cleaning & Denoising Pipeline:
            try:
                import noisereduce as nr
                wav = nr.reduce_noise(y=wav, sr=sampling_rate, stationary=True, prop_decrease=0.85)
            except Exception:
                pass

            # Trim leading/trailing ambient silence and room reverb
            try:
                wav, _ = librosa.effects.trim(wav, top_db=30)
            except Exception:
                pass

            if len(wav) < int(sampling_rate * 0.5):
                continue

            gender = re.sub(r"[^a-zA-Z0-9]", "", str(item.get("gender") or "Spk"))
            lang_bg = re.sub(r"[^a-zA-Z0-9]", "", str(item.get("primary_language") or "IN"))
            spk = f"{gender}_{lang_bg}"
            utt_id = f"svarah_{idx:05d}"

            spk_dir = os.path.join(raw_path, spk)
            os.makedirs(spk_dir, exist_ok=True)

            peak = np.max(np.abs(wav))
            if peak > 0:
                wav_int16 = (wav / peak * max_wav_value).astype(np.int16)
            else:
                wav_int16 = (wav * max_wav_value).astype(np.int16)

            wavfile.write(os.path.join(spk_dir, f"{utt_id}.wav"), sampling_rate, wav_int16)
            with open(os.path.join(spk_dir, f"{utt_id}.lab"), "w", encoding="utf-8") as f:
                f.write(cleaned_txt)

            words = re.findall(r"[a-zA-Z']+", cleaned_txt.lower())
            words_and_phones = []
            for w in words:
                if w not in lexicon:
                    if g2p is not None:
                        p_list = [p for p in g2p(w) if p != " " and re.match(r"^[A-Z0-9]+$", p)]
                    else:
                        p_list = ["AH0"]
                    if not p_list:
                        p_list = ["AH0"]
                    lexicon[w] = p_list
                words_and_phones.append((w, lexicon[w]))

            if fallback_align:
                tg_path = os.path.join(preprocessed_path, "TextGrid", spk, f"{utt_id}.TextGrid")
                _write_textgrid(tg_path, len(wav) / float(sampling_rate), words_and_phones, wav, sampling_rate)

            # Continuous utterance-level acoustic speaker embedding (d-vector)
            if voice_encoder is not None:
                spk_emb_dir = os.path.join(preprocessed_path, "spk_emb")
                os.makedirs(spk_emb_dir, exist_ok=True)
                try:
                    wav_proc = preprocess_wav(wav, source_sr=sampling_rate)
                    spk_emb = voice_encoder.embed_utterance(wav_proc).astype(np.float32)
                    np.save(os.path.join(spk_emb_dir, f"{spk}-spk_emb-{utt_id}.npy"), spk_emb)
                    np.save(os.path.join(spk_emb_dir, f"{utt_id}.npy"), spk_emb)
                except Exception as emb_e:
                    print(f"Warning: Failed to extract embedding for {utt_id}: {emb_e}")

            split_manifest[split_role].append(utt_id)
            pbar.update(1)
            pbar.set_postfix(
                train_h=f"{train_sec/3600:.2f}/{train_hours:.1f}h",
                val_h=f"{val_sec/3600:.2f}/{val_hours:.1f}h",
            )
        pbar.close()

    elif local_candidates:
        rng = random.Random(seed)
        rng.shuffle(local_candidates)
        for idx, item in enumerate(tqdm(local_candidates, desc="Processing Local Svarah")):
            if val_sec >= target_val_sec and train_sec >= target_train_sec:
                break
            raw_txt = item["text"].strip()
            cleaned_txt = _clean_text(raw_txt, cleaners).strip()
            if len(cleaned_txt) < 3 or cleaned_txt in seen_texts:
                continue
            wav, _ = librosa.load(item["wav_path"], sr=sampling_rate)
            dur = len(wav) / float(sampling_rate)
            if dur < 1.0 or dur > 15.0:
                continue
            if val_sec < target_val_sec:
                split_role = "val"
                val_sec += dur
            elif train_sec < target_train_sec:
                split_role = "train"
                train_sec += dur
            else:
                break
            seen_texts.add(cleaned_txt)
            utt_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", os.path.splitext(os.path.basename(item["wav_path"]))[0])
            spk = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(item["speaker"])) or "svarah_spk"

            # Data Cleaning & Denoising Pipeline:
            try:
                import noisereduce as nr
                wav = nr.reduce_noise(y=wav, sr=sampling_rate, stationary=True, prop_decrease=0.85)
            except Exception:
                pass
            try:
                wav, _ = librosa.effects.trim(wav, top_db=30)
            except Exception:
                pass
            dur = len(wav) / float(sampling_rate)
            if dur < 0.5:
                continue

            spk_dir = os.path.join(raw_path, spk)
            os.makedirs(spk_dir, exist_ok=True)
            peak = np.max(np.abs(wav))
            wav_int16 = ((wav / peak if peak > 0 else wav) * max_wav_value).astype(np.int16)
            wavfile.write(os.path.join(spk_dir, f"{utt_id}.wav"), sampling_rate, wav_int16)
            with open(os.path.join(spk_dir, f"{utt_id}.lab"), "w", encoding="utf-8") as f:
                f.write(cleaned_txt)
            words = re.findall(r"[a-zA-Z']+", cleaned_txt.lower())
            words_and_phones = []
            for w in words:
                if w not in lexicon:
                    p_list = [p for p in g2p(w) if p != " " and re.match(r"^[A-Z0-9]+$", p)] if g2p else ["AH0"]
                    lexicon[w] = p_list or ["AH0"]
                words_and_phones.append((w, lexicon[w]))
            if fallback_align:
                tg_path = os.path.join(preprocessed_path, "TextGrid", spk, f"{utt_id}.TextGrid")
                _write_textgrid(tg_path, dur, words_and_phones, wav, sampling_rate)

            # Continuous utterance-level acoustic speaker embedding (d-vector)
            if voice_encoder is not None:
                spk_emb_dir = os.path.join(preprocessed_path, "spk_emb")
                os.makedirs(spk_emb_dir, exist_ok=True)
                try:
                    wav_proc = preprocess_wav(wav, source_sr=sampling_rate)
                    spk_emb = voice_encoder.embed_utterance(wav_proc).astype(np.float32)
                    np.save(os.path.join(spk_emb_dir, f"{spk}-spk_emb-{utt_id}.npy"), spk_emb)
                    np.save(os.path.join(spk_emb_dir, f"{utt_id}.npy"), spk_emb)
                except Exception as emb_e:
                    print(f"Warning: Failed to extract embedding for {utt_id}: {emb_e}")

            split_manifest[split_role].append(utt_id)

    with open(os.path.join(raw_path, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    print(
        f"\nSvarah Preparation Complete -> Train: {len(split_manifest['train'])} utts ({train_sec/3600:.2f}h) | "
        f"Val: {len(split_manifest['val'])} utts ({val_sec/3600:.2f}h) | Overlap: 0"
    )
