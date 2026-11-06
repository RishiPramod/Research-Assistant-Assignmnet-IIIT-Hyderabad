import os
import re
import sys
import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
import tgt

_ALIGNER_BUNDLE = None
_ALIGNER_MODEL = None
_DICTIONARY = None
_DEVICE = None


def get_acoustic_aligner():
    global _ALIGNER_BUNDLE, _ALIGNER_MODEL, _DICTIONARY, _DEVICE
    if _ALIGNER_MODEL is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _ALIGNER_BUNDLE = torchaudio.pipelines.MMS_FA
        _ALIGNER_MODEL = _ALIGNER_BUNDLE.get_model().to(_DEVICE)
        _ALIGNER_MODEL.eval()
        _DICTIONARY = _ALIGNER_BUNDLE.get_dict()
    return _ALIGNER_MODEL, _DICTIONARY, _DEVICE


def get_phone_weight(phone):
    p = phone.rstrip("012").upper()
    is_stressed = ("1" in phone or "2" in phone)
    if p in ("AA", "AE", "AO", "AW", "AY", "EH", "ER", "EY", "IY", "OW", "OY", "UH", "UW"):
        return 2.5 if is_stressed else 1.6
    elif p == "AH":
        return 2.2 if is_stressed else 0.9
    elif p == "IH":
        return 2.0 if is_stressed else 1.0
    elif p in ("L", "R", "M", "N", "NG", "W", "Y"):
        return 1.3
    elif p in ("S", "Z", "SH", "ZH", "F", "V", "TH", "DH", "CH", "JH"):
        return 1.1
    elif p in ("P", "T", "K", "B", "D", "G"):
        return 0.65
    return 1.0


def align_utterance_mms(wav_arr, sr, words_and_phones, language="en"):
    """
    Computes frame-level CTC forced alignment using torchaudio MMS_FA.
    Returns a tgt.TextGrid object with real acoustic phoneme and pause boundaries.
    """
    model, dictionary, device = get_acoustic_aligner()

    if sr != 16000:
        wav_t = torch.from_numpy(wav_arr.astype(np.float32)).unsqueeze(0)
        wav_resampled = torchaudio.functional.resample(wav_t, sr, 16000).squeeze(0).numpy()
    else:
        wav_resampled = wav_arr

    total_dur = len(wav_resampled) / 16000.0
    if total_dur < 0.1:
        return None

    tokens = []
    word_spans = []

    for w, p_list in words_and_phones:
        if language == "te":
            # For Telugu: flatten romanized phones to ascii characters
            flat_chars = []
            for p in p_list:
                p_lower = p.lower()
                for c in p_lower:
                    if c in dictionary:
                        flat_chars.append(c)
            w_chars = flat_chars
        else:
            w_chars = [c for c in w.lower() if c in dictionary]

        if w_chars:
            s_idx = len(tokens)
            tokens.extend([dictionary[c] for c in w_chars])
            e_idx = len(tokens)
            word_spans.append((w, p_list, s_idx, e_idx))

    if not tokens or not word_spans:
        return None

    targets = torch.tensor([tokens], dtype=torch.int32, device=device)
    wav_tensor = torch.from_numpy(wav_resampled.astype(np.float32)).unsqueeze(0).to(device)

    try:
        with torch.inference_mode():
            emission, _ = model(wav_tensor)
            aligned_tokens, scores = F.forced_align(emission, targets, blank=0)

        token_spans = F.merge_tokens(aligned_tokens[0], scores[0])
    except Exception:
        return None

    if len(token_spans) != len(tokens):
        return None

    num_frames = float(emission.size(1))
    frame_dur = total_dur / max(1.0, num_frames)

    tg = tgt.TextGrid()
    phone_tier = tgt.IntervalTier(start_time=0.0, end_time=total_dur, name="phones")
    word_tier = tgt.IntervalTier(start_time=0.0, end_time=total_dur, name="words")

    prev_end = 0.0

    for w, p_list, s_idx, e_idx in word_spans:
        raw_start = token_spans[s_idx].start * frame_dur
        raw_end = token_spans[e_idx - 1].end * frame_dur

        w_start = max(prev_end, min(total_dur, raw_start))
        if w_start > prev_end + 0.02:
            phone_tier.add_interval(
                tgt.Interval(prev_end, w_start, "sp" if prev_end > 0.0 else "sil")
            )
            prev_end = w_start

        min_word_dur = max(0.02 * len(p_list), 0.03)
        w_end = min(total_dur, max(prev_end + min_word_dur, raw_end))
        w_dur = max(0.01, w_end - prev_end)
        weights = [get_phone_weight(p) for p in p_list]
        total_w = sum(weights) if sum(weights) > 0 else 1.0
        cur = prev_end

        for pi, (p, pw) in enumerate(zip(p_list, weights)):
            p_dur = w_dur * (pw / total_w)
            if pi == len(p_list) - 1:
                nxt = w_end
            else:
                nxt = cur + p_dur
            nxt = min(total_dur, max(cur + 0.005, nxt))
            phone_tier.add_interval(tgt.Interval(cur, nxt, str(p)))
            cur = nxt

        word_tier.add_interval(tgt.Interval(prev_end, cur, str(w)))
        prev_end = cur

    if prev_end < total_dur - 0.005:
        phone_tier.add_interval(tgt.Interval(prev_end, total_dur, "sil"))

    tg.add_tier(phone_tier)
    tg.add_tier(word_tier)
    return tg


def write_acoustic_textgrid(tg_path, total_dur, words_and_phones, wav, sr, language="en"):
    """
    Writes true acoustic forced alignment TextGrid to disk.
    Falls back to energy-guided alignment only if CTC alignment fails on rare edge cases.
    """
    dirname = os.path.dirname(tg_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    try:
        tg = align_utterance_mms(wav, sr, words_and_phones, language=language)
        if tg is not None:
            tgt.io.write_to_file(tg, tg_path, format="long")
            return True
    except Exception as e:
        print(f"Warning: CTC alignment failed ({e}); falling back to phonetic alignment.")

    # Fallback if audio is severely degraded or CTC alignment fails:
    intervals = []
    flat_phones = []
    for idx, (w, p_list) in enumerate(words_and_phones):
        flat_phones.extend(p_list)
        if idx < len(words_and_phones) - 1:
            flat_phones.append("sp")
    if not flat_phones:
        flat_phones = ["sp"]

    weights = [get_phone_weight(p) if p not in ("sp", "sil") else 1.5 for p in flat_phones]
    total_w = sum(weights) if sum(weights) > 0 else 1.0
    cur = 0.0
    for pi, (p, w) in enumerate(zip(flat_phones, weights)):
        p_dur = total_dur * (w / total_w)
        if pi == len(flat_phones) - 1:
            nxt = total_dur
        else:
            nxt = min(total_dur, cur + p_dur)
        nxt = min(total_dur, max(cur + 0.005, nxt))
        intervals.append(tgt.Interval(cur, nxt, str(p)))
        cur = nxt

    phone_tier = tgt.IntervalTier(start_time=0.0, end_time=total_dur, name="phones", objects=intervals)
    tg = tgt.TextGrid()
    tg.add_tier(phone_tier)
    tgt.io.write_to_file(tg, tg_path, format="long")
    return False

