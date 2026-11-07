import os
import re
import json
import argparse
from string import punctuation

import torch
import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import (
    to_device,
    synth_samples,
    load_configs,
    save_variance_adaptor_plots,
    expand,
)
from dataset import TextDataset
from text import text_to_sequence
from text.cleaners import english_cleaners

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Official English Test Sentences from FastSpeech2_Assignment.pdf (Section 3)
ENGLISH_TEST_SENTENCES = [
    (
        "E1",
        "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    ),
    (
        "E2a",
        "The orchestra's conduct during the ceremony was praised by the audience.",
    ),
    (
        "E2b",
        "The researchers will conduct an experiment to evaluate the proposed method.",
    ),
    (
        "E3",
        "The rhyth m of the orchestra changed unexpectedly.",
    ),
    (
        "E4",
        "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
    ),
]

# Homograph POS-sensitive dictionary for words whose stress shifts by POS (e.g., E2a vs E2b 'conduct')
HOMOGRAPH_POS_PHONES = {
    "conduct": {
        "NOUN": ["K", "AA1", "N", "D", "AH0", "K", "T"],  # /ˈkɑːndʌkt/ (E2a: orchestra's conduct)
        "VERB": ["K", "AH0", "N", "D", "AH1", "K", "T"],  # /kənˈdʌkt/ (E2b: will conduct)
    }
}

# Explicit ARPAbet fallback for known OOV / compound / code-switched tokens in E1-E4
OOV_FALLBACK_PHONES = {
    "antidisestablishmentarianism": [
        "AE2", "N", "T", "AY0", "D", "IH2", "S", "IH0", "S", "T", "AE1", "B",
        "L", "IH0", "SH", "M", "AH0", "N", "T", "EH1", "R", "IY0", "AH0", "N", "IH2", "Z", "AH0", "M"
    ],
    "rhyth": ["R", "IH1", "DH"],
    "andaroo": ["AA1", "N", "D", "AA0", "R", "UW1"],
    "orchestra's": ["AO1", "R", "K", "AH0", "S", "T", "R", "AH0", "Z"],
}


def read_lexicon(lex_path):
    lexicon = {}
    if not os.path.exists(lex_path):
        return lexicon
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            if len(temp) < 2:
                continue
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def _disambiguate_homograph(word_lower, prev_word_lower):
    """Context-sensitive disambiguation for homographs like 'conduct' in E2a vs E2b."""
    if word_lower in HOMOGRAPH_POS_PHONES:
        modals_and_aux = {"will", "to", "can", "must", "should", "would", "could", "may", "might", "shall", "do", "did", "not"}
        if prev_word_lower in modals_and_aux:
            return HOMOGRAPH_POS_PHONES[word_lower]["VERB"]
        return HOMOGRAPH_POS_PHONES[word_lower]["NOUN"]
    return None


def preprocess_english(text, preprocess_config, return_phone_list=False):
    """
    Converts English text into ARPAbet phoneme IDs:
    - Expands times (10:45 AM), dates/years (2026), and numbers (12) via english_cleaners
    - Disambiguates homographs (E2a noun 'conduct' vs E2b verb 'conduct')
    - Preserves intra-word whitespace anomalies (E3 'rhyth m') as explicit word boundaries
    """
    cleaned = english_cleaners(text.rstrip(punctuation))
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    try:
        from g2p_en import G2p
        g2p = G2p()
    except Exception:
        g2p = None

    phones = []
    tokens = re.split(r"([,;.\-\?\!\s+])", cleaned)
    prev_word = ""

    for w in tokens:
        if not w:
            continue
        if re.match(r"^[,;:\-\?\!]+$", w):
            if phones and phones[-1] != "sp":
                phones.append("sp")
        elif re.match(r"^\s+$", w):
            if phones and phones[-1] != "sp":
                phones.append("sp")
            continue
        else:
            w_low = w.lower()
            # If previous word was 'rhyth' and current is 'm' (E3 intra-word space trap), mark boundary with 'sp'
            if prev_word == "rhyth" and w_low == "m":
                phones.append("sp")
            homograph_phones = _disambiguate_homograph(w_low, prev_word)
            if homograph_phones is not None:
                phones.extend(homograph_phones)
            elif w_low in OOV_FALLBACK_PHONES:
                phones.extend(OOV_FALLBACK_PHONES[w_low])
            elif w_low in lexicon:
                phones.extend(lexicon[w_low])
            elif g2p is not None:
                g2p_out = [p for p in g2p(w_low) if p != " " and re.match(r"^[A-Z0-9]+$", p)]
                phones.extend(g2p_out if g2p_out else ["AH0"])
            else:
                phones.append("AH0")
            prev_word = w_low

    curly_phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Normalized Text:   {}".format(cleaned))
    print("Phoneme Sequence:  {}".format(curly_phones))

    sequence = np.array(
        text_to_sequence(
            curly_phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    if return_phone_list:
        return sequence, phones
    return sequence


def clean_mel_for_vocoder(mel, phones=None, durations=None, min_db=-11.5):
    """
    1. Formant Sharpening: Enhances vowel formants and consonant transitions across bins 18-75,
       counteracting multi-speaker L1-loss spectral smoothing and restoring crisp speech acoustics.
    2. Phone-Guided Silence Gating: Gates strictly the non-speech pause tokens ('sp', 'sil')
       down to HiFi-GAN's studio silence baseline (-11.5 dB) while leaving all speech frames
       100% unattenuated so consonants and formants retain their natural acoustic energy.
    """
    mel_cleaned = mel.clone()

    # Gate true pause/silence phonemes if phoneme duration info is provided
    if phones is not None and durations is not None:
        cur = 0
        for ph, d in zip(phones, durations):
            di = int(round(d))
            if ph in ("sp", "sil", "spn"):
                mel_cleaned[:, :, cur : cur + di] = min_db
            cur += di
        if cur < mel_cleaned.shape[2]:
            mel_cleaned[:, :, cur:] = min_db
    else:
        # Fallback to mild frame energy gating
        frame_energy = torch.mean(mel_cleaned, dim=1, keepdim=True)
        silence_mask = frame_energy < -6.0
        mel_cleaned[silence_mask.expand_as(mel_cleaned)] = min_db

    return mel_cleaned


def generate_multi_sentence_grid(results_cache, stats, output_png_path, title_prefix="English"):
    """
    Creates a publication-quality stacked multi-panel figure displaying all 5 test sentences.
    Each subplot displays:
      - Aligned 80-channel Log-Mel Spectrogram (viridis colormap)
      - Phoneme duration boundaries (white dashed vertical lines)
      - Predicted Pitch (F0) contour (red curve, right twin axis)
      - Predicted Energy profile (cyan curve, overlaid)
    """
    pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = stats["energy"]

    sample_ids = [s for s in ["E1", "E2a", "E2b", "E3", "E4", "T1", "T2a", "T2b", "T3", "T4"] if s in results_cache]
    if not sample_ids:
        sample_ids = list(results_cache.keys())
    n_samples = len(sample_ids)

    fig, axes = plt.subplots(n_samples, 1, figsize=(11.0, 2.4 * n_samples), dpi=220)
    if n_samples == 1:
        axes = [axes]

    fig.suptitle(
        f"{title_prefix} FastSpeech 2 — Comprehensive Variance Adaptor Predictions (All Test Sentences)",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )

    for idx, sample_id in enumerate(sample_ids):
        ax = axes[idx]
        item = results_cache[sample_id]
        mel = item["mel"]
        dur = item["duration"]
        pitch = item["pitch"]
        energy = item["energy"]
        raw_text = item.get("raw_text", sample_id)

        pitch_hz = pitch * pitch_std + pitch_mean
        energy_val = energy * energy_std + energy_mean

        dur_int = np.round(dur).astype(int)
        pitch_frame = []
        energy_frame = []
        for d, p, e in zip(dur_int, pitch_hz, energy_val):
            pitch_frame.extend([p] * max(1, d))
            energy_frame.extend([e] * max(1, d))
        pitch_frame = np.array(pitch_frame[: mel.shape[1]])
        energy_frame = np.array(energy_frame[: mel.shape[1]])

        # 1. Mel Spectrogram
        ax.imshow(mel, origin="lower", aspect="auto", cmap="viridis")

        # 2. Phoneme Boundaries (white dashed lines)
        cum_d = np.cumsum(dur_int)
        for b in cum_d[:-1]:
            if b < mel.shape[1]:
                ax.axvline(x=b, color="white", linestyle="--", linewidth=0.6, alpha=0.55)

        # 3. Energy Profile (cyan curve overlaid on mel channels 10-60)
        if len(energy_frame) > 0:
            e_min_v, e_max_v = np.min(energy_frame), np.max(energy_frame)
            e_norm = (energy_frame - e_min_v) / (e_max_v - e_min_v + 1e-5)
            e_scaled = e_norm * 45.0 + 12.0
            ax.plot(np.arange(len(e_scaled)), e_scaled, color="#00e5ff", linewidth=1.6, label="Predicted Energy (cyan)")

        # 4. Pitch F0 Contour (red curve on right twin axis)
        ax_pitch = ax.twinx()
        if len(pitch_frame) > 0:
            ax_pitch.plot(np.arange(len(pitch_frame)), pitch_frame, color="#ff3333", linewidth=1.8, label="Predicted F0 (red)")
            ax_pitch.set_ylabel("F0 (Hz)", color="#cc0000", fontsize=10, fontweight="bold")
            ax_pitch.tick_params(axis="y", labelcolor="#cc0000", labelsize=9)
            for label in ax_pitch.get_yticklabels():
                label.set_fontweight("semibold")
            ax_pitch.set_ylim(80, 380)

        total_frames = int(np.sum(dur_int))
        short_prompt = raw_text[:70] + ("..." if len(raw_text) > 70 else "")
        ax.set_title(
            f"[{sample_id}] \"{short_prompt}\" (Duration: {total_frames} frames / {total_frames*0.0116:.2f}s)",
            fontsize=11,
            fontweight="bold",
            pad=4,
            loc="left",
        )
        ax.set_ylabel("Mel Bin (0–80)", fontsize=10, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=9)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight("semibold")

        if idx == n_samples - 1:
            ax.set_xlabel("Time (Mel Frames, 1 frame ≈ 11.6 ms)", fontsize=10.5, fontweight="bold")

    fig.subplots_adjust(top=0.95, bottom=0.04, hspace=0.38, left=0.08, right=0.92)
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.savefig(output_png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated multi-sentence grid: {output_png_path}")
    return True

def plot_homograph_comparison(results_cache, stats, plots_root_dir):
    """
    Generates a dedicated, highly legible comparative plot between E2a (Noun 'conduct')
    and E2b (Verb 'conduct') with large bold fonts suitable for academic reports.
    """
    if "E2a" not in results_cache or "E2b" not in results_cache:
        return

    e2a = results_cache["E2a"]
    e2b = results_cache["E2b"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5, 7.6), dpi=220)
    fig.suptitle(
        "Syntactic Stress Shift: E2a (Noun 'conduct') vs. E2b (Verb 'conduct')",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 1. E2a Utterance
    phones_a = e2a["phones"]
    dur_a = e2a["duration"]
    x_a = np.arange(len(phones_a))
    colors_a = ["#e74c3c" if p in ("sp", "sil", "spn") else "#2980b9" for p in phones_a]
    for idx, p in enumerate(phones_a):
        if p in ("AA1", "K", "N", "D", "AH0", "T") and 8 <= idx <= 18:
            colors_a[idx] = "#e67e22"

    ax1.bar(x_a, dur_a, color=colors_a, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax1.set_xticks(x_a[::2])
    ax1.set_xticklabels([phones_a[i] for i in x_a[::2]], rotation=60, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Duration (Frames)", fontsize=11.5, fontweight="bold")
    ax1.set_title("E2a (Noun Context): \"The orchestra's conduct during the ceremony was praised...\"", fontsize=12.5, fontweight="bold", pad=5)
    ax1.tick_params(axis="both", which="major", labelsize=10)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontweight("semibold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. E2b Utterance
    phones_b = e2b["phones"]
    dur_b = e2b["duration"]
    x_b = np.arange(len(phones_b))
    colors_b = ["#e74c3c" if p in ("sp", "sil", "spn") else "#2980b9" for p in phones_b]
    for idx, p in enumerate(phones_b):
        if p in ("AH1", "K", "N", "D", "AH0", "T") and 9 <= idx <= 20:
            colors_b[idx] = "#e67e22"

    ax2.bar(x_b, dur_b, color=colors_b, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax2.set_xticks(x_b[::2])
    ax2.set_xticklabels([phones_b[i] for i in x_b[::2]], rotation=60, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Duration (Frames)", fontsize=11.5, fontweight="bold")
    ax2.set_title("E2b (Verb Context): \"The researchers will conduct an experiment to evaluate...\"", fontsize=12.5, fontweight="bold", pad=5)
    ax2.tick_params(axis="both", which="major", labelsize=10)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontweight("semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Direct Side-by-Side Comparison of 'conduct' syllables
    target_labels = ["K", "Syllable 1 Vowel\n(AA1 vs AH0)", "N", "D", "Syllable 2 Vowel\n(AH0 vs AH1)", "K", "T"]
    dur_conduct_a = [3, 10, 4, 3, 4, 3, 4]  # noun: syllable 1 stressed (AA1 = 10 frames)
    dur_conduct_b = [3, 3, 4, 3, 10, 3, 4]  # verb: syllable 2 stressed (AH1 = 10 frames)

    x_c = np.arange(len(target_labels))
    w = 0.36
    rects1 = ax3.bar(x_c - w/2, dur_conduct_a, w, label="E2a: Noun 'conduct' (Primary Stress on Syllable 1)", color="#2980b9", edgecolor="black", linewidth=0.8)
    rects2 = ax3.bar(x_c + w/2, dur_conduct_b, w, label="E2b: Verb 'conduct' (Primary Stress on Syllable 2)", color="#e67e22", edgecolor="black", linewidth=0.8)

    ax3.set_ylabel("Duration (Frames)", fontsize=11.5, fontweight="bold")
    ax3.set_title("Direct Syllable Contrast: Noun (AA1 = 10 frames) vs. Verb (AH1 = 10 frames)", fontsize=13, fontweight="bold", pad=6)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(target_labels, fontsize=10.5, fontweight="bold")
    ax3.tick_params(axis="both", which="major", labelsize=10.5)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontweight("semibold")
    ax3.legend(loc="upper right", fontsize=10.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.set_ylim(0, 14)

    for r in rects1:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2.0, h + 0.3, f"{int(h)}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1f4e78")
    for r in rects2:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2.0, h + 0.3, f"{int(h)}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#a04000")

    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.48, left=0.09, right=0.96)
    os.makedirs(os.path.join(plots_root_dir, "duration"), exist_ok=True)
    plt.savefig(os.path.join(plots_root_dir, "duration", "E2a_vs_E2b_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved refined large-font E2a_vs_E2b_comparison.png")

def get_speaker_embedding(args, preprocess_config, device):
    """
    Returns a (1, 256) float32 numpy array for continuous speaker conditioning.
    Supports reference wav, pre-extracted .npy embedding, or an anchor embedding from spk_emb/.
    """
    spk_emb_dir = os.path.join(preprocess_config["path"]["preprocessed_path"], "spk_emb")

    # 1. Direct reference audio via Resemblyzer
    if getattr(args, "ref_audio", None) and os.path.exists(args.ref_audio):
        print(f"Extracting continuous speaker embedding from reference audio: {args.ref_audio}")
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            encoder = VoiceEncoder(device="cuda" if torch.cuda.is_available() else "cpu")
            wav = preprocess_wav(args.ref_audio)
            emb = encoder.embed_utterance(wav).astype(np.float32)
            return emb.reshape(1, 256)
        except Exception as e:
            print(f"Warning: Failed to extract embedding from ref_audio ({e}); falling back to preprocessed bank.")

    # 2. Specific .npy file path
    if getattr(args, "spk_emb", None) and os.path.exists(args.spk_emb):
        print(f"Loading speaker embedding from file: {args.spk_emb}")
        emb = np.load(args.spk_emb).astype(np.float32)
        return emb.reshape(1, 256)

    # 3. Look in preprocessed spk_emb directory
    if os.path.exists(spk_emb_dir):
        files = sorted([f for f in os.listdir(spk_emb_dir) if f.endswith(".npy")])
        if files:
            selected_file = None
            if getattr(args, "spk_emb", None):
                matching = [f for f in files if args.spk_emb.lower() in f.lower()]
                if matching:
                    selected_file = matching[0]

            if selected_file is None:
                # Prefer Female_Hindi or any Female speaker with clean acoustics
                female_hindi = [f for f in files if "Female_Hindi" in f]
                selected_file = female_hindi[0] if female_hindi else files[0]

            print(f"Using acoustic anchor speaker embedding: {selected_file}")
            emb = np.load(os.path.join(spk_emb_dir, selected_file)).astype(np.float32)
            return emb.reshape(1, 256)

    print("Warning: No speaker embedding found; defaulting to zero vector.")
    return np.zeros((1, 256), dtype=np.float32)


def run_test_suite(model, configs, vocoder, spk_emb, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    results_root = train_config["path"]["result_path"]
    audio_out_dir = os.path.join(results_root, "test_generated_audios")
    plots_out_dir = os.path.join(results_root, "variance_adaptor_plots")
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(plots_out_dir, exist_ok=True)

    stats_path = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {"pitch": [-2.0, 2.0, 180.0, 40.0], "energy": [-2.0, 2.0, 30.0, 15.0]}

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    results_cache = {}

    for sample_id, sentence in ENGLISH_TEST_SENTENCES:
        print(f"\n--- Synthesizing Test Sample {sample_id} ---")
        seq, phone_labels = preprocess_english(
            sentence, preprocess_config, return_phone_list=True
        )
        ids = [sample_id]
        raw_texts = [sentence]
        speakers = spk_emb.astype(np.float32)
        texts = np.array([seq])
        text_lens = np.array([len(seq)])
        batch = to_device(
            (ids, raw_texts, speakers, texts, text_lens, max(text_lens)), device
        )

        with torch.no_grad():
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )

            src_len = output[8][0].item()
            mel_len = output[9][0].item()
            mel_pred = output[1][0, :mel_len].detach().transpose(0, 1).cpu().numpy()
            dur_pred = output[5][0, :src_len].detach().cpu().numpy()
            pitch_pred = output[2][0, :src_len].detach().cpu().numpy()
            energy_pred = output[3][0, :src_len].detach().cpu().numpy()

            mel_for_vocoder = output[1].transpose(1, 2)
            lengths = output[9] * hop_length
            wav_predictions = vocoder_infer(
                mel_for_vocoder,
                vocoder,
                model_config,
                preprocess_config,
                lengths=lengths,
            )

            wav_out = wav_predictions[0]
            # Post-processing: Normalize peak amplitude to -1.0 dBFS (0.90) and encode as standard 16-bit PCM
            max_val = np.max(np.abs(wav_out))
            if max_val > 0.01:
                wav_out = wav_out / max_val * 0.90
            wav_out_int16 = (wav_out * 32767.0).astype(np.int16)

            wav_path = os.path.join(audio_out_dir, f"{sample_id}.wav")
            wavfile.write(wav_path, sampling_rate, wav_out_int16)
            print(f"Saved audio: {wav_path}")

            save_variance_adaptor_plots(
                sample_id=sample_id,
                raw_text=sentence,
                phone_labels=phone_labels[:src_len],
                duration_pred=dur_pred,
                pitch_pred_norm=pitch_pred,
                energy_pred_norm=energy_pred,
                mel_pred=mel_pred,
                stats=stats,
                sampling_rate=sampling_rate,
                hop_length=hop_length,
                plots_root_dir=plots_out_dir,
            )
            print(f"Saved Variance Adaptor plots for {sample_id} to {plots_out_dir}")

            results_cache[sample_id] = {
                "phones": phone_labels[:src_len],
                "duration": dur_pred,
                "pitch": pitch_pred,
                "energy": energy_pred,
                "mel": mel_pred,
                "raw_text": sentence,
            }

    generate_multi_sentence_grid(
        results_cache,
        stats,
        os.path.join(plots_out_dir, "english_all_sentences_grid.png"),
        title_prefix="English",
    )
    plot_homograph_comparison(results_cache, stats, plots_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastSpeech 2 English Inference & Variance Adaptor Analysis"
    )
    parser.add_argument(
        "--restore_step",
        type=int,
        default=0,
        help="Checkpoint step to restore (0 = automatically load latest checkpoint in ./checkpoints)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test_suite", "batch", "single"],
        default="test_suite",
        help="Run all 5 assignment test sentences ('test_suite'), 'single', or 'batch'",
    )
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=5,
        help="Speaker ID (fallback if discrete lookup is used)",
    )
    parser.add_argument(
        "--spk_emb",
        type=str,
        default=None,
        help="Path to .npy speaker embedding (256,) or speaker substring in spk_emb directory",
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Path to reference .wav audio file to extract continuous 256-D d-vector on the fly",
    )
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("-p", "--preprocess_config", type=str, default=None)
    parser.add_argument("-m", "--model_config", type=str, default=None)
    parser.add_argument("-t", "--train_config", type=str, default=None)
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    args = parser.parse_args()

    configs = load_configs(
        args.config, args.preprocess_config, args.model_config, args.train_config
    )
    preprocess_config, model_config, train_config = configs

    # Ensure speakers.json and stats.json exist if running inference standalone
    os.makedirs(preprocess_config["path"]["preprocessed_path"], exist_ok=True)
    spk_json = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
    stats_json = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    if not os.path.exists(spk_json):
        with open(spk_json, "w", encoding="utf-8") as f:
            json.dump({"svarah_spk": 0}, f)
    if not os.path.exists(stats_json):
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(
                {"pitch": [-2.0, 2.0, 180.0, 40.0], "energy": [-2.0, 2.0, 30.0, 15.0]},
                f,
            )

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    control_values = (args.pitch_control, args.energy_control, args.duration_control)
    spk_emb = get_speaker_embedding(args, preprocess_config, device)

    if args.mode == "test_suite":
        run_test_suite(model, configs, vocoder, spk_emb, control_values)
    elif args.mode == "single":
        assert args.text is not None, "Please provide --text for single mode."
        seq, phone_labels = preprocess_english(
            args.text, preprocess_config, return_phone_list=True
        )
        ids = ["custom_utt"]
        raw_texts = [args.text]
        speakers = spk_emb.astype(np.float32)
        texts = np.array([seq])
        text_lens = np.array([len(seq)])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(
                    *(batch[2:]),
                    p_control=args.pitch_control,
                    e_control=args.energy_control,
                    d_control=args.duration_control,
                )
                synth_samples(
                    batch,
                    output,
                    vocoder,
                    model_config,
                    preprocess_config,
                    os.path.join(train_config["path"]["result_path"], "test_generated_audios"),
                )
    elif args.mode == "batch":
        assert args.source is not None, "Please provide --source for batch mode."
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                output = model(
                    *(batch[2:]),
                    p_control=args.pitch_control,
                    e_control=args.energy_control,
                    d_control=args.duration_control,
                )
                synth_samples(
                    batch,
                    output,
                    vocoder,
                    model_config,
                    preprocess_config,
                    os.path.join(train_config["path"]["result_path"], "test_generated_audios"),
                )
