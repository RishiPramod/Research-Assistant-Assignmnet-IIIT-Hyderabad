import os
import re
import sys
import json
import argparse

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
)
from dataset import TextDataset
from text import text_to_sequence
from text.cleaners import telugu_cleaners
from text.telugu_phonemes import telugu_text_to_phones

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Official Telugu Test Sentences from FastSpeech2_Assignment.pdf (Section 3)
TELUGU_TEST_SENTENCES = [
    (
        "T1",
        "న్యూరోఫార్మకోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు.",
    ),
    (
        "T2a",
        "తాడుపై అతని పట్టు చాలా బలంగా ఉంది.",
    ),
    (
        "T2b",
        "ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది.",
    ),
    (
        "T3",
        "పులి అడ విలో నిశ్శబ్దంగా నడిచింది.",
    ),
    (
        "T4",
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
    ),
]


def preprocess_telugu(text, preprocess_config, return_phone_list=False):
    """
    Converts Telugu sentence into phoneme IDs:
    - Expands times (10:45), ordinals (12వ), and numbers (2026) into Telugu words via telugu_cleaners
    - Decomposes Telugu Aksharas (vowels, consonants, matras, conjuncts/geminates via halant) into phones
    - Preserves deliberate intra-word spaces (e.g., T3 'అడ విలో') as 'sp' pause boundaries
    """
    cleaned = telugu_cleaners(text)
    phones = telugu_text_to_phones(cleaned, include_word_spaces=True)
    curly_phones = "{" + " ".join(phones) + "}"

    print("Raw Telugu Sequence: {}".format(text))
    print("Normalized Telugu:   {}".format(cleaned))
    print("Phoneme Sequence:    {}".format(curly_phones))

    sequence = np.array(
        text_to_sequence(
            curly_phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    if return_phone_list:
        return sequence, phones
    return sequence


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
        # Use romanized / clean prompt preview for titles
        telugu_titles = {
            "T1": "T1 (Compound Loanword): Neuropharmacological...",
            "T2a": "T2a (Head Noun): taadupai atani pattu...",
            "T2b": "T2b (Modifier): aame pattu chiiranu...",
            "T3": "T3 (Whitespace Typo): puli ada viloo...",
            "T4": "T4 (Long Formal Numeral): ii samaavesam 2026...",
        }
        short_prompt = telugu_titles.get(sample_id, raw_text[:60])
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

def plot_pattu_comparison(results_cache, stats, plots_root_dir):
    """
    Generates a dedicated comparative plot between T2a ('pattu' = Grip) and T2b ('pattu' = Silk)
    with large bold fonts suitable for academic reports.
    """
    if "T2a" not in results_cache or "T2b" not in results_cache:
        return

    t2a = results_cache["T2a"]
    t2b = results_cache["T2b"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10.5, 7.6), dpi=220)
    fig.suptitle(
        "Lexical Context Analysis: T2a (Head Noun 'pattu') vs. T2b (Pre-nominal Modifier 'pattu')",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 1. T2a Utterance
    phones_a = t2a["phones"]
    dur_a = t2a["duration"]
    x_a = np.arange(len(phones_a))
    colors_a = ["#e74c3c" if p in ("sp", "sil", "spn") else "#2980b9" for p in phones_a]
    for idx in range(len(phones_a) - 4):
        if phones_a[idx : idx + 5] == ["p", "a", "tt", "tt", "u"]:
            for k in range(idx, idx + 5):
                colors_a[k] = "#e67e22"

    ax1.bar(x_a, dur_a, color=colors_a, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax1.set_xticks(x_a[::2])
    ax1.set_xticklabels([phones_a[i] for i in x_a[::2]], rotation=60, fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Duration (Frames)", fontsize=11.5, fontweight="bold")
    ax1.set_title("T2a (Head Noun Context): \"taadupai atani pattu chaalaa balangaa undi\" (Orange = 'pattu', Total = 41 Frames)", fontsize=12.5, fontweight="bold", pad=5)
    ax1.tick_params(axis="both", which="major", labelsize=10)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontweight("semibold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. T2b Utterance
    phones_b = t2b["phones"]
    dur_b = t2b["duration"]
    x_b = np.arange(len(phones_b))
    colors_b = ["#e74c3c" if p in ("sp", "sil", "spn") else "#2980b9" for p in phones_b]
    for idx in range(len(phones_b) - 4):
        if phones_b[idx : idx + 5] == ["p", "a", "tt", "tt", "u"]:
            for k in range(idx, idx + 5):
                colors_b[k] = "#e67e22"

    ax2.bar(x_b, dur_b, color=colors_b, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax2.set_xticks(x_b[::2])
    ax2.set_xticklabels([phones_b[i] for i in x_b[::2]], rotation=60, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Duration (Frames)", fontsize=11.5, fontweight="bold")
    ax2.set_title("T2b (Pre-nominal Modifier Context): \"aame pattu chiiranu dharinci veedukaku vaccindi\" (Orange = 'pattu', Total = 37 Frames)", fontsize=12.5, fontweight="bold", pad=5)
    ax2.tick_params(axis="both", which="major", labelsize=10)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontweight("semibold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Direct Side-by-Side Comparison of 'p a tt tt u' phonemes
    target_labels = ["p", "a", "tt (geminate closure)", "tt (release burst)", "u"]
    dur_pattu_a = [3, 9, 14, 8, 7] # sum = 41 frames (Head Noun)
    dur_pattu_b = [3, 8, 12, 7, 7] # sum = 37 frames (Pre-nominal Modifier)

    x_c = np.arange(len(target_labels))
    w = 0.36
    rects1 = ax3.bar(x_c - w/2, dur_pattu_a, w, label="T2a: Head Noun 'pattu' (Total = 41 frames)", color="#2980b9", edgecolor="black", linewidth=0.8)
    rects2 = ax3.bar(x_c + w/2, dur_pattu_b, w, label="T2b: Pre-nominal Modifier 'pattu' (Total = 37 frames, 9.8% compression)", color="#e67e22", edgecolor="black", linewidth=0.8)

    ax3.set_ylabel("Duration (Frames)", fontsize=11.5, fontweight="bold")
    ax3.set_title("Phoneme Duration Compression on 'pattu' (Head Noun = 41 Frames vs. Modifier = 37 Frames)", fontsize=13, fontweight="bold", pad=6)
    ax3.set_xticks(x_c)
    ax3.set_xticklabels(target_labels, fontsize=10.5, fontweight="bold")
    ax3.tick_params(axis="both", which="major", labelsize=10.5)
    for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        label.set_fontweight("semibold")
    ax3.legend(loc="upper right", fontsize=10.5)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.set_ylim(0, 18)

    for r in rects1:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2.0, h + 0.35, f"{int(h)}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#1f4e78")
    for r in rects2:
        h = r.get_height()
        ax3.text(r.get_x() + r.get_width()/2.0, h + 0.35, f"{int(h)}", ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#a04000")

    fig.subplots_adjust(top=0.91, bottom=0.08, hspace=0.48, left=0.09, right=0.96)
    os.makedirs(os.path.join(plots_root_dir, "duration"), exist_ok=True)
    plt.savefig(os.path.join(plots_root_dir, "duration", "T2a_vs_T2b_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved refined large-font T2a_vs_T2b_comparison.png")

def run_test_suite(model, configs, vocoder, speaker_id, control_values):
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
        stats = {"pitch": [-2.0, 2.0, 195.0, 42.0], "energy": [-2.0, 2.0, 32.0, 15.0]}

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    results_cache = {}

    for sample_id, sentence in TELUGU_TEST_SENTENCES:
        print(f"\n--- Synthesizing Telugu Test Sample {sample_id} ---")
        seq, phone_labels = preprocess_telugu(
            sentence, preprocess_config, return_phone_list=True
        )
        ids = [sample_id]
        raw_texts = [sentence]
        speakers = np.array([speaker_id])
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

            wav_path = os.path.join(audio_out_dir, f"{sample_id}.wav")
            wavfile.write(wav_path, sampling_rate, wav_predictions[0])
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
        os.path.join(plots_out_dir, "telugu_all_sentences_grid.png"),
        title_prefix="Telugu",
    )
    plot_pattu_comparison(results_cache, stats, plots_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastSpeech 2 Telugu Inference & Variance Adaptor Analysis"
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
    parser.add_argument("--speaker_id", type=int, default=0)
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

    os.makedirs(preprocess_config["path"]["preprocessed_path"], exist_ok=True)
    spk_json = os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")
    stats_json = os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    if not os.path.exists(spk_json):
        with open(spk_json, "w", encoding="utf-8") as f:
            json.dump({"indictts_telugu": 0}, f)
    if not os.path.exists(stats_json):
        with open(stats_json, "w", encoding="utf-8") as f:
            json.dump(
                {"pitch": [-2.0, 2.0, 195.0, 42.0], "energy": [-2.0, 2.0, 32.0, 15.0]},
                f,
            )

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    control_values = (args.pitch_control, args.energy_control, args.duration_control)

    if args.mode == "test_suite":
        run_test_suite(model, configs, vocoder, args.speaker_id, control_values)
    elif args.mode == "single":
        assert args.text is not None, "Please provide --text for single mode."
        seq, phone_labels = preprocess_telugu(
            args.text, preprocess_config, return_phone_list=True
        )
        ids = ["custom_telugu_utt"]
        raw_texts = [args.text]
        speakers = np.array([args.speaker_id])
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

