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


def plot_pattu_comparison(results_cache, stats, plots_root_dir):
    """
    Generates a comparative plot between T2a ('అతని పట్టు చాలా బలంగా' - Noun/Grip)
    and T2b ('పట్టు చీరను' - Silk Modifier), highlighting the geminate 'p a tt tt u'.
    """
    if "T2a" not in results_cache or "T2b" not in results_cache:
        return

    pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = stats["energy"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), dpi=150)
    fig.suptitle(
        "Lexical Context Analysis: T2a ('pattu' [p a tt tt u] = Grip/Hold) vs. T2b ('pattu chiiranu' = Silk Saree)",
        fontsize=13,
        fontweight="bold",
    )

    for col_idx, sample_id in enumerate(["T2a", "T2b"]):
        item = results_cache[sample_id]
        phones = item["phones"]
        dur = item["duration"]
        pitch_hz = item["pitch"] * pitch_std + pitch_mean
        energy_val = item["energy"] * energy_std + energy_mean
        x = np.arange(len(phones))

        # Highlight 'p a tt tt u'
        bar_colors = ["#2980b9"] * len(phones)
        for i in range(len(phones) - 4):
            if phones[i : i + 5] == ["p", "a", "tt", "tt", "u"]:
                for k in range(i, i + 5):
                    bar_colors[k] = "#e67e22"

        axes[0, col_idx].bar(x, dur, color=bar_colors, edgecolor="black", linewidth=0.4)
        axes[0, col_idx].set_xticks(x)
        axes[0, col_idx].set_xticklabels(phones, rotation=60, fontsize=7.5)
        axes[0, col_idx].set_title(
            f"{sample_id}: Predicted Phoneme Durations (Orange = 'pattu' [p a tt tt u])",
            fontsize=10,
            fontweight="bold",
        )
        axes[0, col_idx].grid(axis="y", linestyle="--", alpha=0.4)

        axes[1, col_idx].plot(x, pitch_hz[: len(x)], marker="o", markersize=3, color="#c0392b")
        axes[1, col_idx].set_xticks(x)
        axes[1, col_idx].set_xticklabels(phones, rotation=60, fontsize=7.5)
        axes[1, col_idx].set_title(f"{sample_id}: Predicted Pitch F0 (Hz)", fontsize=10, fontweight="bold")
        axes[1, col_idx].grid(True, linestyle="--", alpha=0.4)

        axes[2, col_idx].plot(x, energy_val[: len(x)], marker="s", markersize=3, color="#8e44ad")
        axes[2, col_idx].set_xticks(x)
        axes[2, col_idx].set_xticklabels(phones, rotation=60, fontsize=7.5)
        axes[2, col_idx].set_title(f"{sample_id}: Predicted Energy", fontsize=10, fontweight="bold")
        axes[2, col_idx].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(os.path.join(plots_root_dir, "duration", "T2a_vs_T2b_comparison.png"))
    plt.close(fig)


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
            }

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

