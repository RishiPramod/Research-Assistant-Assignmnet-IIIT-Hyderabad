import os
import re
import json
import yaml

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt

matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_configs(config_path="config.yaml", preprocess_path=None, model_path=None, train_path=None):
    """
    Supports loading either a single unified `config.yaml` (required by the assignment layout)
    or three separate YAML files (`preprocess.yaml`, `model.yaml`, `train.yaml`).
    """
    if preprocess_path and model_path and train_path:
        preprocess_config = yaml.load(open(preprocess_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
        model_config = yaml.load(open(model_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
        train_config = yaml.load(open(train_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
        return preprocess_config, model_config, train_config

    full_cfg = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.FullLoader)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    if "preprocess" in full_cfg and "model" in full_cfg and "train" in full_cfg:
        for sec in ("preprocess", "train"):
            if "path" in full_cfg[sec]:
                for k, v in list(full_cfg[sec]["path"].items()):
                    if isinstance(v, str) and not os.path.isabs(v):
                        full_cfg[sec]["path"][k] = os.path.normpath(os.path.join(base_dir, v))
        return full_cfg["preprocess"], full_cfg["model"], full_cfg["train"]
    return full_cfg, full_cfg, full_cfg


def to_device(data, device):
    if len(data) == 12:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    if len(data) == 6:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len)


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/pitch_loss", losses[3], step)
        logger.add_scalar("Loss/energy_loss", losses[4], step)
        logger.add_scalar("Loss/duration_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        max_val = max(abs(audio))
        if max_val > 0:
            audio = audio / max_val
        logger.add_audio(
            tag,
            audio,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, vocoder, model_config, preprocess_config):
    basename = targets[0][0]
    src_len = predictions[8][0].item()
    mel_len = predictions[9][0].item()
    mel_target = targets[6][0, :mel_len].detach().transpose(0, 1)
    mel_prediction = predictions[1][0, :mel_len].detach().transpose(0, 1)
    duration = targets[11][0, :src_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
        pitch = targets[9][0, :src_len].detach().cpu().numpy()
        pitch = expand(pitch, duration)
    else:
        pitch = targets[9][0, :mel_len].detach().cpu().numpy()
    if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
        energy = targets[10][0, :src_len].detach().cpu().numpy()
        energy = expand(energy, duration)
    else:
        energy = targets[10][0, :mel_len].detach().cpu().numpy()

    with open(
        os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        stats = stats["pitch"] + stats["energy"][:2]

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch, energy),
            (mel_target.cpu().numpy(), pitch, energy),
        ],
        stats,
        ["Synthetized Spectrogram", "Ground-Truth Spectrogram"],
    )

    if vocoder is not None:
        from .model import vocoder_infer

        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
        wav_prediction = vocoder_infer(
            mel_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    else:
        wav_reconstruction = wav_prediction = None

    return fig, wav_reconstruction, wav_prediction, basename


def synth_samples(targets, predictions, vocoder, model_config, preprocess_config, path):
    os.makedirs(path, exist_ok=True)
    basenames = targets[0]
    for i in range(len(predictions[0])):
        basename = basenames[i]
        src_len = predictions[8][i].item()
        mel_len = predictions[9][i].item()
        mel_prediction = predictions[1][i, :mel_len].detach().transpose(0, 1)
        duration = predictions[5][i, :src_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["pitch"]["feature"] == "phoneme_level":
            pitch = predictions[2][i, :src_len].detach().cpu().numpy()
            pitch = expand(pitch, duration)
        else:
            pitch = predictions[2][i, :mel_len].detach().cpu().numpy()
        if preprocess_config["preprocessing"]["energy"]["feature"] == "phoneme_level":
            energy = predictions[3][i, :src_len].detach().cpu().numpy()
            energy = expand(energy, duration)
        else:
            energy = predictions[3][i, :mel_len].detach().cpu().numpy()

        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            stats = stats["pitch"] + stats["energy"][:2]

        fig = plot_mel(
            [
                (mel_prediction.cpu().numpy(), pitch, energy),
            ],
            stats,
            ["Synthetized Spectrogram"],
        )
        plt.savefig(os.path.join(path, "{}.png".format(basename)))
        plt.close()

    from .model import vocoder_infer

    mel_predictions = predictions[1].transpose(1, 2)
    lengths = predictions[9] * preprocess_config["preprocessing"]["stft"]["hop_length"]
    wav_predictions = vocoder_infer(
        mel_predictions, vocoder, model_config, preprocess_config, lengths=lengths
    )

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(path, "{}.wav".format(basename)), sampling_rate, wav)


def save_variance_adaptor_plots(
    sample_id,
    raw_text,
    phone_labels,
    duration_pred,
    pitch_pred_norm,
    energy_pred_norm,
    mel_pred,
    stats,
    sampling_rate,
    hop_length,
    plots_root_dir,
):
    """
    Generates dedicated variance adaptor plots for duration, pitch (F0), and energy
    with large, highly legible fonts suitable for academic reports.
    """
    dur_dir = os.path.join(plots_root_dir, "duration")
    pitch_dir = os.path.join(plots_root_dir, "pitch")
    energy_dir = os.path.join(plots_root_dir, "energy")
    os.makedirs(dur_dir, exist_ok=True)
    os.makedirs(pitch_dir, exist_ok=True)
    os.makedirs(energy_dir, exist_ok=True)

    pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = stats["energy"]

    pitch_hz = pitch_pred_norm * pitch_std + pitch_mean
    energy_val = energy_pred_norm * energy_std + energy_mean

    total_frames = int(np.sum(duration_pred))
    x_idx = np.arange(len(phone_labels))

    # 1. Duration Plot
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=220)
    colors = ["#e74c3c" if p in ("sp", "sil", "spn") else "#2980b9" for p in phone_labels]
    bars = ax.bar(x_idx, duration_pred, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    step_label = max(1, len(phone_labels) // 40)
    ax.set_xticks(x_idx[::step_label])
    ax.set_xticklabels([phone_labels[i] for i in x_idx[::step_label]], rotation=60, fontsize=11, fontweight="bold")
    ax.set_ylabel("Duration (Frames)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Phoneme Sequence (Red = Pause 'sp')", fontsize=12.5, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=11)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight("semibold")
    ax.set_title(
        f"[{sample_id}] Variance Adaptor — Predicted Phoneme Durations",
        fontsize=15,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if len(phone_labels) <= 45:
        for bar, d in zip(bars, duration_pred):
            if d > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.2,
                    f"{int(d)}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="bold",
                )
    plt.tight_layout()
    plt.savefig(os.path.join(dur_dir, f"{sample_id}_duration.png"))
    plt.close(fig)

    # Expand pitch and energy to frame level if phoneme_level
    if len(pitch_hz) == len(duration_pred):
        pitch_frame = expand(pitch_hz, duration_pred)
        pitch_phoneme = pitch_hz
    else:
        pitch_frame = pitch_hz
        pitch_phoneme = []
        pos = 0
        for d in duration_pred:
            d_int = int(d)
            pitch_phoneme.append(np.mean(pitch_frame[pos : pos + d_int]) if d_int > 0 else 0.0)
            pos += d_int
        pitch_phoneme = np.array(pitch_phoneme)

    if len(energy_val) == len(duration_pred):
        energy_frame = expand(energy_val, duration_pred)
        energy_phoneme = energy_val
    else:
        energy_frame = energy_val
        energy_phoneme = []
        pos = 0
        for d in duration_pred:
            d_int = int(d)
            energy_phoneme.append(np.mean(energy_frame[pos : pos + d_int]) if d_int > 0 else 0.0)
            pos += d_int
        energy_phoneme = np.array(energy_phoneme)

    # 2. Pitch (F0) Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 5.8), dpi=220)
    ax1.plot(x_idx, pitch_phoneme, marker="o", markersize=4.5, color="#d35400", linewidth=2.6, label="Predicted F0 (Hz)")
    ax1.fill_between(x_idx, pitch_phoneme, color="#e67e22", alpha=0.18)
    ax1.set_xticks(x_idx[::step_label])
    ax1.set_xticklabels([phone_labels[i] for i in x_idx[::step_label]], rotation=60, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("F0 (Hz)", fontsize=12.5, fontweight="bold")
    ax1.tick_params(axis="both", which="major", labelsize=11)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontweight("semibold")
    ax1.set_title(
        f"[{sample_id}] Variance Adaptor — Predicted Pitch (F0) Contour",
        fontsize=14.5,
        fontweight="bold",
        pad=7,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=11)

    ax2.imshow(mel_pred, origin="lower", aspect="auto", cmap="viridis")
    ax2.set_ylabel("Mel Bin (0-80)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Time (Mel Frames)", fontsize=12.5, fontweight="bold")
    ax2.tick_params(axis="both", which="major", labelsize=11)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontweight("semibold")
    ax2_f0 = ax2.twinx()
    ax2_f0.plot(np.arange(len(pitch_frame)), pitch_frame, color="#ff3333", linewidth=2.4, label="Expanded Frame F0 (Hz)")
    ax2_f0.set_ylabel("Frame F0 (Hz)", color="#cc0000", fontsize=12.5, fontweight="bold")
    ax2_f0.tick_params(axis="y", labelcolor="#cc0000", labelsize=11)
    for label in ax2_f0.get_yticklabels():
        label.set_fontweight("semibold")
    ax2_f0.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(pitch_dir, f"{sample_id}_pitch.png"))
    plt.close(fig)

    # 3. Energy Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 5.8), dpi=220)
    ax1.plot(x_idx, energy_phoneme, marker="s", markersize=4.5, color="#8e44ad", linewidth=2.6, label="Predicted Energy")
    ax1.fill_between(x_idx, energy_phoneme, color="#9b59b6", alpha=0.18)
    ax1.set_xticks(x_idx[::step_label])
    ax1.set_xticklabels([phone_labels[i] for i in x_idx[::step_label]], rotation=60, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("Energy (L2)", fontsize=12.5, fontweight="bold")
    ax1.tick_params(axis="both", which="major", labelsize=11)
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontweight("semibold")
    ax1.set_title(
        f"[{sample_id}] Variance Adaptor — Predicted Energy Contour",
        fontsize=14.5,
        fontweight="bold",
        pad=7,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=11)

    ax2.imshow(mel_pred, origin="lower", aspect="auto", cmap="magma")
    ax2.set_ylabel("Mel Bin (0-80)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("Time (Mel Frames)", fontsize=12.5, fontweight="bold")
    ax2.tick_params(axis="both", which="major", labelsize=11)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontweight("semibold")
    ax2_en = ax2.twinx()
    ax2_en.plot(np.arange(len(energy_frame)), energy_frame, color="#00e5ff", linewidth=2.4, label="Expanded Frame Energy")
    ax2_en.set_ylabel("Frame Energy", color="#00838f", fontsize=12.5, fontweight="bold")
    ax2_en.tick_params(axis="y", labelcolor="#00838f", labelsize=11)
    for label in ax2_en.get_yticklabels():
        label.set_fontweight("semibold")
    ax2_en.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(energy_dir, f"{sample_id}_energy.png"))
    plt.close(fig)


def plot_loss_curves(log_file_path, output_png_path, title_prefix="Training"):
    """
    Parses train/val loss logs and saves a clean, 6-subplot figure with large bold fonts.
    """
    if not os.path.exists(log_file_path):
        return False

    pattern = re.compile(
        r"(?:Step|Validation Step)\s+(\d+)(?:/\d+)?,\s*Total Loss:\s*([\d\.]+),\s*Mel Loss:\s*([\d\.]+),\s*"
        r"Mel PostNet Loss:\s*([\d\.]+),\s*Pitch Loss:\s*([\d\.]+),\s*Energy Loss:\s*([\d\.]+),\s*Duration Loss:\s*([\d\.]+)"
    )

    runs = []
    current_run = {}
    last_step = -1

    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                s = int(m.group(1))
                if last_step > 5000 and s <= 500 and current_run:
                    runs.append(current_run)
                    current_run = {}
                current_run[s] = (
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                    float(m.group(6)),
                    float(m.group(7)),
                )
                last_step = s

    if current_run:
        runs.append(current_run)

    if not runs:
        return False

    active_dict = runs[-1]
    sorted_steps = sorted(active_dict.keys())
    steps = sorted_steps
    totals = [active_dict[k][0] for k in sorted_steps]
    mels = [active_dict[k][1] for k in sorted_steps]
    postnets = [active_dict[k][2] for k in sorted_steps]
    pitches = [active_dict[k][3] for k in sorted_steps]
    energies = [active_dict[k][4] for k in sorted_steps]
    durations = [active_dict[k][5] for k in sorted_steps]

    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), dpi=220)
    fig.suptitle(f"FastSpeech 2 — {title_prefix} Loss Curves", fontsize=18, fontweight="bold", y=0.98)

    panels = [
        (axes[0, 0], totals, "Total Loss", "#c0392b"),
        (axes[0, 1], mels, "Mel MAE Loss", "#2980b9"),
        (axes[0, 2], postnets, "Mel PostNet MAE Loss", "#16a085"),
        (axes[1, 0], durations, "Duration Predictor MSE Loss", "#d35400"),
        (axes[1, 1], pitches, "Pitch (F0) Predictor MSE Loss", "#8e44ad"),
        (axes[1, 2], energies, "Energy Predictor MSE Loss", "#27ae60"),
    ]

    for ax, vals, name, col in panels:
        ax.plot(steps, vals, color=col, linewidth=2.8)
        ax.set_title(name, fontsize=13.5, fontweight="bold", pad=7)
        ax.set_xlabel("Training Step" if "Training" in title_prefix else "Validation Step", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=11)
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontweight("semibold")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    plt.savefig(output_png_path)
    plt.close(fig)
    return True



def plot_mel(data, stats, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    pitch_min, pitch_max, pitch_mean, pitch_std, energy_min, energy_max = stats
    pitch_min = pitch_min * pitch_std + pitch_mean
    pitch_max = pitch_max * pitch_std + pitch_mean

    def add_axis(fig, old_ax):
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch, energy = data[i]
        pitch = pitch * pitch_std + pitch_mean
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylim(0, pitch_max)
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

        ax2 = add_axis(fig, axes[i][0])
        ax2.plot(energy, color="darkviolet")
        ax2.set_xlim(0, mel.shape[1])
        ax2.set_ylim(energy_min, energy_max)
        ax2.set_ylabel("Energy", color="darkviolet")
        ax2.yaxis.set_label_position("right")
        ax2.tick_params(
            labelsize="x-small",
            colors="darkviolet",
            bottom=False,
            labelbottom=False,
            left=False,
            labelleft=False,
            right=True,
            labelright=True,
        )

    return fig


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded

