import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audio as Audio


class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"].get("val_size", 512)
        self.train_hours = config["preprocessing"].get("train_hours", 5.0)
        self.val_hours = config["preprocessing"].get("val_hours", 1.0)
        self.seed = config["preprocessing"].get("seed", 42)
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]

        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )

        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)

        print("Processing Telugu Data (extracting Duration, Pitch F0, Energy, and Mel-Spectrograms) ...")
        records = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        split_manifest_path = os.path.join(self.in_dir, "split_manifest.json")
        preassigned_splits = {}
        if os.path.exists(split_manifest_path):
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
                for u in sm.get("train", []):
                    preassigned_splits[u] = "train"
                for u in sm.get("val", []):
                    preassigned_splits[u] = "val"

        speakers = {}
        speaker_dirs = sorted(
            [
                d
                for d in os.listdir(self.in_dir)
                if os.path.isdir(os.path.join(self.in_dir, d))
            ]
        )
        for i, speaker in enumerate(tqdm(speaker_dirs, desc="Speakers")):
            speakers[speaker] = i
            spk_dir = os.path.join(self.in_dir, speaker)
            for wav_name in sorted(os.listdir(spk_dir)):
                if not wav_name.endswith(".wav"):
                    continue

                basename = wav_name[:-4]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n, raw_text = ret
                    dur_sec = n * self.hop_length / self.sampling_rate
                    records.append(
                        {
                            "basename": basename,
                            "speaker": speaker,
                            "info": info,
                            "n_frames": n,
                            "duration_sec": dur_sec,
                            "raw_text": raw_text.strip(),
                        }
                    )

                    if len(pitch) > 0:
                        pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                    if len(energy) > 0:
                        energy_scaler.partial_fit(energy.reshape((-1, 1)))

                    n_frames += n

        print("Computing statistic quantities ...")
        if self.pitch_normalization and hasattr(pitch_scaler, "mean_"):
            pitch_mean = pitch_scaler.mean_[0]
            pitch_std = pitch_scaler.scale_[0]
        else:
            pitch_mean = 0.0
            pitch_std = 1.0
        if self.energy_normalization and hasattr(energy_scaler, "mean_"):
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0.0
            energy_std = 1.0

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        with open(os.path.join(self.out_dir, "speakers.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(speakers, indent=2))

        with open(os.path.join(self.out_dir, "stats.json"), "w", encoding="utf-8") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                ],
            }
            f.write(json.dumps(stats, indent=2))

        total_hours = n_frames * self.hop_length / self.sampling_rate / 3600.0
        print("Total aligned audio time: {:.2f} hours".format(total_hours))

        train_entries, val_entries, split_summary = self._split_train_val(
            records, preassigned_splits
        )

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for r in train_entries:
                f.write(r["info"] + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for r in val_entries:
                f.write(r["info"] + "\n")

        with open(
            os.path.join(self.out_dir, "dataset_split_summary.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(split_summary, f, indent=2, ensure_ascii=False)

        local_mirror = self.config["path"].get("local_meta_mirror")
        if local_mirror:
            os.makedirs(local_mirror, exist_ok=True)
            import shutil

            for meta_file in [
                "speakers.json",
                "stats.json",
                "train.txt",
                "val.txt",
                "dataset_split_summary.json",
            ]:
                src = os.path.join(self.out_dir, meta_file)
                dst = os.path.join(local_mirror, meta_file)
                if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)

        if self.config["preprocessing"].get("cleanup_raw_wavs_after_preprocess", False):
            removed_wavs = 0
            for speaker in speaker_dirs:
                spk_dir = os.path.join(self.in_dir, speaker)
                for fname in os.listdir(spk_dir):
                    if fname.endswith(".wav"):
                        try:
                            os.remove(os.path.join(spk_dir, fname))
                            removed_wavs += 1
                        except OSError:
                            pass
            print(f"Cleaned up {removed_wavs} raw .wav files from {self.in_dir} to save disk space.")

        print(
            f"Split complete -> Train: {len(train_entries)} utts ({split_summary['train_hours']:.2f}h) | "
            f"Val: {len(val_entries)} utts ({split_summary['val_hours']:.2f}h) | Overlap: 0"
        )
        return [r["info"] for r in train_entries + val_entries]

    def _split_train_val(self, records, preassigned_splits):
        rng = random.Random(self.seed)
        if preassigned_splits:
            train_entries = [
                r for r in records if preassigned_splits.get(r["basename"]) == "train"
            ]
            val_entries = [
                r for r in records if preassigned_splits.get(r["basename"]) == "val"
            ]
        else:
            shuffled = list(records)
            rng.shuffle(shuffled)
            target_val_sec = self.val_hours * 3600.0
            target_train_sec = self.train_hours * 3600.0

            val_entries = []
            train_entries = []
            val_texts = set()
            train_texts = set()
            val_sec = 0.0
            train_sec = 0.0

            for r in shuffled:
                txt = r["raw_text"]
                if val_sec < target_val_sec and txt not in train_texts:
                    val_entries.append(r)
                    val_texts.add(txt)
                    val_sec += r["duration_sec"]
                elif train_sec < target_train_sec and txt not in val_texts:
                    train_entries.append(r)
                    train_texts.add(txt)
                    train_sec += r["duration_sec"]

            if not train_entries and val_entries:
                split_idx = max(1, int(len(val_entries) * 0.16))
                train_entries = val_entries[split_idx:]
                val_entries = val_entries[:split_idx]

        train_ids = {r["basename"] for r in train_entries}
        val_ids = {r["basename"] for r in val_entries}
        assert len(train_ids & val_ids) == 0, "Overlap detected between train and val IDs!"

        train_sec = sum(r["duration_sec"] for r in train_entries)
        val_sec = sum(r["duration_sec"] for r in val_entries)
        summary = {
            "train_utterances": len(train_entries),
            "val_utterances": len(val_entries),
            "train_seconds": round(train_sec, 2),
            "val_seconds": round(val_sec, 2),
            "train_hours": round(train_sec / 3600.0, 4),
            "val_hours": round(val_sec / 3600.0, 4),
            "id_overlap_count": len(train_ids & val_ids),
            "seed": self.seed,
        }
        return train_entries, val_entries, summary

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        if not os.path.exists(wav_path) or not os.path.exists(text_path):
            return None

        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        wav, _ = librosa.load(wav_path, sr=self.sampling_rate)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            raw_text,
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        files = os.listdir(in_dir)
        if not files:
            return -1.0, 1.0
        for filename in files:
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

