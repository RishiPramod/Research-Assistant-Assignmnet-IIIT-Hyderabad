import os
import json
import glob
import zipfile

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def find_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None, 0
    best_p = os.path.join(ckpt_dir, "best_model.pth.tar")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.pth.tar"))
    if not ckpts:
        return None, 0

    def _extract_step(p):
        base = os.path.basename(p).replace(".pth.tar", "")
        digits = "".join(c for c in base if c.isdigit())
        return int(digits) if digits else 0

    ckpts.sort(key=_extract_step)
    latest = ckpts[-1]
    if os.path.exists(best_p):
        return best_p, _extract_step(latest)
    return latest, _extract_step(latest)


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt_dir = train_config["path"]["ckpt_path"]
    restore_step = getattr(args, "restore_step", 0) or 0

    ckpt_path = None
    if restore_step > 0:
        candidate = os.path.join(ckpt_dir, "{}.pth.tar".format(restore_step))
        best_candidate = os.path.join(ckpt_dir, "best_model.pth.tar")
        if os.path.exists(candidate):
            ckpt_path = candidate
        elif os.path.exists(best_candidate):
            ckpt_path = best_candidate
    elif not train:
        ckpt_path, inferred_step = find_latest_checkpoint(ckpt_dir)
        if ckpt_path is not None:
            restore_step = inferred_step
            args.restore_step = inferred_step

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading model checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"]
        clean_state_dict = {
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        model.load_state_dict(clean_state_dict, strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, restore_step
        )
        if ckpt_path is not None and "optimizer" in ckpt:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def _ensure_hifigan_unzipped(hifigan_dir, ckpt_filename):
    target_path = os.path.join(hifigan_dir, ckpt_filename)
    zip_path = target_path + ".zip"
    if not os.path.exists(target_path) and os.path.exists(zip_path):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(hifigan_dir)
    return target_path


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        hifigan_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hifigan")
        config_path = os.path.join(hifigan_dir, "config.json")
        with open(config_path, "r") as f:
            hifi_config = json.load(f)
        hifi_config = hifigan.AttrDict(hifi_config)
        vocoder = hifigan.Generator(hifi_config)
        if speaker == "LJSpeech":
            ckpt_path = _ensure_hifigan_unzipped(hifigan_dir, "generator_LJSpeech.pth.tar")
        else:
            ckpt_path = _ensure_hifigan_unzipped(hifigan_dir, "generator_universal.pth.tar")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

