import sys
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample, load_configs, plot_loss_curves
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare Telugu (IndicTTS) FastSpeech 2 training ...")

    preprocess_config, model_config, train_config = configs

    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    eff_group = group_size if batch_size * group_size < len(dataset) else 1
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(len(dataset), batch_size * eff_group)),
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    model, optimizer = get_model(args, configs, device, train=True)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    vocoder = None

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    results_dir = train_config["path"]["result_path"]
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    best_val_loss = float("inf")
    existing_val_log = os.path.join(val_log_path, "log.txt")
    if os.path.exists(existing_val_log):
        with open(existing_val_log, "r", encoding="utf-8") as vf:
            for vline in vf:
                if "Total Loss:" in vline:
                    try:
                        best_val_loss = min(
                            best_val_loss,
                            float(vline.split("Total Loss:")[1].split(",")[0].strip()),
                        )
                    except Exception:
                        pass

    outer_bar = tqdm(total=total_step, desc="Training", position=0, miniters=50, mininterval=5.0)
    outer_bar.n = args.restore_step
    outer_bar.update(0)

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1, disable=True)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output = model(*(batch[2:]))

                losses = Loss(batch, output)
                total_loss = losses[0]

                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    loss_vals = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                        *loss_vals
                    )

                    train_log_file = os.path.join(train_log_path, "log.txt")
                    with open(train_log_file, "a", encoding="utf-8") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=loss_vals)
                    if step % val_step == 0 or step == total_step:
                        plot_loss_curves(
                            train_log_file,
                            os.path.join(results_dir, "training_loss.png"),
                            title_prefix="Telugu (IndicTTS) Training",
                        )

                if vocoder is not None and step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0 or step == total_step:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    val_log_file = os.path.join(val_log_path, "log.txt")
                    with open(val_log_file, "a", encoding="utf-8") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    plot_loss_curves(
                        val_log_file,
                        os.path.join(results_dir, "validation_loss.png"),
                        title_prefix="Telugu (IndicTTS) Validation",
                    )

                    try:
                        val_total = float(message.split("Total Loss:")[1].split(",")[0].strip())
                        if val_total < best_val_loss:
                            best_val_loss = val_total
                            state_dict = (
                                model.module.state_dict()
                                if hasattr(model, "module")
                                else model.state_dict()
                            )
                            torch.save(
                                {
                                    "model": state_dict,
                                    "step": step,
                                },
                                os.path.join(
                                    train_config["path"]["ckpt_path"],
                                    "best_model.pth.tar",
                                ),
                            )
                    except Exception:
                        pass

                    model.train()

                if step % save_step == 0 or step == total_step:
                    ckpt_dir = train_config["path"]["ckpt_path"]
                    new_ckpt_path = os.path.join(ckpt_dir, "{}.pth.tar".format(step))
                    # Rotate older numeric step checkpoints first to preserve R:\ disk space
                    for fname in os.listdir(ckpt_dir):
                        if fname.endswith(".pth.tar") and fname not in (
                            "{}.pth.tar".format(step),
                            "best_model.pth.tar",
                        ):
                            stem = fname[: -len(".pth.tar")]
                            if stem.isdigit():
                                try:
                                    os.remove(os.path.join(ckpt_dir, fname))
                                except OSError:
                                    pass
                    state_dict = (
                        model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict()
                    )
                    torch.save(
                        {
                            "model": state_dict,
                            "step": step,
                        },
                        new_ckpt_path,
                    )

                if step >= total_step:
                    return
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to unified config.yaml (default: config.yaml)",
    )
    parser.add_argument("-p", "--preprocess_config", type=str, default=None)
    parser.add_argument("-m", "--model_config", type=str, default=None)
    parser.add_argument("-t", "--train_config", type=str, default=None)
    args = parser.parse_args()

    configs = load_configs(
        args.config, args.preprocess_config, args.model_config, args.train_config
    )
    main(args, configs)

