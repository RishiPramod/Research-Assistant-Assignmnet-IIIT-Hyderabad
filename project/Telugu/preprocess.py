import sys
import argparse
from utils.tools import load_configs
from preprocessor.preprocessor import Preprocessor
from preprocessor import indictts

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastSpeech 2 Telugu (IndicTTS) Preprocessing & Alignment Preparation"
    )
    parser.add_argument(
        "config_pos",
        nargs="?",
        default=None,
        help="Optional positional path to config.yaml or preprocess.yaml",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.yaml",
        help="Path to unified config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--prepare_align",
        action="store_true",
        help="Step 1: Subsample IndicTTS Telugu (5h train + 1h val), resample to 22050Hz, and create Telugu .lab + lexicon for MFA",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download IndicTTS Telugu from HuggingFace if local corpus_path is empty",
    )
    parser.add_argument(
        "--fallback_align",
        action="store_true",
        help="Generate acoustic-energy-guided phoneme TextGrids if external MFA binary is not used",
    )
    args = parser.parse_args()

    cfg_path = args.config_pos if args.config_pos else args.config
    preprocess_config, _, _ = load_configs(cfg_path)

    if args.prepare_align:
        indictts.prepare_align(
            preprocess_config,
            download_hf=args.download,
            fallback_align=args.fallback_align,
        )
    else:
        preprocessor = Preprocessor(preprocess_config)
        preprocessor.build_from_path()

