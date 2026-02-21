import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="d")
    default_config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    )
    parser.add_argument("--config", type=str, default=default_config_path, help="Path to YAML config")
    parser.add_argument(
        "--ds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override deep supervision on/off",
    )
    parser.add_argument("--att", type=str, default=None, help="Override att type")
    parser.add_argument(
        "--ema",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override EMA on/off",
    )
    return parser.parse_args()
