import json
from pathlib import Path

import torch


def safe_mkdir(directory: Path) -> None:
    """Ask before using an existing directory."""
    if directory.exists():
        response = input(
            f"Directory '{str(directory)}' exists, continue? (y/n) "
        ).lower()
        if response not in ["yes", "y"]:
            exit()
    directory.mkdir(parents=True, exist_ok=True)


def device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def write_config(config: dict, directory: Path) -> None:
    """Write config text file to specified directory."""
    with open(directory / "config.json", "w") as f:
        json.dump({key: str(value) for key, value in config.items()}, f)
