#!/usr/bin/env python3
"""Remove a training-only runtime import from the isolated LeRobot copy."""

from pathlib import Path
import shutil


path = Path(__file__).resolve().parents[2] / "vendor" / "lerobot_act_compat" / "lerobot" / "policies" / "pretrained.py"
original = Path(
    "/ssd/hhw/cache_backups/root/.cache/uv/git-v0/checkouts/"
    "3854d3ea6f0ea07f/1a4316c/src/lerobot/policies/pretrained.py"
)
shutil.copyfile(original, path)
source = path.read_text(encoding="utf-8")
source = source.replace("from typing import TypedDict, TypeVar", "from typing import TYPE_CHECKING, TypedDict, TypeVar")
source = source.replace(
    "from lerobot.configs.train import TrainPipelineConfig\n",
    "if TYPE_CHECKING:\n    from lerobot.configs.train import TrainPipelineConfig\n",
)
source = source.replace(
    "from lerobot.policies.utils import log_model_loading_keys\n",
    "def log_model_loading_keys(*_args, **_kwargs):\n    return None\n",
)
source = source.replace("cfg: TrainPipelineConfig,", 'cfg: "TrainPipelineConfig",')
path.write_text(source, encoding="utf-8")
