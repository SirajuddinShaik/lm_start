"""Scripts module for lm-start."""

__version__ = "0.1.0"

from lm_start.scripts.fetch_model_info import fetch_model_info_func as fetch_model_info
from lm_start.scripts.download_model import download_model_func as download_model
from lm_start.scripts.extract_vllm_config import (
    extract_vllm_config_func as extract_vllm_config,
)
from lm_start.scripts.extract_vllm_flags import (
    extract_vllm_flags_func as extract_vllm_flags,
)
from lm_start.scripts.generate_model_sh import (
    generate_model_sh_func as generate_model_sh,
)
from lm_start.scripts.generate_pm2_config import (
    generate_pm2_config_func as generate_pm2_config,
)
from lm_start.scripts.deploy import deploy_model as deploy
from lm_start.scripts.optimizer import optimize_vllm_config as optimize

__all__ = [
    "fetch_model_info",
    "download_model",
    "extract_vllm_config",
    "extract_vllm_flags",
    "generate_model_sh",
    "generate_pm2_config",
    "deploy",
    "optimize",
]
