"""
Path generation functions for checkpoints and outputs
"""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict


def get_base_dir(output_dir: str, project_name: str, content_type: str) -> Path:
    """Get base directory for project."""
    return Path(output_dir) / project_name / content_type


def get_mode_dir(base_dir: Path, config: Dict) -> Path:
    """Get mode-specific directory (fast or normal)."""
    fast_mode = config.get("fast_mode", False)
    mode = "fast" if fast_mode else "normal"
    return base_dir / mode


def get_config_name(config: Dict) -> str:
    """Generate config directory name: {output}_{style}_{param}."""
    return _get_config_name(config, include_style_digest=True)


def _get_config_name(config: Dict, include_style_digest: bool) -> str:
    output_type = config.get("output_type", "slides")
    style = config.get("style", "academic")
    
    if output_type == "poster":
        param = config.get("poster_density", "medium")
    else:
        param = config.get("slides_length", "medium")
    
    # Retain a readable prefix, but identify the entire custom style.
    if style == "custom":
        custom = config.get("custom_style", "")
        suffix = custom[:16].replace(" ", "_").replace("/", "_") if custom else "custom"
        if custom and include_style_digest:
            suffix += "_" + hashlib.sha256(custom.encode("utf-8")).hexdigest()
        style = f"custom_{suffix}"
    
    return f"{output_type}_{style}_{param}"


def get_config_dir(base_dir: Path, config: Dict) -> Path:
    """Get config-specific directory for plan and output."""
    mode_dir = get_mode_dir(base_dir, config)
    config_dir = mode_dir / get_config_name(config)
    if config_dir.exists() or config.get("style") != "custom" or not config.get("custom_style"):
        return config_dir

    # Resume old custom-style paths only when their state proves the full
    # configuration matches. Never reuse another style's truncated prefix.
    legacy_dir = mode_dir / _get_config_name(config, include_style_digest=False)
    try:
        state = json.loads((legacy_dir / "state.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return config_dir
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        return config_dir

    saved_config = state["config"]
    if (
        saved_config.get("style") == "custom"
        and saved_config.get("custom_style") == config["custom_style"]
        and get_config_name(saved_config) == config_dir.name
        and get_mode_dir(base_dir, saved_config) == mode_dir
    ):
        return legacy_dir
    return config_dir


def get_rag_checkpoint(base_dir: Path, config: Dict) -> Path:
    """Get path to RAG checkpoint file."""
    mode_dir = get_mode_dir(base_dir, config)
    return mode_dir / "checkpoint_rag.json"


def get_summary_checkpoint(base_dir: Path, config: Dict) -> Path:
    """Get path to summary checkpoint file."""
    mode_dir = get_mode_dir(base_dir, config)
    return mode_dir / "checkpoint_summary.json"


def get_summary_md(base_dir: Path, config: Dict) -> Path:
    """Get path to summary markdown file."""
    mode_dir = get_mode_dir(base_dir, config)
    return mode_dir / "summary.md"


def get_plan_checkpoint(config_dir: Path) -> Path:
    """Get path to plan checkpoint file."""
    return config_dir / "checkpoint_plan.json"


def get_output_dir(config_dir: Path) -> Path:
    """Get output directory with timestamp to preserve history."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return config_dir / timestamp
