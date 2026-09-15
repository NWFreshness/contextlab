"""Slice 8 — sandbox limits and config loader.

The limits pydantic model is the single source of truth for sandbox
behaviour. `load_sandbox_config` resolves `config/sandbox.yaml` and applies
environment overrides for CI / tests, the same pattern Slice 4's tracer
and Slice 7's orchestrator use.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sandbox.yaml"


Backend = Literal["subprocess", "inprocess"]


class SandboxLimits(BaseModel):
    """Resolved sandbox settings. Every field is required — fail-closed."""

    backend: Backend = "subprocess"
    workdir_parent: Path = PROJECT_ROOT / "data" / "sandbox_work"
    timeout_s: float = Field(default=2.0, gt=0)
    max_output_bytes: int = Field(default=8000, gt=0)
    cpu_s: Optional[float] = Field(default=2.0, gt=0)
    memory_mb: Optional[int] = Field(default=256, gt=0)
    env_allowlist: list[str] = Field(
        default_factory=lambda: ["PATH", "LANG", "LC_ALL"]
    )
    path_allowlist: list[str] = Field(default_factory=list)
    tools: dict[str, "ToolSandboxPolicy"] = Field(default_factory=dict)


class ToolSandboxPolicy(BaseModel):
    """Per-tool sandbox routing.

    `sandbox: false` keeps the tool in-process (fast, used for Slice 1
    helpers). `sandbox: true` routes through the subprocess backend with
    the policy / containment layers. Unknown tools default to `sandbox:
    true` — fail-closed.
    """

    sandbox: bool = True
    # The set of argument names the tool accepts. Anything outside this
    # set is rejected by `SandboxPolicy.check_args`. Empty list means the
    # tool takes no args.
    args_schema: list[str] = Field(default_factory=list)


def load_sandbox_config(path: Optional[Path] = None) -> SandboxLimits:
    """Load config/sandbox.yaml. Env overrides for tests.

    Env vars (one knob each):
        CONTEXTLAB_SANDBOX_BACKEND=subprocess|inprocess
        CONTEXTLAB_SANDBOX_TIMEOUT_S=float
        CONTEXTLAB_SANDBOX_WORKDIR=str
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Sandbox config not found: {config_path}")
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}

    # Env overrides — useful for the inprocess backend in unit tests.
    env_backend = os.environ.get("CONTEXTLAB_SANDBOX_BACKEND")
    if env_backend in ("subprocess", "inprocess"):
        raw["backend"] = env_backend
    env_timeout = os.environ.get("CONTEXTLAB_SANDBOX_TIMEOUT_S")
    if env_timeout:
        try:
            raw["timeout_s"] = float(env_timeout)
        except ValueError:
            pass
    env_workdir = os.environ.get("CONTEXTLAB_SANDBOX_WORKDIR")
    if env_workdir:
        raw["workdir_parent"] = env_workdir

    # Resolve to absolute paths so the executor can compare against
    # `Path.resolve()` regardless of cwd.
    if "workdir_parent" in raw and not Path(raw["workdir_parent"]).is_absolute():
        raw["workdir_parent"] = PROJECT_ROOT / raw["workdir_parent"]

    return SandboxLimits.model_validate(raw)


def has_prlimit() -> bool:
    """`prlimit` is on PATH on most Linux systems. Optional — used for
    best-effort CPU / memory caps via `preexec_fn`."""
    return shutil.which("prlimit") is not None