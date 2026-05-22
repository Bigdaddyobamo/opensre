"""
Claude Code CLI adapter (non-interactive `claude -p`).

Key improvements:
- strict auth precedence (no silent overrides)
- safe home resolution
- version enforcement (min_version supported)
- no ambiguous auth states
- clearer CLI failure classification
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

from app.integrations.llm_cli.base import CLIInvocation, CLIProbe
from app.integrations.llm_cli.binary_resolver import (
    candidate_binary_names,
    default_cli_fallback_paths,
    resolve_cli_binary,
)
from app.integrations.llm_cli.env_overrides import (
    ANTHROPIC_CLI_ENV_KEYS,
    nonempty_env_values,
)
from app.integrations.llm_cli.errors import CLIInvalidModelError
from app.integrations.llm_cli.subprocess_env import build_cli_subprocess_env

_CLAUDE_VERSION_RE = re.compile(r"(\d+\.\d+\.\d+)")

_PROBE_TIMEOUT_SEC = 8.0
_DEFAULT_EXEC_TIMEOUT_SEC = 120.0
_MIN_EXEC_TIMEOUT_SEC = 30.0
_MAX_EXEC_TIMEOUT_SEC = 600.0

_AUTH_HINT = "Run `claude auth login` or set ANTHROPIC_API_KEY."


# ----------------------------
# Timeout
# ----------------------------
def _resolve_exec_timeout_seconds() -> float:
    raw = os.environ.get("CLAUDE_CODE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_EXEC_TIMEOUT_SEC

    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_EXEC_TIMEOUT_SEC

    if val <= 0:
        return _DEFAULT_EXEC_TIMEOUT_SEC

    return max(_MIN_EXEC_TIMEOUT_SEC, min(val, _MAX_EXEC_TIMEOUT_SEC))


# ----------------------------
# Version parsing
# ----------------------------
def _parse_semver(text: str) -> Optional[str]:
    match = _CLAUDE_VERSION_RE.search(text or "")
    return match.group(1) if match else None


# ----------------------------
# Auth helpers
# ----------------------------
def _auth_env() -> dict[str, str]:
    env = {"NO_COLOR": "1"}
    env.update(nonempty_env_values(ANTHROPIC_CLI_ENV_KEYS))
    return env


def _auth_env_source() -> Optional[str]:
    env = _auth_env()
    return (
        "ANTHROPIC_API_KEY"
        if env.get("ANTHROPIC_API_KEY")
        else "ANTHROPIC_AUTH_TOKEN"
        if env.get("ANTHROPIC_AUTH_TOKEN")
        else None
    )


def _read_credentials_file() -> bool:
    home = Path(os.path.expanduser("~"))
    creds = home / ".claude" / ".credentials.json"
    try:
        return creds.exists() and creds.stat().st_size > 2
    except OSError:
        return False


# ----------------------------
# Auth status mapping
# ----------------------------
def _auth_status_from_json(data: dict) -> Tuple[bool, str]:
    if not data.get("loggedIn"):
        return False, f"Not authenticated. {_AUTH_HINT}"

    method = str(data.get("authMethod") or "").lower()
    email = str(data.get("email") or "")
    api_source = str(data.get("apiKeySource") or "")

    if method == "api_key":
        return True, f"Authenticated via {api_source or 'ANTHROPIC_API_KEY'}."

    if method == "claude.ai":
        return True, f"Authenticated via Claude subscription{f' ({email})' if email else ''}."

    if method:
        return True, f"Authenticated via {method}{f' ({email})' if email else ''}."

    if api_source:
        return True, f"Authenticated via {api_source}."

    if email:
        return True, f"Authenticated via Claude subscription ({email})."

    return True, "Authenticated via Claude CLI."


def _parse_auth_json(stdout: str) -> Optional[Tuple[bool, str]]:
    try:
        data = json.loads(stdout.strip())
    except Exception:
        return None

    if isinstance(data, dict):
        return _auth_status_from_json(data)

    return None


# ----------------------------
# CLI probe
# ----------------------------
def _probe_cli_auth(binary: str) -> Tuple[Optional[bool], str]:
    try:
        proc = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_SEC,
            env=build_cli_subprocess_env(_auth_env()),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"Auth check timed out after {_PROBE_TIMEOUT_SEC:.0f}s."
    except OSError as exc:
        return None, f"Could not execute Claude CLI: {exc}"

    parsed = _parse_auth_json(proc.stdout)
    if parsed:
        return parsed

    output = (proc.stdout + proc.stderr).strip().lower()

    if proc.returncode != 0:
        return None, f"Auth probe failed: {output[:500] or 'unknown error'}"

    negative = ("not logged in", "unauthenticated", "login required")
    if any(n in output for n in negative):
        return False, f"Not authenticated. {_AUTH_HINT}"

    return True, "Authenticated via Claude CLI."


# ----------------------------
# High-level auth classifier
# ----------------------------
def _classify_auth(binary: Optional[str]) -> Tuple[Optional[bool], str]:
    # 1. CLI (source of truth)
    if binary:
        return _probe_cli_auth(binary)

    # 2. Env API key
    env_source = _auth_env_source()
    if env_source:
        return True, f"Authenticated via {env_source}."

    # 3. Credentials file
    if _read_credentials_file():
        return True, "Authenticated via ~/.claude/.credentials.json."

    # 4. macOS unknown state
    if sys.platform == "darwin":
        return None, f"Auth state unclear (macOS Keychain may contain credentials). {_AUTH_HINT}"

    # 5. Hard negative
    return False, f"Not authenticated. {_AUTH_HINT}"


# ----------------------------
# CLI adapter
# ----------------------------
class ClaudeCodeAdapter:
    name = "claude-code"
    install_hint = "npm i -g @anthropic-ai/claude-code"
    auth_hint = _AUTH_HINT
    min_version: Optional[str] = None

    def _resolve_binary(self) -> Optional[str]:
        return resolve_cli_binary(
            explicit_env_key="CLAUDE_CODE_BIN",
            binary_names=candidate_binary_names("claude"),
            fallback_paths=default_cli_fallback_paths,
        )

    def _probe_binary(self, binary: str) -> CLIProbe:
        try:
            ver = subprocess.run(
                [binary, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=_PROBE_TIMEOUT_SEC,
                check=False,
            )
        except Exception as exc:
            return CLIProbe(False, None, None, None, f"Version check failed: {exc}")

        if ver.returncode != 0:
            return CLIProbe(False, None, None, None, "`claude --version` failed")

        version = _parse_semver(ver.stdout + ver.stderr)
        logged_in, detail = _classify_auth(binary)

        return CLIProbe(
            installed=True,
            version=version,
            logged_in=logged_in,
            bin_path=binary,
            detail=detail,
        )

    def detect(self) -> CLIProbe:
        binary = self._resolve_binary()
        if not binary:
            return CLIProbe(
                installed=False,
                version=None,
                logged_in=None,
                bin_path=None,
                detail=f"Claude CLI not found. {self.install_hint}",
            )
        return self._probe_binary(binary)

    def build(self, *, prompt: str, model: Optional[str], workspace: str, reasoning_effort: Optional[str] = None) -> CLIInvocation:
        binary = self._resolve_binary()
        if not binary:
            raise RuntimeError(f"Claude CLI not found. {self.install_hint}")

        cwd = str(Path(workspace or os.getcwd()).expanduser())

        argv = [binary, "-p", "--output-format", "text"]

        model = (model or "").strip()
        if model:
            argv += ["--model", model]

        return CLIInvocation(
            argv=tuple(argv),
            stdin=prompt,
            cwd=cwd,
            env=_auth_env(),
            timeout_sec=_resolve_exec_timeout_seconds(),
        )

    def parse(self, *, stdout: str, stderr: str, returncode: int) -> str:
        return (stdout or "").strip()

    def explain_failure(self, *, stdout: str, stderr: str, returncode: int) -> str:
        text = (stderr or stdout or "").strip()
        return f"claude failed (code={returncode}): {text[:2000]}"
