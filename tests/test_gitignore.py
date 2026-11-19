import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="precisa do git e de um repositório",
)


# pergunta ao git se o caminho é ignorado e falha em qualquer erro do próprio git
def _is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", path],
        cwd=ROOT,
        check=False,
    )
    assert result.returncode in (0, 1), f"git falhou para {path}"
    return result.returncode == 0


# segredos, ambiente e saídas de execução nunca entram no Git
@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".venv/bin/python",
        "data/cache/abc.json",
        "runs/run.json",
        "comparisons/cmp.json",
        ".streamlit/secrets.toml",
        "support_assistant/__pycache__/config.cpython-312.pyc",
    ],
)
def test_sensitive_or_generated_path_is_ignored(path):
    assert _is_ignored(path)


# arquivos que precisam ser versionados não podem cair no .gitignore
@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "requirements.txt",
        "data/profiles.json",
        "samples/example.json",
        "README.md",
    ],
)
def test_versioned_file_is_not_ignored(path):
    assert not _is_ignored(path)
