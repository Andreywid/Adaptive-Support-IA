import json
import os

import pytest

from support_assistant import files
from support_assistant.files import atomic_write_json


# grava JSON legível, com acentos preservados e quebra de linha no fim
def test_writes_readable_json_with_accents(tmp_path):
    path = tmp_path / "dados.json"
    atomic_write_json(path, {"nome": "João", "lista": [1, 2]})
    text = path.read_text(encoding="utf-8")
    assert "João" in text
    assert text.endswith("\n")
    assert json.loads(text) == {"nome": "João", "lista": [1, 2]}


# cria as pastas que faltam
def test_creates_missing_directories(tmp_path):
    path = tmp_path / "a" / "b" / "dados.json"
    atomic_write_json(path, {"ok": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}


# substitui o arquivo antigo e não deixa temporário
def test_replaces_the_old_file_and_leaves_no_temp(tmp_path):
    path = tmp_path / "dados.json"
    atomic_write_json(path, {"v": 1})
    atomic_write_json(path, {"v": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 2}
    assert [item.name for item in tmp_path.iterdir()] == ["dados.json"]


# se a serialização falhar, o arquivo antigo fica intacto e não sobra temporário
def test_serialization_failure_keeps_the_old_file(tmp_path):
    path = tmp_path / "dados.json"
    atomic_write_json(path, {"v": 1})
    with pytest.raises(TypeError):
        atomic_write_json(path, {"inválido": {1, 2}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}
    assert [item.name for item in tmp_path.iterdir()] == ["dados.json"]


# se a troca final falhar, o temporário é apagado e o arquivo antigo fica intacto
def test_replace_failure_cleans_the_temp_file(tmp_path, monkeypatch):
    path = tmp_path / "dados.json"
    atomic_write_json(path, {"v": 1})

    # simula a falha na troca final do arquivo
    def broken_replace(source, destination):
        raise OSError("disco cheio")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError, match="disco cheio"):
        atomic_write_json(path, {"v": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"v": 1}
    assert [item.name for item in tmp_path.iterdir()] == ["dados.json"]


# o temporário fica na mesma pasta do destino, para a troca ser atômica
def test_temp_file_is_created_next_to_the_destination(tmp_path, monkeypatch):
    seen = {}
    original = files.tempfile.NamedTemporaryFile

    # registra a pasta onde o temporário foi pedido
    def spy(*args, **kwargs):
        seen["dir"] = kwargs["dir"]
        return original(*args, **kwargs)

    monkeypatch.setattr(files.tempfile, "NamedTemporaryFile", spy)
    atomic_write_json(tmp_path / "dados.json", {})
    assert seen["dir"] == tmp_path
