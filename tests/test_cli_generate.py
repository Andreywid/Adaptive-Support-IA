import runpy
import sys

import pytest

from support_assistant import cache as cache_module
from support_assistant import cli, storage
from support_assistant.config import load_settings
from support_assistant.providers.base import ProviderError
from support_assistant.providers.fake import FakeProvider
from support_assistant.storage import list_runs

BASE = [
    "generate",
    "--profile",
    "p01_marta",
    "--ticket",
    "t01_queda_noturna",
    "--type",
    "flow",
    "--version",
    "v1",
    "--provider",
    "fake",
]


# isola o ambiente: pastas temporárias, configuração padrão e pasta atual no tmp_path
@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(cli, "load_settings", lambda: load_settings({}))
    monkeypatch.chdir(tmp_path)


# o fluxo gerado é impresso, o registro é gravado e a segunda chamada é acerto de cache
def test_generate_prints_the_flow_and_then_hits_the_cache(tmp_path, capsys):
    assert cli.main(BASE) == 0
    first = capsys.readouterr().out
    assert "[P1]" in first
    assert "cache: gerado agora" in first
    assert "status: ok" in first
    assert "registro: runs/" in first
    assert len(list(tmp_path.glob("runs/*.json"))) == 1
    assert cli.main(BASE) == 0
    second = capsys.readouterr().out
    assert "cache: acerto" in second
    assert len(list(tmp_path.glob("runs/*.json"))) == 2


# a saída termina com o estado do cache, o status e o caminho do registro
def test_output_layout(capsys):
    cli.main(BASE)
    lines = capsys.readouterr().out.splitlines()
    assert lines[-5:-2] == ["", "cache: gerado agora", "status: ok"]
    assert lines[-2] == cli.FAKE_NOTICE
    assert lines[-1].startswith("registro: runs/") and lines[-1].endswith(".json")


# tipos que não são fluxo saem como JSON indentado
def test_non_flow_output_is_indented_json(capsys):
    args = [item if item != "flow" else "steps" for item in BASE]
    cli.main(args)
    out = capsys.readouterr().out
    assert out.startswith('{\n  "opening":')


# --no-cache mostra desligado e não grava cache; --refresh grava e mostra gerado agora
def test_no_cache_and_refresh(tmp_path, capsys):
    assert cli.main([*BASE, "--no-cache"]) == 0
    assert "cache: desligado" in capsys.readouterr().out
    assert not (tmp_path / "cache").exists()
    assert cli.main([*BASE, "--refresh"]) == 0
    assert "cache: gerado agora" in capsys.readouterr().out
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1


# o registro gravado traz o provedor, o modelo e os parâmetros do ambiente
def test_saved_record_has_provider_and_params(tmp_path):
    cli.main(BASE)
    ((_, record),) = list_runs()
    assert (record.provider, record.model) == ("fake", "openai/gpt-oss-120b")
    assert record.params.temperature == 0.0
    assert record.customer_id == "p01_marta"
    assert record.ticket_id == "t01_queda_noturna"


# perfil inexistente devolve 1 e a mensagem do DataError no stderr
def test_unknown_profile_exits_with_1(capsys):
    args = [item if item != "p01_marta" else "p09" for item in BASE]
    assert cli.main(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("Erro: Perfil 'p09' não existe. Disponíveis: p01_marta")


# ticket inexistente e versão inexistente também viram Erro com código 1
@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("t01_queda_noturna", "t09", "Erro: Ticket 't09' não existe"),
        ("v1", "v99", "Erro: Versão de prompt 'v99' não existe"),
    ],
)
def test_other_known_errors_exit_with_1(old, new, message, capsys):
    args = [new if item == old else item for item in BASE]
    assert cli.main(args) == 1
    assert capsys.readouterr().err.startswith(message)


# texto livre grava o ticket com o id reservado
def test_free_text_ticket_uses_the_reserved_id():
    args = [item for item in BASE if item not in ("--ticket", "t01_queda_noturna")]
    assert cli.main([*args, "--ticket-text", "A internet caiu"]) == 0
    ((_, record),) = list_runs()
    assert record.ticket_id == "texto_livre"
    assert record.ticket == "A internet caiu"


# texto livre vazio devolve 1 com a mensagem padrão
def test_empty_free_text_exits_with_1(capsys):
    args = [item for item in BASE if item not in ("--ticket", "t01_queda_noturna")]
    assert cli.main([*args, "--ticket-text", "   "]) == 1
    assert "O texto do ticket precisa ter de 1 a 2000 caracteres." in capsys.readouterr().err


# erro do provedor vira mensagem curta e código 1
def test_provider_error_exits_with_1(monkeypatch, capsys):
    provider = FakeProvider(responses=[ProviderError("o provedor caiu")])
    monkeypatch.setattr(cli, "make_provider", lambda settings, name: provider)
    assert cli.main(BASE) == 1
    assert capsys.readouterr().err == "Erro: o provedor caiu\n"


# saída fora do contrato grava o registro e avisa que o texto bruto está nele
def test_invalid_output_is_recorded_and_explained(monkeypatch, capsys):
    provider = FakeProvider(responses=["ruim", "ruim"])
    monkeypatch.setattr(cli, "make_provider", lambda settings, name: provider)
    assert cli.main(BASE) == 1
    out = capsys.readouterr().out
    assert out.startswith("Saída fora do contrato. O texto bruto está no registro.\n- resposta:")
    assert "\n\ncache: gerado agora" in out
    assert "status: invalid_output" in out
    ((_, record),) = list_runs()
    assert record.raw_output == "ruim"
    assert record.validation_errors


# Ctrl+C imprime Interrompido. e devolve 130
def test_keyboard_interrupt_exits_with_130(monkeypatch, capsys):
    # simula o usuário apertando Ctrl+C durante a geração
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "generate_content", interrupt)
    assert cli.main(BASE) == 130
    assert capsys.readouterr().err == "Interrompido.\n"


# uso errado sai com código 2 do argparse
@pytest.mark.parametrize(
    "args",
    [
        ["generate"],
        [*BASE, "--no-cache", "--refresh"],
        [*BASE, "--ticket-text", "texto"],
        ["generate", "--profile", "p01_marta", "--type", "flow", "--version", "v1"],
        [*BASE[:-1], "openai"],
    ],
)
def test_wrong_usage_exits_with_2(args, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


# python -m support_assistant executa o main e sai com o código dele
def test_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["support_assistant", *BASE])
    with pytest.raises(SystemExit) as error:
        runpy.run_module("support_assistant", run_name="__main__")
    assert error.value.code == 0
    assert "[P1]" in capsys.readouterr().out
