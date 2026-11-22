import json
from pathlib import Path

import pytest

from support_assistant import cache as cache_module
from support_assistant import cli, storage
from support_assistant.credentials import REDACTED_CREDENTIAL
from support_assistant.evaluation import compare as compare_module
from support_assistant.storage import list_runs

GENERATE = [
    "generate",
    "--profile",
    "p01_marta",
    "--ticket",
    "t01_queda_noturna",
    "--type",
    "flow",
    "--version",
    "v1",
]


# isola o ambiente: pastas temporárias, provedor fake e pasta atual no tmp_path
@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.chdir(tmp_path)


# gera uma execução com o provedor fake, mudando o que for pedido nos argumentos
def _generate(**changes):
    args = list(GENERATE)
    for flag, value in changes.items():
        args[args.index(f"--{flag}") + 1] = value
    assert cli.main(args) == 0


# simula as respostas digitadas no menu e guarda as perguntas feitas
def _typed(monkeypatch, answers):
    prompts = []
    queue = list(answers)

    # entrega a próxima resposta, levantando EOFError quando acabam
    def fake_input(prompt=""):
        prompts.append(prompt)
        if not queue:
            raise EOFError
        return queue.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


# profiles mostra o cabeçalho e uma linha por perfil, com rótulos acentuados e colunas alinhadas
def test_profiles(capsys):
    assert cli.main(["profiles"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split() for line in lines] == [
        ["id", "nome", "nível", "estilo", "canal"],
        ["p01_marta", "Marta", "leigo", "didático", "telefone"],
        ["p02_rafael", "Rafael", "intermediário", "direto", "chat"],
        ["p03_carla", "Carla", "avançado", "detalhado", "e-mail"],
        ["p04_joao", "João", "leigo", "direto", "chat"],
    ]
    assert lines[2].index("intermediário") == 12 + 1 + 8 + 1
    assert lines[2].index("direto") == 12 + 1 + 8 + 1 + 14 + 1


# tickets mostra o id e o início do texto em uma linha, cortado em 70 caracteres com reticências
def test_tickets(capsys):
    assert cli.main(["tickets"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 8
    assert lines[0].startswith("t01_queda_noturna    Boa noite. A internet cai toda noite")
    assert lines[-1].endswith("Minha internet não está boa. Podem ver isso?")
    for line in lines[:-1]:
        assert line.endswith("...")
        assert len(line) == 20 + 1 + 70 + 3
    assert "\n" not in lines[1]
    assert "wlan0" not in lines[1]


# history mostra as mais recentes primeiro, no formato esperado, e o total na última linha
def test_history_lists_the_most_recent_first(capsys):
    _generate(type="steps")
    _generate(type="flow")
    capsys.readouterr()
    assert cli.main(["history"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "2 execuções em runs/"
    first, second = lines[0].split(), lines[1].split()
    assert first[5:] == ["flow", "v1", "ok", "gerado"]
    assert second[5:] == ["steps", "v1", "ok", "gerado"]
    assert first[3:5] == ["p01_marta", "t01_queda_noturna"]
    assert lines[0][:16].count("-") == 2 and lines[0][10] == " "


# history marca acerto de cache e respeita o limite sem esconder o total
def test_history_marks_cache_hits_and_respects_limit(capsys):
    _generate()
    _generate()
    _generate(type="steps")
    capsys.readouterr()
    assert cli.main(["history", "--limit", "2"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 3
    assert lines[-1] == "3 execuções em runs/"
    assert lines[1].split()[-1] == "acerto"


# history filtra por perfil, ticket, tipo, versão e chave de cache
def test_history_filters(capsys):
    _generate(type="steps")
    _generate(type="flow")
    _generate(type="flow", profile="p02_rafael")
    (_, record), *_ = list_runs(content_type=cli.ContentType.STEPS)
    capsys.readouterr()

    # roda o history com filtros e devolve só as linhas de execução
    def history(*filters):
        assert cli.main(["history", *filters]) == 0
        return capsys.readouterr().out.splitlines()[:-1]

    assert len(history("--profile", "p02_rafael")) == 1
    assert len(history("--type", "flow")) == 2
    assert len(history("--ticket", "t01_queda_noturna")) == 3
    assert history("--ticket", "t03_codigo_sms") == []
    assert history("--version", "v3") == []
    assert len(history("--cache-key", record.cache_key)) == 1


# history sem execuções mostra só o total zero
def test_history_without_runs(capsys):
    assert cli.main(["history"]) == 0
    assert capsys.readouterr().out == "0 execuções em runs/\n"


# argumentos inválidos saem com código 2
@pytest.mark.parametrize("args", [["history", "--limit", "0"], ["history", "--limit", "x"]])
def test_invalid_arguments_exit_with_2(args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


# o menu mostra as listas dos dados, pergunta na ordem e gera com as escolhas
def test_menu_with_valid_answers(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_versions", lambda: ["v1", "v2"])
    prompts = _typed(monkeypatch, ["1", "1", "4", "1"])
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith(
        "Perfil:\n"
        "  1) p01_marta (Marta: leigo, didático, telefone)\n"
        "  2) p02_rafael (Rafael: intermediário, direto, chat)\n"
        "  3) p03_carla (Carla: avançado, detalhado, e-mail)\n"
        "  4) p04_joao (João: leigo, direto, chat)\n"
        "Ticket:\n"
        "  1) t01_queda_noturna\n"
        "  2) t02_wifi_lento\n"
        "  3) t03_codigo_sms\n"
        "  4) t04_sem_sinal\n"
        "  5) t05_tv_travando\n"
        "  6) t06_latencia_jogos\n"
        "  7) t07_cobranca_senha\n"
        "  8) t08_relato_vago\n"
        "  9) texto livre\n"
        "Tipo:\n"
        "  1) diagnosis\n"
        "  2) steps\n"
        "  3) questions\n"
        "  4) flow\n"
        "Versão:\n"
        "  1) v1\n"
        "  2) v2\n"
    )
    assert prompts == ["Escolha [1-4]: ", "Escolha [1-9]: ", "Escolha [1-4]: ", "Escolha [1-2]: "]
    assert "[P1]" in out and "cache: gerado agora" in out and "registro: runs/" in out
    ((_, record),) = list_runs()
    assert (record.customer_id, record.ticket_id) == ("p01_marta", "t01_queda_noturna")
    assert (record.content_type.value, record.prompt_version) == ("flow", "v1")


# escolha inválida repete a pergunta até a resposta valer
def test_menu_repeats_the_question_on_invalid_choice(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_versions", lambda: ["v1"])
    prompts = _typed(monkeypatch, ["9", "x", "", "2", "3", "0", "2", "1", "1"])
    assert cli.main([]) == 0
    assert prompts == [
        "Escolha [1-4]: ",
        "Escolha [1-4]: ",
        "Escolha [1-4]: ",
        "Escolha [1-4]: ",
        "Escolha [1-9]: ",
        "Escolha [1-4]: ",
        "Escolha [1-4]: ",
        "Escolha [1-1]: ",
    ]
    ((_, record),) = list_runs()
    assert (record.customer_id, record.ticket_id, record.content_type.value) == (
        "p02_rafael",
        "t03_codigo_sms",
        "steps",
    )


# a opção texto livre pergunta o ticket em uma linha e grava com o id reservado
def test_menu_free_text_ticket(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_versions", lambda: ["v1"])
    prompts = _typed(monkeypatch, ["3", "9", "A internet caiu agora", "2", "1"])
    assert cli.main([]) == 0
    assert prompts[2] == "Digite o ticket em uma linha: "
    ((_, record),) = list_runs()
    assert (record.ticket_id, record.ticket) == ("texto_livre", "A internet caiu agora")


# nas opções de ticket do arquivo a pergunta do texto livre não aparece
def test_menu_does_not_ask_free_text_for_file_tickets(monkeypatch):
    monkeypatch.setattr(cli, "list_versions", lambda: ["v1"])
    prompts = _typed(monkeypatch, ["1", "2", "1", "1"])
    cli.main([])
    assert "Digite o ticket em uma linha: " not in prompts


# EOF ou Ctrl+C no menu imprimem Interrompido. e devolvem 130
def test_menu_interruption_exits_with_130(monkeypatch, capsys):
    _typed(monkeypatch, ["1"])
    assert cli.main([]) == 130
    assert capsys.readouterr().err == "Interrompido.\n"

    # simula o usuário apertando Ctrl+C na primeira pergunta
    def interrupt(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert cli.main([]) == 130
    assert capsys.readouterr().err == "Interrompido.\n"


# o menu usa o provedor das configurações, que aqui é o fake
def test_menu_uses_the_configured_provider(monkeypatch):
    monkeypatch.setattr(cli, "list_versions", lambda: ["v1"])
    _typed(monkeypatch, ["1", "1", "1", "1"])
    cli.main([])
    ((_, record),) = list_runs()
    assert record.provider == "fake"


# compare aceita texto livre e grava o ticket sem credenciais
def test_compare_uses_free_text_and_preserves_a_safe_ticket(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_module, "COMPARISONS_DIR", tmp_path / "comparisons")
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "0")
    args = [
        "compare",
        "--profile",
        "p01_marta",
        "--ticket-text",
        "O aplicativo não funciona. Minha senha é exemplo2468.",
        "--versions",
        "v1",
        "v2",
        "--seeds",
        "1",
        "--types",
        "steps",
    ]
    assert cli.main(args) == 0
    (path,) = (tmp_path / "comparisons").glob("*.json")
    record = json.loads(path.read_text())
    assert record["ticket_id"] == "texto_livre"
    assert REDACTED_CREDENTIAL in record["ticket"]
    assert "exemplo2468" not in path.read_text()
    assert len(list_runs(ticket_id="texto_livre")) == 2
    assert all("exemplo2468" not in run.ticket for _, run in list_runs())


# checks exige uma avaliação completa e preserva o arquivo
def test_delivery_check_requires_scores_without_rewriting_file(tmp_path, capsys):
    fixtures = Path(__file__).resolve().parent.parent / "samples"
    sample = json.loads((fixtures / "comparacao_v1_v3_p01_t01.json").read_text())
    sample["evaluations"] = []
    copy = tmp_path / "comparacao.json"
    original = json.dumps(sample)
    copy.write_text(original)

    assert cli.main(["checks", str(copy)]) == 1
    assert "Falta uma seed com a avaliação das duas versões" in capsys.readouterr().err
    assert copy.read_text() == original
