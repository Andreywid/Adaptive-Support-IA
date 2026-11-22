from datetime import UTC, datetime

import pytest

from support_assistant import cache as cache_module
from support_assistant import cli, storage
from support_assistant.cache import ResponseCache, hash_schema
from support_assistant.config import load_settings
from support_assistant.credentials import REDACTED_CREDENTIAL
from support_assistant.data_loader import get_profile, get_ticket, ticket_from_text
from support_assistant.evaluation import compare as compare_module
from support_assistant.evaluation.compare import (
    ComparisonError,
    comparison_profile,
    load_comparison_argument,
    new_comparison_id,
    run_comparison,
    save_comparison,
)
from support_assistant.evaluation.review import evaluate_blind
from support_assistant.evaluation.rubric import (
    complete_reviews,
    format_rubric,
    has_complete_evaluation,
)
from support_assistant.models import (
    ContentType,
    Evaluator,
    RunStatus,
    output_json_schema,
)
from support_assistant.providers.base import ProviderError
from support_assistant.providers.fake import CANNED_OUTPUTS, FakeProvider

NOW = datetime(2026, 11, 20, 15, 30, 12, tzinfo=UTC)

# trecho que existe na saída fixa do provedor fake, por tipo, para o avaliador copiar
SNIPPETS = {
    "diagnosis": "Instabilidade do sinal da linha em horário de pico",
    "steps": "Desligue o roteador da tomada e espere meio minuto.",
    "questions": "A internet cai em todos os aparelhos ao mesmo tempo?",
    "flow": "[P1] Todos os aparelhos perdem a conexão?",
}


# provedor fake que se apresenta como groq, para testar a espera entre chamadas reais
class SimulatedGroqProvider(FakeProvider):
    name = "groq"


# executa uma comparação com o provedor fake, esperas falsas e pastas temporárias
def _comparison(
    tmp_path, *, provider=None, versions=("v1", "v2"), seeds=(1, 2), types=None, env=None
):
    sleeps, shown = [], []
    record = run_comparison(
        profile=get_profile("p01_marta"),
        ticket=get_ticket("t01_queda_noturna"),
        versions=list(versions),
        seeds=list(seeds),
        content_types=types or list(ContentType),
        provider=provider or FakeProvider(),
        settings=load_settings(env or {}),
        cache=ResponseCache(tmp_path / "cache"),
        runs_dir=tmp_path / "runs",
        sleep=sleeps.append,
        show=shown.append,
        now=NOW,
    )
    return record, sleeps, shown


# avaliador simulado: nota fixa, justificativa fixa e um trecho da saída mostrada no momento
def _evaluator(shown, scores=None):
    prompts = []
    queue = list(scores or [])

    # responde cada pergunta olhando qual tipo está sendo avaliado
    def ask(prompt):
        prompts.append(prompt)
        assert len(prompts) < 500, "o avaliador simulado entrou em laço"
        if prompt.endswith("nota de 1 a 5: "):
            return queue.pop(0) if queue else "4"
        if prompt == "justificativa: ":
            return "Justificativa de teste."
        current = [line for line in shown if line.startswith("== ") and "(seed" in line][-1]
        return SNIPPETS[current.split()[1]]

    return ask, prompts


# a comparação completa grava 16 execuções e 16 itens, na ordem seed, tipo e versão
def test_full_comparison_runs_everything_in_order(tmp_path):
    record, sleeps, shown = _comparison(tmp_path)
    assert len(record.runs) == 16
    assert len(list((tmp_path / "runs").glob("*.json"))) == 16
    first = record.runs[:4]
    assert [(r.seed, r.content_type.value, r.version) for r in first] == [
        (1, "diagnosis", "v1"),
        (1, "diagnosis", "v2"),
        (1, "steps", "v1"),
        (1, "steps", "v2"),
    ]
    assert shown[0] == "[1/16] v1 diagnosis seed 1: gerado agora (ok)"
    assert shown[-1] == "[16/16] v2 flow seed 2: gerado agora (ok)"
    assert all(run.status is RunStatus.OK and not run.cache_hit for run in record.runs)


# há espera depois de cada chamada real, exceto a última
def test_sleep_after_every_real_call_except_the_last(tmp_path):
    _, sleeps, _ = _comparison(tmp_path, provider=SimulatedGroqProvider())
    assert sleeps == [20.0] * 15


# a espera acontece entre as chamadas, inclusive antes do reparo
def test_sleep_between_requests_including_repair(tmp_path):
    provider = SimulatedGroqProvider(
        responses=["resposta inválida", CANNED_OUTPUTS[ContentType.FLOW].model_dump_json()]
    )
    _, sleeps, _ = _comparison(tmp_path, provider=provider, seeds=(1,), types=[ContentType.FLOW])
    assert [request.schema_name for request in provider.calls] == ["flow"] * 3
    assert sleeps == [20.0, 20.0]


# sem espera com intervalo zero ou com o provedor fake
def test_no_sleep_with_interval_zero_or_fake_provider(tmp_path):
    _, sleeps, _ = _comparison(
        tmp_path / "a", env={"LLM_MIN_INTERVAL_S": "0"}, provider=SimulatedGroqProvider()
    )
    assert sleeps == []
    _, sleeps, _ = _comparison(tmp_path / "b", seeds=(1,), types=[ContentType.FLOW])
    assert sleeps == []


# repetir a comparação é tudo acerto de cache e nunca espera
def test_repeating_the_comparison_hits_the_cache_everywhere(tmp_path):
    _comparison(tmp_path, provider=SimulatedGroqProvider())
    provider = SimulatedGroqProvider()
    record, sleeps, shown = _comparison(tmp_path, provider=provider)
    assert all(run.cache_hit for run in record.runs)
    assert sleeps == []
    assert provider.calls == []
    assert shown[0] == "[1/16] v1 diagnosis seed 1: acerto (ok)"
    assert len(list((tmp_path / "runs").glob("*.json"))) == 32


# a configuração travada vem das configurações, sem seed, e os hashes dos schemas entram
def test_locked_config_and_schema_hashes(tmp_path):
    record, _, _ = _comparison(tmp_path)
    locked = record.locked_config
    assert (locked.provider, locked.model) == ("fake", "openai/gpt-oss-120b")
    assert (locked.temperature, locked.top_p, locked.max_tokens) == (0.0, 1.0, 2048)
    assert locked.reasoning_effort.value == "low" and locked.include_reasoning is False
    assert "seed" not in type(locked).model_fields
    assert record.schema_hashes == {
        item: hash_schema(output_json_schema(item)) for item in ContentType
    }
    assert record.evaluations == []
    assert record.comparison_id.startswith("cmp_20261120T153012Z_")
    assert record.created_at == NOW
    assert (record.customer_id, record.ticket_id) == ("p01_marta", "t01_queda_noturna")
    assert record.versions == ["v1", "v2"] and record.seeds == [1, 2]
    assert record.profile == get_profile("p01_marta")
    assert record.rubric == format_rubric()
    for run in record.runs:
        saved_run = storage.load_run(tmp_path / "runs" / f"{run.run_id}.json")
        assert run.prompt_hash == saved_run.prompt_hash


# cada seed usa a sua seed nos parâmetros da execução
def test_each_run_uses_its_own_seed(tmp_path):
    provider = FakeProvider()
    _comparison(tmp_path, provider=provider, types=[ContentType.FLOW])
    assert [call.params.seed for call in provider.calls] == [1, 1, 2, 2]


# o sorteio cego cobre cada par de tipo e seed, com as duas versões em A e B
def test_blind_map_covers_every_pair(tmp_path):
    record, _, _ = _comparison(tmp_path)
    assert len(record.blind_map) == 8
    assert {(item.content_type, item.seed) for item in record.blind_map} == {
        (content_type, seed) for content_type in ContentType for seed in (1, 2)
    }
    assert all(sorted([item.a, item.b]) == ["v1", "v2"] for item in record.blind_map)


# a mesma semente do sorteio gera o mesmo mapa, e sementes diferentes variam a ordem
def test_blind_map_is_reproducible_by_its_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_module.secrets, "randbelow", lambda limit: 4242)
    first, _, _ = _comparison(tmp_path / "x")
    second, _, _ = _comparison(tmp_path / "y")
    assert first.blind_seed == second.blind_seed == 4242
    assert first.blind_map == second.blind_map
    firsts = set()
    for blind_seed in range(30):
        monkeypatch.setattr(compare_module.secrets, "randbelow", lambda limit, s=blind_seed: s)
        record, _, _ = _comparison(
            tmp_path / f"s{blind_seed}", seeds=(1,), types=[ContentType.FLOW]
        )
        firsts.add(record.blind_map[0].a)
    assert firsts == {"v1", "v2"}


# falha do provedor no meio vira ComparisonError com a posição e as runs já feitas ficam gravadas
def test_provider_error_interrupts_with_a_resumable_message(tmp_path):
    ok = CANNED_OUTPUTS[ContentType.DIAGNOSIS].model_dump_json()
    provider = FakeProvider(responses=[ok, ok, ProviderError("falha de rede")])
    with pytest.raises(ComparisonError) as error:
        _comparison(
            tmp_path,
            provider=provider,
            seeds=(1,),
            types=[ContentType.DIAGNOSIS, ContentType.STEPS],
        )
    assert str(error.value) == (
        "Comparação interrompida em v1 steps seed 1: falha de rede. "
        "Rode o mesmo comando para retomar; o cache evita repetir as chamadas já feitas."
    )
    assert len(list((tmp_path / "runs").glob("*.json"))) == 2


# cenário inválido é recusado antes de qualquer chamada
@pytest.mark.parametrize(
    "changes",
    [
        {"versions": ("v1", "v1")},
        {"versions": ("v1",)},
        {"versions": ("v1", "v2", "v3")},
        {"versions": ("v1", "v99")},
        {"seeds": ()},
        {"seeds": (1, 1)},
        {"seeds": (1, -1)},
        {"seeds": (1, True)},
        {"types": []},
        {"types": [ContentType.FLOW, "inventado"]},
    ],
)
def test_invalid_scenario_fails_before_calling_the_provider(tmp_path, changes):
    provider = FakeProvider()
    kwargs = {"types": list(ContentType), **changes}
    if kwargs["types"] == []:
        kwargs["types"] = None
        with pytest.raises(ComparisonError):
            run_comparison(
                profile=get_profile("p01_marta"),
                ticket=get_ticket("t01_queda_noturna"),
                versions=["v1", "v2"],
                seeds=[1],
                content_types=[],
                provider=provider,
                settings=load_settings({}),
                cache=None,
            )
    else:
        with pytest.raises(ComparisonError):
            _comparison(tmp_path, provider=provider, **kwargs)
    assert provider.calls == []


# salvar e carregar devolve a mesma comparação, num arquivo com o nome do id
def test_save_and_load_roundtrip(tmp_path):
    record, _, _ = _comparison(tmp_path)
    path = save_comparison(record, tmp_path / "comparisons")
    assert path.name == f"{record.comparison_id}.json"
    loaded, _ = load_comparison_argument(record.comparison_id, tmp_path / "comparisons")
    assert loaded == record


# a comparação é lida e atualizada pelo caminho do arquivo
def test_compare_reads_and_updates_an_explicit_sample_path(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    path = tmp_path / "samples" / "comparacao_curada.json"
    save_comparison(record, path=path)
    assert load_comparison_argument(str(path)) == (record, path)


# comparação antiga, com id e perfil no formato anterior, ainda abre
def test_compare_reads_a_legacy_id_and_profile(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    legacy_id = "cmp_20261120T153012Z"
    legacy = record.model_copy(update={"comparison_id": legacy_id, "profile": None, "rubric": None})
    save_comparison(legacy, tmp_path)
    restored, path = load_comparison_argument(legacy_id, tmp_path)
    assert path == tmp_path / f"{legacy_id}.json"
    assert restored == legacy
    assert comparison_profile(restored) == get_profile(legacy.customer_id)


# o ticket é gravado sem credenciais numa comparação nova
def test_new_comparison_redacts_the_ticket_before_persistence(tmp_path):
    ticket = ticket_from_text("Minha internet caiu. Senha: exemplo-42.")
    record = run_comparison(
        profile=get_profile("p01_marta"),
        ticket=ticket,
        versions=["v1", "v2"],
        seeds=[1],
        content_types=[ContentType.DIAGNOSIS],
        provider=FakeProvider(),
        settings=load_settings({"LLM_MIN_INTERVAL_S": "0"}),
        cache=None,
        runs_dir=tmp_path,
        show=lambda _: None,
    )
    assert record.ticket != ticket.text
    assert REDACTED_CREDENTIAL in record.ticket


# carregar id ausente, inválido ou arquivo corrompido gera ComparisonError claro
def test_load_comparison_errors(tmp_path):
    with pytest.raises(ComparisonError, match="Comparação 'cmp_x' não encontrada em comparisons/"):
        load_comparison_argument("cmp_x", tmp_path)
    with pytest.raises(ComparisonError, match="Identificador de comparação inválido"):
        load_comparison_argument("../fora", tmp_path)
    (tmp_path / "cmp_quebrada.json").write_text("{ não é json", encoding="utf-8")
    with pytest.raises(ComparisonError, match="ilegível"):
        load_comparison_argument("cmp_quebrada", tmp_path)


# o identificador segue o formato cmp_ mais data e hora em UTC
def test_new_comparison_id_format():
    first, second = new_comparison_id(NOW), new_comparison_id(NOW)
    for item in (first, second):
        prefix, suffix = item.rsplit("_", 1)
        assert prefix == "cmp_20261120T153012Z"
        assert len(suffix) == 12 and int(suffix, 16) >= 0
    assert first != second


# a avaliação cega grava notas, justificativas e trechos e só revela as versões no fim de cada tipo
def test_blind_evaluation_records_scores_and_reveals_at_the_end(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,))
    shown, saved = [], []
    ask, prompts = _evaluator(shown)
    evaluated = evaluate_blind(
        record,
        seed=1,
        ask=ask,
        show=shown.append,
        now=NOW,
        save=lambda r: saved.append(r) or tmp_path,
    )
    assert len(evaluated.evaluations) == 4
    assert len(saved) == 4 and [len(r.evaluations) for r in saved] == [1, 2, 3, 4]
    assert len(prompts) == 4 * 2 * 3 * 3
    first = evaluated.evaluations[0]
    assert (first.content_type, first.seed) == (ContentType.DIAGNOSIS, 1)
    assert [o.position for o in first.outputs] == ["A", "B"]
    item = record.blind_map[0]
    assert [o.version for o in first.outputs] == [item.a, item.b]
    evaluation = first.outputs[0].evaluation
    assert evaluation.clarity.score == 4
    assert evaluation.customer_fit.justification == "Justificativa de teste."
    assert evaluation.actionability.evidence == SNIPPETS["diagnosis"]
    assert evaluation.evaluator is Evaluator.HUMAN and evaluation.evaluated_at == NOW
    reveal = f"Revelado: A = {item.a}, B = {item.b}"
    assert shown.index(reveal) > shown.index("== B ==")
    before_reveal = "\n".join(shown[: shown.index(reveal)])
    assert "v1" not in before_reveal and "v2" not in before_reveal
    assert "Marta, leigo, didático, telefone" in before_reveal and record.ticket in before_reveal
    assert "atendente e cliente, que recebe o resumo" in before_reveal
    assert format_rubric() in shown
    assert prompts[0] == "A, Clareza, nota de 1 a 5: "
    assert prompts[3] == "A, Adequação ao nível do cliente, nota de 1 a 5: "
    assert prompts[9] == "B, Clareza, nota de 1 a 5: "


# a avaliação cega usa o perfil e a rubrica gravados na comparação
def test_blind_evaluation_uses_the_archived_profile_and_rubric(tmp_path, monkeypatch):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    monkeypatch.setattr(compare_module, "get_profile", lambda _: pytest.fail("usou o catálogo"))
    archived = record.model_copy(update={"rubric": "Texto arquivado da rubrica"})
    shown = []
    ask, _ = _evaluator(shown)
    evaluate_blind(archived, seed=1, ask=ask, show=shown.append, now=NOW)
    assert "Texto arquivado da rubrica" in shown


# par incompleto é avisado antes de pedir as notas
def test_incomplete_pair_is_reported_before_asking_for_scores(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    incomplete = record.model_copy(update={"runs": record.runs[:1]})
    with pytest.raises(ComparisonError, match="Par incompleto para flow na seed 1"):
        evaluate_blind(
            incomplete,
            seed=1,
            ask=lambda _: pytest.fail("pediu nota sem as duas saídas"),
            show=lambda _: None,
        )


# nota 6, justificativa vazia e trecho inexistente são perguntados de novo
def test_invalid_answers_are_asked_again(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    shown = []
    answers = [
        "6",
        "x",
        "0",
        "4",
        "",
        "  ",
        "Boa justificativa.",
        "esse trecho não existe",
        "",
        SNIPPETS["flow"],
    ]

    # responde as 3 primeiras perguntas com respostas ruins antes das boas e o resto normalmente
    def ask(prompt):
        if answers:
            return answers.pop(0)
        if prompt.endswith("nota de 1 a 5: "):
            return "3"
        if prompt == "justificativa: ":
            return "ok"
        return SNIPPETS["flow"]

    evaluated = evaluate_blind(record, seed=1, ask=ask, show=shown.append, now=NOW)
    first = evaluated.evaluations[0].outputs[0].evaluation.clarity
    assert (first.score, first.justification, first.evidence) == (
        4,
        "Boa justificativa.",
        SNIPPETS["flow"],
    )
    assert "Trecho não encontrado na saída A. Copie um trecho exato." in shown


# o trecho é comparado com espaços normalizados
def test_evidence_matching_ignores_whitespace_differences(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    shown = []

    # copia o trecho com espaços e quebras diferentes das do texto mostrado
    def ask(prompt):
        if prompt.endswith("nota de 1 a 5: "):
            return "5"
        if prompt == "justificativa: ":
            return "ok"
        return "[P1]   Todos os aparelhos\nperdem a conexão?"

    evaluated = evaluate_blind(record, seed=1, ask=ask, show=shown.append, now=NOW)
    assert evaluated.evaluations[0].outputs[0].evaluation.clarity.score == 5


# rodar de novo pula os tipos já avaliados, e avaliar outra seed continua de onde parou
def test_evaluation_skips_what_is_already_evaluated(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1, 2), types=[ContentType.FLOW, ContentType.STEPS])
    shown = []
    ask, _ = _evaluator(shown)
    once = evaluate_blind(record, seed=1, ask=ask, show=shown.append, now=NOW)
    asked = []
    twice = evaluate_blind(
        once, seed=1, ask=lambda p: asked.append(p) or "", show=shown.append, now=NOW
    )
    assert asked == [] and twice == once
    other = evaluate_blind(once, seed=2, ask=ask, show=shown.append, now=NOW)
    assert [(e.content_type.value, e.seed) for e in other.evaluations] == [
        ("flow", 1),
        ("steps", 1),
        ("flow", 2),
        ("steps", 2),
    ]


# avaliar de novo um par incompleto troca a nota antiga em vez de acumular
def test_new_evaluation_replaces_an_incomplete_pair(tmp_path, capsys):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    shown = []
    ask, _ = _evaluator(shown)
    pair = evaluate_blind(record, seed=1, ask=ask, show=shown.append, now=NOW).evaluations[0]
    swapped = pair.model_copy(
        update={
            "outputs": [
                output.model_copy(update={"position": position})
                for output, position in zip(pair.outputs, ("B", "A"), strict=True)
            ]
        }
    )
    broken = record.model_copy(update={"evaluations": [swapped]})
    assert complete_reviews(broken, 1) == {}
    reviewed = evaluate_blind(broken, seed=1, ask=ask, show=shown.append, now=NOW)
    assert len(reviewed.evaluations) == 1
    assert complete_reviews(reviewed, 1) == {ContentType.FLOW: reviewed.evaluations[0]}
    path = save_comparison(reviewed, tmp_path)
    assert cli.main(["checks", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Avaliação completa" in out and "Clareza" in out and "v1" in out


# para completar, os pares precisam usar a mesma seed e as posições sorteadas
def test_complete_evaluation_respects_seed_and_blind_positions(tmp_path):
    record, _, _ = _comparison(tmp_path, types=[ContentType.FLOW, ContentType.STEPS])
    shown = []
    ask, _ = _evaluator(shown)
    both = evaluate_blind(record, seed=1, ask=ask, show=shown.append, now=NOW)
    flow, steps = both.evaluations
    swapped = steps.model_copy(
        update={
            "outputs": [
                output.model_copy(update={"position": position})
                for output, position in zip(steps.outputs, ("B", "A"), strict=True)
            ]
        }
    )
    for wrong_steps in (swapped, steps.model_copy(update={"seed": 2})):
        incomplete = record.model_copy(update={"evaluations": [flow, wrong_steps]})
        assert complete_reviews(incomplete, 1) == {ContentType.FLOW: flow}
        assert not has_complete_evaluation(incomplete)


# saída fora do contrato é mostrada como texto bruto e pode ser avaliada
def test_invalid_output_is_shown_as_raw_text(tmp_path):
    provider = FakeProvider(responses=["ruim", "ruim"])
    record, _, _ = _comparison(tmp_path, provider=provider, seeds=(1,), types=[ContentType.FLOW])
    assert [run.status for run in record.runs] == [RunStatus.INVALID_OUTPUT, RunStatus.OK]
    item = record.blind_map[0]
    invalid_position = "A" if item.a == "v1" else "B"
    shown, position, asked = [], [], []

    # copia o texto bruto na posição da saída inválida e um trecho do fluxo na outra
    def ask(prompt):
        asked.append(prompt)
        assert len(asked) < 100, "o avaliador simulado entrou em laço"
        if prompt.endswith("nota de 1 a 5: "):
            position.append(prompt[0])
            return "1"
        if prompt == "justificativa: ":
            return "ok"
        return "ruim" if position[-1] == invalid_position else SNIPPETS["flow"]

    evaluate_blind(record, seed=1, ask=ask, show=shown.append, now=NOW)
    assert "Saída fora do contrato. Texto bruto:\nruim" in shown


# seed que não faz parte da comparação gera ComparisonError
def test_evaluating_an_unknown_seed_fails(tmp_path):
    record, _, _ = _comparison(tmp_path, seeds=(1,), types=[ContentType.FLOW])
    with pytest.raises(ComparisonError, match="A seed 7 não faz parte da comparação. Seeds: 1"):
        evaluate_blind(record, seed=7, ask=lambda p: "", show=lambda s: None)


# isola o ambiente da CLI: pastas temporárias, provedor fake, sem espera e pasta atual no tmp_path
@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(compare_module, "COMPARISONS_DIR", tmp_path / "comparisons")
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LLM_MIN_INTERVAL_S", "0")
    monkeypatch.chdir(tmp_path)
    return tmp_path


COMPARE = [
    "compare",
    "--profile",
    "p01_marta",
    "--ticket",
    "t01_queda_noturna",
    "--versions",
    "v1",
    "v2",
    "--seeds",
    "1",
    "2",
    "--types",
    "flow",
]


# compare pela CLI mostra o progresso e o caminho do arquivo gravado
def test_cli_compare_prints_progress_and_the_path(cli_env, capsys):
    assert cli.main(COMPARE) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[1/4] v1 flow seed 1: gerado agora (ok)"
    assert lines[3] == "[4/4] v2 flow seed 2: gerado agora (ok)"
    assert lines[-1].startswith("comparação: comparisons/cmp_") and lines[-1].endswith(".json")
    assert len(list((cli_env / "comparisons").glob("*.json"))) == 1
    assert len(list((cli_env / "runs").glob("*.json"))) == 4


# compare sem --types usa os 4 tipos e com versões iguais devolve erro 1
def test_cli_compare_defaults_to_all_types_and_rejects_equal_versions(cli_env, capsys):
    args = [item for item in COMPARE if item not in ("--types", "flow")]
    args = [*args[: args.index("--seeds")], "--seeds", "1"]
    assert cli.main(args) == 0
    assert capsys.readouterr().out.count("gerado agora") == 8
    same = [*COMPARE]
    same[same.index("v2")] = "v1"
    assert cli.main(same) == 1
    assert (
        "Erro: A comparação precisa de exatamente 2 versões diferentes." in capsys.readouterr().err
    )


# evaluate pela CLI avalia, salva a comparação e imprime o caminho
def test_cli_evaluate_saves_the_evaluation(cli_env, capsys, monkeypatch):
    cli.main(COMPARE)
    path = capsys.readouterr().out.splitlines()[-1].removeprefix("comparação: ")
    comparison_id = path.split("/")[-1].removesuffix(".json")
    shown = []

    # alimenta o input do avaliador com respostas válidas
    def fake_input(prompt=""):
        if prompt.endswith("nota de 1 a 5: "):
            return "5"
        if prompt == "justificativa: ":
            return "ok"
        return SNIPPETS["flow"]

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["evaluate", comparison_id, "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert "Revelado: A = " in out and out.splitlines()[-1] == f"comparação: {path}"
    saved, _ = load_comparison_argument(comparison_id, cli_env / "comparisons")
    assert [(e.content_type.value, e.seed) for e in saved.evaluations] == [("flow", 2)]
    assert shown == []


# evaluate com id inexistente ou seed fora da comparação devolve 1 com a mensagem
def test_cli_evaluate_errors(cli_env, capsys):
    assert cli.main(["evaluate", "cmp_nao_existe"]) == 1
    assert (
        "Erro: Comparação 'cmp_nao_existe' não encontrada em comparisons/"
        in capsys.readouterr().err
    )
    cli.main(COMPARE)
    path = capsys.readouterr().out.splitlines()[-1]
    comparison_id = path.split("/")[-1].removesuffix(".json")
    assert cli.main(["evaluate", comparison_id, "--seed", "9"]) == 1
    assert "A seed 9 não faz parte da comparação" in capsys.readouterr().err
