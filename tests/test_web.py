import json
from datetime import UTC, datetime
from pathlib import Path
from shutil import copyfile

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from support_assistant import cache as cache_module
from support_assistant import config, storage
from support_assistant.config import load_settings
from support_assistant.data_loader import get_profile, get_ticket
from support_assistant.evaluation import compare as compare_module
from support_assistant.evaluation.compare import run_comparison
from support_assistant.models import (
    BlindItem,
    ContentEvaluation,
    ContentType,
    CriterionScore,
    EvaluatedOutput,
    Evaluation,
    Evaluator,
)
from support_assistant.prompts.engine import DEFAULT_VERSION
from support_assistant.providers import factory as provider_factory
from support_assistant.providers.fake import FakeProvider
from web.presentation import PROMPT_LABELS

APP_PATH = Path(__file__).resolve().parent.parent / "web" / "app.py"
COMPARISON_SAMPLE = Path("samples", "comparacao_teste.json")
FLOW_EVIDENCE = "Todos os aparelhos perdem a conexão?"
NOW = datetime(2026, 11, 20, 15, 30, 12, tzinfo=UTC)


# nota humana de uma saída, com o trecho citado
def _human_output(position, version, score, evidence):
    note = CriterionScore(
        score=score, justification="Queda descrita em linguagem simples.", evidence=evidence
    )
    evaluation = Evaluation(
        clarity=note,
        customer_fit=note,
        actionability=note,
        evaluator=Evaluator.HUMAN,
        evaluated_at=NOW,
    )
    return EvaluatedOutput(position=position, version=version, evaluation=evaluation)


# grava em samples/ uma comparação avaliada e uma execução de exemplo
def _write_samples(root):
    samples = root / "samples"
    samples.mkdir()
    evidence_by_type = {
        ContentType.FLOW: FLOW_EVIDENCE,
        ContentType.STEPS: "Desligue o roteador da tomada e espere meio minuto.",
    }
    content_types = list(evidence_by_type)
    record = run_comparison(
        profile=get_profile("p01_marta"),
        ticket=get_ticket("t01_queda_noturna"),
        versions=["v1", "v2"],
        seeds=[1],
        content_types=content_types,
        provider=FakeProvider(),
        settings=load_settings({"LLM_MIN_INTERVAL_S": "0"}),
        cache=None,
        runs_dir=root / "runs_do_exemplo",
        show=lambda _: None,
        now=NOW,
    )
    evaluations = [
        ContentEvaluation(
            content_type=content_type,
            seed=1,
            outputs=[
                _human_output("A", "v2", 4, evidence),
                _human_output("B", "v1", 3, evidence),
            ],
        )
        for content_type, evidence in evidence_by_type.items()
    ]
    record = record.model_copy(
        update={
            "blind_map": [BlindItem(content_type=t, seed=1, a="v2", b="v1") for t in content_types],
            "evaluations": evaluations,
        }
    )
    (root / COMPARISON_SAMPLE).write_text(record.model_dump_json(), encoding="utf-8")
    first_run = sorted((root / "runs_do_exemplo").glob("*.json"))[0]
    copyfile(first_run, samples / "run_exemplo.json")


# isola pastas, provedor e configuração, para a página nunca tocar o projeto
@pytest.fixture
def web_env(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    comparisons = tmp_path / "comparisons"
    comparisons.mkdir()
    monkeypatch.setattr(compare_module, "COMPARISONS_DIR", comparisons)
    settings = load_settings({"LLM_PROVIDER": "fake", "LLM_MIN_INTERVAL_S": "0"})
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    _write_samples(tmp_path)
    return tmp_path


# caminho e conteúdo da comparação de exemplo, para cada teste alterar
@pytest.fixture
def comparison_sample(web_env):
    path = web_env / COMPARISON_SAMPLE
    return path, json.loads(path.read_text(encoding="utf-8"))


# abre a página e falha se ela levantar exceção
def _open():
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception, [error.value for error in app.exception]
    return app


# texto de todos os blocos de markdown da página
def _markdown(app):
    return " ".join(item.value for item in app.markdown)


# a página abre com as três abas e mantém a versão final selecionada nas opções avançadas
def test_page_opens_with_atendimento_and_advanced_versions(web_env):
    app = _open()
    assert [tab.label for tab in app.tabs] == ["Atendimento", "Comparação", "Exemplos"]
    advanced = next(section for section in app.expander if section.label == "Opções avançadas")
    versions = advanced.selectbox(key="version")
    assert versions.options == [
        "Inicial",
        "Adaptado ao perfil",
        "Regras de decisão",
        "Método de decisão",
    ]
    assert versions.value == DEFAULT_VERSION
    assert {"profile", "ticket", "type"} <= {box.key for box in app.tabs[0].selectbox}
    assert app.button(key="generate").label == "Gerar resposta"
    assert not app.text_input


# sem chave do Groq, a página usa a simulação local
def test_local_without_groq_key_defaults_to_working_simulation(web_env, monkeypatch):
    settings = load_settings({"LLM_PROVIDER": "groq"})
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    app = _open()

    assert any("Simulação offline" in item.value for item in app.tabs[0].caption)
    app.button(key="generate").click().run()

    assert not app.exception
    (record_path,) = (web_env / "runs").glob("*.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert (record["provider"], record["status"]) == ("fake", "ok")


# com chave configurada, a página gera com o Groq
def test_local_generation_uses_groq_when_configured(web_env, monkeypatch):
    settings = load_settings({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "chave-de-teste"})
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    providers = []

    # provedor falso que registra qual provedor foi pedido
    def capture_provider(settings, name):
        providers.append(name)
        return FakeProvider()

    monkeypatch.setattr(provider_factory, "make_provider", capture_provider)
    app = _open()
    assert any("Groq" in item.value for item in app.tabs[0].caption)
    app.button(key="generate").click().run()

    assert not app.exception
    assert providers == ["groq"]
    assert len(list((web_env / "runs").glob("*.json"))) == 1


# gerar mostra o fluxo e grava o registro para a versão escolhida
@pytest.mark.parametrize("version", ["v1", "v3", "v4"])
def test_generate_shows_the_flow_and_persists_the_record(web_env, version):
    app = _open()
    app.selectbox(key="type").set_value("flow")
    app.selectbox(key="version").set_value(version)
    app.button(key="generate").click().run()
    assert app.code[0].value.startswith("[P1] Todos os aparelhos")
    details = [item.value for item in app.tabs[0].caption]
    assert "Cache: nova geração" in details
    (path,) = (web_env / "runs").glob("*.json")
    assert f"JSON salvo em runs/{path.name}" in [item.value for item in app.success]
    assert json.loads(path.read_text(encoding="utf-8"))["prompt_version"] == version


# a segunda geração igual é acerto de cache e as duas ficam gravadas
def test_second_generation_is_a_cache_hit_and_both_records_are_saved(web_env):
    app = _open()
    app.button(key="generate").click().run()
    app.button(key="generate").click().run()
    assert any(item.value == "Cache: acerto" for item in app.tabs[0].caption)
    records = [
        json.loads(path.read_text(encoding="utf-8")) for path in (web_env / "runs").glob("*.json")
    ]
    assert sorted(record["cache_hit"] for record in records) == [False, True]


# falha ao gravar mantém a saída na tela e oferece o JSON para baixar
def test_save_failure_keeps_output_and_offers_json_download(web_env, monkeypatch):
    downloads = []

    # gravação que sempre falha, para simular disco sem permissão
    def fail_save(record):
        raise PermissionError("detalhe interno do disco")

    # registra o botão de download oferecido
    def capture_download(label, data, **kwargs):
        downloads.append((label, data, kwargs))

    monkeypatch.setattr(storage, "save_run", fail_save)
    monkeypatch.setattr(st, "download_button", capture_download)
    app = _open()
    app.button(key="generate").click().run()

    assert not app.exception
    assert [item.value for item in app.tabs[0].subheader] == ["Último resultado"]
    assert any("Não foi possível gravar" in item.value for item in app.error)
    assert all("detalhe interno" not in item.value for item in app.error)
    assert not (web_env / "runs").exists()
    matches = [
        (data, options) for label, data, options in downloads if label == "Baixar registro JSON"
    ]
    assert len(matches) == 1
    data, options = matches[0]
    record = json.loads(data)
    assert record["status"] == "ok" and record["output"]
    assert json.loads(app.tabs[0].code[0].value) == record["output"]
    assert record["ticket_id"] == "t01_queda_noturna"
    assert options["file_name"] == f"{record['run_id']}.json"
    assert options["mime"] == "application/json"


# o último resultado continua na tela depois de mudar os filtros
def test_last_generation_survives_a_rerun_without_using_the_new_filters(web_env):
    app = _open()
    app.selectbox(key="type").set_value("questions")
    app.button(key="generate").click().run()
    generated = app.code[0].value

    app.selectbox(key="type").set_value("steps").run()
    app.button(key="reveal").click().run()

    assert app.code[0].value == generated
    assert any("Perguntas de esclarecimento · Marta" in item.value for item in app.tabs[0].caption)
    assert len(list((web_env / "runs").glob("*.json"))) == 1


# texto livre gera com o id reservado do ticket
def test_free_text_ticket_generates_with_the_reserved_id(web_env):
    app = _open()
    assert "ticket_text" not in [box.key for box in app.text_area]
    app.selectbox(key="ticket").set_value("texto livre").run()
    app.text_area(key="ticket_text").set_value("A internet caiu agora").run()
    app.selectbox(key="type").set_value("steps")
    app.button(key="generate").click().run()
    saved = [json.loads(p.read_text(encoding="utf-8")) for p in (web_env / "runs").glob("*.json")]
    assert [item["ticket_id"] for item in saved] == ["texto_livre"]
    assert saved[0]["ticket"] == "A internet caiu agora"


# texto livre vazio mostra o erro conhecido
def test_empty_free_text_shows_a_known_error(web_env):
    app = _open()
    app.selectbox(key="ticket").set_value("texto livre").run()
    app.button(key="generate").click().run()
    assert any("O texto do ticket precisa ter" in item.value for item in app.error)
    assert not (web_env / "runs").exists()


# a comparação mostra o contexto e as notas antes de revelar as versões
def test_comparison_shows_context_and_scores_before_reveal(web_env):
    app = _open()
    assert app.selectbox(key="comparison").value == COMPARISON_SAMPLE.as_posix()
    assert [item.value for item in app.subheader] == ["Saída A", "Saída B"]
    assert "**Clareza · 4/5**" in _markdown(app) and "**Clareza · 3/5**" in _markdown(app)
    shown = [item.value for item in app.text]
    assert any("Marta, leigo, didático, telefone" in line for line in shown)
    assert any(get_ticket("t01_queda_noturna").text in line for line in shown)
    assert shown.count(f"Evidência: {FLOW_EVIDENCE}") == 6
    assert all(item.value not in {"Inicial", "Adaptado ao perfil"} for item in app.tabs[1].caption)


# a comparação usa o perfil gravado e ainda abre arquivos antigos
@pytest.mark.parametrize(
    ("profile_name", "customer_id", "shown"),
    [
        ("Marta histórica", "p01_marta", "Marta histórica, leigo, didático, telefone"),
        (None, "p01_marta", "Marta, leigo, didático, telefone"),
        (None, "perfil_antigo", "perfil_antigo (detalhes indisponíveis)"),
    ],
)
def test_comparison_prefers_its_profile_snapshot_and_handles_legacy(
    comparison_sample, profile_name, customer_id, shown
):
    path, data = comparison_sample
    data["customer_id"] = customer_id
    if profile_name is None:
        data.pop("profile")
    else:
        data["profile"]["name"] = profile_name
    path.write_text(json.dumps(data), encoding="utf-8")

    app = _open()

    assert any(shown in item.value for item in app.text)
    fallback = any("Perfil recuperado do catálogo atual." in item.value for item in app.caption)
    assert fallback is (profile_name is None and customer_id == "p01_marta")


# comparação sem notas mostra que a avaliação está pendente
def test_comparison_without_ratings_shows_a_pending_status(comparison_sample):
    path, data = comparison_sample
    data["evaluations"] = []
    path.write_text(json.dumps(data), encoding="utf-8")
    app = _open()

    assert [item.value for item in app.subheader] == ["Saída A", "Saída B"]
    assert any("Avaliação pendente" in item.value for item in app.tabs[1].caption)


# as notas registradas aparecem nas duas saídas
def test_comparison_displays_the_recorded_scores(comparison_sample):
    app = _open()
    captions = [item.value for item in app.tabs[1].caption]
    assert "Avaliação pendente para este par." not in captions
    assert "**Clareza · 4/5**" in _markdown(app) and "**Clareza · 3/5**" in _markdown(app)


# par incompleto mostra um aviso
def test_incomplete_comparison_pair_shows_a_warning(comparison_sample):
    path, data = comparison_sample
    data["runs"] = data["runs"][1:]
    path.write_text(json.dumps(data), encoding="utf-8")
    app = _open()

    assert any("Par incompleto para flow na seed 1." in item.value for item in app.warning)


# saída quebrada de um lado não impede mostrar o outro
def test_malformed_comparison_output_does_not_break_the_other_side(comparison_sample):
    path, data = comparison_sample
    run = next(
        run for run in data["runs"] if run["version"] == "v2" and run["content_type"] == "flow"
    )
    run["output"] = {"nodes": [], "edges": "lista inválida"}
    path.write_text(json.dumps(data), encoding="utf-8")
    app = _open()

    assert [item.value for item in app.subheader] == ["Saída A", "Saída B"]
    assert app.warning
    assert any(FLOW_EVIDENCE in code.value for code in app.tabs[1].code)
    assert app.button(key="reveal")


# o botão Revelar mostra a versão de cada saída
def test_reveal_shows_the_version_of_each_output(web_env):
    app = _open()
    reveal = app.button(key="reveal")
    assert reveal.label == "Revelar versões"
    reveal.click().run()
    shown = [item.value for item in app.tabs[1].caption]
    assert shown.index("Adaptado ao perfil") < shown.index("Inicial")


# trocar o tipo volta a esconder as versões
def test_changing_the_type_hides_the_versions_again(web_env):
    app = _open()
    app.button(key="reveal").click().run()
    app.selectbox(key="compare_type").set_value(ContentType.STEPS).run()
    assert [item.value for item in app.subheader] == ["Saída A", "Saída B"]
    assert app.button(key="reveal")


# exemplos e comparações abrem mesmo com configuração de geração inválida
def test_examples_and_comparisons_open_with_invalid_generation_settings(web_env, monkeypatch):
    monkeypatch.setattr(
        config, "load_settings", lambda: load_settings({"LLM_TEMPERATURE": "inválida"})
    )
    app = _open()

    assert app.tabs[0].error
    assert app.selectbox(key="sample").value == "run_exemplo.json"
    assert app.tabs[2].code
    assert app.selectbox(key="comparison").value == COMPARISON_SAMPLE.as_posix()
    assert [item.value for item in app.tabs[1].subheader] == ["Saída A", "Saída B"]


# exemplos do mesmo ticket e tipo permitem escolher os dois perfis
def test_examples_with_same_ticket_and_type_allow_selecting_both_profiles(web_env):
    samples = web_env / "samples"
    data = json.loads((samples / "run_exemplo.json").read_text(encoding="utf-8"))
    data["customer_id"] = "p03_carla"
    (samples / "run_outro_perfil.json").write_text(json.dumps(data), encoding="utf-8")
    app = _open()

    examples = app.selectbox(key="sample")
    assert len(set(examples.options)) == 2
    examples.set_value("run_outro_perfil.json").run()
    assert app.selectbox(key="sample").value == "run_outro_perfil.json"


# comparação local e de exemplo iguais aparecem as duas na lista
def test_matching_local_and_curated_comparisons_can_both_be_selected(web_env):
    local_comparison = compare_module.COMPARISONS_DIR / "comparacao_local.json"
    copyfile(web_env / COMPARISON_SAMPLE, local_comparison)
    app = _open()
    choices = app.selectbox(key="comparison")
    assert len(set(choices.options)) == 2
    stored_path = local_comparison.relative_to(web_env).as_posix()
    choices.set_value(stored_path).run()
    assert app.selectbox(key="comparison").value == stored_path


# uma comparação nova aparece junto com a publicada e continua cega até a revelação
@pytest.mark.parametrize("version", ["v3", "v4"])
def test_archive_opens_new_comparison_alongside_the_existing_sample(web_env, version):
    record = run_comparison(
        profile=get_profile("p01_marta"),
        ticket=get_ticket("t01_queda_noturna"),
        versions=["v2", version],
        seeds=[1],
        content_types=[ContentType.FLOW],
        provider=FakeProvider(),
        settings=load_settings({"LLM_MIN_INTERVAL_S": "0"}),
        cache=None,
        runs_dir=web_env / "runs_da_comparacao",
        show=lambda _: None,
        now=NOW,
    )
    path = compare_module.save_comparison(record)
    app = _open()
    assert len(app.selectbox(key="comparison").options) == 2
    app.selectbox(key="comparison").set_value(path.relative_to(web_env).as_posix()).run()
    assert not app.exception
    assert [item.value for item in app.tabs[1].subheader] == ["Saída A", "Saída B"]
    version_labels = {PROMPT_LABELS["v2"], PROMPT_LABELS[version]}
    assert version_labels.isdisjoint(item.value for item in app.tabs[1].caption)
    app.button(key="reveal").click().run()
    assert version_labels <= {item.value for item in app.tabs[1].caption}


# o arquivo ignora comparações de versões removidas e arquivos ilegíveis
def test_archive_skips_removed_versions_and_unreadable_files(comparison_sample):
    _, original = comparison_sample
    original["versions"] = ["v2", "v9"]
    obsolete = compare_module.COMPARISONS_DIR / "comparacao_antiga.json"
    obsolete.write_text(json.dumps(original), encoding="utf-8")
    (compare_module.COMPARISONS_DIR / "quebrada.json").write_text("{", encoding="utf-8")
    app = _open()

    choices = app.selectbox(key="comparison")
    assert len(choices.options) == 1
    assert choices.value == COMPARISON_SAMPLE.as_posix()
