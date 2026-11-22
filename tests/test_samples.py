import hashlib
import json
from pathlib import Path

import pytest

from support_assistant.cache import hash_messages
from support_assistant.evaluation.compare import comparison_pair
from support_assistant.evaluation.rubric import CRITERIA
from support_assistant.formatting import format_record_output
from support_assistant.models import (
    ComparisonRecord,
    ContentType,
    RunRecord,
    RunStatus,
    Ticket,
)
from support_assistant.prompts.engine import build_messages

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
COMPARISON = SAMPLES / "comparacao_v1_v3_p01_t01.json"
RESERVED = SAMPLES / "teste_reservado"
MODEL = "openai/gpt-oss-120b"
RUN_SAMPLES = (
    ("run_v3_diagnosis_p01_t01.json", ContentType.DIAGNOSIS, "p01_marta", "t01_queda_noturna"),
    ("run_v3_steps_p01_t01.json", ContentType.STEPS, "p01_marta", "t01_queda_noturna"),
    ("run_v3_questions_p01_t01.json", ContentType.QUESTIONS, "p01_marta", "t01_queda_noturna"),
    ("run_v3_flow_p01_t01.json", ContentType.FLOW, "p01_marta", "t01_queda_noturna"),
    ("run_v3_steps_p03_t01.json", ContentType.STEPS, "p03_carla", "t01_queda_noturna"),
    ("run_v3_diagnosis_p03_t02.json", ContentType.DIAGNOSIS, "p03_carla", "t02_wifi_lento"),
    ("run_v3_flow_p02_t03.json", ContentType.FLOW, "p02_rafael", "t03_codigo_sms"),
)


# lê um exemplo de execução e valida contra o contrato
def _run(name: str) -> RunRecord:
    return RunRecord.model_validate_json((SAMPLES / name).read_text(encoding="utf-8"))


# lê a comparação publicada e valida contra o contrato
def _comparison() -> ComparisonRecord:
    return ComparisonRecord.model_validate_json(COMPARISON.read_text(encoding="utf-8"))


# os exemplos de execução são do cenário real e do modelo oficial
@pytest.mark.parametrize(("name", "kind", "profile", "ticket"), RUN_SAMPLES)
def test_run_samples_preserve_the_real_scenario(name, kind, profile, ticket):
    record = _run(name)
    assert record.status is RunStatus.OK
    assert (record.prompt_version, record.customer_id) == ("v3", profile)
    assert (record.content_type, record.ticket_id) == (kind, ticket)
    assert (record.provider, record.model) == ("groq", MODEL)
    assert not record.cache_hit and record.usage.output_tokens > 0


# nenhum exemplo guarda chave de API
def test_samples_do_not_contain_api_keys():
    for path in SAMPLES.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "gsk_" not in text and "GROQ_API_KEY" not in text, path.name


# a comparação de exemplo mantém o experimento: versões, seeds, tipos e configuração
def test_comparison_pairs_preserve_the_experiment():
    record = _comparison()
    assert record.versions == ["v1", "v3"]
    assert record.profile is not None and record.profile.id == record.customer_id
    assert (record.locked_config.provider, record.locked_config.model) == ("groq", MODEL)
    assert set(record.content_types) == set(ContentType)
    assert len(record.runs) == 24
    assert {(run.version, run.seed, run.content_type) for run in record.runs} == {
        (version, seed, kind)
        for version in record.versions
        for seed in record.seeds
        for kind in record.content_types
    }

    ticket = Ticket(id=record.ticket_id, text=record.ticket)
    for run in record.runs:
        messages = build_messages(run.version, run.content_type, record.profile, ticket)
        assert run.prompt_hash == hash_messages(messages)


# toda nota registrada tem um trecho conferível na saída avaliada
def test_recorded_scores_have_evidence_in_the_outputs():
    record = _comparison()
    assert record.profile is not None
    ticket = Ticket(id=record.ticket_id, text=record.ticket)
    for entry in record.evaluations:
        pair = comparison_pair(record, entry.seed, entry.content_type)
        for output in entry.outputs:
            run = pair[0 if output.position == "A" else 1]
            assert output.version == run.version
            messages = build_messages(run.version, run.content_type, record.profile, ticket)
            assert run.prompt_hash == hash_messages(messages)
            shown = " ".join(format_record_output(run).split())
            for key, _ in CRITERIA:
                score = getattr(output.evaluation, key)
                assert " ".join(score.evidence.split()) in shown


# o mesmo ticket gera passos diferentes para canais diferentes
def test_same_ticket_examples_show_distinct_channels():
    marta = _run("run_v3_steps_p01_t01.json")
    carla = _run("run_v3_steps_p03_t01.json")
    assert carla.ticket == marta.ticket and carla.ticket_id == marta.ticket_id
    assert (carla.provider, carla.model, carla.schema_hash) == (
        marta.provider,
        marta.model,
        marta.schema_hash,
    )
    assert marta.output and marta.output["subject"] is None
    assert carla.output and carla.output["subject"]


# o teste reservado cobre os tickets de teste com os prompts congelados
def test_reserved_test_keeps_frozen_prompts_and_every_ticket():
    result = json.loads((RESERVED / "resultado.json").read_text(encoding="utf-8"))
    versions = Path(__file__).resolve().parent.parent / "support_assistant/prompts/versions"
    for version, expected in result["hashes_dos_prompts"].items():
        assert hashlib.sha256((versions / f"{version}.toml").read_bytes()).hexdigest() == expected
    records = [
        ComparisonRecord.model_validate_json((RESERVED / name).read_text(encoding="utf-8"))
        for name in result["arquivos"]
    ]
    assert [record.ticket_id[:3] for record in records] == ["t04", "t05", "t06", "t07", "t08"]
    assert all(record.versions == ["v3", "v4"] and len(record.runs) == 8 for record in records)
    assert result["validas_com_validador_corrigido"] == {"v3": 20, "v4": 18}
