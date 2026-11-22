import logging
import re
from datetime import UTC, datetime

import pytest

from support_assistant import storage
from support_assistant.data_loader import get_profile, get_ticket
from support_assistant.generators.core import generate
from support_assistant.models import SCHEMA_VERSION, ContentType, RunStatus
from support_assistant.providers.fake import FakeProvider
from support_assistant.storage import (
    build_run_record,
    list_runs,
    load_run,
    new_run_id,
    save_run,
)

NOW = datetime(2026, 11, 20, 15, 30, 12, tzinfo=UTC)


# gera com o fake e devolve o resultado com o perfil e o ticket usados
def _result(
    params, content_type=ContentType.STEPS, profile="p01_marta", ticket="t01_queda_noturna"
):
    profile_obj, ticket_obj = get_profile(profile), get_ticket(ticket)
    result = generate(
        content_type=content_type,
        profile=profile_obj,
        ticket=ticket_obj,
        version="v1",
        provider=FakeProvider(),
        model="fake-model",
        params=params,
        cache=None,
    )
    return result, profile_obj, ticket_obj


# grava um registro novo com um instante dado
def _save(params, directory, now, **kwargs):
    result, profile, ticket = _result(params, **kwargs)
    record = build_run_record(result, profile, ticket, now=now)
    return save_run(record, directory), record


# o identificador segue o formato de data em UTC mais 6 hexadecimais
def test_new_run_id_format():
    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{6}", new_run_id())
    assert new_run_id(NOW).startswith("20261120T153012Z_")


# o instante com outro fuso é convertido para UTC no identificador
def test_new_run_id_converts_to_utc():
    from datetime import timedelta, timezone

    local = datetime(2026, 11, 20, 12, 30, 12, tzinfo=timezone(timedelta(hours=-3)))
    assert new_run_id(local).startswith("20261120T153012Z_")


# o registro copia os campos do resultado e identifica perfil e ticket
def test_build_run_record_copies_the_result(params):
    result, profile, ticket = _result(params)
    record = build_run_record(result, profile, ticket, now=NOW)
    assert record.schema_version == SCHEMA_VERSION
    assert record.run_id.startswith("20261120T153012Z_")
    assert record.timestamp == NOW
    assert (record.customer_id, record.ticket_id, record.ticket) == (
        "p01_marta",
        ticket.id,
        ticket.text,
    )
    assert record.evaluation is None
    for field in (
        "content_type",
        "prompt_version",
        "prompt_hash",
        "schema_hash",
        "provider",
        "model",
        "params",
        "cache_key",
        "cache_hit",
        "usage",
        "latency_ms",
        "finish_reason",
        "repaired",
        "status",
        "output",
        "raw_output",
    ):
        assert getattr(record, field) == getattr(result, field), field


# um resultado inválido vira registro com o texto bruto
def test_invalid_output_result_becomes_a_valid_record(params):
    profile, ticket = get_profile("p01_marta"), get_ticket("t01_queda_noturna")
    result = generate(
        content_type=ContentType.STEPS,
        profile=profile,
        ticket=ticket,
        version="v1",
        provider=FakeProvider(responses=["ruim", "ruim"]),
        model="fake-model",
        params=params,
        cache=None,
    )
    record = build_run_record(result, profile, ticket)
    assert record.status is RunStatus.INVALID_OUTPUT
    assert record.raw_output == "ruim"


# salvar e carregar devolve o mesmo registro, num arquivo com o nome do run_id e sem sobras
def test_save_and_load_roundtrip(params, tmp_path):
    path, record = _save(params, tmp_path, NOW)
    assert path == tmp_path / f"{record.run_id}.json"
    assert load_run(path) == record
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


# a pasta padrão é resolvida na hora do uso
def test_default_runs_directory_is_resolved_at_call_time(params, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "RUNS_DIR", tmp_path / "runs")
    result, profile, ticket = _result(params)
    path = save_run(build_run_record(result, profile, ticket))
    assert path.parent == tmp_path / "runs"
    assert len(list_runs()) == 1


# list_runs filtra por cada campo
def test_list_runs_filters(params, tmp_path):
    _save(params, tmp_path, NOW, content_type=ContentType.STEPS)
    _save(params, tmp_path, NOW, content_type=ContentType.FLOW, profile="p02_rafael")
    _, other = _save(params, tmp_path, NOW, ticket="t03_codigo_sms")
    assert len(list_runs(tmp_path)) == 3
    assert len(list_runs(tmp_path, customer_id="p02_rafael")) == 1
    assert len(list_runs(tmp_path, ticket_id="t03_codigo_sms")) == 1
    assert len(list_runs(tmp_path, content_type=ContentType.FLOW)) == 1
    assert len(list_runs(tmp_path, prompt_version="v1")) == 3
    assert list_runs(tmp_path, prompt_version="v2") == []
    assert len(list_runs(tmp_path, cache_key=other.cache_key)) == 1
    assert list_runs(tmp_path, customer_id="p01_marta", content_type=ContentType.FLOW) == []


# a lista sai em ordem crescente de timestamp
def test_list_runs_is_sorted_by_timestamp(params, tmp_path):
    later = datetime(2026, 11, 21, tzinfo=UTC)
    earlier = datetime(2026, 11, 19, tzinfo=UTC)
    _, first = _save(params, tmp_path, later)
    _, second = _save(params, tmp_path, earlier)
    assert [r.run_id for _, r in list_runs(tmp_path)] == [second.run_id, first.run_id]


# pasta inexistente lista nada
def test_list_runs_without_directory_is_empty(tmp_path):
    assert list_runs(tmp_path / "nao_existe") == []


# arquivo corrompido ou fora do contrato é pulado com aviso
def test_corrupted_run_files_are_skipped_with_a_warning(params, tmp_path, caplog):
    _save(params, tmp_path, NOW)
    (tmp_path / "quebrado.json").write_text("{ não é json", encoding="utf-8")
    (tmp_path / "fora.json").write_text('{"schema_version": "1.0"}', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert len(list_runs(tmp_path)) == 1
    assert "runs: arquivo ignorado em quebrado.json" in caplog.text
    assert "runs: arquivo ignorado em fora.json" in caplog.text


# o filtro por chave de cache acha as execuções equivalentes
def test_list_runs_finds_equivalent_runs_by_cache_key(params, tmp_path):
    _, record = _save(params, tmp_path, NOW)
    found = list_runs(tmp_path, cache_key=record.cache_key)
    assert [r.run_id for _, r in found] == [record.run_id]
    assert list_runs(tmp_path, cache_key="sha256:" + "0" * 64) == []


# o carregamento de um registro inválido levanta erro em vez de devolver lixo
def test_load_run_rejects_invalid_content(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_run(path)
