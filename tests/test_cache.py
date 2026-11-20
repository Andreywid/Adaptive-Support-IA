import json
import logging
import re

import pytest

from support_assistant import cache as cache_module
from support_assistant.cache import (
    CacheEntry,
    Fingerprint,
    ResponseCache,
    build_fingerprint,
    canonical_json,
    compute_cache_key,
    hash_messages,
    hash_schema,
    sha256_text,
)
from support_assistant.models import (
    ContentType,
    GenerationParams,
    Message,
    ReasoningEffort,
    RunRecord,
    output_json_schema,
)
from support_assistant.providers.fake import CANNED_OUTPUTS, FakeProvider

HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

# forma canônica do fingerprint de referência, exatamente como entra no hash
GOLDEN_TEXT = (
    '{"model":"openai/gpt-oss-120b","params":{"include_reasoning":false,"max_tokens":2048,'
    '"reasoning_effort":"low","seed":1,"temperature":1,"top_p":1},'
    '"prompt_hash":"sha256:' + "1" * 64 + '","prompt_version":"v2","provider":"groq",'
    '"schema_hash":"sha256:' + "2" * 64 + '"}'
)
GOLDEN_KEY = "sha256:e9a200b2e606504eab93f3dcfbcf6c12ab5f12a15bb5460a0b7b22fc3d2ea99a"


# reconstrói a impressão digital com os campos gravados no registro
def _fingerprint(record):
    return Fingerprint(
        provider=record.provider,
        model=record.model,
        params=record.params,
        prompt_version=record.prompt_version,
        prompt_hash=record.prompt_hash,
        schema_hash=record.schema_hash,
    )


# calcula a chave de cache de uma requisição com provedor e versão de prompt
def _key(request, provider="fake", version="v2") -> str:
    return compute_cache_key(build_fingerprint(provider, request, version))


# registro válido cujos campos de impressão digital vêm da requisição dada
def _record_from_request(request, make_record_data) -> RunRecord:
    fingerprint = build_fingerprint("fake", request, "v2")
    return RunRecord.model_validate(
        make_record_data(
            provider=fingerprint.provider,
            model=fingerprint.model,
            params=fingerprint.params.model_dump(mode="json"),
            prompt_version=fingerprint.prompt_version,
            prompt_hash=fingerprint.prompt_hash,
            schema_hash=fingerprint.schema_hash,
            cache_key=compute_cache_key(fingerprint),
        )
    )


# a ordem das chaves não muda o texto canônico, em nenhum nível
def test_canonical_json_ignores_key_order():
    first = canonical_json({"b": 1, "a": {"y": 2, "x": 3}})
    second = canonical_json({"a": {"x": 3, "y": 2}, "b": 1})
    assert first == second == '{"a":{"x":3,"y":2},"b":1}'


# o texto canônico é compacto e mantém acentos em UTF-8
def test_canonical_json_is_compact_and_keeps_utf8():
    text = canonical_json({"texto": "didático e ótimo", "lista": [1, 2]})
    assert text == '{"lista":[1,2],"texto":"didático e ótimo"}'


# float inteiro vira inteiro, mas decimais e tipos diferentes continuam distintos
def test_canonical_json_normalizes_integral_floats_only():
    assert canonical_json({"t": 1.0}) == canonical_json({"t": 1}) == '{"t":1}'
    assert canonical_json([0.0, -0.0, 0.7]) == "[0,0,0.7]"
    assert canonical_json({"a": True}) != canonical_json({"a": 1})
    assert canonical_json(None) == "null"


# NaN e infinito não existem em JSON e são recusados
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value):
    with pytest.raises(ValueError):
        canonical_json({"t": value})


# exemplo da RFC 8785 com números, escapes de controle e ordem das chaves
def test_canonical_json_matches_the_rfc_8785_example():
    data = {
        "numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 0.000000000000000000000000001],
        "string": '€$\x0f\nA\'B"\\\\"/',
        "literals": [None, True, False],
    }
    expected = (
        r"""{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],"""
        r""""string":"€$\u000f\nA'B\"\\\\\"/"}"""
    )
    assert canonical_json(data) == expected


# o hash do texto vazio é o SHA-256 conhecido, com o prefixo do projeto
def test_sha256_text_known_value():
    expected = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert sha256_text("") == expected


# o hash das mensagens muda com conteúdo, papel e ordem
def test_hash_messages_is_sensitive_to_content_role_and_order():
    system = Message(role="system", content="regras")
    user = Message(role="user", content="ticket")
    base = hash_messages([system, user])
    assert HASH_PATTERN.match(base)
    assert hash_messages([system, user]) == base
    assert hash_messages([user, system]) != base
    assert hash_messages([system, Message(role="user", content="ticket 2")]) != base
    assert hash_messages([system, Message(role="assistant", content="ticket")]) != base


# o hash do schema ignora a ordem das chaves e muda com o conteúdo
def test_hash_schema_ignores_key_order_but_not_content():
    assert hash_schema({"a": 1, "b": 2}) == hash_schema({"b": 2, "a": 1})
    steps = hash_schema(output_json_schema(ContentType.STEPS))
    assert steps != hash_schema(output_json_schema(ContentType.FLOW))


# campos da impressão digital
def test_fingerprint_fields():
    assert list(Fingerprint.model_fields) == [
        "provider",
        "model",
        "params",
        "prompt_version",
        "prompt_hash",
        "schema_hash",
    ]


# requisições equivalentes montadas separadamente geram a mesma chave
def test_equivalent_requests_have_the_same_key(make_request):
    assert _key(make_request()) == _key(make_request())


# a chave tem o formato sha256 com 64 hexadecimais
def test_key_format(make_request):
    assert HASH_PATTERN.match(_key(make_request()))


# trocar o modelo muda a chave
def test_changing_the_model_changes_the_key(make_request):
    big = make_request(model="openai/gpt-oss-120b")
    small = make_request(model="openai/gpt-oss-20b")
    assert _key(big) != _key(small)


# trocar qualquer parâmetro de geração muda a chave
@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": 0.0},
        {"top_p": 0.9},
        {"seed": 2},
        {"max_tokens": 4096},
        {"reasoning_effort": ReasoningEffort.HIGH},
        {"include_reasoning": True},
    ],
)
def test_changing_any_param_changes_the_key(changes, make_request, params):
    changed = params.model_copy(update=changes)
    assert _key(make_request(params=changed)) != _key(make_request())


# trocar a versão do prompt muda a chave
def test_changing_the_prompt_version_changes_the_key(make_request):
    request = make_request()
    assert _key(request, version="v1") != _key(request, version="v2")


# trocar o provedor muda a chave
def test_changing_the_provider_changes_the_key(make_request):
    request = make_request()
    assert _key(request, provider="groq") != _key(request, provider="fake")


# trocar o texto do prompt muda a chave
def test_changing_the_prompt_text_changes_the_key(make_request):
    other = make_request(messages=[Message(role="user", content="outro ticket")])
    assert _key(other) != _key(make_request())


# trocar o schema de saída muda a chave
def test_changing_the_output_schema_changes_the_key(make_request):
    other = make_request(response_schema=output_json_schema(ContentType.STEPS))
    assert _key(other) != _key(make_request())


# a grafia do número não importa, 1 e 1.0 são a mesma temperatura
def test_integer_and_float_temperature_give_the_same_key(make_request, params):
    base = params.model_dump(mode="json")
    as_int = GenerationParams.model_validate({**base, "temperature": 1})
    as_text = GenerationParams.model_validate({**base, "temperature": "1.0"})
    assert _key(make_request(params=as_int)) == _key(make_request(params=as_text))


# chave de referência fixa: se mudar, as chaves já gravadas deixam de bater
def test_golden_key_and_canonical_form_are_frozen():
    fingerprint = Fingerprint(
        provider="groq",
        model="openai/gpt-oss-120b",
        params=GenerationParams(
            temperature=1.0,
            top_p=1.0,
            seed=1,
            max_tokens=2048,
            reasoning_effort=ReasoningEffort.LOW,
            include_reasoning=False,
        ),
        prompt_version="v2",
        prompt_hash="sha256:" + "1" * 64,
        schema_hash="sha256:" + "2" * 64,
    )
    assert canonical_json(fingerprint.model_dump(mode="json")) == GOLDEN_TEXT
    assert compute_cache_key(fingerprint) == GOLDEN_KEY


# a chave gravada no registro é recalculada só com os campos do próprio registro
def test_recorded_key_is_recomputable_from_the_record(make_request, make_record_data):
    record = _record_from_request(make_request(), make_record_data)
    assert compute_cache_key(_fingerprint(record)) == record.cache_key


# a chave continua batendo depois de gravar e ler o registro em JSON
def test_recorded_key_survives_a_json_roundtrip(make_request, make_record_data):
    record = _record_from_request(make_request(), make_record_data)
    reloaded = RunRecord.model_validate_json(record.model_dump_json())
    assert compute_cache_key(_fingerprint(reloaded)) == record.cache_key


# adulterar um campo do registro faz a chave recalculada divergir da gravada
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "outro-modelo"),
        ("provider", "groq"),
        ("prompt_version", "v99"),
        ("prompt_hash", "sha256:" + "f" * 64),
        ("schema_hash", "sha256:" + "f" * 64),
    ],
)
def test_tampering_with_a_record_field_is_detected(field, value, make_request, make_record_data):
    record = _record_from_request(make_request(), make_record_data)
    tampered = record.model_copy(update={field: value})
    assert compute_cache_key(_fingerprint(tampered)) != tampered.cache_key


# a chave evita a segunda chamada equivalente e exige chamada nova ao trocar o modelo
def test_key_lets_a_memo_skip_equivalent_calls(make_request):
    fake = FakeProvider()
    memo = {}

    # responde pelo memo quando a chave já existe, senão chama o provedor
    def generate(request):
        key = _key(request)
        if key not in memo:
            memo[key] = fake.complete(request)
        return memo[key]

    generate(make_request())
    generate(make_request())
    assert len(fake.calls) == 1
    generate(make_request(model="openai/gpt-oss-120b"))
    assert len(fake.calls) == 2


# impressão digital, chave e saída fixa de um diagnóstico para os testes do cache em disco
def _parts(make_request):
    fingerprint = build_fingerprint("fake", make_request(), "v1")
    return fingerprint, compute_cache_key(fingerprint), CANNED_OUTPUTS[ContentType.DIAGNOSIS]


# regrava a entrada de uma chave com campos alterados
def _tamper(path, **changes):
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**data, **changes}), encoding="utf-8")


# put seguido de get devolve a mesma saída, e o arquivo é uma entrada válida
def test_put_then_get_roundtrip(tmp_path, make_request):
    cache = ResponseCache(tmp_path)
    fingerprint, key, output = _parts(make_request)
    path = cache.put(key, fingerprint, ContentType.DIAGNOSIS, output)
    assert cache.get(key, ContentType.DIAGNOSIS) == output.model_dump(mode="json")
    entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
    assert entry.cache_key == key
    assert entry.fingerprint == fingerprint
    assert [item.name for item in tmp_path.iterdir()] == [path.name]


# chave desconhecida devolve None, sem aviso
def test_unknown_key_returns_none_without_warning(tmp_path, make_request, caplog):
    _, key, _ = _parts(make_request)
    with caplog.at_level(logging.WARNING):
        assert ResponseCache(tmp_path).get(key, ContentType.DIAGNOSIS) is None
    assert caplog.records == []


# arquivo corrompido devolve None e registra aviso
def test_corrupted_file_is_a_miss_with_a_warning(tmp_path, make_request, caplog):
    cache = ResponseCache(tmp_path)
    fingerprint, key, output = _parts(make_request)
    path = cache.put(key, fingerprint, ContentType.DIAGNOSIS, output)
    path.write_text("{ isto não é json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert cache.get(key, ContentType.DIAGNOSIS) is None
    assert f"cache: entrada ignorada em {path.name}" in caplog.text


# bytes que não são UTF-8 também viram falha de leitura tratada
def test_undecodable_file_is_a_miss(tmp_path, make_request):
    cache = ResponseCache(tmp_path)
    fingerprint, key, output = _parts(make_request)
    cache.put(key, fingerprint, ContentType.DIAGNOSIS, output).write_bytes(b"\xff\xfe\x00")
    assert cache.get(key, ContentType.DIAGNOSIS) is None


# entrada com outra cache_key, com saída inválida ou com outro tipo devolve None
@pytest.mark.parametrize(
    "changes",
    [
        {"cache_key": "sha256:" + "0" * 64},
        {"output": {"campo": "sem sentido"}},
        {"content_type": "steps"},
        {"schema_version": "9.9"},
    ],
)
def test_inconsistent_entries_are_misses(changes, tmp_path, make_request, caplog):
    cache = ResponseCache(tmp_path)
    fingerprint, key, output = _parts(make_request)
    _tamper(cache.put(key, fingerprint, ContentType.DIAGNOSIS, output), **changes)
    with caplog.at_level(logging.WARNING):
        assert cache.get(key, ContentType.DIAGNOSIS) is None
    assert "cache: entrada ignorada" in caplog.text


# pedir a entrada com outro tipo de conteúdo também é um erro de cache
def test_get_with_another_content_type_is_a_miss(tmp_path, make_request):
    cache = ResponseCache(tmp_path)
    fingerprint, key, output = _parts(make_request)
    cache.put(key, fingerprint, ContentType.DIAGNOSIS, output)
    assert cache.get(key, ContentType.STEPS) is None


# o caminho usa os 64 hexadecimais da chave, sem o prefixo
def test_path_for_uses_the_hex_digest(tmp_path, make_request):
    _, key, _ = _parts(make_request)
    assert ResponseCache(tmp_path).path_for(key) == tmp_path / f"{key.removeprefix('sha256:')}.json"


# chave fora do formato é recusada, para nunca escapar da pasta do cache
@pytest.mark.parametrize("key", ["../fora", "sha256:abc", "sha256:" + "G" * 64, "md5:" + "0" * 64])
def test_path_for_rejects_malformed_keys(key, tmp_path):
    with pytest.raises(ValueError):
        ResponseCache(tmp_path).path_for(key)


# o cache só aceita saída validada do tipo certo e chave coerente com a impressão digital
def test_put_rejects_wrong_output_type_and_wrong_key(tmp_path, make_request):
    cache = ResponseCache(tmp_path)
    fingerprint, key, output = _parts(make_request)
    with pytest.raises(TypeError):
        cache.put(key, fingerprint, ContentType.DIAGNOSIS, CANNED_OUTPUTS[ContentType.STEPS])
    with pytest.raises(ValueError, match="não corresponde"):
        cache.put("sha256:" + "1" * 64, fingerprint, ContentType.DIAGNOSIS, output)
    assert list(tmp_path.iterdir()) == []


# a pasta padrão é resolvida na hora do uso, para os testes poderem trocá-la
def test_default_directory_is_resolved_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path)
    assert ResponseCache().directory == tmp_path
