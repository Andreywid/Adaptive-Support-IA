import json

import pytest

from support_assistant.cache import ResponseCache, build_fingerprint, compute_cache_key
from support_assistant.data_loader import get_profile, get_ticket, ticket_from_text
from support_assistant.generators.core import REPAIR_MESSAGE, UNSAFE_OUTPUT, generate
from support_assistant.generators.diagnosis import generate_diagnosis
from support_assistant.generators.flow import generate_flow
from support_assistant.generators.questions import generate_questions
from support_assistant.generators.steps import generate_steps
from support_assistant.models import OUTPUT_MODELS, ContentType, RunStatus
from support_assistant.prompts.engine import PromptError, build_messages
from support_assistant.providers.base import ProviderError, ProviderRequest
from support_assistant.providers.fake import CANNED_OUTPUTS, FakeProvider

GENERATORS = {
    ContentType.DIAGNOSIS: generate_diagnosis,
    ContentType.STEPS: generate_steps,
    ContentType.QUESTIONS: generate_questions,
    ContentType.FLOW: generate_flow,
}


# argumentos comuns de uma geração com o perfil p01 e o ticket t01
def _kwargs(provider, params, cache, **overrides):
    return {
        "profile": get_profile("p01_marta"),
        "ticket": get_ticket("t01_queda_noturna"),
        "version": "v1",
        "provider": provider,
        "model": "fake-model",
        "params": params,
        "cache": cache,
        **overrides,
    }


# JSON válido da saída fixa de um tipo
def _valid_json(content_type):
    return CANNED_OUTPUTS[content_type].model_dump_json()


# os 4 geradores devolvem status ok com saída válida para o contrato do tipo
@pytest.mark.parametrize("content_type", list(ContentType))
def test_each_generator_returns_a_valid_output(content_type, params, tmp_path):
    result = GENERATORS[content_type](**_kwargs(FakeProvider(), params, ResponseCache(tmp_path)))
    assert result.status is RunStatus.OK
    assert result.content_type is content_type
    assert result.cache_hit is False
    assert result.repaired is False
    assert result.raw_output is None
    OUTPUT_MODELS[content_type].model_validate(result.output)


# o resultado traz a impressão digital da requisição e os campos do provedor
def test_result_carries_the_request_fingerprint(params, tmp_path):
    provider = FakeProvider()
    result = generate(
        content_type=ContentType.STEPS, **_kwargs(provider, params, ResponseCache(tmp_path))
    )
    request = provider.calls[0]
    fingerprint = build_fingerprint("fake", request, "v1")
    assert result.cache_key == compute_cache_key(fingerprint)
    assert result.prompt_hash == fingerprint.prompt_hash
    assert result.schema_hash == fingerprint.schema_hash
    assert (result.provider, result.model, result.prompt_version) == ("fake", "fake-model", "v1")
    assert result.params == params
    assert result.finish_reason == "stop"
    assert request.schema_name == "steps"


# a segunda geração equivalente é acerto de cache e o provedor é chamado uma vez só
def test_second_equivalent_generation_hits_the_cache(params, tmp_path):
    provider, cache = FakeProvider(), ResponseCache(tmp_path)
    first = generate_diagnosis(**_kwargs(provider, params, cache))
    second = generate_diagnosis(**_kwargs(provider, params, cache))
    assert (first.cache_hit, second.cache_hit) == (False, True)
    assert len(provider.calls) == 1
    assert second.output == first.output
    assert second.cache_key == first.cache_key
    assert second.usage.input_tokens == 0
    assert second.usage.output_tokens == 0
    assert second.latency_ms == 0
    assert second.finish_reason is None
    assert second.status is RunStatus.OK


# refresh chama o provedor de novo e regrava o cache
def test_refresh_calls_the_provider_and_rewrites_the_cache(params, tmp_path):
    provider, cache = FakeProvider(), ResponseCache(tmp_path)
    generate_flow(**_kwargs(provider, params, cache))
    refreshed = generate_flow(**_kwargs(provider, params, cache, refresh=True))
    assert refreshed.cache_hit is False
    assert len(provider.calls) == 2
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert generate_flow(**_kwargs(provider, params, cache)).cache_hit is True


# sem cache o provedor é chamado toda vez e nada é gravado
def test_no_cache_calls_every_time_and_writes_nothing(params, tmp_path):
    provider = FakeProvider()
    for _ in range(2):
        assert generate_steps(**_kwargs(provider, params, None)).cache_hit is False
    assert len(provider.calls) == 2
    assert list(tmp_path.iterdir()) == []


# saída inválida seguida de saída válida faz um reparo com as mensagens certas e soma o consumo
def test_one_repair_attempt_fixes_the_output(params, tmp_path):
    provider = FakeProvider(responses=["{inválido", _valid_json(ContentType.STEPS)])
    cache = ResponseCache(tmp_path)
    result = generate_steps(**_kwargs(provider, params, cache))
    assert (result.repaired, result.status) == (True, RunStatus.OK)
    assert len(provider.calls) == 2
    original = provider.calls[0].messages
    repair = provider.calls[1].messages
    assert repair[: len(original)] == original
    assert repair[-2].role == "assistant"
    assert repair[-2].content == "{inválido"
    assert repair[-1].role == "user"
    assert repair[-1].content.startswith("A resposta anterior não passou na validação:\n")
    assert repair[-1].content.endswith("\nResponda de novo apenas com o JSON corrigido.")
    assert provider.calls[1].response_schema == provider.calls[0].response_schema
    assert provider.calls[1].params == provider.calls[0].params
    first_usage = FakeProvider(responses=["{inválido"]).complete(provider.calls[0]).usage
    second_usage = (
        FakeProvider(responses=[_valid_json(ContentType.STEPS)]).complete(provider.calls[1]).usage
    )
    assert result.usage.input_tokens == first_usage.input_tokens + second_usage.input_tokens
    assert result.usage.output_tokens == first_usage.output_tokens + second_usage.output_tokens
    assert generate_steps(**_kwargs(FakeProvider(), params, cache)).cache_hit is True


# duas saídas inválidas dão invalid_output com o texto da segunda resposta e nada no cache
def test_two_invalid_outputs_become_invalid_output(params, tmp_path):
    provider = FakeProvider(responses=["ruim", "ruim"])
    cache = ResponseCache(tmp_path)
    result = generate_diagnosis(**_kwargs(provider, params, cache))
    assert result.status is RunStatus.INVALID_OUTPUT
    assert result.raw_output == "ruim"
    assert result.output is None
    assert result.repaired is False
    assert len(provider.calls) == 2
    assert not list(tmp_path.glob("*.json"))


# a segunda resposta inválida é a que vai para raw_output
def test_raw_output_is_the_second_response(params, tmp_path):
    provider = FakeProvider(responses=["primeira ruim", "segunda ruim"])
    result = generate_diagnosis(**_kwargs(provider, params, ResponseCache(tmp_path)))
    assert result.raw_output == "segunda ruim"


# o pedido de reparo lista no máximo 20 erros
def test_repair_message_lists_at_most_20_errors(params, tmp_path):
    many_errors = '{"questions": [' + ",".join(["{}"] * 10) + "]}"
    provider = FakeProvider(responses=[many_errors, _valid_json(ContentType.QUESTIONS)])
    generate_questions(**_kwargs(provider, params, ResponseCache(tmp_path)))
    prefix = REPAIR_MESSAGE.split("{errors}")[0]
    body = provider.calls[1].messages[-1].content.removeprefix(prefix)
    error_lines = body.split("\nResponda de novo")[0].splitlines()
    assert len(error_lines) == 20
    assert error_lines[0].startswith("questions.0.")


# exceção do provedor sobe como ProviderError, no primeiro ou no segundo pedido
@pytest.mark.parametrize("queue", [[ProviderError("falha")], ["ruim", ProviderError("falha")]])
def test_provider_errors_propagate(queue, params, tmp_path):
    with pytest.raises(ProviderError, match="falha"):
        generate_diagnosis(
            **_kwargs(FakeProvider(responses=queue), params, ResponseCache(tmp_path))
        )


# seed diferente gera outra cache_key
def test_different_seed_changes_the_cache_key(params, tmp_path):
    cache = ResponseCache(tmp_path)
    first = generate_steps(**_kwargs(FakeProvider(), params, cache))
    other = params.model_copy(update={"seed": params.seed + 1})
    second = generate_steps(**_kwargs(FakeProvider(), other, cache))
    assert first.cache_key != second.cache_key
    assert second.cache_hit is False


# versão ou perfil diferentes também geram outra cache_key
def test_different_version_or_profile_changes_the_cache_key(params, tmp_path):
    cache = ResponseCache(tmp_path)
    base = generate_steps(**_kwargs(FakeProvider(), params, cache))
    other_profile = generate_steps(
        **_kwargs(FakeProvider(), params, cache, profile=get_profile("p03_carla"))
    )
    assert base.cache_key != other_profile.cache_key


# versão inexistente sobe PromptError antes de qualquer chamada
def test_unknown_version_fails_before_calling_the_provider(params, tmp_path):
    provider = FakeProvider()
    with pytest.raises(PromptError, match="não existe"):
        generate_steps(**_kwargs(provider, params, ResponseCache(tmp_path), version="v99"))
    assert provider.calls == []


# o provedor recebe as mensagens do motor de prompt, o schema do tipo e o modelo pedido
def test_provider_receives_the_built_request(params, tmp_path):
    provider = FakeProvider()
    generate_questions(**_kwargs(provider, params, None))
    request = provider.calls[0]
    assert isinstance(request, ProviderRequest)
    assert request.messages == build_messages(
        "v1", ContentType.QUESTIONS, get_profile("p01_marta"), get_ticket("t01_queda_noturna")
    )
    assert request.model == "fake-model"


# resposta válida no schema, mas incoerente com as regras de uso, passa por reparo
@pytest.mark.parametrize(
    ("content_type", "update", "error_field"),
    [
        (ContentType.DIAGNOSIS, {"hypotheses": []}, "conclusion.hypothesis_id"),
        (ContentType.STEPS, {"steps": []}, "steps:"),
        (ContentType.QUESTIONS, {"questions": [2, 1, 3]}, "questions:"),
        (ContentType.FLOW, {"edges": []}, "flow:"),
    ],
)
def test_schema_valid_responses_are_repaired_if_semantically_inconsistent(
    content_type, update, error_field, params, tmp_path
):
    output = CANNED_OUTPUTS[content_type].model_dump(mode="json")
    if content_type is ContentType.QUESTIONS:
        for question, priority in zip(output["questions"], update["questions"], strict=True):
            question["priority"] = priority
    else:
        output.update(update)
    invalid = OUTPUT_MODELS[content_type].model_validate(output).model_dump_json()
    provider = FakeProvider(responses=[invalid, _valid_json(content_type)])
    result = GENERATORS[content_type](**_kwargs(provider, params, ResponseCache(tmp_path)))
    assert (result.status, result.repaired) == (RunStatus.OK, True)
    assert error_field in provider.calls[1].messages[-1].content


# saída do cache é validada de novo contra o ticket
def test_cached_output_is_revalidated_against_ticket(params, tmp_path):
    provider, cache = FakeProvider(), ResponseCache(tmp_path)
    first = generate_diagnosis(**_kwargs(provider, params, cache))
    fingerprint = build_fingerprint("fake", provider.calls[0], "v1")
    bad_output = CANNED_OUTPUTS[ContentType.DIAGNOSIS].model_dump()
    bad_output["hypotheses"][0]["evidence"][0]["quote"] = "Citação ausente do ticket"
    diagnosis = OUTPUT_MODELS[ContentType.DIAGNOSIS].model_validate(bad_output)
    cache.put(first.cache_key, fingerprint, ContentType.DIAGNOSIS, diagnosis)

    regenerated = generate_diagnosis(**_kwargs(provider, params, cache))
    assert regenerated.cache_hit is False
    assert regenerated.output == first.output
    assert len(provider.calls) == 2


# falha ao gravar o cache não descarta a saída
def test_cache_write_failure_does_not_discard_output(params, tmp_path, caplog):
    file_instead_of_cache = tmp_path / "cache"
    file_instead_of_cache.touch()
    provider = FakeProvider()
    result = generate_diagnosis(**_kwargs(provider, params, ResponseCache(file_instead_of_cache)))
    assert (result.status, result.cache_hit) == (RunStatus.OK, False)
    assert len(provider.calls) == 1
    assert "a gravação falhou" in caplog.text


# credenciais ficam escondidas no prompt, no reparo e no registro inválido
def test_credential_values_are_hidden_in_prompt_repair_and_invalid_record(params, tmp_path):
    secret = "Violeta22"
    ticket = ticket_from_text(f"A internet não funciona. Minha senha do roteador é {secret}.")
    steps = CANNED_OUTPUTS[ContentType.STEPS].model_dump()
    steps["escalation_condition"] = f"Referência interna {secret.lower()}"
    raw = OUTPUT_MODELS[ContentType.STEPS].model_validate(steps).model_dump_json()
    cache = ResponseCache(tmp_path)
    provider = FakeProvider(responses=[raw, f"JSON inválido com {secret.upper()}"])
    result = generate_steps(**_kwargs(provider, params, cache, ticket=ticket))

    assert result.status is RunStatus.INVALID_OUTPUT
    assert result.raw_output == UNSAFE_OUTPUT
    assert not list(tmp_path.glob("*.json"))
    assert "[credencial removida]" in provider.calls[0].messages[1].content
    assert provider.calls[1].messages[-2].content == UNSAFE_OUTPUT
    assert all(
        secret.casefold() not in message.content.casefold()
        for call in provider.calls
        for message in call.messages
    )


# sem chave, o diagnóstico offline funciona em qualquer ticket e versão
@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4"])
@pytest.mark.parametrize(
    "ticket",
    [
        get_ticket("t01_queda_noturna"),
        get_ticket("t02_wifi_lento"),
        get_ticket("t03_codigo_sms"),
        ticket_from_text("A internet caiu agora"),
    ],
    ids=["t01", "t02", "t03", "texto_livre"],
)
def test_offline_diagnosis_is_valid_for_any_ticket(ticket, version, params, tmp_path):
    kwargs = _kwargs(FakeProvider(), params, ResponseCache(tmp_path), ticket=ticket)
    result = generate_diagnosis(**{**kwargs, "version": version})
    assert result.status is RunStatus.OK
    assert result.output["hypotheses"]


# opções em pergunta de sim ou não são descartadas sem gastar uma chamada de reparo
def test_options_outside_choice_are_cleared_without_repair(params, tmp_path):
    data = CANNED_OUTPUTS[ContentType.QUESTIONS].model_dump(mode="json")
    data["questions"][0]["options"] = ["Sim", "Não"]
    provider = FakeProvider([json.dumps(data)])
    result = generate_questions(**_kwargs(provider, params, ResponseCache(tmp_path)))
    assert result.status is RunStatus.OK and not result.repaired
    assert len(provider.calls) == 1
    assert result.output["questions"][0]["options"] == []


# o resultado guarda o motivo que levou ao reparo
def test_generation_keeps_the_repair_reason(params, tmp_path):
    provider = FakeProvider(["ruim", _valid_json(ContentType.FLOW)])
    result = generate_flow(**_kwargs(provider, params, ResponseCache(tmp_path)))
    assert result.status is RunStatus.OK and result.repaired
    assert result.validation_errors and result.validation_errors[0].startswith("resposta:")
