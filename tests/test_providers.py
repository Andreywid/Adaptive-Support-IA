import pytest
from pydantic import ValidationError

from support_assistant.models import OUTPUT_MODELS, ContentType, Message, Usage, output_json_schema
from support_assistant.providers.base import (
    BadRequestError,
    EmptyResponseError,
    Provider,
    ProviderError,
    ProviderResponse,
    RateLimitError,
    TransientError,
)
from support_assistant.providers.fake import CANNED_OUTPUTS, FakeProvider

# texto do ticket t01, referência das citações da saída fixa
T01_TEXT = "A internet cai toda noite e volta sozinha"


# provedor incompleto: tem complete, mas não define name
class _WithoutName(Provider):
    # método exigido pela interface
    def complete(self, request):
        raise NotImplementedError


# provedor incompleto: tem name, mas não define complete
class _WithoutComplete(Provider):
    name = "incompleto"


# a interface abstrata não pode ser instanciada
def test_provider_interface_is_abstract():
    with pytest.raises(TypeError):
        Provider()


# provedor sem name ou sem complete não pode ser instanciado
@pytest.mark.parametrize("incomplete", [_WithoutName, _WithoutComplete])
def test_incomplete_provider_cannot_be_instantiated(incomplete):
    with pytest.raises(TypeError):
        incomplete()


# o fake se identifica como fake
def test_fake_name():
    assert FakeProvider().name == "fake"


# para cada tipo o fake devolve um JSON válido no contrato correspondente
@pytest.mark.parametrize("content_type", list(ContentType))
def test_fake_returns_valid_output_for_each_content_type(content_type, make_request):
    request = make_request(
        schema_name=content_type.value,
        response_schema=output_json_schema(content_type),
    )
    response = FakeProvider().complete(request)
    assert isinstance(response, ProviderResponse)
    assert response.finish_reason == "stop"
    OUTPUT_MODELS[content_type].model_validate_json(response.content)


# a mesma requisição produz sempre a mesma resposta
def test_fake_is_deterministic(make_request):
    request = make_request()
    assert FakeProvider().complete(request) == FakeProvider().complete(request)


# toda chamada fica registrada em calls, na ordem
def test_fake_records_every_call(make_request):
    fake = FakeProvider()
    first, second = make_request(), make_request(model="outro-modelo")
    fake.complete(first)
    fake.complete(second)
    assert fake.calls == [first, second]


# os tokens estimados crescem com o prompt e nunca são zero
def test_fake_usage_grows_with_prompt_size(make_request):
    request = make_request()
    short = FakeProvider().complete(request)
    longer = FakeProvider().complete(
        make_request(messages=[*request.messages, Message(role="user", content="x" * 4000)])
    )
    assert short.usage.input_tokens >= 1
    assert longer.usage.input_tokens > short.usage.input_tokens
    assert short.usage.output_tokens == longer.usage.output_tokens >= 1


# respostas programadas saem na ordem e depois volta a saída fixa
def test_fake_follows_programmed_responses_in_order(make_request):
    fake = FakeProvider(responses=["primeira", "segunda"])
    request = make_request()
    contents = [fake.complete(request).content for _ in range(3)]
    assert contents[:2] == ["primeira", "segunda"]
    assert contents[2] == CANNED_OUTPUTS[ContentType.DIAGNOSIS].model_dump_json()


# exceção programada é levantada, a chamada fica registrada e a seguinte funciona
def test_fake_raises_programmed_errors(make_request):
    fake = FakeProvider(responses=[ProviderError("falha de rede")])
    request = make_request()
    with pytest.raises(ProviderError, match="falha de rede"):
        fake.complete(request)
    assert len(fake.calls) == 1
    assert fake.complete(request).content
    assert len(fake.calls) == 2


# schema desconhecido vira um ProviderError claro
def test_fake_rejects_unknown_schema_name(make_request):
    with pytest.raises(ProviderError, match="desconhecido"):
        FakeProvider().complete(make_request(schema_name="outro_tipo"))


# as citações do diagnóstico fixo existem no ticket t01
def test_fake_diagnosis_quotes_exist_in_the_t01_ticket():
    diagnosis = CANNED_OUTPUTS[ContentType.DIAGNOSIS]
    quotes = [item.quote for hypothesis in diagnosis.hypotheses for item in hypothesis.evidence]
    assert quotes
    assert all(quote in T01_TEXT for quote in quotes)


# a conclusão fixa aponta para uma hipótese que existe
def test_fake_conclusion_points_to_an_existing_hypothesis():
    diagnosis = CANNED_OUTPUTS[ContentType.DIAGNOSIS]
    ids = {hypothesis.id for hypothesis in diagnosis.hypotheses}
    assert diagnosis.conclusion.hypothesis_id in ids


# o fluxo fixo só liga nós que existem e tem um único início
def test_fake_flow_is_a_connected_graph_with_one_start():
    flow = CANNED_OUTPUTS[ContentType.FLOW]
    ids = {node.id for node in flow.nodes}
    assert all(edge.source in ids and edge.target in ids for edge in flow.edges)
    targets = {edge.target for edge in flow.edges}
    assert ids - targets == {"P1"}


# requisição sem mensagens é rejeitada
def test_request_requires_at_least_one_message(make_request):
    with pytest.raises(ValidationError):
        make_request(messages=[])


# nome de schema fora do padrão é rejeitado
def test_request_rejects_invalid_schema_name(make_request):
    with pytest.raises(ValidationError):
        make_request(schema_name="Diagnóstico 1")


# campo extra na requisição é rejeitado
def test_request_rejects_extra_field(make_request):
    with pytest.raises(ValidationError):
        make_request(temperatura=0.5)


# resposta sem motivo de parada é rejeitada
def test_response_requires_finish_reason():
    usage = Usage(input_tokens=1, output_tokens=1)
    with pytest.raises(ValidationError):
        ProviderResponse(content="x", usage=usage, finish_reason="", latency_ms=0)


# resposta sem latência ou com latência negativa é rejeitada
@pytest.mark.parametrize("changes", [{}, {"latency_ms": -1}])
def test_response_requires_a_non_negative_latency(changes):
    data = {
        "content": "x",
        "usage": Usage(input_tokens=1, output_tokens=1),
        "finish_reason": "stop",
    }
    with pytest.raises(ValidationError):
        ProviderResponse(**{**data, **changes})


# o fake devolve latência zero
def test_fake_latency_is_zero(make_request):
    assert FakeProvider().complete(make_request()).latency_ms == 0


# todo erro tipado descende de ProviderError, e o de limite guarda a espera pedida
def test_typed_errors_share_the_base_class():
    for error in (RateLimitError, TransientError, BadRequestError, EmptyResponseError):
        assert issubclass(error, ProviderError)
    assert RateLimitError("limite").retry_after is None
    assert RateLimitError("limite", 2.5).retry_after == 2.5
