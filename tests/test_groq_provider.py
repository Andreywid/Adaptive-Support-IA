import json
import logging
import random

import httpx
import pytest

from support_assistant.cache import ResponseCache
from support_assistant.credentials import REDACTED_CREDENTIAL
from support_assistant.data_loader import get_profile, get_ticket
from support_assistant.generators.core import generate
from support_assistant.models import ContentType, RunStatus
from support_assistant.providers.base import (
    BadRequestError,
    EmptyResponseError,
    Provider,
    ProviderError,
    RateLimitError,
    TransientError,
)
from support_assistant.providers.groq import (
    BACKOFF_CAP_S,
    GROQ_URL,
    MAX_RETRY_AFTER_S,
    GroqProvider,
)

API_KEY = "test-key"


# resposta 200 válida do Groq, com consumo de tokens
def _ok(content='{"resultado": 1}', finish_reason="stop", usage=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}
    body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7} if usage is None else usage
    return httpx.Response(200, json=body)


# transporte simulado que entrega as respostas em ordem e guarda as requisições
def _scripted(*items):
    requests = []
    queue = list(items)

    # devolve o próximo item da fila, levantando se ele for uma exceção
    def handler(request):
        requests.append(request)
        item = queue.pop(0)
        if isinstance(item, type) and issubclass(item, Exception):
            raise item("falha simulada", request=request)
        return item

    return handler, requests


# provedor com transporte simulado, espera falsa que registra os tempos e rng fixo
def _provider(handler, max_retries=3):
    sleeps = []
    provider = GroqProvider(
        API_KEY,
        5.0,
        max_retries,
        sleep=sleeps.append,
        transport=httpx.MockTransport(handler),
        rng=random.Random(0),
    )
    return provider, sleeps


# o cliente segue a interface de provedor e se chama groq
def test_groq_provider_implements_the_interface():
    provider, _ = _provider(_scripted()[0])
    assert isinstance(provider, Provider)
    assert provider.name == "groq"


# sucesso: conteúdo, tokens, motivo de parada e latência mapeados
def test_success_maps_the_response(make_request):
    handler, requests = _scripted(_ok(finish_reason="length"))
    provider, sleeps = _provider(handler)
    response = provider.complete(make_request())
    assert response.content == '{"resultado": 1}'
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7
    assert response.finish_reason == "length"
    assert response.latency_ms >= 0
    assert len(requests) == 1
    assert sleeps == []


# sem usage e sem finish_reason, tokens valem 0 e o motivo vira unknown
def test_missing_usage_and_finish_reason_use_defaults(make_request):
    body = {"choices": [{"message": {"content": "{}"}}]}
    handler, _ = _scripted(httpx.Response(200, json=body))
    response = _provider(handler)[0].complete(make_request())
    assert (response.usage.input_tokens, response.usage.output_tokens) == (0, 0)
    assert response.finish_reason == "unknown"


# o corpo enviado tem exatamente as chaves esperadas e o cabeçalho de autorização
def test_request_body_and_headers(make_request, params):
    handler, requests = _scripted(_ok())
    request = make_request()
    _provider(handler)[0].complete(request)
    sent = requests[0]
    body = json.loads(sent.content)
    assert str(sent.url) == GROQ_URL
    assert sent.headers["authorization"] == f"Bearer {API_KEY}"
    assert set(body) == {
        "model",
        "messages",
        "temperature",
        "top_p",
        "seed",
        "max_completion_tokens",
        "reasoning_effort",
        "include_reasoning",
        "response_format",
    }
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == params.max_tokens
    assert body["include_reasoning"] is False
    assert body["reasoning_effort"] == "low"
    assert body["messages"][0] == {
        "role": "system",
        "content": "Você é um assistente de suporte técnico.",
    }
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "diagnosis",
            "strict": True,
            "schema": request.response_schema,
        },
    }


# 429 com retry-after de 2 s e depois 200: uma espera de 2.0 e sucesso
def test_rate_limit_waits_the_requested_time_and_retries(make_request):
    handler, requests = _scripted(httpx.Response(429, headers={"retry-after": "2"}), _ok())
    provider, sleeps = _provider(handler)
    assert provider.complete(make_request()).content
    assert sleeps == [2.0]
    assert len(requests) == 2


# 429 com espera acima do teto falha na hora, sem esperar
def test_rate_limit_above_the_cap_fails_immediately(make_request):
    handler, requests = _scripted(httpx.Response(429, headers={"retry-after": "3600"}))
    provider, sleeps = _provider(handler)
    with pytest.raises(RateLimitError) as error:
        provider.complete(make_request())
    assert str(error.value) == "Limite do Groq atingido; tente de novo em 3600 s."
    assert error.value.retry_after == 3600
    assert 3600 > MAX_RETRY_AFTER_S
    assert sleeps == []
    assert len(requests) == 1


# 429 sem retry-after usa o backoff e, esgotadas as tentativas, vira RateLimitError
def test_rate_limit_without_header_exhausts_retries(make_request):
    handler, requests = _scripted(*[httpx.Response(429)] * 3)
    provider, sleeps = _provider(handler, max_retries=2)
    with pytest.raises(RateLimitError, match="depois de 3 tentativas"):
        provider.complete(make_request())
    assert len(requests) == 3
    assert len(sleeps) == 2


# 429 de pedido grande demais para o limite do modelo falha na hora, sem repetir nem esperar
def test_request_too_large_fails_immediately_without_retrying(make_request):
    message = (
        "Request too large for model `qwen/qwen3.8-27b` on output tokens per minute (OTPM): "
        "Limit 1000, Requested 1734."
    )
    body = {"error": {"message": message, "type": "tokens", "code": "rate_limit_exceeded"}}
    handler, requests = _scripted(httpx.Response(429, json=body, headers={"retry-after": "2"}))
    provider, sleeps = _provider(handler)
    with pytest.raises(BadRequestError) as error:
        provider.complete(make_request())
    assert not isinstance(error.value, RateLimitError)
    assert "maior que o limite do modelo (429)" in str(error.value)
    assert "Limit 1000, Requested 1734" in str(error.value)
    assert len(requests) == 1 and sleeps == []


# 429 com outra mensagem no corpo continua sendo espera, com nova tentativa
def test_rate_limit_with_a_different_message_still_retries(make_request):
    body = {"error": {"message": "Rate limit reached for model on requests per minute"}}
    handler, requests = _scripted(httpx.Response(429, json=body), _ok())
    provider, sleeps = _provider(handler)
    assert provider.complete(make_request()).content
    assert len(requests) == 2 and len(sleeps) == 1


# 500 três vezes com 2 repetições: TransientError depois de 3 tentativas e 2 esperas dentro do teto
def test_server_errors_exhaust_retries_with_jittered_waits(make_request):
    handler, requests = _scripted(*[httpx.Response(500)] * 3)
    provider, sleeps = _provider(handler, max_retries=2)
    with pytest.raises(TransientError) as error:
        provider.complete(make_request())
    assert str(error.value) == "O Groq não respondeu depois de 3 tentativas: HTTP 500"
    assert len(requests) == 3
    assert len(sleeps) == 2
    assert 0 <= sleeps[0] <= 1.0
    assert 0 <= sleeps[1] <= 2.0


# o teto do backoff nunca passa do limite configurado
def test_backoff_never_exceeds_the_cap(make_request):
    handler, _ = _scripted(*[httpx.Response(503)] * 8)
    provider, sleeps = _provider(handler, max_retries=7)
    with pytest.raises(TransientError):
        provider.complete(make_request())
    assert len(sleeps) == 7
    assert max(sleeps) <= BACKOFF_CAP_S


# timeout seguido de 200 recupera sem erro
def test_timeout_then_success(make_request):
    handler, requests = _scripted(httpx.ReadTimeout, _ok())
    provider, sleeps = _provider(handler)
    assert provider.complete(make_request()).content
    assert len(requests) == 2
    assert len(sleeps) == 1


# falha de conexão esgotada vira TransientError citando o tipo da falha
def test_connection_failure_becomes_transient_error(make_request):
    handler, _ = _scripted(*[httpx.ConnectError] * 2)
    provider, _ = _provider(handler, max_retries=1)
    with pytest.raises(TransientError, match="ConnectError"):
        provider.complete(make_request())


# corpo que não é JSON ou sem choices conta como falha temporária
@pytest.mark.parametrize(
    "bad",
    [httpx.Response(200, text="não é json"), httpx.Response(200, json={"sem": "choices"})],
)
def test_malformed_success_body_is_retried(bad, make_request):
    handler, requests = _scripted(bad, _ok())
    provider, _ = _provider(handler)
    assert provider.complete(make_request()).content
    assert len(requests) == 2


# 400 falha na hora com a mensagem do provedor e sem nova tentativa
def test_bad_request_uses_the_provider_message(make_request):
    body = {"error": {"message": "schema inválido na propriedade x"}}
    handler, requests = _scripted(httpx.Response(400, json=body))
    provider, sleeps = _provider(handler)
    with pytest.raises(BadRequestError) as error:
        provider.complete(make_request())
    assert str(error.value) == "Groq recusou a requisição (400): schema inválido na propriedade x"
    assert len(requests) == 1
    assert sleeps == []


# erro 400 não expõe a chave configurada na mensagem
def test_bad_request_redacts_the_configured_key_in_the_provider_message(make_request):
    handler, _ = _scripted(httpx.Response(400, text=f"Credencial inválida ({API_KEY})"))
    with pytest.raises(BadRequestError) as error:
        _provider(handler)[0].complete(make_request())
    assert API_KEY not in str(error.value)
    assert REDACTED_CREDENTIAL in str(error.value)


# 4xx sem mensagem no corpo mostra os 200 primeiros caracteres do texto
def test_bad_request_without_message_truncates_the_text(make_request):
    handler, _ = _scripted(httpx.Response(422, text="x" * 500))
    with pytest.raises(BadRequestError) as error:
        _provider(handler)[0].complete(make_request())
    assert str(error.value) == "Groq recusou a requisição (422): " + "x" * 200


# 401 cita a variável da chave e nunca repete a chave
def test_unauthorized_names_the_variable_not_the_key(make_request):
    handler, requests = _scripted(httpx.Response(401, json={"error": {"message": "invalid"}}))
    with pytest.raises(BadRequestError) as error:
        _provider(handler)[0].complete(make_request())
    assert "GROQ_API_KEY" in str(error.value)
    assert API_KEY not in str(error.value)
    assert len(requests) == 1


# conteúdo vazio ou ausente vira EmptyResponseError, sem nova tentativa
@pytest.mark.parametrize("content", ["", "   ", None])
def test_empty_content_raises_without_retry(content, make_request):
    handler, requests = _scripted(_ok(content=content))
    with pytest.raises(EmptyResponseError):
        _provider(handler)[0].complete(make_request())
    assert len(requests) == 1


# a chave não aparece em exceção nem no log, em nenhum cenário de falha
def test_api_key_never_leaks_to_errors_or_logs(make_request, caplog):
    scenarios = [
        [httpx.Response(401)],
        [httpx.Response(400, text="erro")],
        [httpx.Response(500)] * 2,
        [httpx.Response(429, headers={"retry-after": "1"})] * 2,
    ]
    with caplog.at_level(logging.DEBUG):
        for items in scenarios:
            handler, _ = _scripted(*items)
            provider, _ = _provider(handler, max_retries=1)
            with pytest.raises(ProviderError) as error:
                provider.complete(make_request())
            assert API_KEY not in str(error.value)
    assert API_KEY not in caplog.text
    assert "authorization" not in caplog.text.lower()


# cada nova tentativa gera um aviso com o motivo e a espera
def test_each_retry_logs_one_warning(make_request, caplog):
    handler, _ = _scripted(httpx.Response(500), httpx.Response(502), _ok())
    provider, _ = _provider(handler)
    with caplog.at_level(logging.WARNING, logger="support_assistant.providers.groq"):
        provider.complete(make_request())
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert messages[0].startswith("groq: tentativa 1 falhou (HTTP 500); nova tentativa em ")
    assert messages[1].startswith("groq: tentativa 2 falhou (HTTP 502); nova tentativa em ")


# com zero repetições, a primeira falha temporária já esgota
def test_zero_retries_means_a_single_attempt(make_request):
    handler, requests = _scripted(httpx.Response(500))
    provider, sleeps = _provider(handler, max_retries=0)
    with pytest.raises(TransientError, match="depois de 1 tentativas"):
        provider.complete(make_request())
    assert len(requests) == 1
    assert sleeps == []


# fluxo com uma aresta sem label: é o tipo de JSON que o Groq recusa com json_validate_failed
INVALID_FLOW = (
    '{"nodes":[{"id":"P1","kind":"question","text":"Pergunta?"},'
    '{"id":"F1","kind":"end","text":"Fim"},{"id":"F2","kind":"end","text":"Outro fim"}],'
    '"edges":[{"source":"P1","target":"F1","label":"sim"},{"source":"P1","target":"F2"}]}'
)
VALID_FLOW = (
    '{"nodes":[{"id":"P1","kind":"question","text":"Pergunta?"},'
    '{"id":"F1","kind":"end","text":"Fim"},{"id":"F2","kind":"end","text":"Outro fim"}],'
    '"edges":[{"source":"P1","target":"F1","label":"sim"},'
    '{"source":"P1","target":"F2","label":"não"}]}'
)


# resposta 400 do Groq quando o JSON gerado não bate com o schema, com o texto recusado
def _validation_failure(text=INVALID_FLOW):
    error = {
        "message": "Generated JSON does not match the expected schema.",
        "type": "invalid_request_error",
        "code": "json_validate_failed",
        "failed_generation": text,
    }
    return httpx.Response(400, json={"error": error})


# json_validate_failed vira uma resposta com o texto gerado, para seguir à validação e ao reparo
def test_json_validate_failed_becomes_a_response_with_the_generated_text(make_request, caplog):
    handler, requests = _scripted(_validation_failure())
    provider, sleeps = _provider(handler)
    with caplog.at_level(logging.WARNING, logger="support_assistant.providers.groq"):
        response = provider.complete(make_request())
    assert response.content == INVALID_FLOW
    assert response.finish_reason == "json_validate_failed"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (0, 0)
    assert len(requests) == 1 and sleeps == []
    assert "o modelo gerou JSON fora do schema" in caplog.text


# 400 sem o texto gerado, com outro código ou sem corpo JSON continua sendo erro de requisição
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(
            400, json={"error": {"code": "json_validate_failed", "message": "sem texto"}}
        ),
        httpx.Response(
            400,
            json={
                "error": {
                    "code": "json_validate_failed",
                    "failed_generation": "   ",
                    "message": "m",
                }
            },
        ),
        httpx.Response(
            400, json={"error": {"code": "outro_codigo", "failed_generation": "{}", "message": "m"}}
        ),
        httpx.Response(400, text="corpo que não é json"),
        httpx.Response(400, json={"error": "texto solto"}),
    ],
)
def test_other_400_responses_still_fail_as_bad_requests(response, make_request):
    handler, requests = _scripted(response)
    with pytest.raises(BadRequestError):
        _provider(handler)[0].complete(make_request())
    assert len(requests) == 1


# pelo generate, o JSON recusado vai para o reparo e a segunda resposta válida fecha a execução
def test_generate_repairs_a_schema_validation_failure(params, tmp_path):
    handler, requests = _scripted(_validation_failure(), _ok(content=VALID_FLOW))
    provider, _ = _provider(handler)
    result = generate(
        content_type=ContentType.FLOW,
        profile=get_profile("p01_marta"),
        ticket=get_ticket("t01_queda_noturna"),
        version="v1",
        provider=provider,
        model="openai/gpt-oss-20b",
        params=params,
        cache=ResponseCache(tmp_path),
    )
    assert (result.status, result.repaired) == (RunStatus.OK, True)
    assert len(requests) == 2
    messages = json.loads(requests[1].content)["messages"]
    assert messages[-2] == {"role": "assistant", "content": INVALID_FLOW}
    assert messages[-1]["content"].startswith("A resposta anterior não passou na validação:")
    assert "edges.1.label" in messages[-1]["content"]


# duas recusas seguidas viram invalid_output com o texto bruto da segunda, sem derrubar a execução
def test_generate_records_invalid_output_after_two_schema_failures(params, tmp_path):
    second = INVALID_FLOW.replace("Pergunta?", "Segunda tentativa?")
    handler, requests = _scripted(_validation_failure(), _validation_failure(second))
    cache = ResponseCache(tmp_path)
    result = generate(
        content_type=ContentType.FLOW,
        profile=get_profile("p01_marta"),
        ticket=get_ticket("t01_queda_noturna"),
        version="v1",
        provider=_provider(handler)[0],
        model="openai/gpt-oss-20b",
        params=params,
        cache=cache,
    )
    assert result.status is RunStatus.INVALID_OUTPUT
    assert result.raw_output == second and result.output is None
    assert result.finish_reason == "json_validate_failed"
    assert len(requests) == 2 and not list(tmp_path.glob("*.json"))
