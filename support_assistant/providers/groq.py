import logging
import math
import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from support_assistant.credentials import redact_credentials
from support_assistant.models import Usage
from support_assistant.providers.base import (
    BadRequestError,
    EmptyResponseError,
    Provider,
    ProviderRequest,
    ProviderResponse,
    RateLimitError,
    TransientError,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRY_AFTER_S = 60.0
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 30.0
JSON_VALIDATE_FAILED = "json_validate_failed"
TOO_LARGE_PREFIX = "Request too large"

logger = logging.getLogger(__name__)


# falha que vale uma nova tentativa, com o motivo e a espera pedida pelo provedor
class _Retry(Exception):
    # guarda o motivo, a espera pedida e se foi limite de uso
    def __init__(self, reason: str, retry_after: float | None = None, rate_limited: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after
        self.rate_limited = rate_limited


# lê o cabeçalho retry-after em segundos, nulo se ausente ou inválido
def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers["retry-after"])
    except (KeyError, ValueError):
        return None


# mensagem de erro do corpo da resposta, ou os 200 primeiros caracteres do texto
def _error_message(response: httpx.Response) -> str:
    try:
        message = response.json()["error"]["message"]
        if isinstance(message, str) and message:
            return message
    except (ValueError, KeyError, TypeError):
        pass
    return response.text[:200]


# texto gerado que o Groq recusou por não bater com o schema, nulo para qualquer outro 400
def _failed_generation(response: httpx.Response) -> str | None:
    try:
        error = response.json()["error"]
    except (ValueError, KeyError, TypeError):
        return None
    if not isinstance(error, dict) or error.get("code") != JSON_VALIDATE_FAILED:
        return None
    text = error.get("failed_generation")
    return text if isinstance(text, str) and text.strip() else None


# número inteiro de tokens, zero quando ausente
def _tokens(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key)
    return value if isinstance(value, int) and value >= 0 else 0


# cliente do Groq com JSON Schema estrito, timeout, retry com backoff e erros tipados
class GroqProvider(Provider):
    name = "groq"

    # configura a chave, o tempo máximo, as tentativas e o cliente HTTP
    def __init__(
        self,
        api_key: str,
        timeout_s: float,
        max_retries: int,
        *,
        sleep: Callable[[float], None] = time.sleep,
        transport: httpx.BaseTransport | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._client = httpx.Client(timeout=httpx.Timeout(timeout_s), transport=transport)

    # corpo do POST com as chaves exatas que o Groq espera
    def _body(self, request: ProviderRequest) -> dict[str, Any]:
        params = request.params
        return {
            "model": request.model,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "temperature": params.temperature,
            "top_p": params.top_p,
            "seed": params.seed,
            "max_completion_tokens": params.max_tokens,
            "reasoning_effort": params.reasoning_effort.value,
            "include_reasoning": params.include_reasoning,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": True,
                    "schema": request.response_schema,
                },
            },
        }

    # espera aleatória entre 0 e o teto exponencial, o full jitter
    def _backoff(self, failed_attempt: int) -> float:
        ceiling = min(BACKOFF_CAP_S, BACKOFF_BASE_S * 2 ** (failed_attempt - 1))
        return self._rng.uniform(0, ceiling)

    # uma tentativa: devolve a resposta, ou levanta _Retry ou um erro definitivo
    def _attempt(self, body: dict[str, Any]) -> ProviderResponse:
        started = time.perf_counter()
        try:
            response = self._client.post(
                GROQ_URL, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
            )
        except httpx.TransportError as error:
            raise _Retry(type(error).__name__) from None
        latency_ms = int((time.perf_counter() - started) * 1000)
        status = response.status_code
        if status == 429:
            message = _error_message(response)
            if message.startswith(TOO_LARGE_PREFIX):
                detail = redact_credentials(message, (self._api_key,))
                raise BadRequestError(
                    f"O Groq recusou o pedido por ser maior que o limite do modelo (429): {detail}"
                )
            retry_after = _retry_after(response)
            if retry_after is not None and retry_after > MAX_RETRY_AFTER_S:
                raise RateLimitError(
                    f"Limite do Groq atingido; tente de novo em {math.ceil(retry_after)} s.",
                    retry_after,
                )
            raise _Retry("limite de requisições (429)", retry_after, rate_limited=True)
        if status >= 500:
            raise _Retry(f"HTTP {status}")
        if status == 401:
            raise BadRequestError("GROQ_API_KEY recusada pelo Groq (401). Confira a chave no .env.")
        if status == 400 and (failed := _failed_generation(response)) is not None:
            logger.warning(
                "groq: o modelo gerou JSON fora do schema; segue para validação e reparo"
            )
            return ProviderResponse(
                content=failed,
                usage=Usage(input_tokens=0, output_tokens=0),
                finish_reason=JSON_VALIDATE_FAILED,
                latency_ms=latency_ms,
            )
        if status >= 400:
            detail = redact_credentials(_error_message(response), (self._api_key,))
            raise BadRequestError(f"Groq recusou a requisição ({status}): {detail}")
        return self._parse(response, latency_ms)

    # converte uma resposta 200 em ProviderResponse
    def _parse(self, response: httpx.Response, latency_ms: int) -> ProviderResponse:
        try:
            data = response.json()
            choice = data["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError
        except (ValueError, KeyError, IndexError, TypeError):
            raise _Retry("resposta fora do formato esperado") from None
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise EmptyResponseError("O Groq devolveu uma resposta sem conteúdo.")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ProviderResponse(
            content=content,
            usage=Usage(
                input_tokens=_tokens(usage, "prompt_tokens"),
                output_tokens=_tokens(usage, "completion_tokens"),
            ),
            finish_reason=str(choice.get("finish_reason") or "unknown"),
            latency_ms=latency_ms,
        )

    # envia a requisição, repete em falha temporária e limite de uso, e falha rápido no resto
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        body = self._body(request)
        attempts = 1 + self._max_retries
        last: _Retry | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._attempt(body)
            except _Retry as retry:
                last = retry
            if attempt == attempts:
                break
            delay = last.retry_after if last.retry_after is not None else self._backoff(attempt)
            logger.warning(
                "groq: tentativa %d falhou (%s); nova tentativa em %.1f s",
                attempt,
                last.reason,
                delay,
            )
            self._sleep(delay)
        if last.rate_limited:
            raise RateLimitError(
                f"Limite do Groq atingido depois de {attempts} tentativas.", last.retry_after
            )
        raise TransientError(f"O Groq não respondeu depois de {attempts} tentativas: {last.reason}")
