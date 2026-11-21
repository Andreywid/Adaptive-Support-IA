import json
import logging
from typing import Any

from pydantic import ValidationError

from support_assistant.cache import (
    ResponseCache,
    build_fingerprint,
    canonical_json,
    compute_cache_key,
)
from support_assistant.credentials import (
    contains_credential,
    credential_values,
    output_texts,
    redact_credentials,
    requests_credential,
    sanitize_ticket,
)
from support_assistant.models import (
    OUTPUT_MODELS,
    ContentType,
    GenerationParams,
    Message,
    NonEmptyText,
    Profile,
    PromptVersion,
    RunStatus,
    Sha256Hash,
    StrictModel,
    Ticket,
    Usage,
    output_json_schema,
)
from support_assistant.output_validation import normalize_output, output_errors
from support_assistant.prompts.engine import build_messages
from support_assistant.providers.base import Provider, ProviderRequest

REPAIR_MESSAGE = (
    "A resposta anterior não passou na validação:\n"
    "{errors}\n"
    "Responda de novo apenas com o JSON corrigido."
)
MAX_REPAIR_ERROR_LINES = 20
UNSAFE_OUTPUT = "Resposta omitida por conter ou solicitar uma credencial."
logger = logging.getLogger(__name__)


# resultado de uma geração, com tudo que o registro de execução precisa
class GenerationResult(StrictModel):
    content_type: ContentType
    prompt_version: PromptVersion
    prompt_hash: Sha256Hash
    schema_hash: Sha256Hash
    cache_key: Sha256Hash
    cache_hit: bool
    provider: NonEmptyText
    model: NonEmptyText
    params: GenerationParams
    usage: Usage
    latency_ms: int
    finish_reason: str | None
    repaired: bool
    status: RunStatus
    output: dict[str, Any] | None
    raw_output: str | None
    validation_errors: list[str] = []


# uma linha "campo: motivo" por erro de validação, até o limite do pedido de reparo
def _error_lines(error: ValidationError) -> list[str]:
    lines = []
    for item in error.errors(include_input=False, include_url=False)[:MAX_REPAIR_ERROR_LINES]:
        field = ".".join(str(part) for part in item["loc"]) or "resposta"
        lines.append(f"{field}: {item['msg']}")
    return lines


# diz se a resposta contém ou pede uma credencial em algum campo
def _unsafe_response(raw: str, known_values: tuple[str, ...]) -> bool:
    try:
        fields = output_texts(json.loads(raw))
    except ValueError:
        fields = iter((raw,))
    return any(
        contains_credential(field, known_values) or requests_credential(field) for field in fields
    )


# valida a resposta: credenciais, schema e regras de uso; devolve a saída ou os erros
def _validate_response(
    raw: str,
    output_model: type[StrictModel],
    profile: Profile,
    ticket: Ticket,
    known_values: tuple[str, ...],
) -> tuple[StrictModel | None, list[str], str]:
    if _unsafe_response(raw, known_values):
        return None, ["Remova qualquer credencial ou pedido de credencial."], UNSAFE_OUTPUT
    try:
        output = normalize_output(output_model.model_validate_json(raw))
    except ValidationError as error:
        return None, _error_lines(error), raw
    problems = output_errors(output, profile, ticket)
    return (None if problems else output), problems, raw


# gera o conteúdo de um tipo, usando o cache, validando a saída e tentando um reparo
def generate(
    *,
    content_type: ContentType,
    profile: Profile,
    ticket: Ticket,
    version: str,
    provider: Provider,
    model: str,
    params: GenerationParams,
    cache: ResponseCache | None,
    refresh: bool = False,
) -> GenerationResult:
    known_values = credential_values(ticket.text)
    safe_ticket = sanitize_ticket(ticket)
    messages = build_messages(version, content_type, profile, safe_ticket)
    request = ProviderRequest(
        model=model,
        messages=messages,
        schema_name=content_type.value,
        response_schema=output_json_schema(content_type),
        params=params,
    )
    fingerprint = build_fingerprint(provider.name, request, version)
    key = compute_cache_key(fingerprint)
    common: dict[str, Any] = {
        "content_type": content_type,
        "prompt_version": version,
        "prompt_hash": fingerprint.prompt_hash,
        "schema_hash": fingerprint.schema_hash,
        "cache_key": key,
        "provider": provider.name,
        "model": model,
        "params": params,
    }

    output_model = OUTPUT_MODELS[content_type]
    if cache is not None and not refresh:
        hit = cache.get(key, content_type)
        if hit is not None:
            hit_output, _, _ = _validate_response(
                canonical_json(hit), output_model, profile, safe_ticket, known_values
            )
            if hit_output is not None:
                return GenerationResult(
                    **common,
                    cache_hit=True,
                    usage=Usage(input_tokens=0, output_tokens=0),
                    latency_ms=0,
                    finish_reason=None,
                    repaired=False,
                    status=RunStatus.OK,
                    output=hit,
                    raw_output=None,
                )
            logger.warning("cache: saída ignorada após validação semântica ou de credenciais")

    response = provider.complete(request)
    usage = response.usage
    latency_ms = response.latency_ms
    repaired = False
    validated, errors, safe_raw = _validate_response(
        response.content, output_model, profile, safe_ticket, known_values
    )
    validation_errors = [redact_credentials(error, known_values) for error in errors]
    if validated is None:
        repair_errors = "\n".join(errors[:MAX_REPAIR_ERROR_LINES])
        repair_request = request.model_copy(
            update={
                "messages": [
                    *messages,
                    Message(role="assistant", content=safe_raw),
                    Message(role="user", content=REPAIR_MESSAGE.format(errors=repair_errors)),
                ]
            }
        )
        response = provider.complete(repair_request)
        usage = Usage(
            input_tokens=usage.input_tokens + response.usage.input_tokens,
            output_tokens=usage.output_tokens + response.usage.output_tokens,
        )
        latency_ms += response.latency_ms
        validated, final_errors, safe_raw = _validate_response(
            response.content, output_model, profile, safe_ticket, known_values
        )
        repaired = validated is not None
        for error in final_errors:
            safe_error = redact_credentials(error, known_values)
            if safe_error not in validation_errors:
                validation_errors.append(safe_error)

    if cache is not None and validated is not None:
        try:
            cache.put(key, fingerprint, content_type, validated)
        except OSError as error:
            logger.warning("cache: saída gerada, mas a gravação falhou (%s)", type(error).__name__)
    return GenerationResult(
        **common,
        cache_hit=False,
        usage=usage,
        latency_ms=latency_ms,
        finish_reason=redact_credentials(response.finish_reason, known_values),
        repaired=repaired,
        status=RunStatus.OK if validated is not None else RunStatus.INVALID_OUTPUT,
        output=validated.model_dump(mode="json") if validated is not None else None,
        raw_output=safe_raw if validated is None else None,
        validation_errors=validation_errors,
    )
