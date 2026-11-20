import hashlib
import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from support_assistant.config import PROJECT_ROOT
from support_assistant.files import atomic_write_json
from support_assistant.models import (
    OUTPUT_MODELS,
    SCHEMA_VERSION,
    ContentType,
    GenerationParams,
    Message,
    NonEmptyText,
    PromptVersion,
    SchemaVersion,
    Sha256Hash,
    StrictModel,
    UtcDatetime,
)
from support_assistant.providers.base import ProviderRequest

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_KEY_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

logger = logging.getLogger(__name__)


# troca float inteiro por int, para 1.0 e 1 gerarem o mesmo texto
def _normalize(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e21:
        return int(value)
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize(item) for item in value]
    return value


# JSON canônico no espírito da RFC 8785: chaves ordenadas, sem espaços e em UTF-8
def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


# hash SHA-256 de um texto no formato sha256:<hex>
def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


# hash SHA-256 do JSON canônico de um valor
def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


# hash das mensagens exatamente como são enviadas ao modelo
def hash_messages(messages: Sequence[Message]) -> str:
    return sha256_json([message.model_dump(mode="json") for message in messages])


# hash do JSON Schema de saída enviado ao modelo
def hash_schema(schema: dict[str, Any]) -> str:
    return sha256_json(schema)


# impressão digital da requisição: tudo que pode mudar a resposta do modelo
class Fingerprint(StrictModel):
    provider: NonEmptyText
    model: NonEmptyText
    params: GenerationParams
    prompt_version: PromptVersion
    prompt_hash: Sha256Hash
    schema_hash: Sha256Hash


# monta a impressão digital de uma requisição do provedor
def build_fingerprint(provider: str, request: ProviderRequest, prompt_version: str) -> Fingerprint:
    return Fingerprint(
        provider=provider,
        model=request.model,
        params=request.params,
        prompt_version=prompt_version,
        prompt_hash=hash_messages(request.messages),
        schema_hash=hash_schema(request.response_schema),
    )


# chave de cache: SHA-256 do JSON canônico da impressão digital
def compute_cache_key(fingerprint: Fingerprint) -> str:
    return sha256_json(fingerprint.model_dump(mode="json"))


# arquivo do cache: a saída validada e a impressão digital da requisição que a gerou
class CacheEntry(StrictModel):
    schema_version: SchemaVersion
    cache_key: Sha256Hash
    created_at: UtcDatetime
    content_type: ContentType
    fingerprint: Fingerprint
    output: dict[str, Any]


# motivo curto de uma entrada recusada, sem repetir o conteúdo do arquivo
def _reason(error: Exception) -> str:
    if isinstance(error, ValidationError):
        first = error.errors(include_input=False, include_url=False)[0]
        place = ".".join(str(part) for part in first["loc"])
        return f"{place}: {first['msg']}" if place else first["msg"]
    return str(error)


# cache de respostas em disco, um arquivo por chave, que nunca derruba a execução
class ResponseCache:
    # guarda a pasta do cache, ou usa a padrão
    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory

    # pasta do cache, resolvida na hora do uso
    @property
    def directory(self) -> Path:
        return self._directory or CACHE_DIR

    # caminho do arquivo de uma chave: 64 hexadecimais, sem o prefixo sha256
    def path_for(self, key: str) -> Path:
        if not CACHE_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"chave de cache inválida: {key}")
        return self.directory / f"{key.removeprefix('sha256:')}.json"

    # devolve a saída guardada para a chave, ou None se faltar ou estiver inválida
    def get(self, key: str, content_type: ContentType) -> dict[str, Any] | None:
        path = self.path_for(key)
        try:
            entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
            if entry.cache_key != key:
                raise ValueError("cache_key diferente da pedida")
            if compute_cache_key(entry.fingerprint) != key:
                raise ValueError("a chave não corresponde à impressão digital")
            if entry.content_type is not content_type:
                raise ValueError("content_type diferente do pedido")
            OUTPUT_MODELS[content_type].model_validate(entry.output)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as error:
            logger.warning("cache: entrada ignorada em %s: %s", path.name, _reason(error))
            return None
        return entry.output

    # grava a saída já validada, de forma atômica
    def put(
        self,
        key: str,
        fingerprint: Fingerprint,
        content_type: ContentType,
        output: StrictModel,
    ) -> Path:
        if not isinstance(output, OUTPUT_MODELS[content_type]):
            raise TypeError(f"o cache só recebe a saída validada do tipo {content_type.value}")
        if compute_cache_key(fingerprint) != key:
            raise ValueError("a chave não corresponde à impressão digital")
        entry = CacheEntry(
            schema_version=SCHEMA_VERSION,
            cache_key=key,
            created_at=datetime.now(UTC),
            content_type=content_type,
            fingerprint=fingerprint,
            output=output.model_dump(mode="json"),
        )
        path = self.path_for(key)
        atomic_write_json(path, entry.model_dump(mode="json"))
        return path
