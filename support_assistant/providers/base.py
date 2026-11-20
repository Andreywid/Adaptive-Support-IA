from abc import ABC, abstractmethod
from typing import Annotated, Any

from pydantic import Field

from support_assistant.models import (
    GenerationParams,
    Message,
    NonEmptyText,
    Slug,
    StrictModel,
    Usage,
)


# erro de qualquer provedor, base para os erros tipados do cliente real
class ProviderError(Exception):
    pass


# limite de requisições atingido (429), com o tempo de espera que o provedor pediu
class RateLimitError(ProviderError):
    # guarda a mensagem e a espera pedida pelo provedor
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# falha temporária: 5xx, timeout ou falha de conexão
class TransientError(ProviderError):
    pass


# requisição recusada pelo provedor (4xx), sem nova tentativa
class BadRequestError(ProviderError):
    pass


# o provedor respondeu sem conteúdo
class EmptyResponseError(ProviderError):
    pass


# requisição enviada ao provedor: mensagens, schema de saída e parâmetros fixos
class ProviderRequest(StrictModel):
    model: NonEmptyText
    messages: Annotated[list[Message], Field(min_length=1)]
    schema_name: Slug
    response_schema: dict[str, Any]
    params: GenerationParams


# resposta do provedor: texto bruto, consumo de tokens, motivo de parada e latência
class ProviderResponse(StrictModel):
    content: str
    usage: Usage
    finish_reason: NonEmptyText
    latency_ms: Annotated[int, Field(ge=0)]


# interface que todo provedor implementa
class Provider(ABC):
    # nome curto do provedor, gravado em cada execução
    @property
    @abstractmethod
    def name(self) -> str: ...

    # envia a requisição e devolve a resposta bruta, falhas viram ProviderError
    @abstractmethod
    def complete(self, request: ProviderRequest) -> ProviderResponse: ...
