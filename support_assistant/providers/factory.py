from support_assistant.config import ConfigError, Settings
from support_assistant.providers.base import Provider
from support_assistant.providers.fake import FakeProvider
from support_assistant.providers.groq import GroqProvider


# cria o provedor pedido, ou o das configurações quando nenhum nome é dado
def make_provider(settings: Settings, name: str | None = None) -> Provider:
    chosen = name or settings.provider
    if chosen == "fake":
        return FakeProvider()
    if chosen == "groq":
        return GroqProvider(
            api_key=settings.require_groq_api_key(),
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
        )
    raise ConfigError(f"Provedor desconhecido: {chosen}")
