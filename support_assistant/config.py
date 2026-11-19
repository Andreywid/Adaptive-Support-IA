import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from dotenv import dotenv_values
from pydantic import Field, SecretStr, ValidationError

from support_assistant.models import GenerationParams, NonEmptyText, StrictModel

# raiz do projeto e caminho do .env padrão
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# variável de ambiente de cada campo da configuração
ENV_NAMES: dict[str, str] = {
    "provider": "LLM_PROVIDER",
    "model": "LLM_MODEL",
    "groq_api_key": "GROQ_API_KEY",
    "temperature": "LLM_TEMPERATURE",
    "top_p": "LLM_TOP_P",
    "seed": "LLM_SEED",
    "max_tokens": "LLM_MAX_TOKENS",
    "reasoning_effort": "LLM_REASONING_EFFORT",
    "include_reasoning": "LLM_INCLUDE_REASONING",
    "timeout_s": "LLM_TIMEOUT_S",
    "max_retries": "LLM_MAX_RETRIES",
    "min_interval_s": "LLM_MIN_INTERVAL_S",
}

# valor usado quando a variável não está definida, o .env.example repete os mesmos
DEFAULTS: dict[str, str] = {
    "provider": "fake",
    "model": "openai/gpt-oss-120b",
    "temperature": "0",
    "top_p": "1.0",
    "seed": "1",
    "max_tokens": "2048",
    "reasoning_effort": "low",
    "include_reasoning": "false",
    "timeout_s": "60",
    "max_retries": "3",
    "min_interval_s": "20",
}


# erro de configuração com mensagem pronta para mostrar ao usuário
class ConfigError(Exception):
    pass


# configuração validada da aplicação
class Settings(StrictModel):
    provider: Literal["groq", "fake"]
    model: NonEmptyText
    groq_api_key: SecretStr | None
    params: GenerationParams
    timeout_s: Annotated[float, Field(gt=0, le=600, allow_inf_nan=False)]
    max_retries: Annotated[int, Field(ge=0, le=10)]
    min_interval_s: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    # devolve a chave do Groq ou explica como configurá-la
    def require_groq_api_key(self) -> str:
        if self.groq_api_key is None:
            raise ConfigError(
                f"{ENV_NAMES['groq_api_key']} não definida. "
                "Copie .env.example para .env e preencha a chave."
            )
        return self.groq_api_key.get_secret_value()


# lê uma variável tratando valor vazio ou só com espaços como ausente
def _read(environ: Mapping[str, str | None], name: str) -> str | None:
    value = (environ.get(name) or "").strip()
    return value or None


# lista cada variável inválida com o motivo, sem repetir o valor recebido
def _describe(error: ValidationError) -> str:
    lines = []
    for item in error.errors(include_input=False, include_url=False):
        field = str(item["loc"][-1])
        lines.append(f"  {ENV_NAMES.get(field, field)}: {item['msg']}")
    return "Configuração inválida:\n" + "\n".join(lines)


# carrega e valida a configuração, as variáveis do processo vencem as do arquivo .env
def load_settings(
    environ: Mapping[str, str | None] | None = None, env_file: Path = ENV_FILE
) -> Settings:
    if environ is None:
        environ = {**dotenv_values(env_file), **os.environ}
    values = {
        field: _read(environ, name) or DEFAULTS.get(field) for field, name in ENV_NAMES.items()
    }
    try:
        return Settings(
            provider=values["provider"],
            model=values["model"],
            groq_api_key=values["groq_api_key"],
            params={field: values[field] for field in GenerationParams.model_fields},
            timeout_s=values["timeout_s"],
            max_retries=values["max_retries"],
            min_interval_s=values["min_interval_s"],
        )
    except ValidationError as error:
        raise ConfigError(_describe(error)) from None
