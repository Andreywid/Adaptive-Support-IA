import os

import pytest
from dotenv import dotenv_values
from pydantic import ValidationError

from support_assistant.config import (
    DEFAULTS,
    ENV_NAMES,
    PROJECT_ROOT,
    ConfigError,
    load_settings,
)
from support_assistant.models import GenerationParams, ReasoningEffort

SECRET = "chave-falsa-de-teste"


# remove do ambiente do teste qualquer variável que a configuração leia
@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ENV_NAMES.values():
        monkeypatch.delenv(name, raising=False)


# sem variáveis, a configuração usa os valores padrão
def test_defaults_when_environment_is_empty():
    settings = load_settings({})
    assert settings.provider == "fake"
    assert settings.model == "openai/gpt-oss-120b"
    assert settings.groq_api_key is None
    assert settings.params == GenerationParams(
        temperature=0.0,
        top_p=1.0,
        seed=1,
        max_tokens=2048,
        reasoning_effort=ReasoningEffort.LOW,
        include_reasoning=False,
    )


# cada variável sobrescreve o seu valor padrão
def test_environment_overrides_defaults():
    settings = load_settings(
        {
            "LLM_PROVIDER": "groq",
            "LLM_MODEL": "openai/gpt-oss-120b",
            "GROQ_API_KEY": SECRET,
            "LLM_TEMPERATURE": "0",
            "LLM_TOP_P": "0.9",
            "LLM_SEED": "7",
            "LLM_MAX_TOKENS": "4096",
            "LLM_REASONING_EFFORT": "high",
            "LLM_INCLUDE_REASONING": "true",
        }
    )
    assert settings.provider == "groq"
    assert settings.model == "openai/gpt-oss-120b"
    assert settings.params.model_dump(mode="json") == {
        "temperature": 0.0,
        "top_p": 0.9,
        "seed": 7,
        "max_tokens": 4096,
        "reasoning_effort": "high",
        "include_reasoning": True,
    }


# valor vazio ou só com espaços conta como variável ausente
def test_blank_values_fall_back_to_defaults():
    settings = load_settings({"LLM_MODEL": "   ", "LLM_TEMPERATURE": "", "GROQ_API_KEY": " "})
    assert settings.model == DEFAULTS["model"]
    assert settings.params.temperature == 0.0
    assert settings.groq_api_key is None


# valor inválido gera ConfigError que nomeia a variável com problema
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_PROVIDER", "openai"),
        ("LLM_TEMPERATURE", "9"),
        ("LLM_TEMPERATURE", "nan"),
        ("LLM_TOP_P", "0"),
        ("LLM_SEED", "abc"),
        ("LLM_MAX_TOKENS", "0"),
        ("LLM_REASONING_EFFORT", "none"),
        ("LLM_INCLUDE_REASONING", "talvez"),
    ],
)
def test_invalid_value_raises_config_error_naming_the_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value})


# a mensagem de erro lista todas as variáveis inválidas de uma vez
def test_error_lists_every_invalid_variable():
    with pytest.raises(ConfigError) as caught:
        load_settings({"LLM_TEMPERATURE": "9", "LLM_SEED": "abc"})
    assert "LLM_TEMPERATURE" in str(caught.value)
    assert "LLM_SEED" in str(caught.value)


# a chave nunca aparece na mensagem de erro, mesmo com outra variável inválida
def test_error_message_never_contains_the_api_key():
    with pytest.raises(ConfigError) as caught:
        load_settings({"GROQ_API_KEY": SECRET, "LLM_TEMPERATURE": "9"})
    assert SECRET not in str(caught.value)
    assert caught.value.__suppress_context__


# a chave fica oculta em repr, str e nos dumps da configuração
def test_api_key_is_hidden_in_repr_and_dumps():
    settings = load_settings({"GROQ_API_KEY": SECRET})
    for text in (
        repr(settings),
        str(settings),
        settings.model_dump_json(),
        str(settings.model_dump()),
    ):
        assert SECRET not in text


# a chave é entregue só pelo método explícito
def test_require_groq_api_key_returns_the_key():
    assert load_settings({"GROQ_API_KEY": f"  {SECRET}  "}).require_groq_api_key() == SECRET


# sem chave, o erro explica como configurar e cita a variável
def test_require_groq_api_key_without_key_explains_how_to_configure():
    with pytest.raises(ConfigError, match=r"GROQ_API_KEY.*\.env"):
        load_settings({}).require_groq_api_key()


# tempo limite e tentativas têm padrão 60 e 3 e ficam fora dos parâmetros de geração
def test_timeout_and_retries_defaults():
    settings = load_settings({})
    assert settings.timeout_s == 60.0
    assert settings.max_retries == 3
    assert "timeout_s" not in GenerationParams.model_fields
    assert "max_retries" not in GenerationParams.model_fields


# tempo limite e tentativas lidos do ambiente
def test_timeout_and_retries_from_environment():
    settings = load_settings({"LLM_TIMEOUT_S": "12.5", "LLM_MAX_RETRIES": "0"})
    assert (settings.timeout_s, settings.max_retries) == (12.5, 0)


# valores fora dos limites ou ilegíveis geram ConfigError citando a variável
@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_TIMEOUT_S", "abc"),
        ("LLM_TIMEOUT_S", "0"),
        ("LLM_TIMEOUT_S", "601"),
        ("LLM_MAX_RETRIES", "-1"),
        ("LLM_MAX_RETRIES", "11"),
        ("LLM_MAX_RETRIES", "1.5"),
    ],
)
def test_invalid_timeout_or_retries_name_the_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value})


# a espera entre chamadas reais da comparação tem padrão 20 segundos
def test_min_interval_default_and_override():
    assert load_settings({}).min_interval_s == 20.0
    assert load_settings({"LLM_MIN_INTERVAL_S": "0"}).min_interval_s == 0.0
    assert load_settings({"LLM_MIN_INTERVAL_S": "7.5"}).min_interval_s == 7.5


# espera negativa ou ilegível gera ConfigError citando a variável
@pytest.mark.parametrize("value", ["-1", "abc", "inf", "nan"])
def test_invalid_min_interval_names_the_variable(value):
    with pytest.raises(ConfigError, match="LLM_MIN_INTERVAL_S"):
        load_settings({"LLM_MIN_INTERVAL_S": value})


# a configuração é imutável depois de carregada
def test_settings_are_immutable():
    settings = load_settings({})
    with pytest.raises(ValidationError):
        settings.model = "outro"


# toda variável aparece uma única vez e cada parâmetro do contrato tem a sua
def test_every_generation_param_has_an_environment_variable():
    assert set(GenerationParams.model_fields) <= set(ENV_NAMES)
    assert len(set(ENV_NAMES.values())) == len(ENV_NAMES)


# variáveis de ambiente genéricas não colidem com as de outros programas
def test_environment_names_are_prefixed():
    for field, name in ENV_NAMES.items():
        assert name.startswith(("LLM_", "GROQ_")), field


# o arquivo .env é lido quando não se passa o ambiente
def test_env_file_is_read(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_MODEL=openai/gpt-oss-120b\nLLM_SEED=9\nGROQ_API_KEY=chave-do-arquivo\n"
    )
    settings = load_settings(env_file=env_file)
    assert settings.model == "openai/gpt-oss-120b"
    assert settings.params.seed == 9
    assert settings.require_groq_api_key() == "chave-do-arquivo"


# variável do processo vence a do arquivo
def test_process_environment_wins_over_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=do-arquivo\n")
    monkeypatch.setenv("LLM_MODEL", "do-processo")
    assert load_settings(env_file=env_file).model == "do-processo"


# arquivo .env inexistente não é erro
def test_missing_env_file_is_not_an_error(tmp_path):
    assert load_settings(env_file=tmp_path / "nao_existe").model == DEFAULTS["model"]


# ler o arquivo não altera o ambiente do processo
def test_loading_does_not_mutate_the_process_environment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=do-arquivo\n")
    load_settings(env_file=env_file)
    assert "LLM_MODEL" not in os.environ


# o .env.example tem todas as variáveis, com os mesmos padrões do código
def test_env_example_matches_the_code_defaults():
    values = dotenv_values(PROJECT_ROOT / ".env.example")
    expected = {ENV_NAMES[field]: default for field, default in DEFAULTS.items()}
    expected[ENV_NAMES["groq_api_key"]] = ""
    assert values == expected


# o .env.example não carrega chave real, prefixo montado em partes para a varredura de segredos
def test_env_example_has_no_secret():
    text = (PROJECT_ROOT / ".env.example").read_text()
    assert "gsk" + "_" not in text
    assert dotenv_values(PROJECT_ROOT / ".env.example")["GROQ_API_KEY"] == ""
