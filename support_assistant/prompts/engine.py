import re
import tomllib
from datetime import date
from pathlib import Path
from string import Template
from typing import Annotated

from pydantic import Field, ValidationError, model_validator

from support_assistant.models import (
    Channel,
    CommunicationStyle,
    ContentType,
    Message,
    NonEmptyText,
    Profile,
    PromptVersion,
    StrictModel,
    TechnicalLevel,
    Ticket,
)
from support_assistant.prompts.blocks import render_profile, render_ticket

VERSIONS_DIR = Path(__file__).parent / "versions"
VERSION_FILE = re.compile(r"v([0-9]+)\.toml")
DEFAULT_VERSION = "v3"


# erro de prompt com mensagem pronta para o usuário
class PromptError(Exception):
    pass


# dados de rastreabilidade da versão, que não entram nas mensagens
class PromptMeta(StrictModel):
    id: PromptVersion
    parent: PromptVersion | None = None
    created: date
    hypothesis: NonEmptyText
    changes: Annotated[list[NonEmptyText], Field(min_length=1)]


# bloco de texto literal, sem placeholders
class TextBlock(StrictModel):
    text: NonEmptyText


# bloco de template com placeholders ${nome}
class TemplateBlock(StrictModel):
    template: NonEmptyText


# regras por valor de cada dimensão do perfil, cada dimensão com todos os valores do enum
class ProfileRules(StrictModel):
    channel: dict[Channel, str] | None = None
    technical_level: dict[TechnicalLevel, str] | None = None
    communication_style: dict[CommunicationStyle, str] | None = None

    # exige todos os valores do enum em cada dimensão presente
    @model_validator(mode="after")
    def _check_complete(self) -> "ProfileRules":
        for name, enum in (
            ("channel", Channel),
            ("technical_level", TechnicalLevel),
            ("communication_style", CommunicationStyle),
        ):
            by_value = getattr(self, name)
            if by_value is not None:
                missing = sorted(item.value for item in set(enum) - set(by_value))
                if missing:
                    raise ValueError(f"regra faltando em {name}: {', '.join(missing)}")
        return self


# bloco do perfil: template e regras opcionais por dimensão
class ProfileBlock(StrictModel):
    template: NonEmptyText
    rules: ProfileRules = Field(default_factory=ProfileRules)


# instrução de cada tipo de conteúdo
class TasksBlock(StrictModel):
    diagnosis: NonEmptyText
    steps: NonEmptyText
    questions: NonEmptyText
    flow: NonEmptyText


# versão de prompt completa, como lida do arquivo TOML
class PromptSpec(StrictModel):
    meta: PromptMeta
    system: TextBlock
    user: TemplateBlock
    profile: ProfileBlock
    ticket: TemplateBlock
    tasks: TasksBlock


# versões disponíveis em ordem numérica: v1, v2, v10
def list_versions(directory: Path | None = None) -> list[str]:
    folder = directory or VERSIONS_DIR
    found = [m for p in folder.glob("v*.toml") if (m := VERSION_FILE.fullmatch(p.name))]
    return [f"v{number}" for number in sorted(int(m.group(1)) for m in found)]


# lê, valida e devolve uma versão, com erro claro para ausente, inválida ou inconsistente
def load_version(name: str, directory: Path | None = None) -> PromptSpec:
    folder = directory or VERSIONS_DIR
    path = folder / f"{name}.toml"
    if not re.fullmatch(r"v[0-9]+", name) or not path.is_file():
        available = ", ".join(list_versions(folder))
        raise PromptError(f"Versão de prompt '{name}' não existe. Disponíveis: {available}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PromptError(f"TOML inválido em {path.name}: {error}") from None
    try:
        spec = PromptSpec.model_validate(data)
    except ValidationError as error:
        lines = []
        for item in error.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in item["loc"]) or "arquivo"
            lines.append(f"  {field}: {item['msg'].removeprefix('Value error, ')}")
        raise PromptError(f"Versão {name} fora do contrato:\n" + "\n".join(lines)) from None
    if spec.meta.id != name:
        raise PromptError(f"meta.id '{spec.meta.id}' difere do nome do arquivo {path.name}")
    return spec


# troca os placeholders do template, falhando se faltar algum valor
def _substitute(template: str, version: str, **values: str) -> str:
    try:
        return Template(template).substitute(**values)
    except KeyError as error:
        raise PromptError(
            f"Placeholder ${{{error.args[0]}}} sem valor na versão {version}"
        ) from None
    except ValueError as error:
        raise PromptError(f"Template inválido na versão {version}: {error}") from None


# monta as mensagens de sistema e de usuário de uma versão para um perfil e um ticket
def build_messages(
    version: str,
    content_type: ContentType,
    profile: Profile,
    ticket: Ticket,
    directory: Path | None = None,
) -> list[Message]:
    spec = load_version(version, directory)
    try:
        profile_text = render_profile(spec.profile, profile)
        ticket_text = render_ticket(spec.ticket, ticket)
    except KeyError as error:
        raise PromptError(
            f"Placeholder ${{{error.args[0]}}} sem valor na versão {version}"
        ) from None
    except ValueError as error:
        raise PromptError(f"Template inválido na versão {version}: {error}") from None
    task = getattr(spec.tasks, content_type.value).strip()
    user = _substitute(
        spec.user.template,
        version,
        profile=profile_text,
        ticket=ticket_text,
        task=task,
    )
    user = re.sub(r"\n{3,}", "\n\n", user).strip()
    return [
        Message(role="system", content=spec.system.text.strip()),
        Message(role="user", content=user),
    ]
