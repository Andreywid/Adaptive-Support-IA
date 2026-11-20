import json
from pathlib import Path

from pydantic import ValidationError

from support_assistant.config import PROJECT_ROOT
from support_assistant.models import (
    TICKET_MAX_CHARS,
    Profile,
    ProfileCatalog,
    Ticket,
    TicketCatalog,
)

DATA_DIR = PROJECT_ROOT / "data"
PROFILES_FILE = DATA_DIR / "profiles.json"
TICKETS_FILE = DATA_DIR / "tickets.json"
FREE_TEXT_TICKET_ID = "texto_livre"


# erro de dados com mensagem pronta para o usuário
class DataError(Exception):
    pass


# caminho relativo à raiz do projeto, quando possível
def _relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


# uma linha por erro de validação, no formato "  campo: motivo"
def _describe(error: ValidationError) -> str:
    lines = []
    for item in error.errors(include_input=False, include_url=False):
        field = ".".join(str(part) for part in item["loc"]) or "arquivo"
        lines.append(f"  {field}: {item['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


# lê o arquivo e devolve o JSON já convertido, com erro claro para ausente ou inválido
def _read_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise DataError(f"Arquivo de dados não encontrado: {_relative(path)}") from None
    except OSError as error:
        raise DataError(f"Não foi possível ler {_relative(path)}: {error.strerror}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise DataError(
            f"JSON inválido em {_relative(path)}, linha {error.lineno}: {error.msg}"
        ) from None


# carrega o catálogo de perfis, validado pelo contrato
def load_profiles(path: Path | None = None) -> ProfileCatalog:
    path = path or PROFILES_FILE
    data = _read_json(path)
    try:
        return ProfileCatalog.model_validate(data)
    except ValidationError as error:
        raise DataError(f"Dados inválidos em {_relative(path)}:\n{_describe(error)}") from None


# carrega o catálogo de tickets, validado pelo contrato
def load_tickets(path: Path | None = None) -> TicketCatalog:
    path = path or TICKETS_FILE
    data = _read_json(path)
    try:
        return TicketCatalog.model_validate(data)
    except ValidationError as error:
        raise DataError(f"Dados inválidos em {_relative(path)}:\n{_describe(error)}") from None


# acha um perfil pelo id, listando os disponíveis quando não existe
def get_profile(profile_id: str, catalog: ProfileCatalog | None = None) -> Profile:
    catalog = catalog or load_profiles()
    for profile in catalog.profiles:
        if profile.id == profile_id:
            return profile
    available = ", ".join(profile.id for profile in catalog.profiles)
    raise DataError(f"Perfil '{profile_id}' não existe. Disponíveis: {available}")


# acha um ticket pelo id, listando os disponíveis quando não existe
def get_ticket(ticket_id: str, catalog: TicketCatalog | None = None) -> Ticket:
    catalog = catalog or load_tickets()
    for ticket in catalog.tickets:
        if ticket.id == ticket_id:
            return ticket
    available = ", ".join(ticket.id for ticket in catalog.tickets)
    raise DataError(f"Ticket '{ticket_id}' não existe. Disponíveis: {available}")


# cria um ticket a partir de texto livre digitado pelo usuário
def ticket_from_text(text: str) -> Ticket:
    try:
        return Ticket(id=FREE_TEXT_TICKET_ID, text=text)
    except ValidationError:
        raise DataError(
            f"O texto do ticket precisa ter de 1 a {TICKET_MAX_CHARS} caracteres."
        ) from None
