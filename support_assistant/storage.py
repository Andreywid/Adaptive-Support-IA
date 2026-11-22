import logging
import secrets
from datetime import UTC, datetime
from pathlib import Path

from support_assistant.config import PROJECT_ROOT
from support_assistant.credentials import sanitize_ticket
from support_assistant.files import atomic_write_json
from support_assistant.generators.core import GenerationResult
from support_assistant.models import SCHEMA_VERSION, ContentType, Profile, RunRecord, Ticket

RUNS_DIR = PROJECT_ROOT / "runs"

logger = logging.getLogger(__name__)


# identificador de execução: data e hora em UTC mais 6 hexadecimais aleatórios
def new_run_id(now: datetime | None = None) -> str:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ_") + secrets.token_hex(3)


# converte o resultado de uma geração em registro de execução, sem avaliação
def build_run_record(
    result: GenerationResult,
    profile: Profile,
    ticket: Ticket,
    now: datetime | None = None,
) -> RunRecord:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return RunRecord(
        schema_version=SCHEMA_VERSION,
        run_id=new_run_id(moment),
        timestamp=moment,
        customer_id=profile.id,
        ticket_id=ticket.id,
        ticket=sanitize_ticket(ticket).text,
        evaluation=None,
        **result.model_dump(),
    )


# grava o registro em <pasta>/<run_id>.json, de forma atômica
def save_run(record: RunRecord, directory: Path | None = None) -> Path:
    path = (directory or RUNS_DIR) / f"{record.run_id}.json"
    atomic_write_json(path, record.model_dump(mode="json"))
    return path


# lê e valida um registro de execução
def load_run(path: Path) -> RunRecord:
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


# lista as execuções que passam nos filtros, da mais antiga para a mais recente
def list_runs(
    directory: Path | None = None,
    *,
    customer_id: str | None = None,
    ticket_id: str | None = None,
    content_type: ContentType | None = None,
    prompt_version: str | None = None,
    cache_key: str | None = None,
) -> list[tuple[Path, RunRecord]]:
    folder = directory or RUNS_DIR
    if not folder.exists():
        return []
    found: list[tuple[Path, RunRecord]] = []
    for path in folder.glob("*.json"):
        try:
            record = load_run(path)
        except (OSError, ValueError) as error:
            logger.warning("runs: arquivo ignorado em %s: %s", path.name, type(error).__name__)
            continue
        if customer_id is not None and record.customer_id != customer_id:
            continue
        if ticket_id is not None and record.ticket_id != ticket_id:
            continue
        if content_type is not None and record.content_type is not content_type:
            continue
        if prompt_version is not None and record.prompt_version != prompt_version:
            continue
        if cache_key is not None and record.cache_key != cache_key:
            continue
        found.append((path, record))
    return sorted(found, key=lambda item: (item[1].timestamp, item[1].run_id))
