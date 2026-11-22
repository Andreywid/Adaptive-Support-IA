import random
import re
import secrets
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from support_assistant.cache import ResponseCache, hash_schema
from support_assistant.config import PROJECT_ROOT, Settings
from support_assistant.credentials import sanitize_ticket
from support_assistant.data_loader import get_profile
from support_assistant.evaluation.rubric import format_rubric
from support_assistant.files import atomic_write_json
from support_assistant.generators.dispatch import generate_content
from support_assistant.models import (
    SCHEMA_VERSION,
    BlindItem,
    ComparisonRecord,
    ComparisonRun,
    ContentType,
    LockedConfig,
    Profile,
    Ticket,
    output_json_schema,
)
from support_assistant.prompts.blocks import CHANNEL_LABELS, LEVEL_LABELS, STYLE_LABELS
from support_assistant.prompts.engine import PromptError, build_messages
from support_assistant.providers.base import Provider, ProviderError
from support_assistant.providers.pacing import SpacedProvider
from support_assistant.storage import build_run_record, save_run

COMPARISONS_DIR = PROJECT_ROOT / "comparisons"
COMPARISON_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


# erro de comparação com mensagem pronta para o usuário
class ComparisonError(Exception):
    pass


# UTC facilita ordenar os registros e o sufixo diferencia execuções simultâneas.
def new_comparison_id(now: datetime | None = None) -> str:
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return moment.strftime("cmp_%Y%m%dT%H%M%SZ_") + secrets.token_hex(6)


# recusa cenário inválido antes de gastar qualquer chamada
def _check_scenario(
    versions: list[str], seeds: list[int], content_types: list[ContentType]
) -> None:
    if (
        len(versions) != 2
        or not all(isinstance(v, str) for v in versions)
        or versions[0] == versions[1]
    ):
        raise ComparisonError("A comparação precisa de exatamente 2 versões diferentes.")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ComparisonError("As seeds da comparação precisam ser inteiras e não negativas.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ComparisonError("A comparação precisa de ao menos 1 seed, sem repetição.")
    if any(not isinstance(content_type, ContentType) for content_type in content_types):
        raise ComparisonError("A comparação recebeu um tipo de conteúdo inválido.")
    if not content_types or len(set(content_types)) != len(content_types):
        raise ComparisonError("A comparação precisa de ao menos 1 tipo, sem repetição.")


# monta os prompts de todas as versões e tipos antes de chamar o provedor
def _validate_prompts(
    versions: list[str], content_types: list[ContentType], profile: Profile, ticket: Ticket
) -> None:
    for version in versions:
        try:
            for content_type in content_types:
                build_messages(version, content_type, profile, ticket)
        except PromptError as error:
            raise ComparisonError(f"Prompt inválido para a comparação: {error}") from None


# sorteia qual versão é A e qual é B em cada par de seed e tipo, de forma reproduzível
def _blind_map(
    versions: list[str], seeds: list[int], content_types: list[ContentType], blind_seed: int
) -> list[BlindItem]:
    rng = random.Random(blind_seed)
    items = []
    for seed in seeds:
        for content_type in content_types:
            order = list(versions)
            rng.shuffle(order)
            items.append(BlindItem(content_type=content_type, seed=seed, a=order[0], b=order[1]))
    return items


# gera as duas versões em cada seed e tipo, com parâmetros travados, e grava as execuções
def run_comparison(
    *,
    profile: Profile,
    ticket: Ticket,
    versions: list[str],
    seeds: list[int],
    content_types: list[ContentType],
    provider: Provider,
    settings: Settings,
    cache: ResponseCache | None,
    runs_dir: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    show: Callable[[str], None] = print,
    now: datetime | None = None,
) -> ComparisonRecord:
    _check_scenario(versions, seeds, content_types)
    _validate_prompts(versions, content_types, profile, ticket)
    if provider.name != "fake":
        provider = SpacedProvider(provider, settings.min_interval_s, sleep)
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    total = len(seeds) * len(content_types) * len(versions)
    runs: list[ComparisonRun] = []
    for seed in seeds:
        params = settings.params.model_copy(update={"seed": seed})
        for content_type in content_types:
            for version in versions:
                try:
                    result = generate_content(
                        content_type,
                        profile=profile,
                        ticket=ticket,
                        version=version,
                        provider=provider,
                        model=settings.model,
                        params=params,
                        cache=cache,
                    )
                except ProviderError as error:
                    raise ComparisonError(
                        f"Comparação interrompida em {version} {content_type.value} seed {seed}: "
                        f"{error}. Rode o mesmo comando para retomar; o cache evita repetir as "
                        "chamadas já feitas."
                    ) from None
                record = build_run_record(result, profile, ticket)
                save_run(record, runs_dir)
                runs.append(
                    ComparisonRun(
                        version=version,
                        seed=seed,
                        content_type=content_type,
                        run_id=record.run_id,
                        cache_hit=result.cache_hit,
                        status=result.status,
                        output=result.output,
                        raw_output=result.raw_output,
                        prompt_hash=result.prompt_hash,
                    )
                )
                state = "acerto" if result.cache_hit else "gerado agora"
                step = f"[{len(runs)}/{total}] {version} {content_type.value} seed {seed}"
                show(f"{step}: {state} ({result.status.value})")
    blind_seed = secrets.randbelow(10**6)
    return ComparisonRecord(
        schema_version=SCHEMA_VERSION,
        comparison_id=new_comparison_id(moment),
        created_at=moment,
        customer_id=profile.id,
        ticket_id=ticket.id,
        ticket=sanitize_ticket(ticket).text,
        versions=versions,
        seeds=seeds,
        content_types=content_types,
        locked_config=LockedConfig(
            provider=provider.name,
            model=settings.model,
            temperature=settings.params.temperature,
            top_p=settings.params.top_p,
            max_tokens=settings.params.max_tokens,
            reasoning_effort=settings.params.reasoning_effort,
            include_reasoning=settings.params.include_reasoning,
        ),
        schema_hashes={item: hash_schema(output_json_schema(item)) for item in content_types},
        runs=runs,
        blind_seed=blind_seed,
        blind_map=_blind_map(versions, seeds, content_types, blind_seed),
        evaluations=[],
        profile=profile,
        rubric=format_rubric(),
    )


# grava a comparação em JSON, de forma atômica
def save_comparison(
    record: ComparisonRecord, directory: Path | None = None, *, path: Path | None = None
) -> Path:
    path = path or (directory or COMPARISONS_DIR) / f"{record.comparison_id}.json"
    atomic_write_json(path, record.model_dump(mode="json"))
    return path


# abre uma comparação pelo caminho do arquivo ou pelo id
def load_comparison_argument(
    value: str, directory: Path | None = None
) -> tuple[ComparisonRecord, Path]:
    path = Path(value)
    explicit_path = path.suffix.lower() == ".json"
    if not explicit_path:
        if not COMPARISON_ID.fullmatch(value):
            raise ComparisonError(f"Identificador de comparação inválido: {value}")
        path = (directory or COMPARISONS_DIR) / f"{value}.json"
    try:
        return ComparisonRecord.model_validate_json(path.read_text(encoding="utf-8")), path
    except FileNotFoundError:
        if explicit_path:
            raise ComparisonError(f"Arquivo de comparação '{path}' não encontrado.") from None
        raise ComparisonError(f"Comparação '{value}' não encontrada em comparisons/") from None
    except (OSError, ValueError) as error:
        raise ComparisonError(
            f"Comparação '{path.name}' ilegível: {type(error).__name__}"
        ) from None


# perfil gravado na comparação, ou o do arquivo de dados nas comparações antigas
def comparison_profile(record: ComparisonRecord) -> Profile:
    return record.profile or get_profile(record.customer_id)


# perfil em uma linha: nome, nível, estilo e canal
def describe_profile(profile: Profile) -> str:
    return (
        f"{profile.name}, {LEVEL_LABELS[profile.technical_level]}, "
        f"{STYLE_LABELS[profile.communication_style]}, {CHANNEL_LABELS[profile.channel]}"
    )


# as execuções que aparecem como A e B num par de seed e tipo
def comparison_pair(
    record: ComparisonRecord, seed: int, content_type: ContentType
) -> tuple[ComparisonRun, ComparisonRun]:
    matches = [b for b in record.blind_map if b.content_type is content_type and b.seed == seed]
    runs = [r for r in record.runs if r.content_type is content_type and r.seed == seed]
    if len(matches) == 1 and len(runs) == 2:
        item = matches[0]
        by_version = {run.version: run for run in runs}
        if item.a != item.b and {item.a, item.b} == by_version.keys():
            return by_version[item.a], by_version[item.b]
    raise ComparisonError(f"Par incompleto para {content_type.value} na seed {seed}.")
