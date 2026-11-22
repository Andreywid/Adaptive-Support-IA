import argparse
import sys
from pathlib import Path

from support_assistant.cache import ResponseCache
from support_assistant.config import ConfigError, load_settings
from support_assistant.credentials import redact_credentials
from support_assistant.data_loader import (
    DataError,
    get_profile,
    get_ticket,
    load_profiles,
    load_tickets,
    ticket_from_text,
)
from support_assistant.evaluation.compare import (
    ComparisonError,
    load_comparison_argument,
    run_comparison,
    save_comparison,
)
from support_assistant.evaluation.review import evaluate_blind
from support_assistant.evaluation.rubric import has_complete_evaluation, rubric_summary
from support_assistant.formatting import format_output
from support_assistant.generators.dispatch import generate_content
from support_assistant.models import ContentType, Profile, RunStatus, Ticket
from support_assistant.prompts.blocks import CHANNEL_LABELS, LEVEL_LABELS, STYLE_LABELS
from support_assistant.prompts.engine import DEFAULT_VERSION, PromptError, list_versions
from support_assistant.providers.base import ProviderError
from support_assistant.providers.factory import make_provider
from support_assistant.storage import build_run_record, list_runs, save_run

KNOWN_ERRORS = (
    ConfigError,
    DataError,
    PromptError,
    ProviderError,
    ComparisonError,
)
INVALID_OUTPUT_TEXT = "Saída fora do contrato. O texto bruto está no registro."
TICKET_PREVIEW_CHARS = 70
FREE_TEXT_OPTION = "texto livre"
FAKE_NOTICE = "Simulação offline: o provedor fake devolve exemplos fixos."


# caminho relativo à pasta atual quando possível, senão o caminho completo
def _display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return str(path)


# ticket do arquivo de dados pelo id, ou ticket de texto livre
def _resolve_ticket(ticket_id: str | None, ticket_text: str | None) -> Ticket:
    return get_ticket(ticket_id) if ticket_id is not None else ticket_from_text(ticket_text or "")


# gera um conteúdo, grava o registro e imprime a saída, o cache e o caminho do JSON
def _run_generation(
    *,
    profile: Profile,
    ticket: Ticket,
    content_type: ContentType,
    version: str,
    provider_name: str | None = None,
    no_cache: bool = False,
    refresh: bool = False,
) -> int:
    settings = load_settings()
    provider = make_provider(settings, provider_name)
    cache = None if no_cache else ResponseCache()
    result = generate_content(
        content_type,
        profile=profile,
        ticket=ticket,
        version=version,
        provider=provider,
        model=settings.model,
        params=settings.params,
        cache=cache,
        refresh=refresh,
    )
    record = build_run_record(result, profile, ticket)
    path = save_run(record)
    if result.status is RunStatus.OK and result.output is not None:
        body = format_output(result.content_type, result.output)
    else:
        reasons = "\n".join(f"- {error}" for error in result.validation_errors)
        body = f"{INVALID_OUTPUT_TEXT}\n{reasons}" if reasons else INVALID_OUTPUT_TEXT
    if result.cache_hit:
        cache_state = "acerto"
    elif no_cache:
        cache_state = "desligado"
    else:
        cache_state = "gerado agora"
    print(f"{body}\n\ncache: {cache_state}\nstatus: {result.status.value}")
    if result.provider == "fake":
        print(FAKE_NOTICE)
    print(f"registro: {_display_path(path)}")
    return 0 if result.status is RunStatus.OK else 1


# comando generate: gera um tipo de conteúdo para um perfil e um ticket
def _cmd_generate(args: argparse.Namespace) -> int:
    return _run_generation(
        profile=get_profile(args.profile),
        ticket=_resolve_ticket(args.ticket, args.ticket_text),
        content_type=ContentType(args.type),
        version=args.version,
        provider_name=args.provider,
        no_cache=args.no_cache,
        refresh=args.refresh,
    )


# comando profiles: lista os perfis em colunas
def _cmd_profiles(args: argparse.Namespace) -> int:
    rows = [("id", "nome", "nível", "estilo", "canal")]
    for profile in load_profiles().profiles:
        rows.append(
            (
                profile.id,
                profile.name,
                LEVEL_LABELS[profile.technical_level],
                STYLE_LABELS[profile.communication_style],
                CHANNEL_LABELS[profile.channel],
            )
        )
    for ident, name, level, style, channel in rows:
        print(f"{ident:<12} {name:<8} {level:<14} {style:<10} {channel}")
    return 0


# ticket em uma linha, cortado no tamanho da prévia
def _preview(text: str) -> str:
    one_line = " ".join(text.split())
    if len(one_line) > TICKET_PREVIEW_CHARS:
        return one_line[:TICKET_PREVIEW_CHARS] + "..."
    return one_line


# comando tickets: lista os tickets sem mostrar credenciais
def _cmd_tickets(args: argparse.Namespace) -> int:
    for ticket in load_tickets().tickets:
        print(f"{ticket.id:<20} {_preview(redact_credentials(ticket.text))}")
    return 0


# comando history: lista as execuções gravadas, com filtros
def _cmd_history(args: argparse.Namespace) -> int:
    runs = list_runs(
        customer_id=args.profile,
        ticket_id=args.ticket,
        content_type=ContentType(args.type) if args.type else None,
        prompt_version=args.version,
        cache_key=args.cache_key,
    )
    for _, record in list(reversed(runs))[: args.limit]:
        cache = "acerto" if record.cache_hit else "gerado"
        print(
            f"{record.timestamp:%Y-%m-%d %H:%M}  {record.run_id}  {record.customer_id}  "
            f"{record.ticket_id}  {record.content_type.value:<9}  {record.prompt_version}  "
            f"{record.status.value:<14}  {cache}"
        )
    print(f"{len(runs)} execuções em runs/")
    return 0


# comando compare: gera as duas versões no mesmo cenário e grava a comparação
def _cmd_compare(args: argparse.Namespace) -> int:
    settings = load_settings()
    profile = get_profile(args.profile)
    ticket = _resolve_ticket(args.ticket, args.ticket_text)
    provider = make_provider(settings, args.provider)
    content_types = [ContentType(item) for item in args.types] if args.types else list(ContentType)
    record = run_comparison(
        profile=profile,
        ticket=ticket,
        versions=args.versions,
        seeds=args.seeds,
        content_types=content_types,
        provider=provider,
        settings=settings,
        cache=ResponseCache(),
    )
    if provider.name == "fake":
        print(FAKE_NOTICE)
    print(f"comparação: {_display_path(save_comparison(record))}")
    return 0


# comando evaluate: avaliação cega de uma comparação, salva a cada par
def _cmd_evaluate(args: argparse.Namespace) -> int:
    record, path = load_comparison_argument(args.comparison_id)
    evaluate_blind(
        record, seed=args.seed, ask=input, save=lambda item: save_comparison(item, path=path)
    )
    print(f"comparação: {_display_path(path)}")
    return 0


# comando checks: confere se há uma seed com todos os tipos avaliados nas duas versões
def _cmd_checks(args: argparse.Namespace) -> int:
    record, path = load_comparison_argument(args.comparison_id)
    if not has_complete_evaluation(record):
        raise ComparisonError("Falta uma seed com a avaliação das duas versões de todos os tipos.")
    print(f"Avaliação completa em {_display_path(path)}")
    print(rubric_summary(record))
    return 0


# mostra uma lista numerada e repete a pergunta até a escolha ser válida
def _choose(title: str, options: list[str]) -> int:
    print(f"{title}:")
    for number, label in enumerate(options, 1):
        print(f"  {number}) {label}")
    while True:
        answer = input(f"Escolha [1-{len(options)}]: ").strip()
        if not answer.isdecimal():
            continue
        number = int(answer)
        if 1 <= number <= len(options):
            return number - 1


# menu interativo: pergunta perfil, ticket, tipo e versão e gera
def _menu() -> int:
    profiles = load_profiles().profiles
    labels = [
        f"{p.id} ({p.name}: {LEVEL_LABELS[p.technical_level]}, "
        f"{STYLE_LABELS[p.communication_style]}, {CHANNEL_LABELS[p.channel]})"
        for p in profiles
    ]
    profile = profiles[_choose("Perfil", labels)]
    tickets = load_tickets().tickets
    picked = _choose("Ticket", [t.id for t in tickets] + [FREE_TEXT_OPTION])
    if picked < len(tickets):
        ticket = tickets[picked]
    else:
        ticket = ticket_from_text(input("Digite o ticket em uma linha: "))
    content_types = list(ContentType)
    content_type = content_types[_choose("Tipo", [item.value for item in content_types])]
    versions = list_versions()
    version = versions[_choose("Versão", versions)]
    return _run_generation(
        profile=profile,
        ticket=ticket,
        content_type=content_type,
        version=version,
    )


# argumentos de ticket: id do arquivo ou texto livre, um dos dois
def _add_ticket_arguments(parser: argparse.ArgumentParser) -> None:
    ticket = parser.add_mutually_exclusive_group(required=True)
    ticket.add_argument("--ticket", help="id de um ticket do arquivo de dados")
    ticket.add_argument("--ticket-text", help="texto livre de um ticket")


# monta o parser com todos os subcomandos
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m support_assistant",
        description="Assistente de suporte técnico com prompts versionados.",
    )
    commands = parser.add_subparsers(dest="command")

    generate_parser = commands.add_parser(
        "generate", help="gera um conteúdo para um perfil e um ticket"
    )
    generate_parser.add_argument("--profile", required=True, help="id do perfil, como p01_marta")
    _add_ticket_arguments(generate_parser)
    generate_parser.add_argument(
        "--type",
        required=True,
        choices=[item.value for item in ContentType],
        help="tipo de conteúdo",
    )
    generate_parser.add_argument(
        "--version", default=DEFAULT_VERSION, help=f"versão do prompt, padrão {DEFAULT_VERSION}"
    )
    generate_parser.add_argument(
        "--provider", choices=["groq", "fake"], help="troca o provedor do .env"
    )
    cache = generate_parser.add_mutually_exclusive_group()
    cache.add_argument("--no-cache", action="store_true", help="não lê nem grava o cache")
    cache.add_argument(
        "--refresh", action="store_true", help="ignora o cache, mas grava o resultado"
    )
    generate_parser.set_defaults(handler=_cmd_generate)

    commands.add_parser("profiles", help="lista os perfis").set_defaults(handler=_cmd_profiles)
    commands.add_parser("tickets", help="lista os tickets").set_defaults(handler=_cmd_tickets)

    history = commands.add_parser("history", help="lista as execuções gravadas")
    history.add_argument("--profile", help="filtra pelo id do perfil")
    history.add_argument("--ticket", help="filtra pelo id do ticket")
    history.add_argument(
        "--type", choices=[item.value for item in ContentType], help="filtra pelo tipo"
    )
    history.add_argument("--version", help="filtra pela versão do prompt")
    history.add_argument("--cache-key", help="filtra pela chave de cache")
    history.add_argument(
        "--limit", type=_positive_int, default=20, help="máximo de linhas, padrão 20"
    )
    history.set_defaults(handler=_cmd_history)

    compare = commands.add_parser("compare", help="compara duas versões de prompt")
    compare.add_argument("--profile", required=True, help="id do perfil")
    _add_ticket_arguments(compare)
    compare.add_argument(
        "--versions",
        nargs=2,
        default=["v1", DEFAULT_VERSION],
        metavar=("VA", "VB"),
        help=f"padrão v1 {DEFAULT_VERSION}",
    )
    compare.add_argument("--seeds", nargs="+", type=int, default=[1], help="seeds, padrão 1")
    compare.add_argument(
        "--types",
        nargs="+",
        choices=[item.value for item in ContentType],
        help="tipos, padrão todos",
    )
    compare.add_argument("--provider", choices=["groq", "fake"], help="troca o provedor do .env")
    compare.set_defaults(handler=_cmd_compare)

    evaluate = commands.add_parser("evaluate", help="registra a avaliação às cegas")
    evaluate.add_argument("comparison_id", help="id da comparação ou caminho do JSON")
    evaluate.add_argument("--seed", type=int, default=1, help="seed a avaliar, padrão 1")
    evaluate.set_defaults(handler=_cmd_evaluate)

    checks = commands.add_parser("checks", help="confere se a avaliação está completa")
    checks.add_argument("comparison_id", help="id da comparação ou caminho do JSON")
    checks.set_defaults(handler=_cmd_checks)
    return parser


# converte o argumento em inteiro positivo, para o argparse recusar o resto
def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("precisa ser maior que zero")
    return value


# ponto de entrada: sem subcomando abre o menu; erros conhecidos viram mensagem curta e código 1
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command is None:
            return _menu()
        return args.handler(args)
    except (KeyboardInterrupt, EOFError):
        print("Interrompido.", file=sys.stderr)
        return 130
    except KNOWN_ERRORS as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        target = f" ({error.filename})" if error.filename else ""
        reason = error.strerror or type(error).__name__
        print(f"Erro ao acessar o armazenamento{target}: {reason}", file=sys.stderr)
        return 1
