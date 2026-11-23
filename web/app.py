# ruff: noqa: E402
import sys
from pathlib import Path

# Permite iniciar o Streamlit a partir da raiz do projeto.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from support_assistant.cache import ResponseCache
from support_assistant.config import PROJECT_ROOT, ConfigError, Settings, load_settings
from support_assistant.data_loader import DataError, load_profiles, load_tickets, ticket_from_text
from support_assistant.evaluation.compare import describe_profile
from support_assistant.generators.dispatch import generate_content
from support_assistant.models import TICKET_MAX_CHARS, ContentType, Profile, RunRecord, Ticket
from support_assistant.prompts.engine import DEFAULT_VERSION, PromptError, list_versions
from support_assistant.providers.base import Provider, ProviderError
from support_assistant.providers.factory import make_provider
from support_assistant.storage import build_run_record, save_run
from web.archive import show_comparisons, show_examples
from web.presentation import CONTENT_LABELS, PROMPT_LABELS, display_path, show_output, ticket_label

FREE_TEXT = "texto livre"
LAST_GENERATION = "last_generation"


# gera com as mesmas funções da linha de comando e grava o registro
def _run_generation(
    settings: Settings,
    provider: Provider,
    profile: Profile,
    ticket: Ticket,
    content_type: ContentType,
    version: str,
) -> tuple[RunRecord, Path | None]:
    result = generate_content(
        content_type,
        profile=profile,
        ticket=ticket,
        version=version,
        provider=provider,
        model=settings.model,
        params=settings.params,
        cache=ResponseCache(),
    )
    record = build_run_record(result, profile, ticket)
    try:
        return record, save_run(record)
    except OSError:
        return record, None


# mostra o último resultado, o caminho do registro e o botão de download
def _show_result(record: RunRecord, path: Path | None, profile_name: str) -> None:
    st.subheader("Último resultado", anchor=False)
    st.caption(f"{CONTENT_LABELS[record.content_type]} · {profile_name}")
    if path is None:
        st.error("Não foi possível gravar o registro local. Baixe o JSON para guardá-lo.")
    else:
        st.success(f"JSON salvo em {display_path(path, PROJECT_ROOT)}")
    show_output(record)
    st.download_button(
        "Baixar registro JSON",
        data=record.model_dump_json(indent=2),
        file_name=f"{record.run_id}.json",
        mime="application/json",
        icon=":material/download:",
        on_click="ignore",
    )

    with st.expander("Detalhes da execução"):
        st.text(record.ticket)
        version = PROMPT_LABELS[record.prompt_version]
        st.caption(f"{version} · {record.provider} / {record.model}")
        cache = "acerto" if record.cache_hit else "nova geração"
        st.caption(f"Cache: {cache}")


# aba Atendimento: escolhe perfil, ticket, tipo e versão e gera
def _show_generation() -> None:
    st.header("Novo atendimento", anchor=False)
    try:
        settings = load_settings()
        profiles = {profile.id: profile for profile in load_profiles().profiles}
        tickets = {ticket.id: ticket for ticket in load_tickets().tickets}
    except (ConfigError, DataError) as error:
        st.error(str(error))
        return

    versions = [version for version in list_versions() if version in PROMPT_LABELS]
    if not profiles or not versions:
        st.warning("Cadastre um perfil e uma versão de prompt para gerar.")
        return

    left, right = st.columns(2)
    profile_id = left.selectbox(
        "Perfil do cliente",
        list(profiles),
        format_func=lambda key: describe_profile(profiles[key]),
        key="profile",
    )
    content_type = right.selectbox(
        "Tipo de resposta",
        [kind.value for kind in ContentType],
        format_func=lambda kind: CONTENT_LABELS[ContentType(kind)],
        key="type",
    )
    ticket_choice = st.selectbox(
        "Ticket",
        [*tickets, FREE_TEXT],
        format_func=lambda key: "Escrever um ticket" if key == FREE_TEXT else ticket_label(key),
        key="ticket",
    )
    ticket_text = ""
    if ticket_choice == FREE_TEXT:
        ticket_text = st.text_area(
            "Descrição do problema", max_chars=TICKET_MAX_CHARS, key="ticket_text"
        )
    else:
        with st.expander("Ver descrição do ticket"):
            st.text(tickets[ticket_choice].text)

    with st.expander("Opções avançadas"):
        version = st.selectbox(
            "Versão do prompt",
            versions,
            index=versions.index(DEFAULT_VERSION) if DEFAULT_VERSION in versions else 0,
            format_func=PROMPT_LABELS.__getitem__,
            key="version",
        )

    use_groq = settings.provider == "groq" and settings.groq_api_key is not None
    if use_groq:
        st.caption("O ticket será enviado à Groq.")
    else:
        st.caption("Simulação offline com respostas fixas.")
    if st.button("Gerar resposta", type="primary", icon=":material/play_arrow:", key="generate"):
        try:
            ticket = (
                ticket_from_text(ticket_text)
                if ticket_choice == FREE_TEXT
                else tickets[ticket_choice]
            )
            provider = make_provider(settings, "groq" if use_groq else "fake")
            with st.spinner("Gerando resposta", show_time=True):
                st.session_state[LAST_GENERATION] = _run_generation(
                    settings,
                    provider,
                    profiles[profile_id],
                    ticket,
                    ContentType(content_type),
                    version,
                )
        except (ConfigError, DataError, PromptError, ProviderError) as error:
            st.error(str(error))

    if LAST_GENERATION in st.session_state:
        record, path = st.session_state[LAST_GENERATION]
        profile = profiles.get(record.customer_id)
        st.divider()
        _show_result(record, path, profile.name if profile else record.customer_id)


# monta a página: título e as três abas
def main() -> None:
    st.set_page_config(
        page_title="Assistente de suporte · Conecta Fibra",
        page_icon=":material/support_agent:",
        layout="wide",
    )
    with (
        st.container(horizontal=True, horizontal_alignment="center"),
        st.container(width=1120),
    ):
        st.title("Assistente de suporte", anchor=False)
        st.caption("Conecta Fibra · Provedor de internet fictício")
        generation, comparisons, examples = st.tabs(["Atendimento", "Comparação", "Exemplos"])
        with generation:
            _show_generation()
        with comparisons:
            show_comparisons(PROJECT_ROOT)
        with examples:
            show_examples(PROJECT_ROOT)


main()
