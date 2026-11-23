from collections import Counter
from pathlib import Path

import streamlit as st

from support_assistant.data_loader import DataError, load_profiles
from support_assistant.evaluation import compare
from support_assistant.evaluation.rubric import CRITERIA, complete_reviews
from support_assistant.models import ComparisonRecord, Evaluation, RunRecord
from support_assistant.storage import load_run
from web.presentation import CONTENT_LABELS, PROMPT_LABELS, display_path, show_output, ticket_label

PRIMARY_COMPARISON = Path("samples/comparacao_v1_v3_p01_t01.json").as_posix()


# acrescenta o arquivo ao rótulo quando dois rótulos se repetem
def _distinct_labels(labels: dict[str, str]) -> dict[str, str]:
    counts = Counter(labels.values())
    return {
        key: f"{label} · {key}" if counts[label] > 1 else label for key, label in labels.items()
    }


# comparações que podem ser abertas, locais e de exemplo
def _comparison_files(project_root: Path) -> dict[str, ComparisonRecord]:
    archive: dict[str, ComparisonRecord] = {}
    for path in (
        *compare.COMPARISONS_DIR.glob("*.json"),
        *(project_root / "samples").glob("comparacao_*.json"),
    ):
        try:
            record, _ = compare.load_comparison_argument(str(path))
        except compare.ComparisonError:
            continue
        if set(record.versions) <= PROMPT_LABELS.keys():
            archive[display_path(path, project_root)] = record
    return archive


# mostra as notas, as justificativas e os trechos de uma saída
def _show_scores(evaluation: Evaluation) -> None:
    with st.expander("Avaliação da rubrica", expanded=True):
        for key, label in CRITERIA:
            score = getattr(evaluation, key)
            st.markdown(f"**{label} · {score.score}/5**")
            st.text(score.justification)
            st.text(f"Evidência: {score.evidence}")


# mostra o perfil, o ticket e o arquivo da comparação
def _show_context(record: ComparisonRecord, path: str) -> None:
    try:
        profile = compare.comparison_profile(record)
    except DataError:
        st.text(f"Perfil: {record.customer_id} (detalhes indisponíveis)")
    else:
        st.text(compare.describe_profile(profile))
        if record.profile is None:
            st.caption("Perfil recuperado do catálogo atual.")
    st.text(record.ticket)
    st.caption(f"{record.locked_config.model} · {path}")


# aba Comparação: mostra A e B lado a lado e só revela as versões no botão
def show_comparisons(project_root: Path) -> None:
    st.header("Comparar respostas", anchor=False)
    archive = _comparison_files(project_root)
    if not archive:
        st.info("Nenhuma comparação disponível.")
        return

    paths = sorted(archive, key=lambda path: (path != PRIMARY_COMPARISON, path))
    labels = {}
    for path, record in archive.items():
        customer = record.profile.name if record.profile else record.customer_id
        origin = (
            "Referência" if path == PRIMARY_COMPARISON else f"{record.created_at:%d/%m/%Y %H:%M}"
        )
        labels[path] = f"{customer} · {ticket_label(record.ticket_id)} · {origin}"
    labels = _distinct_labels(labels)
    path = st.selectbox("Cenário", paths, format_func=labels.__getitem__, key="comparison")
    record = archive[path]
    with st.expander("Perfil e ticket"):
        _show_context(record, path)
    left, right = st.columns([2, 1])
    content_type = left.selectbox(
        "Tipo de resposta",
        record.content_types,
        format_func=CONTENT_LABELS.__getitem__,
        key="compare_type",
    )
    seed = right.selectbox("Repetição (seed)", record.seeds, key="seed")
    try:
        pair = compare.comparison_pair(record, seed, content_type)
    except compare.ComparisonError as error:
        st.warning(str(error))
        return

    review = complete_reviews(record, seed).get(content_type)
    ratings = {output.version: output.evaluation for output in review.outputs} if review else {}
    if review is None:
        st.caption("Avaliação pendente para este par.")
    revealed_key = f"revealed:{path}:{seed}:{content_type.value}"
    revealed = st.session_state.get(revealed_key, False)
    if not revealed and st.button("Revelar versões", icon=":material/visibility:", key="reveal"):
        st.session_state[revealed_key] = True
        st.rerun()

    for column, position, run in zip(st.columns(2), ("A", "B"), pair, strict=True):
        with column:
            st.subheader(f"Saída {position}", anchor=False)
            if revealed:
                st.caption(PROMPT_LABELS.get(run.version, run.version))
            with st.container(height=440, border=True):
                show_output(run)
            if run.version in ratings:
                _show_scores(ratings[run.version])


# aba Exemplos: mostra as respostas reais guardadas em samples/
def show_examples(project_root: Path) -> None:
    st.header("Respostas de exemplo", anchor=False)
    samples: dict[str, RunRecord] = {}
    for path in sorted((project_root / "samples").glob("run_*.json")):
        try:
            record = load_run(path)
        except (OSError, ValueError):
            continue
        if record.prompt_version in PROMPT_LABELS:
            samples[path.name] = record
    if not samples:
        st.info("Nenhum exemplo de geração disponível.")
        return

    try:
        profiles = {profile.id: profile.name for profile in load_profiles().profiles}
    except DataError:
        profiles = {}
    labels = _distinct_labels(
        {
            name: (
                f"{CONTENT_LABELS[record.content_type]} · {ticket_label(record.ticket_id)} · "
                f"{profiles.get(record.customer_id, record.customer_id)}"
            )
            for name, record in samples.items()
        }
    )
    name = st.selectbox("Exemplo", list(samples), format_func=labels.__getitem__, key="sample")
    record = samples[name]
    with st.expander("Ticket do exemplo"):
        st.text(record.ticket)
    show_output(record)
