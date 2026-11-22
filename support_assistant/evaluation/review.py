from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from support_assistant.credentials import redact_credentials
from support_assistant.evaluation.compare import (
    ComparisonError,
    comparison_pair,
    comparison_profile,
    describe_profile,
)
from support_assistant.evaluation.rubric import (
    CONTENT_READERS,
    CRITERIA,
    complete_reviews,
    format_rubric,
)
from support_assistant.formatting import format_record_output
from support_assistant.models import (
    ComparisonRecord,
    ContentEvaluation,
    CriterionScore,
    EvaluatedOutput,
    Evaluation,
    Evaluator,
)


# pede nota, justificativa e trecho de um critério, repetindo cada pergunta até a resposta valer
def _ask_criterion(
    position: str,
    label: str,
    shown: str,
    ask: Callable[[str], str],
    show: Callable[[str], None],
) -> CriterionScore:
    while True:
        answer = ask(f"{position}, {label}, nota de 1 a 5: ").strip()
        if answer in ("1", "2", "3", "4", "5"):
            break
    while True:
        justification = ask("justificativa: ").strip()
        if justification:
            break

    shown = " ".join(shown.split())
    while True:
        evidence = ask("trecho da saída que motivou a nota: ").strip()
        if evidence:
            if " ".join(evidence.split()) in shown:
                return CriterionScore(
                    score=int(answer), justification=justification, evidence=evidence
                )
            show(f"Trecho não encontrado na saída {position}. Copie um trecho exato.")


# avalia às cegas, um tipo por vez, e revela as versões só depois das notas do tipo
def evaluate_blind(
    record: ComparisonRecord,
    *,
    seed: int,
    ask: Callable[[str], str] = input,
    show: Callable[[str], None] = print,
    now: datetime | None = None,
    save: Callable[[ComparisonRecord], Path] | None = None,
) -> ComparisonRecord:
    if seed not in record.seeds:
        seeds = ", ".join(str(item) for item in record.seeds)
        raise ComparisonError(f"A seed {seed} não faz parte da comparação. Seeds: {seeds}")

    show(f"Perfil do cliente\n{describe_profile(comparison_profile(record))}")
    show(f"Ticket\n{redact_credentials(record.ticket)}")
    rubric = record.rubric or format_rubric()
    reviewed_types = complete_reviews(record, seed)
    current = record
    for content_type in record.content_types:
        if content_type in reviewed_types:
            continue

        first, second = comparison_pair(record, seed, content_type)
        candidates = [
            (position, run.version, format_record_output(run))
            for position, run in (("A", first), ("B", second))
        ]
        show(f"== {content_type.value} (seed {seed}) ==")
        show(f"Leitor esperado\n{CONTENT_READERS[content_type]}")
        for position, _, text in candidates:
            show(f"== {position} ==")
            show(text)
        show(rubric)

        outputs = []
        for position, version, text in candidates:
            scores = {
                key: _ask_criterion(position, label, text, ask, show) for key, label in CRITERIA
            }
            evaluation = Evaluation(
                **scores,
                evaluator=Evaluator.HUMAN,
                evaluated_at=(now or datetime.now(UTC)).astimezone(UTC),
            )
            outputs.append(
                EvaluatedOutput(position=position, version=version, evaluation=evaluation)
            )

        evaluated = ContentEvaluation(content_type=content_type, seed=seed, outputs=outputs)
        others = [
            item
            for item in current.evaluations
            if (item.content_type, item.seed) != (content_type, seed)
        ]
        current = current.model_copy(update={"evaluations": [*others, evaluated]})
        if save is not None:
            save(current)
        show(f"Revelado: A = {first.version}, B = {second.version}")
    return current
