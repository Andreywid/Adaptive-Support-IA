import json
from pathlib import Path

import streamlit as st
from pydantic import ValidationError

from support_assistant.flow_render import render_flow
from support_assistant.models import (
    OUTPUT_MODELS,
    ComparisonRun,
    ContentType,
    Degree,
    DiagnosisOutput,
    FlowOutput,
    QuestionKind,
    QuestionsOutput,
    RunRecord,
    RunStatus,
    StepsOutput,
)

CONTENT_LABELS = {
    ContentType.DIAGNOSIS: "Diagnóstico",
    ContentType.STEPS: "Passo a passo",
    ContentType.QUESTIONS: "Perguntas de esclarecimento",
    ContentType.FLOW: "Fluxo de decisão",
}
PROMPT_LABELS = {
    "v1": "Inicial",
    "v2": "Adaptado ao perfil",
    "v3": "Regras de decisão",
    "v4": "Método de decisão",
}
TICKET_LABELS = {
    "t01_queda_noturna": "Internet cai à noite",
    "t02_wifi_lento": "Wi-Fi lento em um cômodo",
    "t03_codigo_sms": "Código por SMS não chega",
    "t04_sem_sinal": "Sem internet após tempestade",
    "t05_tv_travando": "TV trava nos filmes",
    "t06_latencia_jogos": "Ping alto nos jogos",
    "t07_cobranca_senha": "Cobrança em dobro e lentidão",
    "t08_relato_vago": "Relato com poucos dados",
}

DEGREE_LABELS = {Degree.LOW: "baixa", Degree.MEDIUM: "média", Degree.HIGH: "alta"}
QUESTION_LABELS = {
    QuestionKind.OPEN: "Resposta aberta",
    QuestionKind.YES_NO: "Sim ou não",
    QuestionKind.CHOICE: "Escolha entre opções",
}


# nome legível de um ticket
def ticket_label(ticket_id: str) -> str:
    return TICKET_LABELS.get(ticket_id, ticket_id.replace("_", " ").capitalize())


# caminho relativo à raiz do projeto, para mostrar ao usuário
def display_path(path: Path, project_root: Path) -> str:
    if path.is_relative_to(project_root):
        return path.relative_to(project_root).as_posix()
    return str(path)


# um campo com rótulo em negrito e o texto embaixo
def _field(label: str, text: str) -> None:
    st.markdown(f"**{label}**")
    st.text(text, width="stretch")


# mostra o diagnóstico: resumo, conclusão, hipóteses e o que falta saber
def _show_diagnosis(output: DiagnosisOutput) -> None:
    _field("Resumo para o cliente", output.customer_summary)
    _field("Conclusão", output.conclusion.statement)
    st.caption(f"Confiança {DEGREE_LABELS[output.conclusion.confidence]}")
    if output.conclusion.hypothesis_id is not None:
        st.text(f"Hipótese selecionada · {output.conclusion.hypothesis_id}", width="stretch")

    if output.hypotheses:
        st.markdown("**Hipóteses e evidências**")
        for hypothesis in output.hypotheses:
            st.text(f"{hypothesis.id} · {hypothesis.description}", width="stretch")
            st.caption(f"Plausibilidade {DEGREE_LABELS[hypothesis.plausibility]}")
            for evidence in hypothesis.evidence:
                st.text(f"“{evidence.quote}”\n{evidence.reading}", width="stretch")

    if output.missing_information:
        missing = "\n".join(f"• {item}" for item in output.missing_information)
        _field("Informações que faltam", missing)

    escalation = "Escalonamento recomendado" if output.escalation.needed else "Sem escalonamento"
    _field(escalation, output.escalation.reason)


# mostra o passo a passo: assunto, abertura, passos e fechamento
def _show_steps(output: StepsOutput) -> None:
    if output.subject:
        _field("Assunto", output.subject)
    st.text(output.opening, width="stretch")
    for position, step in enumerate(output.steps, start=1):
        _field(f"Passo {position}", step.action)
        st.text(
            f"Resultado esperado: {step.expected_result}\nSe o teste falhar: {step.if_fails}",
            width="stretch",
        )

    st.text(output.closing, width="stretch")
    _field("Quando escalar", output.escalation_condition)


# mostra as perguntas com o tipo e as opções
def _show_questions(output: QuestionsOutput) -> None:
    if not output.questions:
        st.info("A resposta não trouxe perguntas de esclarecimento.")
    for question in output.questions:
        st.caption(f"Prioridade {question.priority} · {QUESTION_LABELS[question.kind]}")
        st.text(question.text, width="stretch")
        if question.options:
            st.text("\n".join(f"• {option}" for option in question.options), width="stretch")
        st.text(f"O que esclarece: {question.resolves}", width="stretch")


# mostra uma saída fora do contrato, com o texto bruto
def _show_invalid(record: RunRecord | ComparisonRun) -> None:
    st.warning("A saída está fora do contrato esperado.")
    if record.output is not None:
        st.code(
            json.dumps(record.output, ensure_ascii=False, indent=2),
            language="json",
            wrap_lines=True,
        )
    elif record.raw_output is not None:
        st.code(record.raw_output, language=None, wrap_lines=True)


# mostra uma saída no formato do tipo dela
def show_output(record: RunRecord | ComparisonRun) -> None:
    if record.status is not RunStatus.OK or record.output is None:
        _show_invalid(record)
        return
    try:
        output = OUTPUT_MODELS[record.content_type].model_validate(record.output)
    except ValidationError:
        _show_invalid(record)
        return

    if isinstance(output, FlowOutput):
        flow = render_flow(output)
        if not flow.drawn:
            st.warning("Não foi possível desenhar este fluxo.")
        st.code(flow.text, language=None, wrap_lines=True)
        return
    if isinstance(output, DiagnosisOutput):
        _show_diagnosis(output)
    elif isinstance(output, StepsOutput):
        _show_steps(output)
    elif isinstance(output, QuestionsOutput):
        _show_questions(output)

    with st.expander("Ver JSON da saída"):
        st.code(
            json.dumps(record.output, ensure_ascii=False, indent=2),
            language="json",
            wrap_lines=True,
        )
