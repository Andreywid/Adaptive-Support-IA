import re

from support_assistant.credentials import clauses, is_negated, output_texts
from support_assistant.flow_render import validate_flow
from support_assistant.models import (
    Channel,
    DiagnosisOutput,
    FlowOutput,
    NodeKind,
    Profile,
    QuestionKind,
    QuestionsOutput,
    StepsOutput,
    StrictModel,
    Ticket,
)
from support_assistant.prompts.blocks import neutralize_ticket_tags

TECHNICAL_VALUE = re.compile(
    r"(?<![\w.,])-?(\d+(?:[.,]\d+)?)\s*(?:mbps|mb/s|gbps|gb/s|megas?|dbm|ms)(?!\w)",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
MENU_PATH = re.compile(r"[A-ZÀ-Ý][^\W\d_]*\s*(?:>|→|»)\s*[A-ZÀ-Ý][^\W\d_]*")
HYPHENS = str.maketrans(dict.fromkeys("\u2010\u2011\u2012\u2013\u2212", "-"))
PROMISE = re.compile(
    r"\b(?:agend\w+|marc(?:ar|amos|aremos)|garant\w+|promet\w+|conced\w+|receber[aá]|"
    r"ter[aá]\s+direito)\b[^.!?\n]{0,40}\b(?:visita|reembolso|desconto|cr[eé]dito|estorno)\b",
    re.IGNORECASE,
)


# espaços e hifens normalizados, para comparar trechos com o texto original
def normalize_text(text: str) -> str:
    return " ".join(text.translate(HYPHENS).split())


# regras do diagnóstico: ids únicos, conclusão coerente e citações copiadas do ticket
def _diagnosis_errors(output: DiagnosisOutput, ticket: Ticket) -> list[str]:
    errors = []
    if not output.hypotheses:
        errors.append("hypotheses: inclua ao menos uma hipótese com evidência do ticket")
    ids = [hypothesis.id for hypothesis in output.hypotheses]
    if any(not identifier.strip() for identifier in ids) or len(set(ids)) != len(ids):
        errors.append("hypotheses: use identificadores não vazios e únicos")
    if output.conclusion.hypothesis_id is not None and output.conclusion.hypothesis_id not in ids:
        errors.append("conclusion.hypothesis_id: use um id existente ou null")
    ticket_text = normalize_text(neutralize_ticket_tags(ticket.text))
    for index, hypothesis in enumerate(output.hypotheses):
        if not hypothesis.description.strip():
            errors.append(f"hypotheses.{index}.description: descreva a hipótese")
        if not hypothesis.evidence or any(
            not normalize_text(item.quote)
            or normalize_text(item.quote) not in ticket_text
            or not item.reading.strip()
            for item in hypothesis.evidence
        ):
            errors.append(f"hypotheses.{index}.evidence: cite e interprete um trecho do ticket")
    if not output.conclusion.statement.strip() or not output.customer_summary.strip():
        errors.append("diagnosis: preencha a conclusão e o resumo para o cliente")
    return errors


# regras do passo a passo: campos preenchidos e assunto só no e-mail
def _steps_errors(output: StepsOutput, profile: Profile) -> list[str]:
    errors = []
    if not output.steps or any(
        not all(field.strip() for field in (step.action, step.expected_result, step.if_fails))
        for step in output.steps
    ):
        errors.append("steps: inclua ação, resultado esperado e alternativa em cada passo")
    if profile.channel is Channel.EMAIL and not (output.subject or "").strip():
        errors.append("subject: preencha o assunto para e-mail")
    if profile.channel is not Channel.EMAIL and output.subject is not None:
        errors.append("subject: use null fora do canal e-mail")
    return errors


# regras das perguntas: prioridades em sequência e opções coerentes com o tipo
def _questions_errors(output: QuestionsOutput) -> list[str]:
    errors = []
    if [question.priority for question in output.questions] != list(
        range(1, len(output.questions) + 1)
    ):
        errors.append("questions: numere as prioridades em sequência a partir de 1")
    for index, question in enumerate(output.questions):
        if not question.text.strip() or not question.resolves.strip():
            errors.append(f"questions.{index}: preencha o texto e o motivo da pergunta")
        options = [normalize_text(option).casefold() for option in question.options]
        if question.kind is QuestionKind.CHOICE:
            if len(options) < 2 or "" in options or len(set(options)) != len(options):
                errors.append(f"questions.{index}.options: informe alternativas distintas")
        elif options:
            errors.append(f"questions.{index}.options: use uma lista vazia neste tipo de pergunta")
    return errors


# regras do fluxo: grafo válido, rótulos claros e mais de um destino por pergunta
def _flow_errors(output: FlowOutput) -> list[str]:
    blocks, warnings = validate_flow(output)
    if any(not node.id.strip() or not node.text.strip() for node in output.nodes):
        blocks.append("todo nó precisa de id e texto")
    node_ids = {node.id for node in output.nodes}
    targets_by_question: dict[str, list[str]] = {
        node.id: [] for node in output.nodes if node.kind is NodeKind.QUESTION
    }
    labels = []
    for edge in output.edges:
        if edge.source not in targets_by_question:
            continue
        labels.append((edge.source, normalize_text(edge.label).casefold()))
        if edge.target in node_ids:
            targets_by_question[edge.source].append(edge.target)
    if len(labels) != len(set(labels)) or any(not label for _, label in labels):
        blocks.append("use rótulos não vazios e distintos nas saídas de cada pergunta")
    blocks += [
        f"pergunta {question_id}: direcione as respostas para pelo menos 2 destinos distintos"
        for question_id, targets in targets_by_question.items()
        if len(targets) >= 2 and len(set(targets)) < 2
    ]
    return [f"flow: {problem}" for problem in (*blocks, *warnings)]


# rejeita medidas técnicas que a resposta cita sem que o ticket as tenha informado
def _invented_value_errors(output: StrictModel, ticket: Ticket) -> list[str]:
    known = {number.replace(",", ".") for number in NUMBER.findall(ticket.text)}
    invented = sorted(
        {
            match.group(0).strip()
            for text in output_texts(output.model_dump(mode="json"))
            for match in TECHNICAL_VALUE.finditer(text)
            if match.group(1).replace(",", ".") not in known
        }
    )
    if not invented:
        return []
    return [f"valores: remova {', '.join(invented)}, que não aparecem no ticket"]


# rejeita caminhos de menu que o ticket não confirmou
def _invented_menu_errors(output: StrictModel, ticket: Ticket) -> list[str]:
    ticket_text = normalize_text(ticket.text).casefold()
    invented = sorted(
        {
            match.group(0)
            for text in output_texts(output.model_dump(mode="json"))
            for match in MENU_PATH.finditer(text)
            if normalize_text(match.group(0)).casefold() not in ticket_text
        }
    )
    if not invented:
        return []
    return [f"menus: remova o caminho {', '.join(invented)}, que o ticket não confirma"]


# rejeita promessas de visita, reembolso, desconto ou crédito, sem contar citações do ticket
def _promise_errors(output: StrictModel) -> list[str]:
    data = output.model_dump(mode="json")
    if isinstance(output, DiagnosisOutput):
        for hypothesis in data["hypotheses"]:
            for item in hypothesis["evidence"]:
                item["quote"] = ""
    for text in output_texts(data):
        for clause in clauses(text):
            for match in PROMISE.finditer(clause):
                if not is_negated(clause, match.start()):
                    return ["limites: remova a promessa de visita, reembolso, desconto ou crédito"]
    return []


# descarta opções de perguntas que não são de múltipla escolha
def normalize_output(output: StrictModel) -> StrictModel:
    if not isinstance(output, QuestionsOutput):
        return output
    questions = [
        question.model_copy(update={"options": []})
        if question.kind is not QuestionKind.CHOICE and question.options
        else question
        for question in output.questions
    ]
    return output.model_copy(update={"questions": questions})


# erros de uso de uma saída, conforme o tipo de conteúdo
def output_errors(output: StrictModel, profile: Profile, ticket: Ticket) -> list[str]:
    errors = [
        *_invented_value_errors(output, ticket),
        *_invented_menu_errors(output, ticket),
        *_promise_errors(output),
    ]
    if isinstance(output, DiagnosisOutput):
        return [*_diagnosis_errors(output, ticket), *errors]
    if isinstance(output, StepsOutput):
        return [*_steps_errors(output, profile), *errors]
    if isinstance(output, QuestionsOutput):
        return [*_questions_errors(output), *errors]
    if isinstance(output, FlowOutput):
        return [*_flow_errors(output), *errors]
    raise TypeError(f"sem validação de saída para {type(output).__name__}")
