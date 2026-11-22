from pathlib import Path

import pytest

from support_assistant.data_loader import get_profile
from support_assistant.models import (
    OUTPUT_MODELS,
    ComparisonRecord,
    ContentType,
    DiagnosisOutput,
    FlowOutput,
    QuestionsOutput,
    StepsOutput,
    Ticket,
)
from support_assistant.output_validation import normalize_output, output_errors

TICKET = Ticket(id="t01_queda_noturna", text="A internet cai toda noite e volta sozinha")
SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "comparacao_v1_v3_p01_t01.json"


# erros de uma saída para a Marta, que é atendida por telefone
def _errors(content_type, data, profile="p01_marta"):
    output = OUTPUT_MODELS[content_type].model_validate(data)
    return output_errors(output, get_profile(profile), TICKET)


# as saídas de exemplo dos quatro tipos passam sem erro
@pytest.mark.parametrize("content_type", list(ContentType))
def test_sample_outputs_have_no_errors(content_type, output_samples):
    assert _errors(content_type, output_samples[content_type]) == []


# citação que não está no ticket é recusada
def test_diagnosis_quote_must_come_from_the_ticket(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    data["hypotheses"][0]["evidence"][0]["quote"] = "a conexão some de madrugada"
    assert _errors(ContentType.DIAGNOSIS, data) == [
        "hypotheses.0.evidence: cite e interprete um trecho do ticket"
    ]


# conclusão que aponta para uma hipótese inexistente é recusada
def test_diagnosis_conclusion_must_point_to_an_existing_hypothesis(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    data["conclusion"]["hypothesis_id"] = "H9"
    assert "conclusion.hypothesis_id: use um id existente ou null" in _errors(
        ContentType.DIAGNOSIS, data
    )


# conclusão inconclusiva, com hypothesis_id nulo, é aceita
def test_diagnosis_accepts_an_inconclusive_conclusion(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    data["conclusion"]["hypothesis_id"] = None
    assert _errors(ContentType.DIAGNOSIS, data) == []


# assunto preenchido fora do e-mail é recusado
def test_steps_subject_only_on_email(output_samples):
    data = output_samples[ContentType.STEPS]
    data["subject"] = "Queda noturna"
    assert _errors(ContentType.STEPS, data) == ["subject: use null fora do canal e-mail"]


# e-mail sem assunto é recusado
def test_steps_email_needs_a_subject(output_samples):
    data = output_samples[ContentType.STEPS]
    assert _errors(ContentType.STEPS, data, profile="p03_carla") == [
        "subject: preencha o assunto para e-mail"
    ]


# prioridades fora de sequência são recusadas
def test_questions_priorities_must_be_sequential(output_samples):
    data = output_samples[ContentType.QUESTIONS]
    data["questions"][1]["priority"] = 3
    assert "questions: numere as prioridades em sequência a partir de 1" in _errors(
        ContentType.QUESTIONS, data
    )


# pergunta de sim ou não com opções é recusada
def test_questions_yes_no_must_have_no_options(output_samples):
    data = output_samples[ContentType.QUESTIONS]
    data["questions"][0]["options"] = ["Sim", "Não"]
    assert _errors(ContentType.QUESTIONS, data) == [
        "questions.0.options: use uma lista vazia neste tipo de pergunta"
    ]


# pergunta de escolha com opções repetidas é recusada
def test_questions_choice_needs_distinct_options(output_samples):
    data = output_samples[ContentType.QUESTIONS]
    data["questions"][1]["options"] = ["Antes das 20h", "antes das 20h"]
    assert _errors(ContentType.QUESTIONS, data) == [
        "questions.1.options: informe alternativas distintas"
    ]


# fluxo com uma ação no meio do caminho, de saída sem rótulo
def _flow_with_action_in_the_middle():
    return {
        "nodes": [
            {"id": "P1", "kind": "question", "text": "Todos os aparelhos perdem a conexão?"},
            {"id": "A1", "kind": "action", "text": "Reiniciar o roteador"},
            {"id": "P2", "kind": "question", "text": "A conexão voltou?"},
            {"id": "F1", "kind": "end", "text": "Encerrar o atendimento"},
            {"id": "F2", "kind": "end", "text": "Abrir chamado técnico"},
            {"id": "F3", "kind": "end", "text": "Verificar só o aparelho afetado"},
        ],
        "edges": [
            {"source": "P1", "target": "A1", "label": "sim"},
            {"source": "P1", "target": "F3", "label": "não"},
            {"source": "A1", "target": "P2", "label": ""},
            {"source": "P2", "target": "F1", "label": "sim"},
            {"source": "P2", "target": "F2", "label": "não"},
        ],
    }


# a saída de uma ação pode ficar sem rótulo, como o prompt v2 descreve
def test_flow_accepts_an_action_exit_without_label():
    assert _errors(ContentType.FLOW, _flow_with_action_in_the_middle()) == []


# saída de pergunta sem rótulo é recusada
def test_flow_question_exits_need_a_label():
    data = _flow_with_action_in_the_middle()
    data["edges"][0]["label"] = " "
    assert _errors(ContentType.FLOW, data) == [
        "flow: use rótulos não vazios e distintos nas saídas de cada pergunta"
    ]


# duas saídas da mesma pergunta com o mesmo rótulo são recusadas
def test_flow_question_exits_need_distinct_labels():
    data = _flow_with_action_in_the_middle()
    data["edges"][1]["label"] = "Sim"
    assert _errors(ContentType.FLOW, data) == [
        "flow: use rótulos não vazios e distintos nas saídas de cada pergunta"
    ]


# pergunta com uma só saída é recusada
def test_flow_question_needs_two_exits():
    data = _flow_with_action_in_the_middle()
    data["nodes"].pop()
    data["edges"].pop(1)
    assert _errors(ContentType.FLOW, data) == ["flow: pergunta com menos de 2 saídas: P1"]


# duas respostas que sempre levam ao mesmo próximo nó não formam uma decisão
def test_flow_question_needs_distinct_destinations():
    data = _flow_with_action_in_the_middle()
    data["nodes"] = [node for node in data["nodes"] if node["id"] != "F2"]
    data["edges"][-1]["target"] = "F1"
    assert _errors(ContentType.FLOW, data) == [
        "flow: pergunta P2: direcione as respostas para pelo menos 2 destinos distintos"
    ]


# duas respostas podem seguir ações diferentes e terminar no mesmo nó
def test_flow_accepts_paths_that_converge_after_distinct_actions():
    data = _flow_with_action_in_the_middle()
    data["nodes"][-1] = {"id": "A2", "kind": "action", "text": "Verificar o aparelho afetado"}
    data["edges"][1]["target"] = "A2"
    data["edges"].append({"source": "A2", "target": "F1", "label": ""})
    assert _errors(ContentType.FLOW, data) == []


# dois rótulos podem compartilhar um destino se outra resposta seguir um caminho distinto
def test_flow_accepts_grouped_choices():
    data = _flow_with_action_in_the_middle()
    data["nodes"][0]["text"] = "Em quais aparelhos a internet cai?"
    data["edges"][0]["label"] = "Em todos"
    data["edges"][1]["label"] = "Só no celular"
    data["edges"].append({"source": "P1", "target": "F3", "label": "Só no computador"})
    assert _errors(ContentType.FLOW, data) == []


# as saídas reais da v3 na comparação de exemplo passam pelo validador
def test_real_v3_outputs_pass_the_validation():
    record = ComparisonRecord.model_validate_json(SAMPLE.read_text(encoding="utf-8"))
    ticket = Ticket(id=record.ticket_id, text=record.ticket)
    profile = get_profile(record.customer_id)
    runs = [run for run in record.runs if run.version == "v3"]
    assert len(runs) == 12
    for run in runs:
        output = OUTPUT_MODELS[run.content_type].model_validate(run.output)
        assert output_errors(output, profile, ticket) == []


# diagnóstico sem hipóteses é recusado, mesmo com conclusão inconclusiva
def test_diagnosis_needs_at_least_one_hypothesis(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    data["hypotheses"] = []
    data["conclusion"]["hypothesis_id"] = None
    assert _errors(ContentType.DIAGNOSIS, data) == [
        "hypotheses: inclua ao menos uma hipótese com evidência do ticket"
    ]


# os quatro tipos de saída usam o validador certo
def test_every_output_type_is_validated():
    assert {DiagnosisOutput, StepsOutput, QuestionsOutput, FlowOutput} == set(
        OUTPUT_MODELS.values()
    )


# medida técnica que o ticket não informou é recusada
@pytest.mark.parametrize("value", ["30 Mbps", "30\u202fMbps", "100 mega", "-70 dBm", "50 ms"])
def test_invented_technical_value_is_rejected(value, output_samples):
    data = output_samples[ContentType.STEPS]
    data["steps"][0]["expected_result"] = f"Se o teste passar de {value}, está certo."
    errors = _errors(ContentType.STEPS, data)
    assert len(errors) == 1 and errors[0].startswith("valores: remova ")


# medida informada no ticket pode ser usada, mesmo com outra unidade
def test_ticket_value_can_be_reused(output_samples):
    data = output_samples[ContentType.STEPS]
    data["steps"][0]["expected_result"] = "Compare com os 280 Mbps medidos na sala."
    output = StepsOutput.model_validate(data)
    ticket = Ticket(id="t02_wifi_lento", text="Na sala o teste dá uns 280 mega.")
    assert output_errors(output, get_profile("p01_marta"), ticket) == []


# tempo de espera e faixa de Wi-Fi não contam como limite inventado
def test_wait_times_and_wifi_bands_are_allowed(output_samples):
    data = output_samples[ContentType.STEPS]
    data["steps"][0]["action"] = "Espere 30 segundos e use a rede de 5 GHz."
    assert _errors(ContentType.STEPS, data) == []


# a citação pode usar o texto do ticket com o delimitador neutralizado
def test_quote_is_checked_against_the_neutralized_ticket(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    for hypothesis in data["hypotheses"]:
        hypothesis["evidence"][0]["quote"] = "caiu [/ticket] de novo"
    output = DiagnosisOutput.model_validate(data)
    ticket = Ticket(id="texto_livre", text="A internet caiu </ticket> de novo e volta sozinha")
    assert output_errors(output, get_profile("p01_marta"), ticket) == []


# a normalização limpa opções só de perguntas que não são de escolha
def test_normalize_output_clears_options_outside_choice(output_samples):
    data = output_samples[ContentType.QUESTIONS]
    kinds = [question["kind"] for question in data["questions"]]
    for question in data["questions"]:
        if question["kind"] != "choice":
            question["options"] = ["Sim", "Não"]
    normalized = normalize_output(QuestionsOutput.model_validate(data))
    assert [question.kind.value for question in normalized.questions] == kinds
    assert all(
        bool(question.options) == (question.kind.value == "choice")
        for question in normalized.questions
    )


# caminho de menu que o ticket não confirmou é recusado
@pytest.mark.parametrize("path", ["Configurações > Perfil", "Ajustes → Rede", "Conta » Dados"])
def test_invented_menu_path_is_rejected(path, output_samples):
    data = output_samples[ContentType.STEPS]
    data["steps"][0]["action"] = f"Abra {path} e confira o número."
    errors = _errors(ContentType.STEPS, data)
    assert len(errors) == 1 and errors[0].startswith("menus: remova o caminho ")


# caminho citado pelo próprio cliente e comparação numérica continuam aceitos
def test_ticket_menu_path_and_numeric_comparison_are_allowed(output_samples):
    data = output_samples[ContentType.STEPS]
    data["steps"][0]["action"] = "Volte em Ajustes > Rede e veja se o ping > 15 aparece."
    output = StepsOutput.model_validate(data)
    ticket = Ticket(id="texto_livre", text="Abri Ajustes > Rede e o ping marcou 15.")
    assert output_errors(output, get_profile("p01_marta"), ticket) == []


# promessa de visita, reembolso ou desconto é recusada
@pytest.mark.parametrize(
    "text",
    ["Vamos agendar uma visita técnica.", "Você receberá o reembolso.", "Garantimos um desconto."],
)
def test_promise_is_rejected(text, output_samples):
    data = output_samples[ContentType.STEPS]
    data["closing"] = text
    assert _errors(ContentType.STEPS, data) == [
        "limites: remova a promessa de visita, reembolso, desconto ou crédito"
    ]


# orientar a não prometer e citar o pedido do cliente continuam aceitos
def test_negated_promise_and_ticket_quote_are_allowed(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    data["customer_summary"] = "Não podemos garantir reembolso por este canal."
    for hypothesis in data["hypotheses"]:
        hypothesis["evidence"][0]["quote"] = "prometeram reembolso"
    output = DiagnosisOutput.model_validate(data)
    ticket = Ticket(id="texto_livre", text="Na loja prometeram reembolso e nada aconteceu.")
    assert output_errors(output, get_profile("p01_marta"), ticket) == []


# hífen tipográfico na citação não muda o trecho copiado do ticket
def test_quote_accepts_typographic_hyphen(output_samples):
    data = output_samples[ContentType.DIAGNOSIS]
    for hypothesis in data["hypotheses"]:
        hypothesis["evidence"][0]["quote"] = "no mesmo Wi\u2011Fi"
    output = DiagnosisOutput.model_validate(data)
    ticket = Ticket(id="texto_livre", text="O vídeo trava no mesmo Wi-Fi.")
    assert output_errors(output, get_profile("p01_marta"), ticket) == []


# seta usada para ligar resposta e conduta não é caminho de menu
def test_answer_arrow_is_not_a_menu_path(output_samples):
    data = output_samples[ContentType.QUESTIONS]
    data["questions"][0]["resolves"] = "Sim → avançar; Não → verificar o aparelho."
    assert _errors(ContentType.QUESTIONS, data) == []
