import re
from collections import deque
from collections.abc import Iterable

from support_assistant.models import (
    Conclusion,
    ContentType,
    Degree,
    DiagnosisOutput,
    Escalation,
    Evidence,
    FlowEdge,
    FlowNode,
    FlowOutput,
    Hypothesis,
    NodeKind,
    Question,
    QuestionKind,
    QuestionsOutput,
    Step,
    StepsOutput,
    StrictModel,
    Usage,
)
from support_assistant.providers.base import (
    Provider,
    ProviderError,
    ProviderRequest,
    ProviderResponse,
)

# saída válida e fixa de cada tipo, escrita para o ticket t01 e um cliente leigo ao telefone
CANNED_OUTPUTS: dict[ContentType, StrictModel] = {
    ContentType.DIAGNOSIS: DiagnosisOutput(
        hypotheses=[
            Hypothesis(
                id="H1",
                description="Instabilidade do sinal da linha em horário de pico",
                evidence=[
                    Evidence(
                        quote="cai toda noite",
                        reading="o problema se repete no mesmo período do dia",
                    )
                ],
                plausibility=Degree.MEDIUM,
            ),
            Hypothesis(
                id="H2",
                description="Reinício automático do roteador",
                evidence=[
                    Evidence(
                        quote="volta sozinha",
                        reading="a conexão se recupera sem intervenção do cliente",
                    )
                ],
                plausibility=Degree.LOW,
            ),
        ],
        missing_information=[
            "Horário exato em que a internet cai",
            "Se todos os aparelhos perdem a conexão ao mesmo tempo",
            "Cor das luzes do roteador durante a queda",
        ],
        conclusion=Conclusion(
            hypothesis_id="H1",
            confidence=Degree.LOW,
            statement="O padrão noturno aponta para instabilidade da linha, mas faltam dados.",
        ),
        customer_summary="Vamos investigar por que a internet cai à noite. Precisamos de detalhes.",
        escalation=Escalation(
            needed=False,
            reason="Não há sinal de risco de segurança nem de perda de dados.",
        ),
    ),
    ContentType.STEPS: StepsOutput(
        opening="Vou pedir um teste simples. É como reiniciar o celular quando ele trava.",
        steps=[
            Step(
                action="Desligue o roteador da tomada e espere meio minuto.",
                expected_result="Todas as luzes do roteador apagam.",
                if_fails="Se alguma luz continuar acesa, me diga qual.",
            ),
            Step(
                action="Ligue o roteador na tomada e espere dois minutos.",
                expected_result="A luz de internet fica verde.",
                if_fails="Se a luz ficar vermelha, me avise.",
            ),
        ],
        closing="Se a internet cair de novo hoje à noite, anote o horário e ligue para nós.",
        escalation_condition="Luz de internet vermelha depois do reinício: abrir chamado técnico.",
        subject=None,
    ),
    ContentType.QUESTIONS: QuestionsOutput(
        questions=[
            Question(
                priority=1,
                text="A internet cai em todos os aparelhos ao mesmo tempo?",
                kind=QuestionKind.YES_NO,
                options=[],
                resolves="Se o problema está na rede ou em um aparelho",
            ),
            Question(
                priority=2,
                text="Em que horário a internet costuma cair?",
                kind=QuestionKind.CHOICE,
                options=["Antes das 20h", "Entre 20h e 23h", "Depois das 23h"],
                resolves="Horário exato em que a internet cai",
            ),
            Question(
                priority=3,
                text="De que cor ficam as luzes do roteador quando a internet cai?",
                kind=QuestionKind.OPEN,
                options=[],
                resolves="Cor das luzes do roteador durante a queda",
            ),
        ]
    ),
    ContentType.FLOW: FlowOutput(
        nodes=[
            FlowNode(id="P1", kind=NodeKind.QUESTION, text="Todos os aparelhos perdem a conexão?"),
            FlowNode(
                id="P2", kind=NodeKind.QUESTION, text="A luz de sinal do roteador fica vermelha?"
            ),
            FlowNode(
                id="A1",
                kind=NodeKind.ACTION,
                text="Reiniciar o roteador e observar por uma noite",
            ),
            FlowNode(id="F1", kind=NodeKind.END, text="Abrir chamado técnico de linha"),
            FlowNode(id="F2", kind=NodeKind.END, text="Verificar só o aparelho afetado"),
        ],
        edges=[
            FlowEdge(source="P1", target="P2", label="sim"),
            FlowEdge(source="P1", target="F2", label="não"),
            FlowEdge(source="P2", target="F1", label="sim"),
            FlowEdge(source="P2", target="A1", label="não"),
        ],
    ),
}


# pega as primeiras palavras do ticket na mensagem, com ou sem delimitador
def _ticket_excerpt(user_message: str) -> str:
    found = re.search(r"<ticket>\s*(.+?)\s*</ticket>", user_message, re.S) or re.search(
        r"Ticket do cliente:\s*(.+?)(?:\n\n|\Z)", user_message, re.S
    )
    text = found.group(1) if found else user_message
    return " ".join(text.split()[:8])


# exemplos offline usam evidências do cenário conhecido e um trecho do próprio ticket nos demais
def _canned_content(request: ProviderRequest) -> str:
    try:
        content_type = ContentType(request.schema_name)
    except ValueError:
        raise ProviderError(f"fake: schema desconhecido: {request.schema_name}") from None
    output = CANNED_OUTPUTS[content_type]
    user_message = next(
        (message.content for message in request.messages if message.role == "user"), ""
    )
    if isinstance(output, DiagnosisOutput) and any(
        item.quote not in user_message
        for hypothesis in output.hypotheses
        for item in hypothesis.evidence
    ):
        output = DiagnosisOutput(
            hypotheses=[
                Hypothesis(
                    id="H1",
                    description="Causa ainda não identificada neste exemplo offline",
                    evidence=[
                        Evidence(
                            quote=_ticket_excerpt(user_message),
                            reading="o trecho descreve o sintoma e ainda não separa as causas",
                        )
                    ],
                    plausibility=Degree.LOW,
                )
            ],
            missing_information=["Quando o problema começou e quais aparelhos foram afetados"],
            conclusion=Conclusion(
                hypothesis_id="H1",
                confidence=Degree.LOW,
                statement="Este exemplo offline ainda não tem dados para confirmar a causa.",
            ),
            customer_summary="Precisamos de mais detalhes para orientar o próximo teste.",
            escalation=Escalation(needed=False, reason="O exemplo offline pede mais detalhes."),
        )
    if isinstance(output, StepsOutput):
        context = user_message.split("\n\n", 1)[0]
        if "canal: e-mail" in context or "atendimento por e-mail" in context:
            output = output.model_copy(update={"subject": "Próximas etapas do atendimento"})
    return output.model_dump_json()


# estima os tokens como caracteres divididos por 4, de forma determinista
def _estimate_usage(request: ProviderRequest, content: str) -> Usage:
    prompt_chars = sum(len(message.content) for message in request.messages)
    return Usage(input_tokens=max(1, prompt_chars // 4), output_tokens=max(1, len(content) // 4))


# provedor falso e determinista, sem rede, para testes e demonstração offline
class FakeProvider(Provider):
    name = "fake"

    # recebe respostas programadas em ordem: texto vira a resposta e exceção é levantada
    def __init__(self, responses: Iterable[str | Exception] | None = None) -> None:
        self.calls: list[ProviderRequest] = []
        self._responses = deque(responses or [])

    # registra a chamada e devolve a próxima resposta programada ou a saída fixa do tipo
    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        if self._responses:
            item = self._responses.popleft()
            if isinstance(item, Exception):
                raise item
            content = item
        else:
            content = _canned_content(request)
        return ProviderResponse(
            content=content,
            usage=_estimate_usage(request, content),
            finish_reason="stop",
            latency_ms=0,
        )
