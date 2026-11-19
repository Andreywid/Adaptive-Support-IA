from datetime import UTC
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# versão dos contratos gravados em disco
SchemaVersion = Literal["1.0"]
SCHEMA_VERSION: SchemaVersion = "1.0"

# limite de caracteres do texto de um ticket
TICKET_MAX_CHARS = 2000

# identificador em snake_case minúsculo, seguro para nome de arquivo e argumento de CLI
Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$", max_length=64)]

# nome de versão de prompt no formato vN
PromptVersion = Annotated[str, StringConstraints(pattern=r"^v[0-9]+$")]

# hash no formato sha256:<64 caracteres hexadecimais>
Sha256Hash = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

# identificador de execução, seguro para nome de arquivo
RunId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,128}$")]

# texto obrigatório, sem espaços nas pontas
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# texto do ticket, obrigatório e com limite de tamanho
TicketText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=TICKET_MAX_CHARS)
]

# data e hora com fuso, sempre normalizada para UTC
UtcDatetime = Annotated[AwareDatetime, AfterValidator(lambda value: value.astimezone(UTC))]


# nível técnico do cliente
class TechnicalLevel(StrEnum):
    LEIGO = "leigo"
    INTERMEDIARIO = "intermediario"
    AVANCADO = "avancado"


# estilo de comunicação preferido pelo cliente
class CommunicationStyle(StrEnum):
    DIRETO = "direto"
    DETALHADO = "detalhado"
    DIDATICO = "didatico"


# canal pelo qual o cliente é atendido
class Channel(StrEnum):
    TELEFONE = "telefone"
    CHAT = "chat"
    EMAIL = "email"


# tipo de conteúdo que um gerador produz
class ContentType(StrEnum):
    DIAGNOSIS = "diagnosis"
    STEPS = "steps"
    QUESTIONS = "questions"
    FLOW = "flow"


# resultado de uma execução: saída válida ou fora do contrato
class RunStatus(StrEnum):
    OK = "ok"
    INVALID_OUTPUT = "invalid_output"


# esforço de raciocínio aceito pelos modelos gpt-oss
class ReasoningEffort(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# grau em três níveis, usado em plausibilidade e confiança
class Degree(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# forma de resposta esperada de uma pergunta de esclarecimento
class QuestionKind(StrEnum):
    OPEN = "open"
    YES_NO = "yes_no"
    CHOICE = "choice"


# papel de um nó na árvore de decisão
class NodeKind(StrEnum):
    QUESTION = "question"
    ACTION = "action"
    END = "end"


# quem atribuiu uma avaliação
class Evaluator(StrEnum):
    HUMAN = "human"


# base de todos os contratos: campos extras proibidos e instâncias imutáveis
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# perfil de cliente usado no experimento
class Profile(StrictModel):
    id: Slug
    name: NonEmptyText
    technical_level: TechnicalLevel
    communication_style: CommunicationStyle
    channel: Channel


# ticket de suporte: identificador e texto escrito pelo cliente
class Ticket(StrictModel):
    id: Slug
    text: TicketText


# devolve os ids que aparecem mais de uma vez
def _duplicated_ids(ids: list[str]) -> list[str]:
    return sorted({item for item in ids if ids.count(item) > 1})


# arquivo de perfis: versão do contrato e lista de perfis
class ProfileCatalog(StrictModel):
    schema_version: SchemaVersion
    profiles: list[Profile]

    # rejeita perfis com id repetido
    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        duplicated = _duplicated_ids([profile.id for profile in self.profiles])
        if duplicated:
            raise ValueError(f"ids de perfil repetidos: {', '.join(duplicated)}")
        return self


# arquivo de tickets: versão do contrato e lista de tickets
class TicketCatalog(StrictModel):
    schema_version: SchemaVersion
    tickets: list[Ticket]

    # rejeita tickets com id repetido
    @model_validator(mode="after")
    def _check_unique_ids(self) -> Self:
        duplicated = _duplicated_ids([ticket.id for ticket in self.tickets])
        if duplicated:
            raise ValueError(f"ids de ticket repetidos: {', '.join(duplicated)}")
        return self


# trecho literal do ticket e o que ele indica
class Evidence(StrictModel):
    quote: str
    reading: str


# causa possível do problema, com evidências e plausibilidade
class Hypothesis(StrictModel):
    id: str
    description: str
    evidence: list[Evidence]
    plausibility: Degree


# conclusão do diagnóstico, hypothesis_id nulo significa inconclusivo
class Conclusion(StrictModel):
    hypothesis_id: str | None
    confidence: Degree
    statement: str


# decisão de escalar ou não o atendimento, com o motivo
class Escalation(StrictModel):
    needed: bool
    reason: str


# saída do gerador de diagnóstico, evidências antes da conclusão
class DiagnosisOutput(StrictModel):
    hypotheses: list[Hypothesis]
    missing_information: list[str]
    conclusion: Conclusion
    customer_summary: str
    escalation: Escalation


# um passo da instrução ao cliente
class Step(StrictModel):
    action: str
    expected_result: str
    if_fails: str


# saída do gerador de passo a passo, subject só vale para o canal e-mail
class StepsOutput(StrictModel):
    opening: str
    steps: list[Step]
    closing: str
    escalation_condition: str
    subject: str | None


# pergunta de esclarecimento, priority 1 é a primeira a perguntar
class Question(StrictModel):
    priority: int
    text: str
    kind: QuestionKind
    options: list[str]
    resolves: str


# saída do gerador de perguntas de esclarecimento
class QuestionsOutput(StrictModel):
    questions: list[Question]


# nó da árvore de decisão
class FlowNode(StrictModel):
    id: str
    kind: NodeKind
    text: str


# aresta da árvore de decisão, com o rótulo da resposta que leva ao destino
class FlowEdge(StrictModel):
    source: str
    target: str
    label: str


# saída do gerador do resumo visual do fluxo
class FlowOutput(StrictModel):
    nodes: list[FlowNode]
    edges: list[FlowEdge]


# parâmetros de geração fixados no .env e gravados em cada execução
class GenerationParams(StrictModel):
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    seed: Annotated[int, Field(ge=0)]
    max_tokens: Annotated[int, Field(gt=0)]
    reasoning_effort: ReasoningEffort
    include_reasoning: bool


# mensagem enviada ao modelo
class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


# consumo de tokens de uma chamada
class Usage(StrictModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


# nota de 1 a 5 de um critério, com justificativa e o trecho da saída que a motivou
class CriterionScore(StrictModel):
    score: Annotated[int, Field(ge=1, le=5)]
    justification: NonEmptyText
    evidence: NonEmptyText


# avaliação de uma saída pela rubrica de 3 critérios
class Evaluation(StrictModel):
    clarity: CriterionScore
    customer_fit: CriterionScore
    actionability: CriterionScore
    evaluator: Evaluator
    evaluated_at: UtcDatetime


# contrato de saída de cada tipo de conteúdo
OUTPUT_MODELS: dict[ContentType, type[StrictModel]] = {
    ContentType.DIAGNOSIS: DiagnosisOutput,
    ContentType.STEPS: StepsOutput,
    ContentType.QUESTIONS: QuestionsOutput,
    ContentType.FLOW: FlowOutput,
}


# registro de uma execução, gravado como um JSON em runs/
class RunRecord(StrictModel):
    schema_version: SchemaVersion
    run_id: RunId
    timestamp: UtcDatetime
    customer_id: Slug
    ticket_id: Slug
    ticket: TicketText
    content_type: ContentType
    prompt_version: PromptVersion
    prompt_hash: Sha256Hash
    schema_hash: Sha256Hash
    provider: NonEmptyText
    model: NonEmptyText
    params: GenerationParams
    cache_key: Sha256Hash
    cache_hit: bool
    usage: Usage
    latency_ms: Annotated[int, Field(ge=0)]
    finish_reason: str | None
    repaired: bool
    status: RunStatus
    output: dict[str, Any] | None
    raw_output: str | None
    evaluation: Evaluation | None
    validation_errors: list[str] = []

    # status ok exige saída válida para o tipo, invalid_output exige o texto bruto
    @model_validator(mode="after")
    def _check_output(self) -> Self:
        if self.status is RunStatus.OK:
            if self.output is None or self.raw_output is not None:
                raise ValueError("status ok exige output e não aceita raw_output")
            OUTPUT_MODELS[self.content_type].model_validate(self.output)
        elif self.output is not None or self.raw_output is None:
            raise ValueError("status invalid_output exige raw_output e não aceita output")
        return self


# configuração fixa de uma comparação, sem a seed que varia entre as repetições
class LockedConfig(StrictModel):
    provider: NonEmptyText
    model: NonEmptyText
    temperature: float
    top_p: float
    max_tokens: int
    reasoning_effort: ReasoningEffort
    include_reasoning: bool


# uma execução de uma comparação: versão, seed, tipo e o resultado
class ComparisonRun(StrictModel):
    version: PromptVersion
    seed: int
    content_type: ContentType
    run_id: RunId
    cache_hit: bool
    status: RunStatus
    output: dict[str, Any] | None
    raw_output: str | None
    prompt_hash: Sha256Hash | None = None


# sorteio de qual versão aparece como A e qual como B num par de tipo e seed
class BlindItem(StrictModel):
    content_type: ContentType
    seed: int
    a: PromptVersion
    b: PromptVersion


# avaliação de uma saída, com a posição em que ela foi mostrada e a versão revelada depois
class EvaluatedOutput(StrictModel):
    position: Literal["A", "B"]
    version: PromptVersion
    evaluation: Evaluation


# as duas avaliações de um par de tipo e seed
class ContentEvaluation(StrictModel):
    content_type: ContentType
    seed: int
    outputs: list[EvaluatedOutput]


# arquivo de uma comparação: cenário, configuração travada, execuções, sorteio cego e avaliações
class ComparisonRecord(StrictModel):
    schema_version: SchemaVersion
    comparison_id: RunId
    created_at: UtcDatetime
    customer_id: Slug
    ticket_id: Slug
    ticket: TicketText
    versions: list[PromptVersion]
    seeds: list[int]
    content_types: list[ContentType]
    locked_config: LockedConfig
    schema_hashes: dict[ContentType, Sha256Hash]
    runs: list[ComparisonRun]
    blind_seed: int
    blind_map: list[BlindItem]
    evaluations: list[ContentEvaluation]
    profile: Profile | None = None
    rubric: NonEmptyText | None = None

    # exige duas versões diferentes, ao menos uma seed e nenhuma repetição de seed ou de tipo
    @model_validator(mode="after")
    def _check_scenario(self) -> Self:
        if len(self.versions) != 2 or self.versions[0] == self.versions[1]:
            raise ValueError("a comparação precisa de exatamente 2 versões diferentes")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("a comparação precisa de ao menos 1 seed, sem repetição")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("as seeds da comparação precisam ser não negativas")
        if not self.content_types or len(set(self.content_types)) != len(self.content_types):
            raise ValueError("a comparação precisa de ao menos 1 tipo de conteúdo, sem repetição")
        if self.profile is not None and self.profile.id != self.customer_id:
            raise ValueError("o perfil registrado precisa corresponder a customer_id")
        return self


# converte uma união X | None em {"type": [X, "null"]}, único formato de opcional aceito
def _nullable(options: list[dict[str, Any]]) -> dict[str, Any]:
    others = [option for option in options if option != {"type": "null"}]
    if len(options) == 2 and len(others) == 1 and set(others[0]) == {"type"}:
        return {"type": [others[0]["type"], "null"]}
    raise ValueError("união não suportada no schema de saída, use apenas X | None com X simples")


# simplifica um nó do schema: sem título, sem referências e com opcional como união com null
def _simplify_schema(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in schema:
        return _simplify_schema(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    result = {key: value for key, value in schema.items() if key not in ("title", "$defs")}
    if "properties" in result:
        result["properties"] = {
            name: _simplify_schema(sub, defs) for name, sub in result["properties"].items()
        }
    if "items" in result:
        result["items"] = _simplify_schema(result["items"], defs)
    if "anyOf" in result:
        options = [_simplify_schema(sub, defs) for sub in result.pop("anyOf")]
        result.update(_nullable(options))
    return result


# JSON Schema enviado ao modelo para um contrato, só com as palavras-chave básicas
def strict_json_schema(model: type[StrictModel]) -> dict[str, Any]:
    raw = model.model_json_schema()
    return _simplify_schema(raw, raw.get("$defs", {}))


# JSON Schema enviado ao modelo para um tipo de conteúdo
def output_json_schema(content_type: ContentType) -> dict[str, Any]:
    return strict_json_schema(OUTPUT_MODELS[content_type])
