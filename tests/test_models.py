import json
from datetime import UTC, datetime, timedelta, timezone
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

import support_assistant.models as models_module
from support_assistant.models import (
    OUTPUT_MODELS,
    SCHEMA_VERSION,
    TICKET_MAX_CHARS,
    Channel,
    CommunicationStyle,
    ComparisonRecord,
    ContentType,
    EvaluatedOutput,
    Evaluation,
    GenerationParams,
    Profile,
    ProfileCatalog,
    RunRecord,
    SchemaVersion,
    StrictModel,
    TechnicalLevel,
    Ticket,
    TicketCatalog,
    Usage,
    _simplify_schema,
    output_json_schema,
    strict_json_schema,
)

# campos de cada saída, na ordem do schema (evidências antes da conclusão)
EXPECTED_FIELDS = {
    ContentType.DIAGNOSIS: [
        "hypotheses",
        "missing_information",
        "conclusion",
        "customer_summary",
        "escalation",
    ],
    ContentType.STEPS: ["opening", "steps", "closing", "escalation_condition", "subject"],
    ContentType.QUESTIONS: ["questions"],
    ContentType.FLOW: ["nodes", "edges"],
}

# palavras-chave de schema permitidas no que é enviado ao modelo
ALLOWED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum"}

PROFILE = {
    "id": "p01_marta",
    "name": "Marta",
    "technical_level": "leigo",
    "communication_style": "didatico",
    "channel": "telefone",
}

# campos do registro de execução
RUN_RECORD_FIELDS = [
    "schema_version",
    "run_id",
    "timestamp",
    "customer_id",
    "ticket_id",
    "ticket",
    "content_type",
    "prompt_version",
    "prompt_hash",
    "schema_hash",
    "provider",
    "model",
    "params",
    "cache_key",
    "cache_hit",
    "usage",
    "latency_ms",
    "finish_reason",
    "repaired",
    "status",
    "output",
    "raw_output",
    "evaluation",
    "validation_errors",
]


# percorre o schema e entrega todo nó que é um objeto
def _object_nodes(node):
    if node.get("type") == "object":
        yield node
    for sub in node.get("properties", {}).values():
        yield from _object_nodes(sub)
    if "items" in node:
        yield from _object_nodes(node["items"])


# reúne as palavras-chave usadas no schema, sem contar nomes de propriedades
def _keywords(node):
    found = set(node)
    for sub in node.get("properties", {}).values():
        found |= _keywords(sub)
    if "items" in node:
        found |= _keywords(node["items"])
    return found


# todo objeto do schema de cada saída tem additionalProperties falso e todos os campos em required
@pytest.mark.parametrize("content_type", list(ContentType))
def test_output_schema_is_strict(content_type):
    objects = list(_object_nodes(output_json_schema(content_type)))
    assert objects
    for node in objects:
        assert node["additionalProperties"] is False
        assert node["required"] == list(node["properties"])


# a checagem de rigor alcança os objetos aninhados e não só a raiz
def test_strict_check_reaches_nested_objects():
    objects = list(_object_nodes(output_json_schema(ContentType.DIAGNOSIS)))
    assert len(objects) == 5


# o schema enviado usa só o vocabulário básico, sem referências, títulos nem descrições
@pytest.mark.parametrize("content_type", list(ContentType))
def test_output_schema_uses_only_basic_keywords(content_type):
    assert _keywords(output_json_schema(content_type)) <= ALLOWED_KEYWORDS


# os campos de cada saída aparecem no schema na ordem esperada
@pytest.mark.parametrize("content_type", list(ContentType))
def test_output_fields_order(content_type):
    schema = output_json_schema(content_type)
    assert list(schema["properties"]) == EXPECTED_FIELDS[content_type]


# no diagnóstico as evidências vêm antes da conclusão, porque a saída segue a ordem das chaves
def test_diagnosis_puts_evidence_before_conclusion():
    properties = output_json_schema(ContentType.DIAGNOSIS)["properties"]
    fields = list(properties)
    assert fields.index("hypotheses") < fields.index("conclusion")
    inner = list(properties["hypotheses"]["items"]["properties"])
    assert inner.index("evidence") < inner.index("plausibility")


# campo opcional vira união com null e continua em required
def test_optional_fields_are_nullable_unions_and_still_required():
    steps = output_json_schema(ContentType.STEPS)
    assert steps["properties"]["subject"] == {"type": ["string", "null"]}
    assert "subject" in steps["required"]
    conclusion = output_json_schema(ContentType.DIAGNOSIS)["properties"]["conclusion"]
    assert conclusion["properties"]["hypothesis_id"] == {"type": ["string", "null"]}
    assert "hypothesis_id" in conclusion["required"]


# enums aparecem inline, sem $ref
def test_enums_are_inlined_without_references():
    hypothesis = output_json_schema(ContentType.DIAGNOSIS)["properties"]["hypotheses"]["items"]
    assert hypothesis["properties"]["plausibility"] == {
        "enum": ["low", "medium", "high"],
        "type": "string",
    }


# modelo com união que o simplificador não sabe converter
class _BrokenOutput(StrictModel):
    value: int | str


# união fora do padrão X | None falha alto em vez de gerar um schema diferente
def test_schema_simplifier_rejects_unsupported_unions():
    raw = _BrokenOutput.model_json_schema()
    with pytest.raises(ValueError, match="união não suportada"):
        _simplify_schema(raw, raw.get("$defs", {}))


# o exemplo válido de cada saída passa pelo contrato e volta idêntico em JSON
@pytest.mark.parametrize("content_type", list(ContentType))
def test_valid_sample_roundtrips(content_type, output_samples):
    sample = output_samples[content_type]
    model = OUTPUT_MODELS[content_type].model_validate(sample)
    assert model.model_dump(mode="json") == sample
    assert OUTPUT_MODELS[content_type].model_validate_json(json.dumps(sample)) == model


# campo extra na saída do modelo é rejeitado
@pytest.mark.parametrize("content_type", list(ContentType))
def test_output_rejects_extra_field(content_type, output_samples):
    sample = {**output_samples[content_type], "campo_extra": 1}
    with pytest.raises(ValidationError):
        OUTPUT_MODELS[content_type].model_validate(sample)


# faltar qualquer campo de qualquer saída é rejeitado
@pytest.mark.parametrize(
    ("content_type", "field"),
    [(kind, name) for kind, model in OUTPUT_MODELS.items() for name in model.model_fields],
)
def test_output_requires_every_field(content_type, field, output_samples):
    sample = output_samples[content_type]
    del sample[field]
    with pytest.raises(ValidationError):
        OUTPUT_MODELS[content_type].model_validate(sample)


# campo opcional aninhado precisa estar presente, mesmo que nulo
def test_nullable_field_must_still_be_present(output_samples):
    sample = output_samples[ContentType.DIAGNOSIS]
    del sample["conclusion"]["hypothesis_id"]
    with pytest.raises(ValidationError):
        OUTPUT_MODELS[ContentType.DIAGNOSIS].model_validate(sample)


# diagnóstico inconclusivo é uma resposta válida
def test_inconclusive_diagnosis_is_valid(output_samples):
    sample = output_samples[ContentType.DIAGNOSIS]
    sample["conclusion"]["hypothesis_id"] = None
    model = OUTPUT_MODELS[ContentType.DIAGNOSIS].model_validate(sample)
    assert model.conclusion.hypothesis_id is None


# valores aceitos em cada dimensão do perfil
def test_profile_dimension_values():
    assert {item.value for item in TechnicalLevel} == {"leigo", "intermediario", "avancado"}
    assert {item.value for item in CommunicationStyle} == {"direto", "detalhado", "didatico"}
    assert {item.value for item in Channel} == {"telefone", "chat", "email"}


# perfil válido é aceito
def test_valid_profile_is_accepted():
    profile = Profile.model_validate(PROFILE)
    assert profile.channel is Channel.TELEFONE


# valor fora do enum, id fora do padrão e campo extra são rejeitados
@pytest.mark.parametrize(
    "changes",
    [
        {"technical_level": "especialista"},
        {"communication_style": "formal"},
        {"channel": "whatsapp"},
        {"id": "P01 Marta"},
        {"id": ""},
        {"campo_extra": 1},
    ],
)
def test_invalid_profile_is_rejected(changes):
    with pytest.raises(ValidationError):
        Profile.model_validate({**PROFILE, **changes})


# faltar qualquer campo do perfil é rejeitado
@pytest.mark.parametrize("field", list(PROFILE))
def test_profile_requires_every_field(field):
    data = dict(PROFILE)
    del data[field]
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


# catálogo de perfis válido é aceito
def test_valid_profile_catalog_is_accepted():
    other = {**PROFILE, "id": "p02_rafael"}
    catalog = ProfileCatalog.model_validate({"schema_version": "1.0", "profiles": [PROFILE, other]})
    assert [profile.id for profile in catalog.profiles] == ["p01_marta", "p02_rafael"]


# catálogo com id repetido ou versão desconhecida é rejeitado
@pytest.mark.parametrize(
    "data",
    [
        {"schema_version": "1.0", "profiles": [PROFILE, PROFILE]},
        {"schema_version": "9.9", "profiles": [PROFILE]},
        {"profiles": [PROFILE]},
    ],
)
def test_invalid_profile_catalog_is_rejected(data):
    with pytest.raises(ValidationError):
        ProfileCatalog.model_validate(data)


# a mensagem de id repetido diz qual id se repetiu
def test_duplicated_profile_id_is_named_in_the_error():
    data = {"schema_version": "1.0", "profiles": [PROFILE, PROFILE]}
    with pytest.raises(ValidationError, match="p01_marta"):
        ProfileCatalog.model_validate(data)


# catálogo de tickets rejeita id repetido
def test_duplicated_ticket_id_is_rejected():
    ticket = {"id": "t01_queda_noturna", "text": "A internet cai toda noite"}
    with pytest.raises(ValidationError, match="t01_queda_noturna"):
        TicketCatalog.model_validate({"schema_version": "1.0", "tickets": [ticket, ticket]})


# catálogo de tickets válido é aceito
def test_valid_ticket_catalog_is_accepted():
    ticket = {"id": "t01_queda_noturna", "text": "A internet cai toda noite"}
    catalog = TicketCatalog.model_validate({"schema_version": "1.0", "tickets": [ticket]})
    assert catalog.tickets[0].id == "t01_queda_noturna"


# espaços nas pontas do ticket são removidos
def test_ticket_text_is_trimmed():
    ticket = Ticket(id="t01_queda_noturna", text="  A internet cai  \n")
    assert ticket.text == "A internet cai"


# ticket vazio ou só com espaços é rejeitado antes de qualquer chamada
@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_ticket_is_rejected(text):
    with pytest.raises(ValidationError):
        Ticket(id="t01_queda_noturna", text=text)


# ticket no limite passa e um caractere acima é rejeitado
def test_ticket_length_limit():
    assert Ticket(id="t01_queda_noturna", text="x" * TICKET_MAX_CHARS)
    with pytest.raises(ValidationError):
        Ticket(id="t01_queda_noturna", text="x" * (TICKET_MAX_CHARS + 1))


# campos dos parâmetros de geração, na ordem gravada
def test_generation_params_fields():
    assert list(GenerationParams.model_fields) == [
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "reasoning_effort",
        "include_reasoning",
    ]


# temperatura inteira é aceita e vira float
def test_integer_temperature_becomes_float(params):
    data = {**params.model_dump(mode="json"), "temperature": 1}
    assert GenerationParams.model_validate(data).temperature == 1.0


# valores fora dos limites dos parâmetros são rejeitados
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", -0.1),
        ("temperature", 2.1),
        ("temperature", float("nan")),
        ("top_p", 0),
        ("top_p", 1.1),
        ("seed", -1),
        ("max_tokens", 0),
        ("reasoning_effort", "none"),
        ("include_reasoning", "talvez"),
    ],
)
def test_invalid_params_are_rejected(field, value, params):
    data = {**params.model_dump(mode="json"), field: value}
    with pytest.raises(ValidationError):
        GenerationParams.model_validate(data)


# consumo de tokens negativo é rejeitado
@pytest.mark.parametrize("field", ["input_tokens", "output_tokens"])
def test_negative_usage_is_rejected(field):
    data = {"input_tokens": 1, "output_tokens": 1, field: -1}
    with pytest.raises(ValidationError):
        Usage.model_validate(data)


# avaliação válida é aceita
def test_valid_evaluation_is_accepted(evaluation_data):
    evaluation = Evaluation.model_validate(evaluation_data)
    assert evaluation.clarity.score == 4


# nota fora de 1 a 5 é rejeitada
@pytest.mark.parametrize("score", [-1, 0, 6])
def test_score_out_of_range_is_rejected(score, evaluation_data):
    evaluation_data["clarity"]["score"] = score
    with pytest.raises(ValidationError):
        Evaluation.model_validate(evaluation_data)


# justificativa ou trecho citado em branco é rejeitado
@pytest.mark.parametrize("field", ["justification", "evidence"])
def test_blank_justification_or_evidence_is_rejected(field, evaluation_data):
    evaluation_data["actionability"][field] = "   "
    with pytest.raises(ValidationError):
        Evaluation.model_validate(evaluation_data)


# os 3 critérios são obrigatórios
@pytest.mark.parametrize("criterion", ["clarity", "customer_fit", "actionability"])
def test_evaluation_requires_the_three_criteria(criterion, evaluation_data):
    del evaluation_data[criterion]
    with pytest.raises(ValidationError):
        Evaluation.model_validate(evaluation_data)


# data da avaliação sem fuso é rejeitada
def test_naive_evaluation_time_is_rejected(evaluation_data):
    evaluation_data["evaluated_at"] = datetime(2026, 11, 20, 12, 30)
    with pytest.raises(ValidationError):
        Evaluation.model_validate(evaluation_data)


# registro válido é aceito
def test_valid_record_is_accepted(make_record_data):
    record = RunRecord.model_validate(make_record_data())
    assert record.content_type is ContentType.DIAGNOSIS


# registro grava e lê em JSON sem perder nada
def test_record_roundtrips_through_json(make_record_data):
    record = RunRecord.model_validate(make_record_data())
    assert RunRecord.model_validate_json(record.model_dump_json()) == record


# horário com outro fuso é gravado em UTC
def test_record_timestamp_is_serialized_in_utc(make_record_data):
    brasilia = timezone(timedelta(hours=-3))
    data = make_record_data(timestamp=datetime(2026, 11, 20, 9, 0, tzinfo=brasilia))
    payload = json.loads(RunRecord.model_validate(data).model_dump_json())
    assert payload["timestamp"] == "2026-11-20T12:00:00Z"


# o registro tem exatamente estes campos, nesta ordem
def test_run_record_fields():
    assert list(RunRecord.model_fields) == RUN_RECORD_FIELDS


# em cache hit não há chamada nova, então finish_reason pode ser nulo
def test_record_accepts_null_finish_reason(make_record_data):
    record = RunRecord.model_validate(make_record_data(cache_hit=True, finish_reason=None))
    assert record.finish_reason is None


# registro sem finish_reason ou sem repaired é rejeitado
@pytest.mark.parametrize("field", ["finish_reason", "repaired"])
def test_record_requires_finish_reason_and_repaired(field, make_record_data):
    data = make_record_data()
    del data[field]
    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


# saída que só passou depois do reparo fica marcada no registro
def test_record_keeps_the_repaired_flag(make_record_data):
    assert RunRecord.model_validate(make_record_data(repaired=True)).repaired is True


# perfil sem nome é rejeitado
def test_profile_without_name_is_rejected():
    data = {key: value for key, value in PROFILE.items() if key != "name"}
    with pytest.raises(ValidationError):
        Profile.model_validate(data)


# perfil com nome em branco é rejeitado
def test_profile_with_blank_name_is_rejected():
    with pytest.raises(ValidationError):
        Profile.model_validate({**PROFILE, "name": "   "})


# todo campo do registro é obrigatório, menos os motivos de validação de registros antigos
@pytest.mark.parametrize(
    "field", [name for name in RunRecord.model_fields if name != "validation_errors"]
)
def test_record_requires_every_field(field, make_record_data):
    data = make_record_data()
    del data[field]
    with pytest.raises(ValidationError):
        RunRecord.model_validate(data)


# valores inválidos em campos do registro são rejeitados
@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "2.0"},
        {"run_id": "../fora"},
        {"prompt_version": "versao2"},
        {"prompt_hash": "md5:abc"},
        {"cache_key": "sha256:" + "A" * 64},
        {"customer_id": "Marta"},
        {"ticket": "   "},
        {"latency_ms": -1},
        {"timestamp": datetime(2026, 11, 20, 12, 0)},
        {"provider": ""},
        {"campo_extra": 1},
    ],
)
def test_invalid_record_is_rejected(changes, make_record_data):
    with pytest.raises(ValidationError):
        RunRecord.model_validate(make_record_data(**changes))


# status ok sem saída é rejeitado
def test_ok_status_requires_output(make_record_data):
    with pytest.raises(ValidationError, match="status ok"):
        RunRecord.model_validate(make_record_data(output=None))


# status ok com texto bruto é rejeitado
def test_ok_status_rejects_raw_output(make_record_data):
    with pytest.raises(ValidationError, match="status ok"):
        RunRecord.model_validate(make_record_data(raw_output="texto"))


# status invalid_output guarda só o texto bruto, mesmo que ele seja vazio
@pytest.mark.parametrize("raw", ['{"quebrado": ', ""])
def test_invalid_output_status_keeps_only_raw_text(raw, make_record_data):
    data = make_record_data(status="invalid_output", output=None, raw_output=raw)
    assert RunRecord.model_validate(data).raw_output == raw


# status invalid_output com saída ou sem texto bruto é rejeitado
@pytest.mark.parametrize("changes", [{"raw_output": None}, {"raw_output": "texto"}])
def test_invalid_output_status_rejects_inconsistent_fields(
    changes, make_record_data, output_samples
):
    base = {"status": "invalid_output", "output": output_samples[ContentType.DIAGNOSIS]}
    with pytest.raises(ValidationError, match="invalid_output"):
        RunRecord.model_validate(make_record_data(**{**base, **changes}))


# a saída precisa respeitar o contrato do tipo declarado no registro
def test_output_must_match_the_content_type(make_record_data):
    with pytest.raises(ValidationError):
        RunRecord.model_validate(make_record_data(content_type="steps"))


# cada saída válida é aceita no registro do seu próprio tipo
@pytest.mark.parametrize("content_type", list(ContentType))
def test_record_accepts_each_output_type(content_type, make_record_data, output_samples):
    data = make_record_data(content_type=content_type, output=output_samples[content_type])
    assert RunRecord.model_validate(data).content_type is content_type


# registro aceita a avaliação embutida
def test_record_accepts_an_evaluation(make_record_data, evaluation_data):
    record = RunRecord.model_validate(make_record_data(evaluation=evaluation_data))
    assert record.evaluation.customer_fit.score == 5


# a constante de versão e o tipo literal dos contratos não podem divergir
def test_schema_version_constant_matches_the_literal_type():
    assert get_args(SchemaVersion) == (SCHEMA_VERSION,)


# modelos Pydantic definidos em um módulo, sem contar os importados
def _models_defined_in(module):
    return [
        item
        for item in vars(module).values()
        if isinstance(item, type)
        and issubclass(item, BaseModel)
        and item.__module__ == module.__name__
    ]


# todo contrato do módulo proíbe campos extras e é imutável
def test_every_contract_forbids_extra_fields_and_is_frozen():
    found = _models_defined_in(models_module)
    assert found
    for model in found:
        assert model.model_config.get("extra") == "forbid", model.__name__
        assert model.model_config.get("frozen") is True, model.__name__


# campos do arquivo de comparação, na ordem gravada
COMPARISON_FIELDS = [
    "schema_version",
    "comparison_id",
    "created_at",
    "customer_id",
    "ticket_id",
    "ticket",
    "versions",
    "seeds",
    "content_types",
    "locked_config",
    "schema_hashes",
    "runs",
    "blind_seed",
    "blind_map",
    "evaluations",
]


# dados mínimos de uma comparação válida, com campos sobrescritos por quem chama
def _comparison_data(**overrides):
    data = {
        "schema_version": "1.0",
        "comparison_id": "cmp_20261120T153012Z",
        "created_at": datetime(2026, 11, 20, 15, 30, 12, tzinfo=UTC),
        "customer_id": "p01_marta",
        "ticket_id": "t01_queda_noturna",
        "ticket": "A internet cai toda noite",
        "versions": ["v1", "v2"],
        "seeds": [1, 2, 3],
        "content_types": ["diagnosis", "flow"],
        "locked_config": {
            "provider": "groq",
            "model": "openai/gpt-oss-120b",
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": 2048,
            "reasoning_effort": "low",
            "include_reasoning": False,
        },
        "schema_hashes": {"diagnosis": "sha256:" + "a" * 64, "flow": "sha256:" + "b" * 64},
        "runs": [],
        "blind_seed": 123456,
        "blind_map": [{"content_type": "flow", "seed": 1, "a": "v2", "b": "v1"}],
        "evaluations": [],
    }
    return {**data, **overrides}


# a comparação mínima válida é aceita e volta idêntica depois de ida e volta em JSON
def test_comparison_record_roundtrips():
    record = ComparisonRecord.model_validate(_comparison_data())
    assert ComparisonRecord.model_validate_json(record.model_dump_json()) == record
    assert record.schema_hashes[ContentType.FLOW].endswith("b" * 64)


# campos do arquivo de comparação, na ordem gravada
def test_comparison_record_fields():
    assert list(ComparisonRecord.model_fields) == [*COMPARISON_FIELDS, "profile", "rubric"]


# outros campos extras continuam recusados
def test_comparison_still_rejects_other_extra_fields():
    with pytest.raises(ValidationError, match="other"):
        ComparisonRecord.model_validate(_comparison_data(other=[]))


# versões iguais, em número diferente de 2, são rejeitadas
@pytest.mark.parametrize("versions", [["v1", "v1"], ["v1"], ["v1", "v2", "v3"], []])
def test_comparison_needs_exactly_two_different_versions(versions):
    with pytest.raises(ValidationError, match="2 versões diferentes"):
        ComparisonRecord.model_validate(_comparison_data(versions=versions))


# seeds vazias ou repetidas são rejeitadas
@pytest.mark.parametrize("seeds", [[], [1, 1], [1, 2, 1]])
def test_comparison_needs_unique_seeds(seeds):
    with pytest.raises(ValidationError, match="ao menos 1 seed, sem repetição"):
        ComparisonRecord.model_validate(_comparison_data(seeds=seeds))


# seeds negativas são recusadas
def test_comparison_rejects_negative_seeds():
    with pytest.raises(ValidationError, match="não negativas"):
        ComparisonRecord.model_validate(_comparison_data(seeds=[1, -1]))


# tipos vazios ou repetidos são rejeitados
@pytest.mark.parametrize("types", [[], ["flow", "flow"]])
def test_comparison_needs_unique_content_types(types):
    with pytest.raises(ValidationError, match="tipo de conteúdo"):
        ComparisonRecord.model_validate(_comparison_data(content_types=types))


# todo campo da comparação é obrigatório
@pytest.mark.parametrize("field", COMPARISON_FIELDS)
def test_comparison_requires_every_field(field):
    data = _comparison_data()
    del data[field]
    with pytest.raises(ValidationError):
        ComparisonRecord.model_validate(data)


# a posição de uma saída avaliada é só A ou B
@pytest.mark.parametrize("position", ["C", "a", ""])
def test_evaluated_output_position_is_a_or_b(position, evaluation_data):
    with pytest.raises(ValidationError):
        EvaluatedOutput.model_validate(
            {"position": position, "version": "v1", "evaluation": evaluation_data}
        )


# o schema público de qualquer contrato é o mesmo que o de saída gera para os 4 tipos
@pytest.mark.parametrize("content_type", list(ContentType))
def test_output_schema_is_the_strict_schema_of_its_model(content_type):
    assert output_json_schema(content_type) == strict_json_schema(
        models_module.OUTPUT_MODELS[content_type]
    )
