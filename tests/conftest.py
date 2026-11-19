import copy
from datetime import UTC, datetime
from typing import Any

import pytest

from support_assistant.models import (
    ContentType,
    GenerationParams,
    Message,
    ReasoningEffort,
    output_json_schema,
)

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64

# exemplos válidos de cada saída, no formato JSON que o modelo devolve
OUTPUT_SAMPLES: dict[ContentType, dict[str, Any]] = {
    ContentType.DIAGNOSIS: {
        "hypotheses": [
            {
                "id": "H1",
                "description": "Instabilidade do sinal em horário de pico",
                "evidence": [
                    {"quote": "cai toda noite", "reading": "o padrão se repete no mesmo período"}
                ],
                "plausibility": "medium",
            },
            {
                "id": "H2",
                "description": "Reinício automático do roteador",
                "evidence": [
                    {"quote": "volta sozinha", "reading": "a conexão se recupera sem ajuda"}
                ],
                "plausibility": "low",
            },
        ],
        "missing_information": [
            "Horário exato das quedas",
            "Se todos os aparelhos perdem a conexão",
        ],
        "conclusion": {
            "hypothesis_id": "H1",
            "confidence": "low",
            "statement": "Há indício de instabilidade noturna, mas faltam dados para confirmar.",
        },
        "customer_summary": "Vamos investigar por que a conexão cai à noite e precisamos de dados.",
        "escalation": {"needed": False, "reason": "Sem sinal de risco, o caso segue no nível 1."},
    },
    ContentType.STEPS: {
        "opening": "Vamos fazer um teste simples no roteador.",
        "steps": [
            {
                "action": "Desligue o roteador da tomada por 30 segundos e ligue de novo.",
                "expected_result": "As luzes acendem e a luz de internet fica verde.",
                "if_fails": "Me diga qual luz ficou vermelha ou apagada.",
            }
        ],
        "closing": "Se a internet cair de novo hoje à noite, ligue para nós.",
        "escalation_condition": "Luz de internet vermelha após o reinício: abrir chamado técnico.",
        "subject": None,
    },
    ContentType.QUESTIONS: {
        "questions": [
            {
                "priority": 1,
                "text": "A internet cai em todos os aparelhos ao mesmo tempo?",
                "kind": "yes_no",
                "options": [],
                "resolves": "Se o problema está na rede ou em um aparelho",
            },
            {
                "priority": 2,
                "text": "Em que horário a queda costuma começar?",
                "kind": "choice",
                "options": ["Antes das 20h", "Entre 20h e 23h", "Depois das 23h"],
                "resolves": "Horário exato das quedas",
            },
        ]
    },
    ContentType.FLOW: {
        "nodes": [
            {"id": "P1", "kind": "question", "text": "Todos os aparelhos perdem a conexão?"},
            {"id": "P2", "kind": "question", "text": "A luz de sinal do roteador fica vermelha?"},
            {"id": "A1", "kind": "action", "text": "Reiniciar o roteador e observar uma noite"},
            {"id": "F1", "kind": "end", "text": "Abrir chamado técnico de linha"},
            {"id": "F2", "kind": "end", "text": "Verificar só o aparelho afetado"},
        ],
        "edges": [
            {"source": "P1", "target": "P2", "label": "sim"},
            {"source": "P1", "target": "F2", "label": "não"},
            {"source": "P2", "target": "F1", "label": "sim"},
            {"source": "P2", "target": "A1", "label": "não"},
        ],
    },
}


# cópia dos exemplos de saída, segura para cada teste alterar
@pytest.fixture
def output_samples() -> dict[ContentType, dict[str, Any]]:
    return copy.deepcopy(OUTPUT_SAMPLES)


# parâmetros de geração válidos, usados como base nos testes
@pytest.fixture
def params() -> GenerationParams:
    return GenerationParams(
        temperature=1.0,
        top_p=1.0,
        seed=1,
        max_tokens=2048,
        reasoning_effort=ReasoningEffort.LOW,
        include_reasoning=False,
    )


# fábrica de requisições de provedor válidas, com campos sobrescritos por quem chama
@pytest.fixture
def make_request(params):
    # import tardio: o conftest carrega mesmo antes de existir o pacote de provedores
    from support_assistant.providers.base import ProviderRequest

    # monta uma requisição de diagnóstico com o schema real da saída
    def _make(**overrides: Any) -> ProviderRequest:
        data = {
            "model": "fake-model",
            "messages": [
                Message(role="system", content="Você é um assistente de suporte técnico."),
                Message(role="user", content="Ticket: A internet cai toda noite e volta sozinha"),
            ],
            "schema_name": "diagnosis",
            "response_schema": output_json_schema(ContentType.DIAGNOSIS),
            "params": params,
        }
        return ProviderRequest(**{**data, **overrides})

    return _make


# dados de uma avaliação válida
@pytest.fixture
def evaluation_data() -> dict[str, Any]:
    return {
        "clarity": {
            "score": 4,
            "justification": "Cada passo tem um só sentido e segue ordem lógica.",
            "evidence": "Desligue o roteador da tomada por 30 segundos",
        },
        "customer_fit": {
            "score": 5,
            "justification": "Frases curtas e sem jargão, próprias para leitura ao telefone.",
            "evidence": "As luzes acendem e a luz de internet fica verde.",
        },
        "actionability": {
            "score": 3,
            "justification": "Há resultado esperado, mas falta uma alternativa clara.",
            "evidence": "Me diga qual luz ficou vermelha ou apagada.",
        },
        "evaluator": "human",
        "evaluated_at": datetime(2026, 11, 20, 12, 30, tzinfo=UTC),
    }


# fábrica dos dados de um registro de execução válido, com campos sobrescritos por quem chama
@pytest.fixture
def make_record_data(output_samples, params):
    # monta o dicionário de um registro ok do tipo diagnóstico
    def _make(**overrides: Any) -> dict[str, Any]:
        data = {
            "schema_version": "1.0",
            "run_id": "20261120T120000Z-a1b2c3d4",
            "timestamp": datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
            "customer_id": "p01_marta",
            "ticket_id": "t01_queda_noturna",
            "ticket": "A internet cai toda noite e volta sozinha",
            "content_type": "diagnosis",
            "prompt_version": "v2",
            "prompt_hash": HASH_A,
            "schema_hash": HASH_B,
            "provider": "fake",
            "model": "fake-model",
            "params": params.model_dump(mode="json"),
            "cache_key": HASH_C,
            "cache_hit": False,
            "usage": {"input_tokens": 120, "output_tokens": 80},
            "latency_ms": 350,
            "finish_reason": "stop",
            "repaired": False,
            "status": "ok",
            "output": output_samples[ContentType.DIAGNOSIS],
            "raw_output": None,
            "evaluation": None,
        }
        return {**data, **overrides}

    return _make


# guarda as comparações de teste fora do workspace, quando o módulo já existe
@pytest.fixture(autouse=True)
def isolated_output_folders(tmp_path_factory, monkeypatch):
    try:
        from support_assistant.evaluation import compare
    except ImportError:
        return

    monkeypatch.setattr(compare, "COMPARISONS_DIR", tmp_path_factory.mktemp("comparisons"))
