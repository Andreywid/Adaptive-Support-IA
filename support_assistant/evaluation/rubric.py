from support_assistant.models import ComparisonRecord, ContentEvaluation, ContentType

CRITERIA = (
    ("clarity", "Clareza"),
    ("customer_fit", "Adequação ao nível do cliente"),
    ("actionability", "Acionabilidade"),
)

CONTENT_READERS = {
    ContentType.DIAGNOSIS: "atendente e cliente, que recebe o resumo",
    ContentType.STEPS: "cliente",
    ContentType.QUESTIONS: "cliente",
    ContentType.FLOW: "atendente",
}

# âncoras de cada critério para as notas 1, 3 e 5
ANCHORS = {
    "clarity": {
        1: "Desorganizado, contraditório ou ambíguo em ponto central.",
        3: "Compreensível, com 1 ou 2 trechos ambíguos.",
        5: "Cada item tem um só sentido, em ordem lógica, sem redundância.",
    },
    "customer_fit": {
        1: "Registro errado para o perfil (ex.: comando de terminal para leigo ao telefone).",
        3: "Adequado na maior parte, com 1 ou 2 desvios (ex.: jargão sem explicação).",
        5: (
            "Vocabulário, profundidade e formato combinam com nível, estilo e canal "
            "em todos os itens."
        ),
    },
    "actionability": {
        1: "Não dá para saber o que fazer.",
        3: "Próximo passo claro, sem critério de sucesso nem alternativa.",
        5: "Próximo passo inequívoco, com resultado esperado e o que fazer se falhar.",
    },
}


# texto da rubrica para quem avalia: um bloco por critério, com as âncoras
def format_rubric() -> str:
    blocks = []
    for key, label in CRITERIA:
        lines = [f"{label} ({key})"]
        lines += [f"  {score}: {ANCHORS[key][score]}" for score in (1, 3, 5)]
        lines.append("  2 e 4: entre as âncoras vizinhas.")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# pares completos de uma seed, por tipo, com as duas versões nas posições sorteadas
def complete_reviews(record: ComparisonRecord, seed: int) -> dict[ContentType, ContentEvaluation]:
    expected_versions = set(record.versions)
    positions = {
        item.content_type: {"A": item.a, "B": item.b}
        for item in record.blind_map
        if item.seed == seed
    }
    return {
        item.content_type: item
        for item in record.evaluations
        if item.seed == seed
        and len(item.outputs) == 2
        and {output.version for output in item.outputs} == expected_versions
        and {output.position: output.version for output in item.outputs}
        == positions.get(item.content_type)
    }


# diz se alguma seed tem todos os tipos avaliados nas duas versões
def has_complete_evaluation(record: ComparisonRecord) -> bool:
    expected_types = set(record.content_types)
    return any(expected_types <= complete_reviews(record, seed).keys() for seed in record.seeds)


# tabela com a média de cada critério por versão, nas seeds com avaliação completa
def rubric_summary(record: ComparisonRecord) -> str:
    scores: dict[tuple[str, str], list[int]] = {}
    for seed in record.seeds:
        reviews = complete_reviews(record, seed)
        if set(record.content_types) <= reviews.keys():
            for review in reviews.values():
                for output in review.outputs:
                    for key, _ in CRITERIA:
                        score = getattr(output.evaluation, key).score
                        scores.setdefault((output.version, key), []).append(score)
    first, second = record.versions
    lines = [f"{'critério':<32}{first:>6}{second:>6}"]
    for key, label in CRITERIA:
        means = [sum(scores[(v, key)]) / len(scores[(v, key)]) for v in (first, second)]
        lines.append(f"{label:<32}{means[0]:>6.2f}{means[1]:>6.2f}")
    return "\n".join(lines)
