from support_assistant.evaluation.rubric import ANCHORS, CRITERIA, format_rubric
from support_assistant.models import Evaluation

FIRST_BLOCK = """Clareza (clarity)
  1: Desorganizado, contraditório ou ambíguo em ponto central.
  3: Compreensível, com 1 ou 2 trechos ambíguos.
  5: Cada item tem um só sentido, em ordem lógica, sem redundância.
  2 e 4: entre as âncoras vizinhas."""


# os nomes dos critérios são exatamente os campos de nota da avaliação
def test_criteria_names_are_the_evaluation_score_fields():
    score_fields = [
        name for name in Evaluation.model_fields if name not in ("evaluator", "evaluated_at")
    ]
    assert [key for key, _ in CRITERIA] == score_fields


# cada critério tem as âncoras 1, 3 e 5, e nenhuma outra
def test_each_criterion_has_anchors_one_three_and_five():
    assert set(ANCHORS) == {key for key, _ in CRITERIA}
    for anchors in ANCHORS.values():
        assert set(anchors) == {1, 3, 5}
        assert all(text.endswith(".") and text.strip() for text in anchors.values())


# a rubrica formatada traz os 3 nomes, com o rótulo e o campo
def test_format_rubric_contains_the_three_criteria():
    text = format_rubric()
    for key, label in CRITERIA:
        assert f"{label} ({key})" in text
    assert text.count("2 e 4: entre as âncoras vizinhas.") == 3


# o primeiro bloco tem exatamente o formato da especificação
def test_format_rubric_first_block_layout():
    assert format_rubric().split("\n\n")[0] == FIRST_BLOCK


# os blocos são separados por uma linha vazia e o texto não termina com quebra de linha
def test_format_rubric_blocks_are_separated_by_one_blank_line():
    text = format_rubric()
    assert len(text.split("\n\n")) == 3
    assert "\n\n\n" not in text
    assert not text.endswith("\n")


# as 9 âncoras pré-registradas não mudam sem querer
def test_anchor_texts_are_frozen():
    assert ANCHORS == {
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
