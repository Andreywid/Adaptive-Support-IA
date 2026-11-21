from support_assistant.flow_render import render_flow, validate_flow
from support_assistant.models import ContentType, FlowOutput
from support_assistant.providers.fake import CANNED_OUTPUTS

EXPECTED_CANNED_DRAWING = """[P1] Todos os aparelhos perdem a conexão?
  sim -> [P2] A luz de sinal do roteador fica vermelha?
           sim -> [F1] Abrir chamado técnico de linha
           não -> [A1] Reiniciar o roteador e observar por uma noite
  não -> [F2] Verificar só o aparelho afetado"""


# monta um fluxo a partir de nós (id, tipo) e arestas (origem, destino, rótulo)
def _flow(nodes, edges):
    return FlowOutput.model_validate(
        {
            "nodes": [{"id": i, "kind": k, "text": f"texto {i}"} for i, k in nodes],
            "edges": [{"source": s, "target": t, "label": label} for s, t, label in edges],
        }
    )


# a saída fixa de fluxo do fake desenha exatamente a árvore esperada
def test_canned_flow_is_drawn_exactly():
    render = render_flow(CANNED_OUTPUTS[ContentType.FLOW])
    assert render.drawn is True
    assert render.warnings == []
    assert render.text == EXPECTED_CANNED_DRAWING


# fluxo sem nós é um bloqueio
def test_flow_without_nodes_is_blocked():
    assert validate_flow(_flow([], [])) == (["fluxo sem nós"], [])


# ids repetidos e aresta para nó inexistente são bloqueios, nesta ordem
def test_repeated_ids_and_dangling_edge_are_blocks():
    flow = _flow(
        [("P1", "question"), ("P1", "question"), ("F1", "end")],
        [("P1", "F1", "sim"), ("P1", "X9", "não")],
    )
    blocks, _ = validate_flow(flow)
    assert blocks[:2] == ["ids repetidos: P1", "aresta para nó inexistente: P1 -> X9"]


# aresta para nó inexistente não desenha e lista nós e arestas
def test_dangling_edge_blocks_the_drawing():
    flow = _flow(
        [("P1", "question"), ("F1", "end")],
        [("P1", "F1", "sim"), ("P1", "X9", "não")],
    )
    render = render_flow(flow)
    assert render.drawn is False
    assert render.text == (
        "Fluxo não desenhado:\n"
        "- aresta para nó inexistente: P1 -> X9\n"
        "\n"
        "Nós:\n"
        "[P1] (question) texto P1\n"
        "[F1] (end) texto F1\n"
        "\n"
        "Arestas:\n"
        "P1 -> F1 (sim)\n"
        "P1 -> X9 (não)"
    )


# todo nó com aresta de entrada significa que não há início
def test_no_start_is_blocked():
    flow = _flow(
        [("P1", "question"), ("P2", "question")],
        [("P1", "P2", "sim"), ("P2", "P1", "não")],
    )
    assert validate_flow(flow)[0] == ["nenhum início (todo nó tem aresta de entrada)"]


# dois inícios são um bloqueio que lista os dois
def test_two_starts_are_blocked():
    flow = _flow(
        [("P1", "question"), ("P3", "question"), ("F1", "end")],
        [("P1", "F1", "sim"), ("P3", "F1", "sim")],
    )
    render = render_flow(flow)
    assert render.drawn is False
    assert "- mais de um início: P1, P3" in render.text


# ciclo alcançável a partir do início é um bloqueio que mostra o caminho
def test_cycle_is_blocked_with_its_path():
    flow = _flow(
        [("S1", "question"), ("P1", "question"), ("P2", "question")],
        [("S1", "P1", "sim"), ("P1", "P2", "sim"), ("P2", "P1", "não")],
    )
    render = render_flow(flow)
    assert render.drawn is False
    assert "- ciclo no fluxo: P1 -> P2 -> P1" in render.text


# pergunta com uma só saída é aviso e o fluxo ainda é desenhado
def test_question_with_one_exit_warns_and_still_draws():
    flow = _flow(
        [("P1", "question"), ("F1", "end")],
        [("P1", "F1", "sim")],
    )
    render = render_flow(flow)
    assert render.drawn is True
    assert render.warnings == ["pergunta com menos de 2 saídas: P1"]
    assert render.text == (
        "[P1] texto P1\n  sim -> [F1] texto F1\n\nAvisos:\n- pergunta com menos de 2 saídas: P1"
    )


# final com saída e ação com duas saídas são avisos, na ordem das categorias
def test_end_with_exit_and_action_with_two_exits_warn():
    flow = _flow(
        [("P1", "question"), ("A1", "action"), ("F1", "end"), ("F2", "end")],
        [
            ("P1", "A1", "sim"),
            ("P1", "F2", "não"),
            ("A1", "F1", "ok"),
            ("A1", "F2", "falhou"),
            ("F1", "F2", "x"),
        ],
    )
    assert validate_flow(flow) == (
        [],
        ["final com saída: F1", "ação com mais de uma saída: A1"],
    )


# nós que só se ligam entre si, sem chegar ao início, são avisos e não impedem o desenho
def test_unreachable_nodes_warn():
    flow = _flow(
        [("P1", "question"), ("F1", "end"), ("F2", "end"), ("A2", "action"), ("A3", "action")],
        [("P1", "F1", "sim"), ("P1", "F2", "não"), ("A2", "A3", "ok"), ("A3", "A2", "ok")],
    )
    blocks, warnings = validate_flow(flow)
    assert blocks == []
    assert warnings == ["nó fora do caminho do início: A2", "nó fora do caminho do início: A3"]
    assert render_flow(flow).drawn is True


# nó alcançado por dois caminhos aparece nos dois
def test_node_reached_by_two_paths_is_drawn_twice():
    flow = _flow(
        [("P1", "question"), ("A1", "action"), ("A2", "action"), ("F1", "end")],
        [("P1", "A1", "sim"), ("P1", "A2", "não"), ("A1", "F1", "ok"), ("A2", "F1", "ok")],
    )
    assert render_flow(flow).text.count("[F1] texto F1") == 2


# o recuo dos filhos acompanha o tamanho do rótulo e quebras de linha do texto viram espaço
def test_indentation_follows_label_length_and_text_is_one_line():
    flow = FlowOutput.model_validate(
        {
            "nodes": [
                {"id": "P1", "kind": "question", "text": "Pergunta\ncom quebra"},
                {"id": "P2", "kind": "question", "text": "Segunda"},
                {"id": "F1", "kind": "end", "text": "Fim 1"},
                {"id": "F2", "kind": "end", "text": "Fim 2"},
                {"id": "F3", "kind": "end", "text": "Fim 3"},
            ],
            "edges": [
                {"source": "P1", "target": "P2", "label": "rótulo longo"},
                {"source": "P1", "target": "F3", "label": "b"},
                {"source": "P2", "target": "F1", "label": "a"},
                {"source": "P2", "target": "F2", "label": "b"},
            ],
        }
    )
    lines = render_flow(flow).text.splitlines()
    assert lines[0] == "[P1] Pergunta com quebra"
    assert lines[1] == "  rótulo longo -> [P2] Segunda"
    assert lines[2] == " " * (2 + len("rótulo longo -> ") + 2) + "a -> [F1] Fim 1"
