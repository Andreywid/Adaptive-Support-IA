from support_assistant.models import FlowEdge, FlowNode, FlowOutput, NodeKind, StrictModel


# fluxo desenhado em texto, com os avisos e se o desenho foi possível
class FlowRender(StrictModel):
    text: str
    warnings: list[str]
    drawn: bool


# ids que aparecem mais de uma vez, na ordem da primeira aparição
def _repeated_ids(nodes: list[FlowNode]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for node in nodes:
        if node.id in seen and node.id not in repeated:
            repeated.append(node.id)
        seen.add(node.id)
    return repeated


# primeiro ciclo achado por busca em profundidade a partir de um nó, na ordem das arestas
def _find_cycle(start: str, children: dict[str, list[str]]) -> list[str] | None:
    path: list[str] = []
    done: set[str] = set()

    # visita um nó e devolve o ciclo se encontrar um nó que já está no caminho
    def visit(node: str) -> list[str] | None:
        if node in path:
            return path[path.index(node) :] + [node]
        if node in done:
            return None
        path.append(node)
        for child in children.get(node, []):
            cycle = visit(child)
            if cycle:
                return cycle
        path.pop()
        done.add(node)
        return None

    return visit(start)


# arestas que ligam nós que existem, base de toda a análise do grafo
def _valid_edges(flow: FlowOutput) -> list[FlowEdge]:
    ids = {node.id for node in flow.nodes}
    return [edge for edge in flow.edges if edge.source in ids and edge.target in ids]


# confere o fluxo e devolve (bloqueios que impedem o desenho, avisos que não impedem)
def validate_flow(flow: FlowOutput) -> tuple[list[str], list[str]]:
    if not flow.nodes:
        return ["fluxo sem nós"], []
    blocks: list[str] = []
    warnings: list[str] = []
    repeated = _repeated_ids(flow.nodes)
    if repeated:
        blocks.append(f"ids repetidos: {', '.join(repeated)}")
    ids = {node.id for node in flow.nodes}
    for edge in flow.edges:
        if edge.source not in ids or edge.target not in ids:
            blocks.append(f"aresta para nó inexistente: {edge.source} -> {edge.target}")
    edges = _valid_edges(flow)
    children: dict[str, list[str]] = {}
    for edge in edges:
        children.setdefault(edge.source, []).append(edge.target)
    targets = {edge.target for edge in edges}
    starts = [node.id for node in flow.nodes if node.id not in targets]
    if not starts:
        blocks.append("nenhum início (todo nó tem aresta de entrada)")
    elif len(starts) > 1:
        blocks.append(f"mais de um início: {', '.join(starts)}")
    for start in starts:
        cycle = _find_cycle(start, children)
        if cycle:
            blocks.append(f"ciclo no fluxo: {' -> '.join(cycle)}")
            break
    outgoing = {node.id: len(children.get(node.id, [])) for node in flow.nodes}
    warnings += [
        f"pergunta com menos de 2 saídas: {n.id}"
        for n in flow.nodes
        if n.kind is NodeKind.QUESTION and outgoing[n.id] < 2
    ]
    warnings += [
        f"final com saída: {n.id}"
        for n in flow.nodes
        if n.kind is NodeKind.END and outgoing[n.id] > 0
    ]
    warnings += [
        f"ação com mais de uma saída: {n.id}"
        for n in flow.nodes
        if n.kind is NodeKind.ACTION and outgoing[n.id] > 1
    ]
    if starts and not any(b.startswith("ciclo") for b in blocks):
        reached: set[str] = set()
        pending = [starts[0]]
        while pending:
            node = pending.pop()
            if node not in reached:
                reached.add(node)
                pending.extend(children.get(node, []))
        warnings += [
            f"nó fora do caminho do início: {n.id}" for n in flow.nodes if n.id not in reached
        ]
    return blocks, warnings


# texto do nó em uma linha só, para não quebrar o alinhamento
def _one_line(text: str) -> str:
    return " ".join(text.split())


# desenha a árvore a partir do início, com o recuo na coluna do colchete do pai mais 2
def _draw_tree(flow: FlowOutput) -> list[str]:
    nodes = {node.id: node for node in flow.nodes}
    children: dict[str, list[tuple[str, str]]] = {}
    targets = set()
    for edge in flow.edges:
        children.setdefault(edge.source, []).append((edge.label, edge.target))
        targets.add(edge.target)
    start = next(node.id for node in flow.nodes if node.id not in targets)
    lines = [f"[{start}] {_one_line(nodes[start].text)}"]

    # acrescenta os filhos de um nó, cada um na sua linha
    def add_children(node_id: str, column: int) -> None:
        for label, target in children.get(node_id, []):
            prefix = " " * (column + 2) + f"{_one_line(label)} -> "
            lines.append(f"{prefix}[{target}] {_one_line(nodes[target].text)}")
            add_children(target, len(prefix))

    add_children(start, 0)
    return lines


# desenha o fluxo em texto, ou explica por que não foi possível desenhar
def render_flow(flow: FlowOutput) -> FlowRender:
    blocks, warnings = validate_flow(flow)
    if blocks:
        lines = ["Fluxo não desenhado:", *[f"- {block}" for block in blocks], "", "Nós:"]
        lines += [f"[{n.id}] ({n.kind.value}) {_one_line(n.text)}" for n in flow.nodes]
        lines += ["", "Arestas:"]
        lines += [f"{e.source} -> {e.target} ({_one_line(e.label)})" for e in flow.edges]
        return FlowRender(text="\n".join(lines), warnings=warnings, drawn=False)
    lines = _draw_tree(flow)
    if warnings:
        lines += ["", "Avisos:", *[f"- {warning}" for warning in warnings]]
    return FlowRender(text="\n".join(lines), warnings=warnings, drawn=True)
