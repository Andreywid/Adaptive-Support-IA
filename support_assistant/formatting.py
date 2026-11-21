import json
from typing import Any

from support_assistant.flow_render import render_flow
from support_assistant.models import ComparisonRun, ContentType, FlowOutput, RunRecord, RunStatus


# texto mostrado ao usuário: fluxo desenhado em árvore, os outros tipos em JSON indentado
def format_output(content_type: ContentType, output: dict[str, Any]) -> str:
    if content_type is ContentType.FLOW:
        return render_flow(FlowOutput.model_validate(output)).text
    return json.dumps(output, ensure_ascii=False, indent=2)


# texto de uma execução: a saída formatada, ou o texto bruto quando saiu fora do contrato
def format_record_output(run: RunRecord | ComparisonRun) -> str:
    if run.status is RunStatus.OK and run.output is not None:
        return format_output(run.content_type, run.output)
    return f"Saída fora do contrato. Texto bruto:\n{run.raw_output}"
