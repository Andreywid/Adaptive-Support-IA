from typing import Any

from support_assistant.generators.core import GenerationResult, generate
from support_assistant.models import ContentType


# gera o conteúdo do tipo flow com os argumentos de generate
def generate_flow(**kwargs: Any) -> GenerationResult:
    return generate(content_type=ContentType.FLOW, **kwargs)
