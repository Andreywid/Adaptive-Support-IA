from typing import Any

from support_assistant.generators.core import GenerationResult, generate
from support_assistant.models import ContentType


# gera o conteúdo do tipo diagnosis com os argumentos de generate
def generate_diagnosis(**kwargs: Any) -> GenerationResult:
    return generate(content_type=ContentType.DIAGNOSIS, **kwargs)
