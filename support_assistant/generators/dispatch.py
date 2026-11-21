from collections.abc import Callable
from typing import Any

from support_assistant.generators.core import GenerationResult
from support_assistant.generators.diagnosis import generate_diagnosis
from support_assistant.generators.flow import generate_flow
from support_assistant.generators.questions import generate_questions
from support_assistant.generators.steps import generate_steps
from support_assistant.models import ContentType

GENERATORS: dict[ContentType, Callable[..., GenerationResult]] = {
    ContentType.DIAGNOSIS: generate_diagnosis,
    ContentType.STEPS: generate_steps,
    ContentType.QUESTIONS: generate_questions,
    ContentType.FLOW: generate_flow,
}


# chama a função de geração do tipo de conteúdo pedido
def generate_content(content_type: ContentType, **kwargs: Any) -> GenerationResult:
    return GENERATORS[content_type](**kwargs)
