import re
from string import Template
from typing import TYPE_CHECKING

from support_assistant.models import Channel, CommunicationStyle, Profile, TechnicalLevel, Ticket

if TYPE_CHECKING:
    from support_assistant.prompts.engine import ProfileBlock, TemplateBlock

LEVEL_LABELS = {
    TechnicalLevel.LEIGO: "leigo",
    TechnicalLevel.INTERMEDIARIO: "intermediário",
    TechnicalLevel.AVANCADO: "avançado",
}
STYLE_LABELS = {
    CommunicationStyle.DIRETO: "direto",
    CommunicationStyle.DETALHADO: "detalhado",
    CommunicationStyle.DIDATICO: "didático",
}
CHANNEL_LABELS = {Channel.TELEFONE: "telefone", Channel.CHAT: "chat", Channel.EMAIL: "e-mail"}
RULE_ORDER = ("channel", "technical_level", "communication_style")
TICKET_TAG = re.compile(r"<\s*(/?)\s*ticket\b[^>]*>", re.IGNORECASE)


# uma linha "- regra" por dimensão que a versão define, na ordem de precedência
def _rules_text(block: "ProfileBlock", profile: Profile) -> str:
    lines = []
    for dimension in RULE_ORDER:
        by_value = getattr(block.rules, dimension)
        if by_value:
            lines.append(f"- {by_value[getattr(profile, dimension)]}")
    return "\n".join(lines)


# renderiza o bloco de perfil com rótulos acentuados e as regras da versão
def render_profile(block: "ProfileBlock", profile: Profile) -> str:
    return (
        Template(block.template)
        .substitute(
            name=profile.name,
            technical_level=LEVEL_LABELS[profile.technical_level],
            communication_style=STYLE_LABELS[profile.communication_style],
            channel=CHANNEL_LABELS[profile.channel],
            rules=_rules_text(block, profile),
        )
        .strip()
    )


# impede que o texto do cliente abra ou feche o delimitador do ticket
def neutralize_ticket_tags(text: str) -> str:
    return TICKET_TAG.sub(lambda match: f"[{match.group(1)}ticket]", text)


# renderiza o bloco do ticket, com o texto do cliente como valor literal
def render_ticket(block: "TemplateBlock", ticket: Ticket) -> str:
    text = neutralize_ticket_tags(ticket.text)
    return Template(block.template).substitute(text=text).strip()
