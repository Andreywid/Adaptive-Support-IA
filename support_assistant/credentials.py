import re
from collections.abc import Iterable, Iterator
from typing import Any

from support_assistant.models import TICKET_MAX_CHARS, Ticket

REDACTED_CREDENTIAL = "[credencial removida]"

_CREDENTIAL_NAME = (
    r"(?:senhas?|c[oó]digos?\s+(?:de\s+)?"
    r"(?:verifica[çc][aã]o|seguran[çc]a|SMS|2FA|recebid[oa]s?|enviad[oa]s?\s+por\s+SMS)"
    r"|OTP|PIN|CVV|CVC|(?:dados|n[uú]mero)\s+(?:do\s+)?cart[aã]o"
    r"|(?:chave\s+(?:de\s+)?API|(?:[A-Z][A-Z0-9]*_)?api[_ -]key)|token\s+de\s+acesso)\b"
)
_LABELED_VALUE = re.compile(
    rf"\b{_CREDENTIAL_NAME}(?:\s+(?:atual|tempor[aá]ri[oa]|antig[oa]|nov[oa]))?"
    r"(?:\s+(?:do|da|de)\s+[\w-]+){0,2}"
    r"\s*(?:[:=]|\bé\b)\s*"
    r"(?:[\"'`](?P<quoted>[^\"'`\r\n]{3,128})[\"'`]"
    r"|(?P<bare>[\w@+!#$%&*/?=-]{3,128})(?![\w-]))",
    re.IGNORECASE,
)
_TOKEN = re.compile(
    r"\b(?:gsk_[\w-]{16,}|sk-[\w-]{16,}|gh[pousr]_[\w]{20,})\b"
    r"|\bBearer\s+([\w.-]{12,})",
    re.IGNORECASE,
)
_CARD_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_NON_VALUES = frozenset(
    {
        "bloqueada",
        "confidencial",
        "correta",
        "desconhecida",
        "essa",
        "esta",
        "fraca",
        "forte",
        "importante",
        "incorreta",
        "inválida",
        "necessária",
        "obrigatória",
        "opcional",
        "pessoal",
        "secreta",
        "temporária",
        "uma",
        "única",
        "válida",
    }
)

_CLAUSE_SPLIT = re.compile(r"[.!?;\n]+|\b(?:mas|porém|contudo|entretanto)\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:n[aã]o|nunca|jamais|sem)\b", re.IGNORECASE)
_REQUEST = re.compile(
    r"\b(?:compartilh[ae]|diga|digite|envie|escreva|forne[çc]a|informe|mande|mostre|passe|"
    r"confirme|repita|precisamos|preciso|solicitamos|solicite|"
    rf"qual\s+(?:[eé]|seria)(?=\s+(?:(?:a|o)\s+)?(?:(?:seu|sua)\s+)?{_CREDENTIAL_NAME})|"
    r"(?:pode|poderia)\s+(?:me\s+)?(?:dizer|informar|enviar|passar|digitar))\b",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(rf"\b{_CREDENTIAL_NAME}", re.IGNORECASE)
_LOCAL_WIFI_ENTRY = re.compile(
    r"\b(?:selecione|escolha|conecte|reconecte)\b.*\b(?:wi\W?fi|rede)\b"
    r"|\b(?:wi\W?fi|rede)\b.*\bn[ao]\s+(?:tv|televis[aã]o|aparelho|celular|notebook|computador|tela)\b",
    re.IGNORECASE,
)
_SHARED_OR_ACCOUNT = re.compile(
    r"\b(?:me|mim|n[oó]s|atendente|aqui|chat|e-?mail|telefone|conta|aplicativo|app|cart[aã]o)\b",
    re.IGNORECASE,
)

_CODE_NAME = re.compile(r"\bc[oó]digos?\b", re.IGNORECASE)
_OTHER_SECRET = re.compile(r"\b(?:senhas?|cart[aã]o|cvv|cvc|pin)\b", re.IGNORECASE)
_RESEND_TARGET = re.compile(
    r"\b(?:nov[oa]|novamente|outr[oa]|reenvi\w+)\b|\bn[oa]\s+(?:aplicativo|app|site)\b",
    re.IGNORECASE,
)
_TO_PERSON = re.compile(
    r"\b(?:ao|à|do|da)\s+(?:cliente|usu[aá]rio|pessoa)\b|\b(?:me|nos|atendente|aqui|chat)\b",
    re.IGNORECASE,
)


# diz se uma sequência de dígitos parece número de cartão
def _is_card(value: str) -> bool:
    digits = [int(char) for char in value if char.isdecimal()]
    if len(set(digits)) < 2:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        doubled = 2 * digit if position % 2 else digit
        total += doubled - 9 if doubled > 9 else doubled
    return total % 10 == 0


# valores de credencial escritos no texto, como a senha depois de "senha:"
def credential_values(text: str) -> tuple[str, ...]:
    values = []
    for match in _LABELED_VALUE.finditer(text):
        value = match["quoted"] or match["bare"]
        if value.casefold() not in _NON_VALUES:
            values.append(value)
    values.extend(match[1] or match[0] for match in _TOKEN.finditer(text))
    values.extend(match[0] for match in _CARD_NUMBER.finditer(text) if _is_card(match[0]))
    return tuple(dict.fromkeys(values))


# expressão que encontra qualquer um dos valores, do mais longo para o mais curto
def _values_pattern(values: Iterable[str]) -> re.Pattern[str] | None:
    ordered = sorted(set(values), key=len, reverse=True)
    if not ordered:
        return None
    alternatives = "|".join(re.escape(value) for value in ordered)
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


# troca as credenciais do texto pelo marcador de credencial removida
def redact_credentials(text: str, known_values: Iterable[str] = ()) -> str:
    pattern = _values_pattern((*known_values, *credential_values(text)))
    return pattern.sub(lambda _: REDACTED_CREDENTIAL, text) if pattern else text


# ticket com as credenciais removidas, dentro do limite de tamanho
def sanitize_ticket(ticket: Ticket) -> Ticket:
    text = redact_credentials(ticket.text)
    boundary = TICKET_MAX_CHARS
    if len(text) > boundary:
        start = text.rfind(REDACTED_CREDENTIAL, 0, boundary + len(REDACTED_CREDENTIAL) - 1)
        if start != -1 and start < boundary < start + len(REDACTED_CREDENTIAL):
            boundary = start
    return ticket.model_copy(update={"text": text[:boundary].rstrip()})


# diz se o texto contém uma credencial conhecida ou escrita com rótulo
def contains_credential(text: str, known_values: Iterable[str] = ()) -> bool:
    pattern = _values_pattern(known_values)
    return bool((pattern and pattern.search(text)) or credential_values(text))


# divide o texto em orações, para avaliar cada pedido separadamente
def clauses(text: str) -> Iterator[str]:
    yield from _CLAUSE_SPLIT.split(text)


# diz se o pedido na posição dada está negado, como em "nunca informe"
def is_negated(clause: str, position: int) -> bool:
    prefix = clause[:position]
    prefix = re.sub(r",\s*em hip[oó]tese alguma\s*,", " ", prefix, flags=re.IGNORECASE)
    prefix = prefix.rsplit(",", 1)[-1]
    return bool(_NEGATION.search(" ".join(prefix.split()[-5:])))


# diz se a oração orienta digitar a senha do Wi-Fi no próprio aparelho, sem passá-la a ninguém
def _is_local_wifi_entry(clause: str, verb: str) -> bool:
    return (
        verb.casefold() in ("digite", "confirme")
        and bool(_LOCAL_WIFI_ENTRY.search(clause))
        and not _SHARED_OR_ACCOUNT.search(clause)
    )


# diz se a oração manda pedir um novo código ao serviço, sem repassá-lo a ninguém
def _is_code_resend(clause: str, verb: str) -> bool:
    return (
        verb.casefold() == "solicite"
        and bool(_CODE_NAME.search(clause))
        and not _OTHER_SECRET.search(clause)
        and bool(_RESEND_TARGET.search(clause))
        and not _TO_PERSON.search(clause)
    )


# diz se o texto pede ao cliente uma senha, um código ou dados de cartão
def requests_credential(text: str) -> bool:
    for clause in clauses(text):
        for request in _REQUEST.finditer(clause):
            following = clause[request.end() :].lstrip()
            if re.match(r"(?:se|quando|como|onde)\b", following, re.IGNORECASE):
                continue
            if _is_local_wifi_entry(clause, request.group()) or _is_code_resend(
                clause, request.group()
            ):
                continue
            if _CREDENTIAL.search(following[:80]) and not is_negated(clause, request.start()):
                return True
    return False


# todos os textos de uma saída, para procurar credenciais em qualquer campo
def output_texts(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for name, field in value.items():
            yield name
            yield from output_texts(field)
    elif isinstance(value, list):
        for item in value:
            yield from output_texts(item)
