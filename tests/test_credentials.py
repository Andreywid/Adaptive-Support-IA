import pytest

from support_assistant.credentials import (
    REDACTED_CREDENTIAL,
    contains_credential,
    credential_values,
    redact_credentials,
    requests_credential,
    sanitize_ticket,
)
from support_assistant.models import TICKET_MAX_CHARS, Ticket


# os valores de credencial e as repetições deles são mascarados
def test_redaction_masks_values_and_their_repetitions():
    for label, value in [
        ("Minha senha do aplicativo é", "Violeta22"),
        ("Senha atual:", "Tulipa56"),
        ("Meu código de verificação:", "921842"),
        ("PIN:", "4018"),
        ("CVV:", "589"),
        ("Número do cartão:", "4111 1111 1111 1111"),
        ("GROQ_API_KEY=", "valor-ficticio"),
    ]:
        text = f"O valor é {value.upper()}. {label} {value}. O SMS não chegou."
        known = credential_values(text)
        assert contains_credential(value.upper(), known)
        sanitized = redact_credentials(text)
        assert sanitized.count(REDACTED_CREDENTIAL) == 2
        assert sanitized.endswith("O SMS não chegou.")
        assert redact_credentials(sanitized) == sanitized


# ticket longo mantém o marcador inteiro e cabe no contrato
def test_sanitized_long_ticket_preserves_complete_marker_and_contract():
    text = "A internet caiu. " + " ".join(f"Senha: {i:03}." for i in range(100, 230))
    original = Ticket(id="texto_livre", text=text)
    sanitized = sanitize_ticket(original)
    assert len(text) <= TICKET_MAX_CHARS
    assert len(sanitized.text) <= TICKET_MAX_CHARS
    assert sanitized.text.startswith("A internet caiu. ")
    assert sanitized.text.count("[") == sanitized.text.count("]")
    assert not credential_values(sanitized.text)
    assert sanitize_ticket(sanitized) == sanitized


# variações de pedido de credencial são detectadas
@pytest.mark.parametrize(
    "text",
    [
        "Qual é a sua senha?",
        "Pode me informar o código de verificação?",
        "Você não está conectado, digite sua senha.",
        "Precisamos do CVV para prosseguir.",
        "Confirme o código recebido.",
        "Me confirme a senha do Wi-Fi.",
        "Informe a senha do Wi-Fi.",
        "Confirme a senha do Wi-Fi.",
        "Selecione a rede e digite a senha da sua conta.",
        "Selecione a rede Wi-Fi e digite a senha aqui no chat.",
    ],
)
def test_credential_request_variants_are_flagged(text):
    assert requests_credential(text)


# orientação de segurança neutra não é tratada como pedido
@pytest.mark.parametrize(
    "text",
    [
        "Nunca envie sua senha.",
        "Nunca, em hipótese alguma, envie sua senha.",
        "Por favor, não informe o código de verificação.",
        "Confirme se o código de verificação chegou.",
        "Qual é o problema com a senha?",
        "O aplicativo pede um código de verificação, mas ele nunca chega.",
        "Sua senha é importante.",
        "Selecione a sua rede Wi\u2011Fi e confirme a senha que você costuma usar.",
        "Digite a senha do Wi-Fi na TV.",
    ],
)
def test_neutral_security_advice_is_preserved(text):
    assert not requests_credential(text)
    assert redact_credentials(text) == text


# pedir um novo código ao aplicativo não é pedir o código ao cliente
@pytest.mark.parametrize(
    "text",
    [
        "Solicite um novo código de verificação no aplicativo.",
        "Solicite novamente o código de verificação.",
    ],
)
def test_requesting_a_new_code_from_the_service_is_allowed(text):
    assert not requests_credential(text)


# pedir o código para a pessoa ou para o atendente continua bloqueado
@pytest.mark.parametrize(
    "text",
    [
        "Solicite o código de verificação ao cliente.",
        "Solicite um novo código de verificação e me envie.",
        "Solicite a nova senha do aplicativo.",
    ],
)
def test_requesting_the_code_from_a_person_stays_blocked(text):
    assert requests_credential(text)
