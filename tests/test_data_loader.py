import json

import pytest

from support_assistant.config import PROJECT_ROOT
from support_assistant.data_loader import (
    FREE_TEXT_TICKET_ID,
    DataError,
    get_profile,
    get_ticket,
    load_profiles,
    load_tickets,
    ticket_from_text,
)
from support_assistant.models import Channel, CommunicationStyle, TechnicalLevel

PROFILE = {
    "id": "p01_marta",
    "name": "Marta",
    "technical_level": "leigo",
    "communication_style": "didatico",
    "channel": "telefone",
}


# grava um JSON de dados num arquivo temporário
def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# os arquivos reais carregam 4 perfis e 8 tickets, nesta ordem de ids
def test_real_files_load_in_order():
    profiles = load_profiles()
    tickets = load_tickets()
    assert [p.id for p in profiles.profiles] == ["p01_marta", "p02_rafael", "p03_carla", "p04_joao"]
    assert [t.id for t in tickets.tickets] == [
        "t01_queda_noturna",
        "t02_wifi_lento",
        "t03_codigo_sms",
        "t04_sem_sinal",
        "t05_tv_travando",
        "t06_latencia_jogos",
        "t07_cobranca_senha",
        "t08_relato_vago",
    ]


# cada valor de cada enum de perfil aparece em pelo menos um perfil
def test_every_profile_enum_value_is_used():
    profiles = load_profiles().profiles
    assert {p.technical_level for p in profiles} == set(TechnicalLevel)
    assert {p.communication_style for p in profiles} == set(CommunicationStyle)
    assert {p.channel for p in profiles} == set(Channel)


# get_profile e get_ticket acham por id
def test_get_by_id():
    assert get_profile("p03_carla").name == "Carla"
    assert get_ticket("t03_codigo_sms").id == "t03_codigo_sms"


# id inexistente gera a mensagem com a lista de disponíveis
def test_unknown_id_lists_available_ones():
    with pytest.raises(DataError) as profile_error:
        get_profile("p09")
    assert str(profile_error.value) == (
        "Perfil 'p09' não existe. Disponíveis: p01_marta, p02_rafael, p03_carla, p04_joao"
    )
    with pytest.raises(DataError) as ticket_error:
        get_ticket("t09")
    assert str(ticket_error.value).startswith(
        "Ticket 't09' não existe. Disponíveis: t01_queda_noturna"
    )


# arquivo ausente cita o caminho relativo à raiz do projeto
def test_missing_file_message_uses_relative_path():
    with pytest.raises(DataError) as error:
        load_profiles(PROJECT_ROOT / "data" / "nao_existe.json")
    assert str(error.value) == "Arquivo de dados não encontrado: data/nao_existe.json"


# JSON inválido cita a linha do problema
def test_invalid_json_message_has_the_line(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text('{\n  "schema_version": ,\n}', encoding="utf-8")
    with pytest.raises(DataError, match=r"JSON inválido em .*profiles\.json, linha 2:"):
        load_profiles(path)


# id repetido, campo extra e versão errada viram DataError com uma linha por erro
@pytest.mark.parametrize(
    ("profiles", "expected"),
    [
        ([PROFILE, PROFILE], "arquivo: ids de perfil repetidos: p01_marta"),
        ([{**PROFILE, "campo_extra": 1}], "profiles.0.campo_extra: Extra inputs are not permitted"),
        ([{**PROFILE, "channel": "whatsapp"}], "profiles.0.channel:"),
    ],
)
def test_invalid_profiles_are_rejected(tmp_path, profiles, expected):
    path = _write(tmp_path / "profiles.json", {"schema_version": "1.0", "profiles": profiles})
    with pytest.raises(DataError) as error:
        load_profiles(path)
    message = str(error.value)
    assert message.startswith("Dados inválidos em ")
    assert f"  {expected}" in message


# ticket vazio e id repetido são recusados
def test_invalid_tickets_are_rejected(tmp_path):
    empty = _write(
        tmp_path / "a.json", {"schema_version": "1.0", "tickets": [{"id": "t01_a", "text": "  "}]}
    )
    with pytest.raises(DataError, match=r"tickets\.0\.text"):
        load_tickets(empty)
    repeated = {"id": "t01_a", "text": "texto"}
    both = _write(tmp_path / "b.json", {"schema_version": "1.0", "tickets": [repeated, repeated]})
    with pytest.raises(DataError, match="ids de ticket repetidos: t01_a"):
        load_tickets(both)


# texto livre válido vira ticket com o id reservado
def test_ticket_from_text_accepts_valid_text():
    ticket = ticket_from_text("  A internet caiu  ")
    assert ticket.id == FREE_TEXT_TICKET_ID
    assert ticket.text == "A internet caiu"


# texto livre vazio ou com 2001 caracteres é recusado com a mensagem padrão
@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_ticket_from_text_rejects_invalid_text(text):
    with pytest.raises(DataError) as error:
        ticket_from_text(text)
    assert str(error.value) == "O texto do ticket precisa ter de 1 a 2000 caracteres."


# o ticket t01 contém os trechos citados pela saída fixa do provedor fake
def test_t01_text_contains_the_fake_provider_quotes():
    text = get_ticket("t01_queda_noturna").text
    assert "cai toda noite" in text
    assert "volta sozinha" in text
