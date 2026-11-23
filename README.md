# Adaptive Support IA

Assistente de suporte técnico que adapta diagnóstico, orientação e comunicação ao perfil de cada cliente, com prompts versionados e comparados por uma rubrica fixa.

Desenvolvi este projeto para o Desafio Técnico de Estágio em IA e Engenharia de Prompt. A operadora **Conecta Fibra**, os clientes e os tickets são fictícios.

**Tecnologias:** Python 3.11+, Pydantic, HTTPX, Streamlit e API da Groq com o modelo `openai/gpt-oss-120b`.

**Decisões de prompt:** as técnicas, as versões, os testes e as limitações estão em [PROMPT_ENGINEERING_NOTES.md](PROMPT_ENGINEERING_NOTES.md).

## Sumário

- [O que o projeto faz](#o-que-o-projeto-faz)
- [Início rápido](#início-rápido)
- [Como usar](#como-usar)
- [Como funciona](#como-funciona)
- [Engenharia de prompt](#engenharia-de-prompt)
- [Avaliação](#avaliação)
- [Requisitos do desafio](#requisitos-do-desafio)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Configuração](#configuração)
- [Qualidade e segurança](#qualidade-e-segurança)
- [Limitações](#limitações)

## O que o projeto faz

A partir de **um perfil de cliente** e **um ticket**, o sistema gera quatro conteúdos:

| Conteúdo | Tipo | Quem lê | O que entrega |
| --- | --- | --- | --- |
| Diagnóstico | `diagnosis` | Atendente e cliente | Hipóteses, trechos do ticket como evidência, dados faltantes, conclusão e resumo |
| Passo a passo | `steps` | Cliente | Orientação adaptada ao nível técnico, ao estilo e ao canal |
| Perguntas | `questions` | Cliente | Só as perguntas que mudam a próxima ação, em ordem de prioridade |
| Fluxo de decisão | `flow` | Atendente | Árvore de decisão desenhada em ASCII |

Cada execução é salva em JSON com data, cliente, ticket, tipo, versão do prompt, provedor, modelo e saída.

Exemplo real de um fluxo gerado para a Marta, cliente leiga atendida por telefone, com o ticket "A internet cai toda noite e volta sozinha":

```text
[P1] A internet está funcionando agora?
  Sim -> [F1] Registre recorrência e peça observar nova queda.
  Não -> [P2] A queda acontece em todos os aparelhos?
           Sim -> [P3] As luzes do modem estão ligadas e piscando?
                    Sim -> [A1] Oriente reiniciar modem e roteador.
                             próximo -> [P5] A internet voltou após reiniciar?
                                          Sim -> [F1] Registre recorrência e peça observar nova queda.
                                          Não -> [F3] Encaminhar avaliação técnica e registrar sinais observados.
                    Não -> [F2] Registre falta de energia no equipamento e encaminhe avaliação técnica.
           Não -> [P4] A internet funciona no outro aparelho?
                    Sim -> [F1] Registre recorrência e peça observar nova queda.
                    Não -> [F3] Encaminhar avaliação técnica e registrar sinais observados.
```

O registro completo dessa saída está em [samples/run_v3_flow_p01_t01.json](samples/run_v3_flow_p01_t01.json).

## Início rápido

Requer Python 3.11 ou superior. O projeto já roda sem chave de API, com um provedor simulado:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m support_assistant generate --profile p01_marta --ticket t01_queda_noturna --type flow
```

A saída mostra o fluxo, o estado do cache e onde o JSON foi salvo:

```text
[P1] Todos os aparelhos perdem a conexão?
  sim -> [P2] A luz de sinal do roteador fica vermelha?
           sim -> [F1] Abrir chamado técnico de linha
           não -> [A1] Reiniciar o roteador e observar por uma noite
  não -> [F2] Verificar só o aparelho afetado

cache: gerado agora
status: ok
Simulação offline: o provedor fake devolve exemplos fixos.
registro: runs/20261123T120000Z_a1b2c3.json
```

Ao repetir o comando, a saída passa a mostrar `cache: acerto`, sem nova chamada ao provedor.

### Usar a API real

Crie uma chave em [console.groq.com/keys](https://console.groq.com/keys) e ajuste o `.env`:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=sua_chave
```

## Como usar

### Linha de comando

| Comando | O que faz |
| --- | --- |
| `python -m support_assistant` | Abre o menu interativo |
| `python -m support_assistant profiles` | Lista os perfis |
| `python -m support_assistant tickets` | Lista os tickets |
| `python -m support_assistant generate` | Gera um conteúdo |
| `python -m support_assistant history` | Lista as execuções salvas |
| `python -m support_assistant compare` | Compara duas versões de prompt |
| `python -m support_assistant evaluate` | Registra a avaliação às cegas |
| `python -m support_assistant checks` | Confere a avaliação e mostra as médias |

```bash
# ticket escrito na hora
python -m support_assistant generate --profile p02_rafael --ticket-text "A internet caiu agora" --type steps

# outra versão do prompt, ignorando o cache
python -m support_assistant generate --profile p03_carla --ticket t02_wifi_lento --type diagnosis --version v2 --refresh

# execuções salvas, com filtros
python -m support_assistant history --profile p01_marta --limit 5

# comparação sem rede
python -m support_assistant compare --provider fake --profile p01_marta --ticket t01_queda_noturna --versions v1 v3

# médias da comparação registrada
python -m support_assistant checks samples/comparacao_v1_v3_p01_t01.json
```

### Interface web

```bash
streamlit run web/app.py
```

A interface abre apenas na sua máquina e tem três abas:

- **Atendimento:** escolhe perfil, ticket, tipo e versão, gera a resposta e mostra onde o JSON foi salvo.
- **Comparação:** abre as comparações, com as respostas A e B e as notas da rubrica.
- **Exemplos:** mostra as saídas reais de `samples/`.

### Perfis e tickets

| Perfil | Nível técnico | Estilo | Canal |
| --- | --- | --- | --- |
| `p01_marta` | Leigo | Didático | Telefone |
| `p02_rafael` | Intermediário | Direto | Chat |
| `p03_carla` | Avançado | Detalhado | E-mail |
| `p04_joao` | Leigo | Direto | Chat |

| Ticket | Cenário | Uso |
| --- | --- | --- |
| `t01_queda_noturna` | Internet cai à noite | Desenvolvimento |
| `t02_wifi_lento` | Wi-Fi lento em um cômodo | Desenvolvimento |
| `t03_codigo_sms` | Código por SMS não chega | Desenvolvimento |
| `t04_sem_sinal` | Sem internet após tempestade | Teste |
| `t05_tv_travando` | TV trava nos filmes | Teste |
| `t06_latencia_jogos` | Ping alto nos jogos | Teste |
| `t07_cobranca_senha` | Cobrança em dobro e senha no relato | Teste |
| `t08_relato_vago` | Relato com poucos dados | Teste |

Os tickets de desenvolvimento servem para escrever e corrigir os prompts. Os de teste ficam reservados para verificar se a versão funciona em casos que não foram usados para ajustá-la.

## Como funciona

```text
perfil + ticket
   │
   ▼
1. Proteção          remove credenciais e neutraliza as tags do ticket
   │
   ▼
2. Motor de prompt   monta sistema, perfil, ticket e tarefa da versão escolhida
   │
   ▼
3. Cache ─────────── acerto ───────────────┐
   │ sem acerto                            │
   ▼                                       │
4. Provedor          Groq ou simulado      │
   │                                       │
   ▼                                       │
5. Validação         schema e regras de uso│
   │   └─ erro: uma tentativa de reparo    │
   ▼                                       │
6. Registro JSON e saída na CLI ou na web ◄┘
```

1. **Proteção:** senhas, códigos e dados de cartão são trocados por um marcador antes de chegar ao modelo.
2. **Motor de prompt:** cada versão é um arquivo TOML com blocos independentes.
3. **Cache:** a chave é o hash do provedor, do modelo, dos parâmetros, da versão, das mensagens e do schema. Qualquer mudança gera uma nova chamada.
4. **Provedor:** a interface é a mesma para a API real e para o provedor simulado.
5. **Validação:** o código confere o schema, as citações do ticket, as credenciais e o grafo do fluxo. Também recusa promessas e medidas ou caminhos de menu que não vieram do ticket.
6. **Registro:** o JSON guarda a resposta, o consumo de tokens e o motivo de qualquer reparo. O raciocínio interno do modelo não é pedido nem gravado.

## Engenharia de prompt

O prompt é montado em camadas, e cada uma tem uma responsabilidade:

```text
Sistema   persona, limites, ordem de prioridade e método de decisão
Cliente   perfil traduzido em regras de canal, nível técnico e estilo
Ticket    relato delimitado e tratado como dado
Tarefa    instrução e formato de cada tipo de conteúdo
```

As principais técnicas são Persona Prompting, hierarquia de instruções, Context Setting, delimitação contra prompt injection, método hipotético-dedutivo, matriz É / Não É, Grounding, incerteza declarada, valor da informação, Output Formatting e validação determinística. As [notas de engenharia de prompt](PROMPT_ENGINEERING_NOTES.md) explicam por que cada uma está aqui.

| Versão | Foco | O que mudou | Situação |
| --- | --- | --- | --- |
| `v1` | Linha de base | Papel em uma frase, perfil como rótulos e tarefa curta | Referência |
| `v2` | Adaptação ao perfil | Limites, regras por perfil, ticket delimitado e sentido de cada campo | Histórico |
| `v3` | Regras de decisão | Condutas por resposta, evidência separada de sintoma e incerteza declarada | **Padrão** |
| `v4` | Método geral | Um método de decisão comum aos quatro conteúdos | Testada, não promovida |

A `v4` foi testada nos tickets reservados e não substituiu a `v3`. O resultado está em [Teste em tickets reservados](#teste-em-tickets-reservados).

## Avaliação

A comparação registrada usa a mesma cliente, o mesmo ticket, o mesmo modelo e os mesmos parâmetros, mudando apenas a versão do prompt. São 24 respostas reais: 2 versões, 4 tipos e 3 seeds.

As respostas da seed 101 receberam notas de 1 a 5 em três critérios, cada uma com justificativa e um trecho literal da resposta:

| Critério | `v1` | `v3` |
| --- | ---: | ---: |
| Clareza | 2,75 | 4,25 |
| Adequação ao nível do cliente | 2,25 | 4,25 |
| Acionabilidade | 2,50 | 3,50 |
| **Média geral** | **2,50** | **4,00** |

Na prática, a diferença aparece assim:

| Ponto observado | `v1` | `v3` |
| --- | --- | --- |
| Primeiro passo | "Desligue o modem/roteador da tomada." | "A internet está funcionando agora?" |
| Linguagem ao telefone | "O modem inicia o processo de boot e os LEDs começam a acender sequencialmente." | "Se ainda estiver sem internet, desligue o modem da tomada, aguarde 30 segundos e ligue novamente." |
| Primeira pergunta | "Qual é o modelo do seu roteador/modem e há quanto tempo ele está em uso?" | "A internet está funcionando neste momento?" |
| Tamanho do roteiro | 7 passos | 3 passos |

O arquivo completo, com as respostas, as notas e as justificativas, está em [samples/comparacao_v1_v3_p01_t01.json](samples/comparacao_v1_v3_p01_t01.json). Para repetir a coleta na API real (cerca de 10 minutos):

```bash
python -m support_assistant compare --provider groq \
  --profile p01_marta --ticket t01_queda_noturna --versions v1 v3 --seeds 101 102 103
```

O comando `evaluate` mostra as respostas como A e B, sem revelar a versão, e pede nota, justificativa e trecho em cada critério:

```bash
python -m support_assistant evaluate comparisons/ARQUIVO.json --seed 101
python -m support_assistant checks comparisons/ARQUIVO.json
```

### Teste em tickets reservados

Para verificar se a `v4` generalizava, congelei as duas versões, gravei a regra de decisão antes de ver os resultados e rodei `v3` e `v4` nos tickets `t04` a `t08`, que não foram usados para ajustar os prompts. São 40 respostas reais.

| Medida | `v3` | `v4` |
| --- | ---: | ---: |
| Válidas na coleta | 18 de 20 | 17 de 20 |
| Válidas após corrigir o validador | 20 de 20 | 18 de 20 |
| Respostas que precisaram de reparo | 1 | 3 |

O teste revelou dois erros no validador, que foram corrigidos. A decisão não mudou: **a `v3` continua como versão final**, porque a `v4` teve mais saídas inválidas. Os arquivos estão em [samples/teste_reservado/](samples/teste_reservado/).

## Requisitos do desafio

| Requisito | Como foi atendido | Onde ver |
| --- | --- | --- |
| Quatro tipos de conteúdo | Uma função e uma tarefa de prompt por tipo | [generators/](support_assistant/generators/) |
| 3 a 5 perfis em JSON | 4 perfis, cobrindo todos os níveis, estilos e canais | [profiles.json](data/profiles.json) |
| Motor de prompt | Persona, contexto, decomposição, formato e versionamento | [prompts/](support_assistant/prompts/) |
| Persistência em JSON | Data, cliente, ticket, tipo, versão, provedor, modelo e saída | [samples/](samples/) |
| Cache | Reaproveita chamadas com entrada, versão e configuração equivalentes | [cache.py](support_assistant/cache.py) |
| Execuções equivalentes | Filtro `history --cache-key` | [storage.py](support_assistant/storage.py) |
| Comparação de versões | Mesmo cenário e mesma configuração, mudando só o prompt | [compare.py](support_assistant/evaluation/compare.py) |
| Rubrica de 3 critérios | Notas de 1 a 5 com justificativa, gravadas no JSON | [comparação](samples/comparacao_v1_v3_p01_t01.json) |
| Interface | CLI completa e interface web | [cli.py](support_assistant/cli.py), [web/](web/) |
| Segredos | `.env` fora do Git e `.env.example` sem chave | [.env.example](.env.example) |
| Extras | Testes, validação adicional, interface web e CI | [tests/](tests/), [output_validation.py](support_assistant/output_validation.py) |

## Estrutura do projeto

```text
.
├── support_assistant/
│   ├── cli.py                 comandos e menu interativo
│   ├── config.py              variáveis de ambiente
│   ├── models.py              contratos de dados e schemas de saída
│   ├── data_loader.py         leitura de perfis e tickets
│   ├── prompts/
│   │   ├── engine.py          carrega a versão e monta as mensagens
│   │   ├── blocks.py          renderiza perfil e ticket
│   │   └── versions/          prompts v1 a v4 em TOML
│   ├── generators/            uma função por tipo de conteúdo e o núcleo da geração
│   ├── providers/             cliente Groq, provedor simulado e intervalo entre chamadas
│   ├── evaluation/            comparação, avaliação às cegas e rubrica
│   ├── output_validation.py   regras de uso de cada saída
│   ├── credentials.py         detecção e remoção de credenciais
│   ├── flow_render.py         validação e desenho do fluxo
│   ├── formatting.py          apresentação das saídas
│   ├── cache.py               cache por hash
│   ├── storage.py             registros de execução
│   └── files.py               escrita atômica de JSON
├── web/                       interface em Streamlit
├── data/                      perfis e tickets fictícios
├── samples/                   saídas reais, comparação com rubrica e teste reservado
├── tests/                     testes automatizados, sem rede
├── .github/workflows/ci.yml   testes e lint em Python 3.11, 3.12 e 3.13
├── PROMPT_ENGINEERING_NOTES.md
└── requirements.txt
```

As pastas `runs/`, `comparisons/` e `data/cache/` são criadas durante o uso e ficam fora do Git.

## Configuração

| Variável | Padrão | O que faz |
| --- | --- | --- |
| `LLM_PROVIDER` | `fake` | `groq` usa a API real, `fake` usa respostas fixas |
| `GROQ_API_KEY` | vazia | Chave da API, nunca versionada |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Modelo usado na geração |
| `LLM_TEMPERATURE` | `0` | Temperatura |
| `LLM_TOP_P` | `1.0` | Top p |
| `LLM_SEED` | `1` | Semente, atendida em melhor esforço pelo provedor |
| `LLM_MAX_TOKENS` | `2048` | Limite de tokens da resposta |
| `LLM_REASONING_EFFORT` | `low` | Esforço de raciocínio: `low`, `medium` ou `high` |
| `LLM_INCLUDE_REASONING` | `false` | O raciocínio do modelo não é pedido nem gravado |
| `LLM_TIMEOUT_S` | `60` | Tempo máximo de cada tentativa, em segundos |
| `LLM_MAX_RETRIES` | `3` | Novas tentativas em falha temporária ou limite de uso |
| `LLM_MIN_INTERVAL_S` | `20` | Espera entre chamadas reais durante uma comparação |

## Qualidade e segurança

```bash
python -m pytest -q
ruff check .
ruff format --check .
```

- Os testes rodam sem rede e sem chave de API.
- A chave fica só no `.env`, que está fora do Git.
- Credenciais escritas no ticket não chegam ao modelo nem aos registros.
- O texto do cliente é tratado como dado e não consegue fechar o delimitador do prompt.
- Respostas que pedem credenciais ou citam medidas fora do ticket voltam para correção.

## Limitações

- A comparação com rubrica cobre um perfil e um ticket.
- A `v3` ainda pode aplicar a triagem de falha intermitente a problemas contínuos e repetir testes que o cliente já fez.
- A validação não detecta todo fato inventado, por isso o atendente revisa a resposta antes de enviar.
- O projeto não tem base de conhecimento da operadora, integração com sistema de tickets nem conversa com várias trocas.

A lista completa e os próximos passos estão nas [notas de engenharia de prompt](PROMPT_ENGINEERING_NOTES.md#limitações).
