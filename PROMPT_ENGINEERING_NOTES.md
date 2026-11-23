# Notas de engenharia de prompt

Este documento explica as decisões por trás dos prompts: quais técnicas usei, por que cada uma entrou, como as versões mudaram e o que os testes mostraram. Para instalar e executar o projeto, veja o [README](README.md).

## Resumo

- **Problema:** gerar diagnóstico, passo a passo, perguntas e fluxo de decisão adaptados ao nível técnico, ao estilo e ao canal de cada cliente.
- **Abordagem:** prompt em camadas, perfil traduzido em regras, saída estruturada e validação em código do que não pode depender do modelo.
- **Versões:** `v1` é a linha de base, `v2` adapta ao perfil, `v3` adiciona regras de decisão e é a versão final, e `v4` testou um método geral.
- **Resultado:** na comparação com rubrica, a média foi de 2,50 na `v1` para 4,00 na `v3`, em uma escala de 1 a 5.
- **Teste de generalização:** em cinco tickets reservados, a `v3` teve 20 de 20 respostas válidas e a `v4`, 18 de 20. Por isso a `v3` ficou como versão final.
- **Principal limite:** a comparação com rubrica cobre um perfil e um ticket, então ela não prova ganho geral.

## O problema central

O assistente prepara material para dois leitores diferentes. O atendente lê o diagnóstico e o fluxo de decisão e precisa de algo que possa checar. O cliente recebe o passo a passo, as perguntas e o resumo do diagnóstico, e esses textos precisam soar como alguém do suporte falando com ele, no canal em que ele está.

Um prompt genérico não resolve isso. A mesma queda de internet pede uma fala curta e sem termos técnicos para a Marta, leiga ao telefone, e uma análise com medições para a Carla, avançada por e-mail. Quase todas as decisões abaixo saem dessa diferença.

## As técnicas e por que cada uma está aqui

O prompt é montado em camadas, e cada camada tem uma responsabilidade:

```text
Sistema   persona, limites, ordem de prioridade e método de decisão
Cliente   perfil traduzido em regras de canal, nível técnico e estilo
Ticket    relato delimitado e tratado como dado
Tarefa    instrução e formato de cada tipo de conteúdo
```

Cada versão é um arquivo TOML em `support_assistant/prompts/versions/`. O motor (`support_assistant/prompts/engine.py`) junta os blocos com o perfil e o ticket da vez. Os quatro tipos compartilham o sistema e o perfil e mudam só a tarefa e o schema de saída.

| Técnica ou método | Para que serve aqui | Onde está |
| --- | --- | --- |
| Persona Prompting | Define papel, responsabilidade e limites | bloco `system` |
| Hierarquia de instruções | Resolve conflito entre regras | bloco `system` |
| Context Setting | Transforma o perfil em regras de escrita | `profile.rules` |
| Delimitação do ticket | Separa o relato das instruções | `prompts/blocks.py` |
| Método hipotético-dedutivo | Organiza o diagnóstico em hipóteses e testes | `tasks.diagnosis` |
| Matriz É / Não É (Kepner-Tregoe) | Delimita quando e onde a falha acontece | método da `v4` |
| Few-Shot | Mostra a decisão esperada com um exemplo curto | tarefas da `v3`, sistema da `v4` |
| Grounding | Liga cada hipótese a um trecho do ticket | `quote` e `reading` |
| Incerteza declarada | Evita conclusão mais forte que a evidência | `conclusion.confidence` |
| Valor da informação | Prioriza as perguntas que mudam a conduta | `tasks.questions` |
| Output Formatting | Fixa campos, tamanho e formato da saída | `models.py` |
| Validação determinística | Confere em código o que não depende do modelo | `output_validation.py` |
| Versionamento de prompt | Identifica qual prompt gerou cada resposta | `[meta]` e hash das mensagens |

### Persona Prompting

Todo prompt começa definindo quem é o assistente (nível 1 da Conecta Fibra), para quem ele escreve (o atendente, que revisa e repassa ao cliente) e o que ele não pode fazer: prometer reembolso ou prazo, pedir senha ou código, citar política da operadora que não está no ticket.

Não dei credenciais à persona, como "especialista com 15 anos de experiência", porque um título não torna a resposta tecnicamente mais correta. Aqui a persona serve para delimitar a função.

### Hierarquia de instruções

Os limites vêm antes de qualquer regra de estilo, e o prompt declara uma ordem para conflitos: limites, formato, canal, nível técnico e estilo. Sem essa ordem, um cliente de estilo "detalhado" atendido por telefone recebe duas regras opostas: explicar cada passo e falar pouco. A ordem orienta o modelo, mas não é uma garantia. O que não pode falhar também é conferido em código.

### Context Setting: o perfil vira regras

Um rótulo depende de o modelo saber o que fazer com ele. A instrução diz o comportamento esperado. Para a mesma cliente, a diferença entre a `v1` e a `v3` é esta:

```text
v1   nome: Marta; nível técnico: leigo; estilo de comunicação: didático; canal: telefone

v3   Marta, nível técnico leigo, estilo didático, atendimento por telefone.
     - Use fala direta com frases de até vinte palavras, uma pergunta ou instrução por vez.
       Sem URLs, endereços IP ou caminhos de menu para ditar.
     - Use palavras do dia a dia e descreva o que a pessoa consegue ver ou fazer.
       Explique um termo técnico se precisar usá-lo.
     - Explique o que observar com palavras conhecidas.
       Use uma comparação curta se ela ajudar a executar a ação.
```

- **Nível técnico:** segue a ideia de reversão da expertise. O detalhe que ajuda um iniciante pode atrapalhar quem já sabe. Por isso a Marta recebe o que ver e fazer, e a Carla recebe medições e onde conferi-las.
- **Canal:** considera a carga cognitiva. Quem só ouve não consegue reler, então o telefone usa uma instrução por vez e no máximo 3 passos.
- **Linguagem:** aplica princípios de Linguagem Simples, como frases curtas, ordem direta, palavras comuns e explicação de termos técnicos.

Os limites de 20 palavras e de 3 passos são decisões de projeto, escolhidas para manter a fala curta.

### Delimitação do ticket e prompt injection

O texto do cliente vai entre `<ticket>` e `</ticket>`, e o sistema avisa que nada escrito ali muda as regras. O delimitador falharia se o próprio cliente escrevesse `</ticket>`, então o código neutraliza essas tags antes de montar o prompt (`prompts/blocks.py`).

Senhas, códigos e números de cartão viram um marcador antes de chegar ao modelo (`credentials.py`). Se a resposta pedir ou repetir uma credencial, ela volta para correção. Isso reduz o risco de prompt injection, mas não o elimina, e por isso o atendente revisa antes de enviar.

### Método hipotético-dedutivo e matriz É / Não É

O diagnóstico segue o método hipotético-dedutivo, comum em troubleshooting: partir do que foi observado, levantar causas possíveis e escolher um teste que ajude a separá-las.

A v3 aplica essa ideia dentro de cada tarefa. Na v4, tentei transformá-la em um método único para os quatro conteúdos, e ela não virou padrão pelo motivo descrito em [Teste em tickets reservados](#teste-em-tickets-reservados). As etapas de padrão e alcance vêm da matriz É / Não É, de Kepner-Tregoe, que compara onde, quando e em que extensão o problema acontece e onde ele não acontece.

| Etapa | O que faz |
| --- | --- |
| Fatos | Parte do que o cliente já informou ou testou |
| Padrão | Identifica se a falha é contínua, intermitente ou indefinida |
| Alcance | Descobre o que é e o que não é afetado |
| Próxima ação | Escolhe o teste simples e reversível que separa as causas restantes |
| Valores | Usa só medidas informadas no ticket |

### Few-Shot

A v3 traz um exemplo curto de passo de triagem. A v4 usa dois exemplos de outros serviços, uma impressora e um telefone fixo, para mostrar o método sem entregar a resposta dos tickets de internet.

### Grounding e incerteza declarada

O desafio dispensa o raciocínio interno, e o projeto não o pede nem o grava (`LLM_INCLUDE_REASONING=false`). Também evitei Chain-of-Thought visível, porque uma explicação convincente pode não mostrar o motivo real da resposta.

No lugar dela, usei grounding. Cada hipótese traz um trecho copiado do ticket (`quote`) e o que ele mostra (`reading`), e o código confere se a citação existe. Assim o atendente verifica a hipótese no relato em vez de confiar no texto.

A conclusão declara a confiança com um critério: alta só quando um teste ou uma medição descarta as outras causas. Esse nível orienta o atendente, mas não é uma probabilidade calibrada.

### Valor da informação nas perguntas

As perguntas seguem a ideia de valor da informação: uma pergunta só vale a pena se a resposta puder mudar a próxima ação. O campo `resolves` obriga o modelo a dizer como cada resposta muda a conduta, e a primeira pergunta deve ser a que mais separa as causas restantes.

### Output Formatting e validação determinística

Cada tipo tem um schema JSON próprio (`models.py`), e a tarefa descreve o sentido e o tamanho de cada campo. Formato rígido pode limitar o modelo, mas aceitei esse custo porque a saída precisa ser verificável e aparecer sempre do mesmo jeito.

O fluxo volta como nós e arestas, e o desenho em ASCII é feito por código (`flow_render.py`). O modelo decide o conteúdo, e o código cuida da parte determinística.

No `gpt-oss`, o modo estrito do Groq ainda devolveu erro 400 com o texto gerado nos meus testes. Por isso trato esse texto como resposta a validar. O `output_validation.py` confere citação literal, credenciais, assunto, grafo do fluxo e promessas de visita, reembolso, desconto ou crédito. Também recusa medidas e caminhos de menu que não vieram do ticket.

Uma resposta inválida recebe uma única tentativa de correção com a lista de erros. Mais tentativas aumentariam o custo e esconderiam prompts ruins. O motivo de cada reparo fica gravado em `validation_errors`.

### Versionamento de prompt

Cada arquivo de versão tem um bloco `[meta]` com a versão de origem, a data, a hipótese e a lista de mudanças. Cada execução grava a versão e o hash das mensagens enviadas, então dá para saber exatamente qual prompt gerou cada resposta.

## Como as versões evoluíram

Cada versão nasceu de uma falha que apareceu nas respostas da anterior.

| Versão | Falha que motivou | O que mudou |
| --- | --- | --- |
| `v1` | Ponto de partida | Papel em uma frase, perfil como rótulos, ticket colado no texto e uma frase de tarefa por tipo |
| `v2` | A v1 dava termos técnicos a uma cliente leiga ao telefone | Limites, perfil em regras, ticket delimitado e sentido de cada campo |
| `v3` | No fluxo da v2, a cliente dizia que a internet estava estável e o chamado era aberto do mesmo jeito. O horário da queda também virava prova de sobrecarga | Cada resposta leva a uma conduta diferente, evidência se separa de sintoma e o diagnóstico sempre traz hipóteses com a incerteza declarada |
| `v4` | Em outras combinações de perfil e ticket, a v3 aplicou a triagem de falha intermitente a problemas contínuos e repetiu testes que o cliente já tinha feito | Um método de decisão comum no bloco de sistema, sem regras escritas para um tipo de falha |

Testei outras variações no caminho, como encadeamento e remoção de blocos do prompt. Só o que melhorou as respostas ficou na v3.

A `v1` é o primeiro prompt razoável que eu escreveria, com o mesmo schema de saída. Ela não é uma versão sabotada, para a comparação medir o efeito das instruções e não o do formato.

Para não ajustar o prompt a um caso específico, separei os tickets. Uso `t01` a `t03` para escrever e corrigir os prompts, e `t04` a `t08` ficam reservados para o teste final. É a lógica de um conjunto de teste reservado (holdout): o que foi usado para ajustar deixa de medir generalização.

## Como comparei

A comparação registrada está em `samples/comparacao_v1_v3_p01_t01.json`. O cenário é a Marta (leiga, estilo didático, atendida por telefone) com o ticket da internet que cai toda noite. São 24 respostas: 2 versões, 4 tipos e as seeds 101, 102 e 103.

Só o prompt muda entre as versões. O modelo (`openai/gpt-oss-120b`), a temperatura (0), o `top_p`, o limite de tokens, o esforço de raciocínio e o schema de cada tipo são os mesmos e ficam gravados no próprio arquivo (`locked_config` e `schema_hashes`). Cada seed é usada nas duas versões, para parear as execuções. As quatro respostas da v3 na seed 101 vieram do cache de uma execução idêntica, com a mesma chave de provedor, modelo, parâmetros, mensagens e schema.

Mesmo com temperatura 0, as respostas não se repetem: entre as seeds, só 1 dos 12 pares da v1 e nenhum dos 12 pares da v3 deram a mesma saída. Por isso as 3 seeds funcionam como repetições, e a seed é só um melhor esforço do provedor.

O comando `evaluate` mostra as respostas como A e B, numa ordem sorteada e gravada (`blind_map`). Para cada critério ele pede a nota de 1 a 5, a justificativa e um trecho literal da saída. As âncoras das notas 1, 3 e 5 estão em `support_assistant/evaluation/rubric.py` e seguem a ideia de escalas ancoradas em comportamento: cada nota descreve algo observável, em vez de um adjetivo solto.

## O que mudou nas respostas

Antes da rubrica, contei o que dá para verificar direto no arquivo, nas 3 seeds:

| O que contei | `v1` | `v3` |
| --- | --- | --- |
| Passos no passo a passo | 7, 7 e 7 | 3, 3 e 3 |
| Frases com mais de 20 palavras no passo a passo | 0, 1 e 1 | 1, 2 e 0 |
| Perguntas de esclarecimento (o telefone pede até 3) | 5 em cada seed | 3 em cada seed |
| Hipóteses no diagnóstico | 4 em cada seed | 3 em cada seed |
| Itens de "o que falta saber" | 7, 7 e 6 | 3 em cada seed |
| Confiança da conclusão | alta nas 3 | média nas 3 |
| Citações copiadas literalmente do ticket | 12 de 12 | 9 de 9 |
| Fluxos válidos | 3 de 3 | 3 de 3 |
| Respostas que precisaram de reparo | 9 de 12 | 1 de 12 |

A v3 não ganhou em tudo. O limite de 20 palavras por frase continua sendo violado. Os empates em citações e fluxos valem para a saída final: na v1, 2 dos 3 diagnósticos e 1 dos 3 fluxos passaram por reparo.

Lendo as respostas da seed 101, as diferenças aparecem assim.

**Diagnóstico.** A v1 conclui com confiança alta a partir de uma única frase do ticket e pede dados que não ajudam a decidir, como "Modelo e número de série do modem/roteador.". O resumo fala da cliente em vez de falar com ela: "Marta relata que a conexão de internet cai todas as noites". A v3 ordena as hipóteses, diz o que o trecho ainda não prova ("mas o ticket não confirma falha de hardware") e fala direto com a cliente. A confiança média ainda é generosa para um ticket que só descreve o sintoma.

**Passo a passo.** A v1 manda desligar o modem logo de início e explica que "Os capacitores internos do modem descarregam", em 7 passos. A v3 começa perguntando se a internet funciona naquele momento. Depois pede para testar mais de um aparelho, mas só dá instruções para falha em todos ou em apenas um. Se a falha for isolada, manda encaminhar antes de testar o aparelho afetado.

**Perguntas.** A v1 faz 5 perguntas, e a primeira é aberta e dupla ("Qual é o modelo do seu roteador/modem e há quanto tempo ele está em uso?"), com uma resposta que não muda o próximo passo. A v3 faz 3 perguntas, e as duas primeiras já definem o caminho. A terceira pergunta o horário, que o ticket já responde em parte.

**Fluxo.** A v1 abre com uma saudação e uma pergunta vaga sobre o padrão das luzes. O nó seguinte volta às luzes, e uma luz piscando já leva ao agendamento de visita. A v3 começa perguntando se a internet funciona naquele momento e separa a falha geral da falha em um aparelho. A orientação de reiniciar o modem ainda deixa a explicação prática a cargo do atendente.

**Outros perfis e tickets.** Os exemplos em `samples/` mostram a v3 em outros cenários. No Wi-Fi lento, o diagnóstico da Carla (avançada, por e-mail) usa as medições do ticket, como `RSSI -78 dBm`. O passo a passo da Carla para a queda noturna traz assunto de e-mail, mas ainda é básico para o nível dela. No fluxo do Rafael, sobre o SMS que não chega, a primeira pergunta saiu como instrução ("Verifique se o SMS chegou agora.").

## Resultado da rubrica

Avaliei as oito respostas da seed 101. As 24 notas foram registradas como `human`, cada uma com justificativa e um trecho da resposta. [Comparação v1 e v3](samples/comparacao_v1_v3_p01_t01.json)

| Critério | v1 | v3 |
| --- | ---: | ---: |
| Clareza | 2,75 | 4,25 |
| Adequação ao nível do cliente | 2,25 | 4,25 |
| Acionabilidade | 2,50 | 3,50 |
| **Média geral** | **2,50** | **4,00** |

A maior diferença foi na adequação ao nível da Marta. As perguntas da v3 são fáceis de fazer por telefone e começam pelo que muda a próxima ação. O passo a passo ficou com 3 em acionabilidade porque ainda deixa lacunas depois do teste em outros aparelhos.

## Teste em tickets reservados

A comparação com rubrica mostra a evolução em um cenário, mas não diz se o prompt funciona em outros. Para testar isso, congelei a `v3` e a `v4`, gravei a regra de decisão antes de ver os resultados e rodei as duas versões em cinco tickets que não foram usados para ajustar os prompts: queda após tempestade, TV travando, ping alto em jogos, cobrança em dobro com senha no relato e um relato vago. São 40 respostas, com o mesmo modelo, a mesma temperatura e a seed 101.

A regra era: a `v4` só viraria padrão se não tivesse mais saídas inválidas que a `v3` e cometesse menos violações do método.

| Medida | `v3` | `v4` |
| --- | ---: | ---: |
| Válidas na coleta | 18 de 20 | 17 de 20 |
| Válidas após corrigir o validador | 20 de 20 | 18 de 20 |
| Respostas que precisaram de reparo | 1 | 3 |

O teste encontrou dois erros no meu validador. Um hífen tipográfico em "Wi-Fi" reprovava citações corretas, e setas usadas para ligar uma resposta a uma conduta eram confundidas com caminhos de menu. Corrigi os dois e reavaliei as mesmas respostas, sem nova chamada ao modelo. A decisão não mudou, porque a `v4` continuou com mais saídas inválidas.

**A `v3` é a versão final.** A `v4` fica no projeto como registro do experimento: concentrar o método no sistema não foi suficiente para o modelo aplicá-lo com consistência. A `v3` também tem falhas nesses tickets. No caso da tempestade, a cliente já tinha reiniciado o modem duas vezes, e o fluxo ainda diz "Oriente o cliente a reiniciar o modem novamente.". Os arquivos estão em [samples/teste_reservado/](samples/teste_reservado/).

## Decisões e trade-offs

- **Cache pela impressão digital da chamada.** A chave é o hash do provedor, do modelo, dos parâmetros, da versão, das mensagens e do schema (`support_assistant/cache.py`). Qualquer mudança gera uma chave nova, e só respostas válidas entram no cache.
- **Provedor simulado.** O provedor `fake` devolve respostas fixas e é o padrão do `.env.example`, para o projeto rodar e ser testado sem chave e sem rede.
- **Modelo padrão.** Uso `gpt-oss-120b` com temperatura 0 porque foi a configuração avaliada. No `gpt-oss-20b`, a v4 ignorou parte do método e um fluxo saiu inválido mesmo depois do reparo.
- **Intervalo entre chamadas reais.** A camada gratuita do Groq limita tokens por minuto, então a comparação espera entre as chamadas, inclusive entre a resposta e o reparo (`support_assistant/providers/pacing.py`).

## Limitações

- A comparação com rubrica cobre um perfil e um ticket, com 3 seeds. Ela mostra diferenças nesse cenário e não prova ganho geral.
- A ordem A/B e o trecho obrigatório reduzem o viés da avaliação, mas quem avalia conhece os prompts. A rubrica pode ser reaplicada por outra pessoa sobre o mesmo arquivo.
- As respostas variam entre execuções mesmo com temperatura 0. Uma nova coleta não vai repetir o texto, só a tendência.
- A v3 ainda escreve frases acima de 20 palavras no telefone, às vezes aplica a triagem de falha intermitente a problemas contínuos e pode repetir testes que o cliente já fez.
- O código recusa medidas e caminhos de menu que não vieram do ticket, mas não detecta todo fato inventado, como um recurso da operadora citado em texto corrido.
- Ficaram de fora uma base de conhecimento da operadora, a integração com um sistema de tickets real e a conversa com várias trocas.

## O que eu mudaria numa próxima versão

1. **Avaliação mais ampla.** Aplicar a rubrica a outros perfis e tickets e registrar as falhas por tipo de resposta.
2. **Fatos em uma etapa separada.** Extrair primeiro o que o cliente já informou e entregar essa lista às tarefas, porque só pedir isso no prompt não bastou na `v4`.
3. **Base de conhecimento.** Dar ao modelo uma referência da operadora, para que valores técnicos e procedimentos venham de uma fonte conferível.
4. **Regra de fala em código.** Validar o limite de palavras por frase nos atendimentos por telefone.
