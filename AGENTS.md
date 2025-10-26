# Pontos de atenção do projeto

1. **Dependências de execução**: o script `script_optimization_3.py` depende de `pandas`, `numpy` e `ortools`. Garanta que o ambiente Python possua esses pacotes instalados antes de executar a otimização.
2. **Planilha de entrada**: mantenha a aba `Participants` alinhada com os parâmetros configurados (número de mesas e cadeiras por mesa). O script agora interrompe a execução quando há mais participantes do que vagas por rodada.
3. **Parâmetros via ambiente**: é possível sobrepor `ROUNDS`, `TABLES`, `SEATS_PER_TABLE`, tempo limite (`MAX_TIME_S`) e número de workers usando as variáveis `NW_ROUNDS`, `NW_TABLES`, `NW_SEATS_PER_TABLE`, `NW_MAX_TIME_S` e `NW_N_WORKERS`.
4. **Grupos must_together**: utilize o relatório de diagnóstico inicial do script para verificar gargalos. Cada pessoa consegue encontrar até `ROUNDS * (SEATS_PER_TABLE - 1)` parceiros distintos; ultrapassar esse limite tende a gerar inviabilidade.
5. **Restrições must_avoid**: o diagnóstico também aponta os participantes com mais bloqueios de encontros. Revise esses casos ao adicionar novas pessoas para evitar restrições conflitantes.
6. **Status do solver**: quando o CP-SAT retornar `UNKNOWN`, significa que não houve solução no tempo limite. Ajuste `NW_MAX_TIME_S` ou simplifique restrições ao ampliar o número de participantes.
7. **Aliases de ramos**: mantenha os arquivos `ramo_alias.csv` e `affinity_overrides.csv` atualizados para reduzir ruídos de classificação e garantir que as preferências sejam avaliadas corretamente.
8. **Debug rápido**: aproveite o diagnóstico impresso antes da otimização para validar fixos, contagens e possíveis gargalos antes de iniciar uma busca que pode consumir tempo considerável.
