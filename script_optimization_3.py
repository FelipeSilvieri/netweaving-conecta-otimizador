import pandas as pd
import numpy as np
import os
import re
import time
import unicodedata
import math
from collections import defaultdict, Counter

from itertools import combinations
from ortools.sat.python import cp_model

participants = pd.read_excel('Rodada_Negocios_Netweaving_27102025_3.xlsx', sheet_name='Participants')
must_together = pd.read_excel('Rodada_Negocios_Netweaving_27102025_3.xlsx', sheet_name='Must_Together_2')
must_avoid = pd.read_excel('Rodada_Negocios_Netweaving_27102025_3.xlsx', sheet_name='Must_Avoid')
ramo_alias = pd.read_csv('ramo_alias.csv')
affinity_overrides = pd.read_csv('affinity_overrides.csv')

# === Otimizador de Rodada de Negócios (CP-SAT / OR-Tools) — versão com fixos robustos ===
# Entradas já carregadas: participants, must_together, must_avoid, ramo_alias, affinity_overrides

# -------------------------
# Parâmetros do evento
# -------------------------
def _env_int(var, default):
    val = os.getenv(var)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


ROUNDS = _env_int('NW_ROUNDS', 4)
TABLES = _env_int('NW_TABLES', 17)
SEATS_PER_TABLE = _env_int('NW_SEATS_PER_TABLE', 4)

# -------------------------
# Pesos (ajuste fino)
# -------------------------
W_TOGETHER = 5000
W_SIM = 10
W_PREF1 = 300
W_PREF2 = 180
W_PREF3 = 90
PREF_FILLED_BOOST = 1.5
W_OVER = 800
DEFAULT_SIM_SAME = 0.30
DEFAULT_SIM_DIFF = 0.55

MAX_TIME_S = _env_int('NW_MAX_TIME_S', 120)
N_WORKERS = _env_int('NW_N_WORKERS', 8)

# -------------------------
# Normalização por alias
# -------------------------
def strip_accents(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s

alias_map = {}
if {'entrada','normalizado'}.issubset({c.lower() for c in ramo_alias.columns}):
    _entrada = [c for c in ramo_alias.columns if c.lower()=='entrada'][0]
    _normalizado = [c for c in ramo_alias.columns if c.lower()=='normalizado'][0]
    for _,row in ramo_alias.iterrows():
        k = strip_accents(row[_entrada])
        v = str(row[_normalizado]).strip().lower()
        if k:
            alias_map[k] = v

def alias_norm(x: str) -> str:
    k = strip_accents(x)
    return alias_map.get(k, k if k else "outros")

# -------------------------
# Preparo dos participants
# -------------------------
cols = {c.lower(): c for c in participants.columns}
getcol = lambda name: cols.get(name)

id_col       = getcol('id')
nome_col     = getcol('nome_pessoa') or getcol('nome') or getcol('pessoa') or getcol('nome completo')
empresa_col  = getcol('nome_empresa') or getcol('empresa')
ramo_col     = getcol('ramo')
pref1_col    = getcol('pref1_ramo')
pref2_col    = getcol('pref2_ramo')
pref3_col    = getcol('pref3_ramo')
fixo_col     = getcol('fixo')          # pode não existir
mesa_fixa_col= getcol('mesa_fixa') or getcol('mesa fixa') or getcol('mesa fixa ')

dfP = participants.copy()

# id
if id_col is None:
    dfP['id'] = np.arange(len(dfP))
    id_col = 'id'

# nome/empresa/ramo (obrigatórios)
if nome_col is None:
    raise ValueError("Coluna 'nome_pessoa' (ou equivalente) não encontrada em Participants.")
if empresa_col is None:
    raise ValueError("Coluna 'nome_empresa' (ou equivalente) não encontrada em Participants.")
if ramo_col is None:
    raise ValueError("Coluna 'ramo' não encontrada em Participants.")

# normaliza ramo e prefs
dfP['ramo_norm'] = dfP[ramo_col].astype(str).apply(alias_norm)
for colname, newname in [(pref1_col,'pref1_norm'), (pref2_col,'pref2_norm'), (pref3_col,'pref3_norm')]:
    if colname:
        dfP[newname] = dfP[colname].astype(str).apply(lambda x: alias_norm(x) if str(x).strip() != "" else "")
    else:
        dfP[newname] = ""

# mesa fixa: força fixo=1 quando houver mesa definida
if mesa_fixa_col is None:
    dfP['mesa_fixa'] = np.nan
    mesa_fixa_col = 'mesa_fixa'
else:
    dfP['mesa_fixa'] = pd.to_numeric(dfP[mesa_fixa_col], errors='coerce')

# Heurística de base 1 → base 0 (se todos valores válidos estiverem em 1..TABLES e não houver 0)
if dfP['mesa_fixa'].notna().any():
    mf_nonnull = dfP['mesa_fixa'].dropna()
    cond_one_based = (mf_nonnull.min() >= 1) and (mf_nonnull.max() <= TABLES) and (0 not in set(mf_nonnull.astype(int)))
    if cond_one_based:
        dfP.loc[dfP['mesa_fixa'].notna(), 'mesa_fixa'] = dfP.loc[dfP['mesa_fixa'].notna(), 'mesa_fixa'].astype(int) - 1

# fixo (coluna pode não existir; aceita números, bools, strings)
def _to_bool01(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, np.integer)): return 1 if int(x)!=0 else 0
    if isinstance(x, float): 
        if np.isnan(x): return 0
        return 1 if int(round(x))!=0 else 0
    s = str(x).strip().lower()
    return 1 if s in {'1','true','t','sim','yes','y'} else 0

if fixo_col is None:
    dfP['fixo'] = 0
else:
    dfP['fixo'] = dfP[fixo_col].apply(_to_bool01)

# **Regra nova**: quem tem mesa_fixa preenchida vira fixo=1 (sobrepõe)
dfP.loc[dfP['mesa_fixa'].notna(), 'fixo'] = 1

# sanity básico
N = len(dfP)
total_slots = TABLES * SEATS_PER_TABLE
if N > total_slots:
    raise RuntimeError(
        f"Existem {N} participantes para apenas {total_slots} vagas por rodada. "
        "Aumente o número de mesas ou assentos antes de rodar o solver.")
if N != total_slots:
    print(
        f"[Aviso] Há {N} participantes, mas a configuração prevê {total_slots} lugares por rodada. "
        "Mesas serão mantidas com 4 participantes por rodada; haverá cadeiras ociosas.")

# mapeamentos
P_ids = list(dfP[id_col].tolist())
id_to_idx = {pid:i for i,pid in enumerate(P_ids)}

names    = dfP[nome_col].astype(str).tolist()
empresas = dfP[empresa_col].astype(str).tolist()
ind_of   = dfP['ramo_norm'].astype(str).tolist()
pref1    = dfP['pref1_norm'].astype(str).tolist()
pref2    = dfP['pref2_norm'].astype(str).tolist()
pref3    = dfP['pref3_norm'].astype(str).tolist()
has_pref = [(str(pref1[i]).strip()!="") or (str(pref2[i]).strip()!="") or (str(pref3[i]).strip()!="") for i in range(N)]

fixed_info = {}
for i in range(N):
    if dfP['fixo'].iloc[i] == 1 and pd.notna(dfP['mesa_fixa'].iloc[i]):
        mf = int(dfP['mesa_fixa'].iloc[i])
        if not (0 <= mf < TABLES):
            raise ValueError(f"Mesa fixa fora do range [0..{TABLES-1}] para {names[i]}: {mf}")
        fixed_info[i] = mf  # fixo por mesa

# -------------------------
# must_together / must_avoid -> índices
# -------------------------
def get_pair_df(df, ca='id_a', cb='id_b'):
    if df is None or df.empty:
        return []
    cols = {c.lower(): c for c in df.columns}
    A = cols.get(ca.lower(), list(df.columns)[0])
    B = cols.get(cb.lower(), list(df.columns)[1])
    out = []
    for _,row in df.iterrows():
        a = row[A]; b = row[B]
        if pd.isna(a) or pd.isna(b): continue
        a = id_to_idx.get(a, None); b = id_to_idx.get(b, None)
        if a is None or b is None or a == b: continue
        if a > b: a,b = b,a
        out.append((a,b))
    return sorted(set(out))

must_together_pairs = get_pair_df(must_together)
must_avoid_pairs    = get_pair_df(must_avoid)

# -------------------------
# Diagnóstico informativo
# -------------------------
def diagnostic_report():
    print("\n=== Diagnóstico das entradas ===")
    print(f"Participantes: {N}")
    print(
        f"Mesas configuradas: {TABLES} | Assentos/mesa: {SEATS_PER_TABLE} | "
        f"Vagas totais por rodada: {total_slots}")
    print(
        f"Pares must_together: {len(must_together_pairs)} | "
        f"pares must_avoid: {len(must_avoid_pairs)}")

    if fixed_info:
        print("- Participantes fixos por mesa:")
        fixed_by_table = defaultdict(list)
        for idx, mesa in fixed_info.items():
            fixed_by_table[mesa].append(idx)
        for mesa in sorted(fixed_by_table):
            nomes = ", ".join(names[i] for i in fixed_by_table[mesa])
            print(f"  Mesa {mesa}: {nomes}")
    else:
        print("- Nenhum participante possui mesa fixa.")

    limit_mt = ROUNDS * (SEATS_PER_TABLE - 1)
    mt_degree = Counter()
    for a, b in must_together_pairs:
        mt_degree[a] += 1
        mt_degree[b] += 1

    alertas = []
    for idx, qtd in mt_degree.items():
        if qtd > limit_mt and idx not in fixed_info:
            alertas.append(
                f"{names[idx]} precisa encontrar {qtd} parceiros must_together, acima do limite teórico de {limit_mt} em {ROUNDS} rodadas.")
        elif idx in fixed_info and qtd > limit_mt:
            alertas.append(
                f"{names[idx]} (fixo na mesa {fixed_info[idx]}) possui {qtd} parceiros must_together; mesa fixa comporta no máximo {limit_mt} convidados distintos.")

    if alertas:
        print("[Alerta] Possível gargalo nos must_together:")
        for msg in alertas:
            print("  - " + msg)
    else:
        print("- Nenhum gargalo teórico de must_together detectado.")

    if must_avoid_pairs:
        ma_degree = Counter()
        for a, b in must_avoid_pairs:
            ma_degree[a] += 1
            ma_degree[b] += 1
        mais_ma = sorted(ma_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        if mais_ma:
            print("- Participantes com mais restrições must_avoid:")
            for idx, qtd in mais_ma:
                print(f"  {names[idx]}: evita {qtd} participante(s)")
    print("=== Fim do diagnóstico ===\n")

diagnostic_report()

# -------------------------
# Pré-checagens (fail-fast)
# -------------------------
# a) capacidade por mesa só com fixos
fix_count = Counter(fixed_info.values())
overcap = {t:c for t,c in fix_count.items() if c > SEATS_PER_TABLE}
if overcap:
    raise RuntimeError(f"Capacidade excedida com fixos: {overcap}")

# b) must_avoid com dois fixos mesma mesa
bad_avoid = []
for (a,b) in must_avoid_pairs:
    if (a in fixed_info) and (b in fixed_info) and (fixed_info[a]==fixed_info[b]):
        bad_avoid.append((P_ids[a], P_ids[b], fixed_info[a]))
if bad_avoid:
    raise RuntimeError(f"Par(es) must_avoid estão fixos na mesma mesa (inviável): {bad_avoid}")

# c) must_together com dois fixos em mesas diferentes
bad_mt = []
for (a,b) in must_together_pairs:
    if (a in fixed_info) and (b in fixed_info) and (fixed_info[a]!=fixed_info[b]):
        bad_mt.append((P_ids[a], P_ids[b], fixed_info[a], fixed_info[b]))
if bad_mt:
    raise RuntimeError(f"Par(es) must_together fixos em mesas distintas (inviável): {bad_mt}")

# -------------------------
# Afinidade (sim) por ramo
# -------------------------
def sim_lookup(ind_a, ind_b):
    ia = str(ind_a).strip().lower()
    ib = str(ind_b).strip().lower()
    if len(affinity_overrides) > 0:
        cols = {c.lower(): c for c in affinity_overrides.columns}
        ra = cols.get('ramo_a', list(affinity_overrides.columns)[0])
        rb = cols.get('ramo_b', list(affinity_overrides.columns)[1])
        rs = cols.get('sim',    list(affinity_overrides.columns)[2])
        mask = ((affinity_overrides[ra].str.strip().str.lower()==ia) & 
                (affinity_overrides[rb].str.strip().str.lower()==ib))
        val = affinity_overrides.loc[mask, rs]
        if len(val)==0:
            mask = ((affinity_overrides[ra].str.strip().str.lower()==ib) & 
                    (affinity_overrides[rb].str.strip().str.lower()==ia))
            val = affinity_overrides.loc[mask, rs]
        if len(val)>0:
            try:
                return float(val.iloc[0])
            except:
                pass
    return DEFAULT_SIM_SAME if ia == ib else DEFAULT_SIM_DIFF

sim100 = {}
inds_unique = sorted(set(ind_of))
for a in inds_unique:
    for b in inds_unique:
        sim100[(a,b)] = int(round(sim_lookup(a,b) * 100))

# -------------------------
# Score de preferência (direcional)
# -------------------------
def pref_dir_score(i_from, i_to):
    ramo_to = ind_of[i_to]
    s = 0.0
    if ramo_to and ramo_to == pref1[i_from]: s += W_PREF1
    if ramo_to and ramo_to == pref2[i_from]: s += W_PREF2
    if ramo_to and ramo_to == pref3[i_from]: s += W_PREF3
    if has_pref[i_from]:
        s *= PREF_FILLED_BOOST
    return int(round(s))

# -------------------------
# Modelo CP-SAT
# -------------------------
R = range(ROUNDS)
T = range(TABLES)
P = range(N)
model = cp_model.CpModel()

# X[i,r,t] = 1 se pessoa i está na mesa t na rodada r
X = {(i,r,t): model.NewBoolVar(f"X_{i}_{r}_{t}") for i in P for r in R for t in T}

# 1) Cada pessoa ocupa exatamente 1 mesa por rodada
for i in P:
    for r in R:
        model.Add(sum(X[i,r,t] for t in T) == 1)

# 2) Capacidade: 4 por mesa/rodada
for r in R:
    for t in T:
        model.Add(sum(X[i,r,t] for i in P) == SEATS_PER_TABLE)

# 3) Fixos: mesa fixa em todas as rodadas
for i, mf in fixed_info.items():
    for r in R:
        model.Add(X[i,r,mf] == 1)

# 3b) Não fixos não podem repetir mesa (cada mesa no máx. 1x para a pessoa)
for i in P:
    if i not in fixed_info:
        for t in T:
            model.Add(sum(X[i,r,t] for r in R) <= 1)

# 4) must_avoid
for (a,b) in must_avoid_pairs:
    for r in R:
        for t in T:
            model.Add(X[a,r,t] + X[b,r,t] <= 1)

# 5) Indicadores de encontro por par/rodada/mesa (somente a<b)
pairs = [(a,b) for (a,b) in combinations(P,2)]
B = {}
for (a,b) in pairs:
    for r in R:
        for t in T:
            B[a,b,r,t] = model.NewBoolVar(f"B_{a}_{b}_{r}_{t}")
            model.AddBoolAnd([X[a,r,t], X[b,r,t]]).OnlyEnforceIf(B[a,b,r,t])
            model.AddBoolOr([X[a,r,t].Not(), X[b,r,t].Not()]).OnlyEnforceIf(B[a,b,r,t].Not())

# 6) Pares não must_together: no máximo 1 encontro no evento
must_together_set = set(must_together_pairs)
for (a,b) in pairs:
    if (a,b) not in must_together_set:
        model.Add(sum(B[a,b,r,t] for r in R for t in T) <= 1)

# 7) Y[a,b] para must_together atendido
Y = {}
for (a,b) in must_together_pairs:
    Y[a,b] = model.NewBoolVar(f"Y_{a}_{b}")
    model.Add(Y[a,b] <= sum(B[a,b,r,t] for r in R for t in T))
    for r in R:
        for t in T:
            model.Add(Y[a,b] >= B[a,b,r,t])

# 8) Diversidade por mesa: penaliza cnt(ramo) > 1
inds_present = sorted(set(ind_of))
over_vars = []
for r in R:
    for t in T:
        for ind in inds_present:
            cnt = model.NewIntVar(0, SEATS_PER_TABLE, f"cnt_{ind}_r{r}_t{t}")
            inds_idx = [i for i in P if ind_of[i] == ind]
            if len(inds_idx) == 0:
                model.Add(cnt == 0)
            else:
                model.Add(cnt == sum(X[i,r,t] for i in inds_idx))
            over = model.NewIntVar(0, SEATS_PER_TABLE, f"over_{ind}_r{r}_t{t}")
            model.Add(over >= cnt - 1)
            model.Add(over >= 0)
            over_vars.append(over)

# -------------------------
# Função objetivo (minimização)
# -------------------------
obj_terms = []

# must_together (max)
for (a,b), yvar in Y.items():
    obj_terms.append((-W_TOGETHER, yvar))

# afinidade + preferências (max)
for (a,b) in pairs:
    s_sim  = W_SIM * sim100[(ind_of[a], ind_of[b])]
    s_pref = pref_dir_score(a, b) + pref_dir_score(b, a)
    s_pair = s_sim + s_pref
    if s_pair != 0:
        for r in R:
            for t in T:
                obj_terms.append((-s_pair, B[a,b,r,t]))

# diversidade (min)
for over in over_vars:
    obj_terms.append((W_OVER, over))

model.Minimize(sum(coef * var for (coef, var) in obj_terms))

# -------------------------
# Solver
# -------------------------
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = MAX_TIME_S
solver.parameters.num_search_workers = N_WORKERS

status = solver.Solve(model)
status_name = solver.StatusName(status)
obj_val = solver.ObjectiveValue() if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
bound = solver.BestObjectiveBound()
print(f"[Solver] Status: {status_name} | Objetivo: {obj_val} | Bound: {bound}")

if status == cp_model.INFEASIBLE:
    raise RuntimeError("Modelo infeasible (verifique pares must_avoid/must_together e distribuição de fixos).")
if status == cp_model.MODEL_INVALID:
    raise RuntimeError("Modelo inválido. Revise se todos os dados foram lidos corretamente.")
if status == cp_model.UNKNOWN:
    raise RuntimeError(
        f"Solver não encontrou solução dentro do limite de {MAX_TIME_S}s (status UNKNOWN). "
        "Considere aumentar NW_MAX_TIME_S ou simplificar as restrições.")

# -------------------------
# Extrai solução
# -------------------------
assign = {(r,t): [] for r in R for t in T}
for i in range(N):
    for r in R:
        for t in T:
            if solver.Value(X[i,r,t]) == 1:
                assign[(r,t)].append(i)

# DataFrame longo
rows = []
for r in R:
    for t in T:
        for i in assign[(r,t)]:
            rows.append({
                "rodada": r,
                "mesa": t,
                "id": P_ids[i],
                "nome_pessoa": names[i],
                "nome_empresa": empresas[i],
                "ramo_norm": ind_of[i],
                "pref1_norm": pref1[i],
                "pref2_norm": pref2[i],
                "pref3_norm": pref3[i],
                "fixo": int(dfP['fixo'].iloc[i]),
                "mesa_fixa": int(dfP['mesa_fixa'].iloc[i]) if pd.notna(dfP['mesa_fixa'].iloc[i]) else np.nan,
            })
schedule_df = pd.DataFrame(rows).sort_values(["rodada","mesa","nome_pessoa"]).reset_index(drop=True)

# Visual compacto
def to_wide(g):
    ps = list(g.sort_values("nome_pessoa")["nome_pessoa"])
    row = {"rodada": int(g["rodada"].iloc[0]), "mesa": int(g["mesa"].iloc[0])}
    for k, nm in enumerate(ps[:SEATS_PER_TABLE], start=1):
        row[f"p{k}"] = nm
    return pd.Series(row)

schedule_wide = (
    schedule_df.groupby(["rodada","mesa"], as_index=False)
    .apply(to_wide)
    .sort_values(["rodada","mesa"])
    .reset_index(drop=True)
)

# Relatórios rápidos
mt_att = []
for (a,b) in must_together_pairs:
    atendeu = 0; r_hit = None; t_hit = None
    for r in R:
        for t in T:
            if solver.Value(B[a,b,r,t]) == 1:
                atendeu = 1; r_hit = r; t_hit = t; break
        if atendeu: break
    mt_att.append({"id_a": P_ids[a], "nome_a": names[a], "id_b": P_ids[b], "nome_b": names[b],
                   "atendido": atendeu, "rodada": r_hit, "mesa": t_hit})
must_together_report = pd.DataFrame(mt_att)

ma_viol = []
for (a,b) in must_avoid_pairs:
    for r in R:
        for t in T:
            if solver.Value(B[a,b,r,t]) == 1:
                ma_viol.append({"id_a": P_ids[a], "nome_a": names[a], "id_b": P_ids[b], "nome_b": names[b],
                                "rodada": r, "mesa": t})
must_avoid_violations = pd.DataFrame(ma_viol)

# Checagem extra: fixos realmente permaneceram na mesma mesa (e na mesa certa)
fixed_check = []
for i, mf in fixed_info.items():
    mesas_i = sorted(schedule_df.query("id == @P_ids[i]")["mesa"].unique().tolist())
    fixed_check.append({
        "id": P_ids[i], "nome": names[i], "mesa_fixa_esperada": mf, "mesas_encontradas": mesas_i,
        "ok": (mesas_i == [mf])
    })
fixed_check_df = pd.DataFrame(fixed_check)

print("Resumo:")
print("- must_together atendidos:", must_together_report['atendido'].sum(), "/", len(must_together_report))
print("- must_avoid violações   :", len(must_avoid_violations))
if not fixed_check_df.empty and not fixed_check_df['ok'].all():
    print("[ATENÇÃO] Algum fixo não ficou 100% na mesa esperada. Veja fixed_check_df.")
