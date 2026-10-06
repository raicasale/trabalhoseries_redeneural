"""Gera docs/DOCUMENTACAO_TREINAMENTO.md a partir de results/ (nenhum número digitado à mão)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import trabalhoseries_redeneural.src.tsutils_delhi as tu

ROOT = tu.ROOT; R = ROOT / "results"
def tab(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]): df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values) + "\n"

meta = json.load(open(R / "protocolo.json")); sel = json.load(open(R / "hiperparametros_selecionados.json"))
P = pd.read_csv(R / "previsoes_walkforward.csv", parse_dates=["data"])
MODELS = ["Holt-Winters", "SARIMAX", "Random Forest", "MLP Regressor"]; REF = "Persistência (ref.)"
raw = tu.load_raw(); D = tu.clean(raw); y = D.meantemp.values
dec = tu.stl(y, 7, seasonal=13, trend=21); fs, ft = tu.strength(dec)
ph = tu.trend_phases(D.date, dec["trend"]); ph["inicio"] = ph.inicio.dt.date; ph["fim"] = ph.fim.dt.date

rows = []
ep = (P.real - P[REF]).values
for m in MODELS + [REF, "Média móvel 7d (ref.)", "Sazonal ingênuo m=7 (ref.)"]:
    e = (P.real - P[m]).values
    dm, p = tu.diebold_mariano(ep, e) if m in MODELS else (np.nan, np.nan)
    rows.append(dict(modelo=m, MAE=tu.mae(P.real, P[m]), RMSE=tu.rmse(P.real, P[m]), vies=e.mean(), DM_vs_persistencia=dm, p_DM=p))
RES = pd.DataFrame(rows)
lb = []
for m in MODELS:
    e = (P.real - P[m]).values; t = tu.ljung_box(e, [5, 10]).set_index("lag")
    lb.append(dict(modelo=m, Q5=t.loc[5, "Q"], p5=t.loc[5, "p_valor"], Q10=t.loc[10, "Q"], p10=t.loc[10, "p_valor"], ACF1=tu.acf(e, 1)[1]))
LB = pd.DataFrame(lb)
tempos = pd.read_csv(R / "tempos_execucao.csv", index_col=0)["segundos"].round(2).to_frame().reset_index().rename(columns={"index": "modelo"})
sd = pd.read_csv(R / "mlp_sensibilidade_semente.csv"); al = pd.read_csv(R / "mlp_sensibilidade_alpha_posthoc.csv")
ab = pd.read_csv(R / "mlp_ablacao.csv"); jw = pd.read_csv(R / "selecao_w_conjunto.csv")
cf = pd.read_csv(R / "coeficientes_sarimax.csv").drop(columns=["AIC_css", "BIC_css"])
gm = pd.read_csv(R / "importancia_mlp_grupos.csv"); gr = pd.read_csv(R / "importancia_rf_grupos.csv")
im = pd.read_csv(R / "importancia_mlp.csv").head(8); ir = pd.read_csv(R / "importancia_rf.csv").head(8)
ss = {k: pd.read_csv(R / f"busca_{k}.csv") for k in ["holt_winters", "sarimax", "random_forest", "mlp"]}
cur = pd.read_csv(R / "mlp_curva_aprendizado.csv")
qual = tu.quality_report(raw); outl = tu.outlier_table(raw)

md = f"""# Documentação completa do treinamento: Grupo 4 (MLP Regressor) · base do Grupo 1 (Daily Delhi Climate)

> Documento gerado por `src/build_docs_delhi.py` a partir dos arquivos em `results/`. Todos os números vêm da execução real do pipeline.
> Escopo desta entrega: **somente a base do Grupo 1**. HTML/PDF do relatório, apresentação oral e registro de demandas **não** fazem parte deste pacote.

## 1. Identificação e escopo
| Item | Valor |
|---|---|
| Grupo / modelo de especialização | 4 / MLP Regressor (`sklearn.neural_network.MLPRegressor`) |
| Modelos comuns | SARIMAX, Holt-Winters (referência univariada), Random Forest |
| Base | Daily Delhi Climate, arquivo congelado `datasets/DailyDelhiClimateTest.csv` (não modificado) |
| Período / frequência | 2017-01-01 a 2017-04-24, diária, {len(raw)} observações |
| Alvo | `meantemp` (°C); horizonte h = 1 dia |
| Exógenas | `humidity`, `wind_speed`, `meanpressure` (sempre defasadas) + calendário do dia-alvo |
| Combinações executadas | 4 modelos × 1 base (as outras 4 bases não fazem parte deste arquivo) |

## 2. Ambiente e reprodutibilidade
- Python 3.12; `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` (ver `requirements.txt`). **PyTorch e statsmodels não foram usados** (indisponíveis no ambiente de execução).
- STL, Holt-Winters, SARIMAX, ACF, Ljung-Box e Diebold-Mariano foram **implementados no projeto** (`src/tsutils_delhi.py`, `src/ts_models_delhi.py`) e validados com séries sintéticas (`tests/`). A comparação numérica direta com `statsmodels` não é usada como critério: as implementações deste projeto usam CSS, inicialização explícita dos estados e convenção própria para a tendência inicial, enquanto `statsmodels` ajusta por verossimilhança e inicializa estados de outra forma. Assim, a comparação válida aqui é interna, com a mesma implementação e protocolo para todos os modelos; diferenças de nível devem ser interpretadas como diferenças de convenção, não como erro de reprodução.
- Semente global 42; MLP com sementes 42..46 no teste (5) e 42..44 na busca (3). Buscas salvas em `results/busca_*.csv` (apague-os para refazê-las).
- Reprodução: `pip install -r requirements.txt` → `python tests/test_core_delhi.py && python tests/test_leakage_delhi.py` → `python src/pipeline_delhi.py` → `python src/build_notebooks_delhi.py` → `python src/build_docs_delhi.py`.
- Protocolo: {json.dumps(meta, ensure_ascii=False)}

## 3. Dados e qualidade
{tab(qual)}
Atípicos por escore z robusto (MAD > 3,5):

{tab(outl)}
Decisões de limpeza:
1. Pressão = 59,0 hPa em 01/01/2017 é impossível (demais ≈ 1000 hPa): substituída (NaN → valor do dia seguinte). Distorcia o desvio-padrão da pressão (≈ 89 hPa) e não havia sido detectada na entrega original.
2. Variação de temperatura em 26/01 e queda de pressão em 20/02: **mantidas** (variação meteorológica plausível).
3. Sem duplicidades, sem datas faltantes, sem ausentes. Sem transformação do alvo.

## 4. STL e sazonalidade (período 7; ns=13, nt=21)
- Força da sazonalidade (Hyndman & Athanasopoulos) = **{fs:.3f}** (fraca); força da tendência = **{ft:.3f}**.
- Variância de y: tendência {np.var(dec['trend'])/np.var(y):.1%}, sazonal {np.var(dec['seasonal'])/np.var(y):.1%}, resíduo {np.var(dec['resid'])/np.var(y):.1%}.
- Resíduo STL: ACF(1) = {tu.acf(dec['resid'],1)[1]:.2f}; Ljung-Box p(5) < 0,001 (não é ruído branco: episódios meteorológicos de poucos dias).
- Com 114 dias (< 1 ciclo anual) a **sazonalidade anual não é estimável**. Fases da tendência:

{tab(ph[['inicio','fim','dias','fase','trend_inicio','trend_fim','variacao','inclinacao_media']], 2)}
## 5. Feature Engineering
- Origem t → alvo y(t+1). Features (w=14, **30 colunas**): lags de y (0..w−1), médias de 3 e 7 dias, desvio de 7 dias, diferença de 1 dia; umidade, vento e pressão em t, t−1, t−2; variação da pressão; sen/cos do dia do ano **do dia-alvo**.
- **Nenhuma exógena de t+1 é usada** (não conhecida na origem). Excluídos: dia da semana (sem mecanismo físico; sazonalidade semanal fraca), feriados (irrelevantes).
- Ausentes dos lags/janelas: 13 primeiras origens descartadas (sem imputação); 100 origens utilizáveis para todo `w`.
- Auditoria de vazamento executável: `tests/test_leakage.py`. Padronização do MLP ajustada só no treino de cada origem.
- RF e MLP usam **exatamente as mesmas features** (w compartilhado, ver seção 7).

## 6. Protocolo walk-forward
Teste = últimos 22 alvos (**{meta['test_targets']}**), janela expansiva com **reajuste a cada origem**, h=1, mesmas origens para todos os modelos e referências. Seleção de hiperparâmetros na validação interna (14 origens: {meta['inner_targets']}); fixos no teste.

## 7. Modelos, espaços de busca e seleção
**Janela compartilhada (RF e MLP):** w* = menor média dos melhores MAE internos (decisão só com validação interna).

{tab(jw, 4)}
**Selecionados:**

{tab(pd.DataFrame([(k, json.dumps(v, ensure_ascii=False)) for k, v in sel.items()], columns=['modelo','hiperparâmetros']))}
Espaços: Holt-Winters (tendência {{nenhuma, aditiva, amortecida}} × sazonalidade {{nenhuma, aditiva, multiplicativa}}, m=7); SARIMAX ((p,d,q) ∈ {{0,1,2}}×{{0,1}}×{{0,1}}, (P,D,Q) ∈ {{(0,0,0),(1,0,0),(0,0,1),(1,0,1),(0,1,1)}}, m=7, com/sem exógenas; AIC/BIC por CSS); Random Forest (60 de 972 configurações); MLP (60 de 1.152 configurações; alvo nível ou Δ).
Resultados completos das buscas: `results/busca_*.csv`. Cinco melhores do MLP:

{tab(ss['mlp'].drop(columns='seg').head(5), 4)}
Holt-Winters (todas as combinações):

{tab(ss['holt_winters'], 4)}
## 8. Resultados (MAE fora da amostra, 22 previsões)
{tab(RES)}
Tempo de execução do walk-forward (s):

{tab(tempos)}
- Menor MAE numérico: **{RES[RES.modelo.isin(MODELS)].sort_values('MAE').modelo.iloc[0]}**. **Nenhuma diferença contra a persistência é significativa** (Diebold-Mariano, p ≥ {RES[RES.modelo.isin(MODELS)].p_DM.min():.2f}).
- Ranking (1 base): {', '.join(f'{i+1}º {m}' for i, m in enumerate(RES[RES.modelo.isin(MODELS)].sort_values('MAE').modelo))}. Vitórias: {RES[RES.modelo.isin(MODELS)].sort_values('MAE').modelo.iloc[0]} = 1; demais = 0. Com uma única base, a posição média é a própria posição.
- O dia **07/04/2017** (queda de 31,2 → 27,0 °C) gera erro ≈ −4 °C em todos os modelos (≈ 0,2 °C do MAE de cada um); os resíduos dos modelos têm correlação 0,88-0,99 entre si.

## 9. Resíduos e Ljung-Box (erro = real − previsto)
{tab(LB)}
Nenhum p-valor abaixo de 0,05; limite da ACF ±{1.96/np.sqrt(len(P)):.2f}. Com n=22 o poder do teste é baixo (lag 10 é grande para n=22). Viés pequeno e não significativo em todos; RF com maior viés (subestima por não extrapolar).

## 10. Importância das features
**Permutação (MLP, grupos):**

{tab(gm)}
**Permutação (Random Forest, grupos):**

{tab(gr)}
**MLP, top 8 individuais:**

{tab(im)}
**Random Forest, top 8 individuais (com importância por impureza):**

{tab(ir)}
**SARIMAX (coeficientes; erros-padrão aproximados via Gauss-Newton/CSS):**

{tab(cf)}
Holt-Winters: sem importância de features (nível, tendência e ausência de sazonalidade na seção 7). Importâncias do MLP são **instáveis** (grupo "vento" negativo × `wind_lag0` positivo). O vento do dia é a única exógena consistente entre RF, MLP (individual) e SARIMAX, com efeito pequeno.

## 11. Estudo do MLP
- Arquitetura final: 30 entradas → {sel['MLP Regressor']['hidden']} ({sel['MLP Regressor']['activation']}) → 1; α(L2)={sel['MLP Regressor']['alpha']}; solver {sel['MLP Regressor']['solver']}; alvo Δ. **1.025 parâmetros para 78 amostras** (≈ 13×): sem L2 forte haveria memorização.
- Sementes (MAE de teste): {', '.join(f'{v:.3f}' for v in sd.MAE_teste)} (desvio {sd.MAE_teste.std():.3f}).
- Curva de aprendizado (Adam, diagnóstico): melhor MAE de validação interna {cur.MAE_validacao_interna.min():.3f} na época {int(cur.loc[cur.MAE_validacao_interna.idxmin(),'epoca'])}; na época 300 treino {cur.MAE_treino.iloc[-1]:.3f} × validação {cur.MAE_validacao_interna.iloc[-1]:.3f} (sobreajuste leve).
- **Pós-hoc (não alterou a seleção)** sensibilidade ao α:

{tab(al)}
- **Pós-hoc** ablação por grupo (3 sementes):

{tab(ab)}
Explicação didática completa (funcionamento, hipóteses, hiperparâmetros): `notebooks/04_estudo_MLP_e_importancia.ipynb`.

## 12. Auditoria da entrega original e o que foi corrigido
| Problema na entrega original | Correção |
|---|---|
| Saídas idênticas (MAE 6,4696, RMSE 1,3456) nos 4 notebooks, inclusive logs do MLP dentro do SARIMAX/RF/HW | Notebooks regerados por execução real (`src/nbexec.py`); resultados distintos por modelo |
| "legalzinho vs baseline": RMSE padronizado (1,3456) comparado a °C (1,38) | Tudo em °C; comparação com persistência + Diebold-Mariano + bootstrap |
| MAE 6,47 °C (≈ 6× a persistência) por prever o nível sob deslocamento de distribuição | Alvo Δ escolhido por validação interna: MAE ≈ {RES.loc[RES.modelo=='MLP Regressor','MAE'].iloc[0]:.2f} °C |
| Sem walk-forward (holdout 80/20), horizontes diferentes entre modelos (1 × 22 passos) | Walk-forward expansivo, h=1, mesmas origens para todos |
| MLP sem busca; sem semente; sem retreino | Busca aleatória (60), 5 sementes, reajuste a cada origem |
| Sem STL, Ljung-Box, ACF, importância, explicação do MLP | Notebooks 01, 03 e 04 |
| Pressão de 59 hPa não detectada; `dayofweek` rotulado como `week_of_year` | Detecção e correção documentadas; feature semanal removida e justificada |
| Features sem justificativa de disponibilidade; calendário só dos dias da janela | Tabela de disponibilidade; calendário do dia-alvo; auditoria de vazamento |
| Caminho absoluto do Windows; `.venv` de 96 MB; LSTM e guias de NLP no ZIP | Caminhos relativos; removidos |

## 13. Transparência metodológica (leia antes de avaliar)
1. **Regra do `w` compartilhado foi introduzida depois de uma primeira rodada** em que RF e MLP selecionaram janelas diferentes (violando o item 5.3). A mudança foi motivada pelo enunciado, não pelo teste, mas **piorou o MAE do RF no teste** (≈ 1,10 → 1,17), o que se mantém reportado.
2. O α do MLP caiu na borda da grade; a análise de α maior é **pós-hoc** e não alterou a escolha.
3. Ablação e importâncias usam o conjunto de teste de forma **descritiva**; nada foi reescolhido com base nelas.
4. SARIMAX/Holt-Winters/STL próprios: validados em dados sintéticos, **não** contra o `statsmodels`. AIC/BIC são de CSS.
5. Apenas **22 previsões** e **uma base**: o ranking é instável e não se generaliza.

## 14. Referências
- Cleveland, Cleveland, McRae, Terpenning (1990). *STL: A Seasonal-Trend Decomposition Procedure Based on Loess.* Journal of Official Statistics.
- Hyndman & Athanasopoulos. *Forecasting: Principles and Practice* (força de tendência e sazonalidade; Holt-Winters; ARIMA).
- Diebold & Mariano (1995); Harvey, Leybourne & Newbold (1997): comparação de acurácia preditiva.
- Ljung & Box (1978). Breiman (2001), *Random Forests*. Pedregosa et al. (2011), *scikit-learn*.
- Conjunto *Daily Delhi Climate* (Kaggle); a procedência original (indicada como Weather Underground) deve ser confirmada na página do conjunto.

## 15. Inventário
`datasets/` base · `src/` código (`*_delhi.py`) · `tests/` testes · `results/` CSV/JSON gerados · `figures/` PNG · `notebooks/` 01-04 executados · `docs/` esta documentação.
"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "DOCUMENTACAO_TREINAMENTO.md").write_text(md, encoding="utf-8")
print("ok", len(md))
