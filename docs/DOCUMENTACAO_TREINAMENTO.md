# Documentação completa do treinamento: Grupo 4 (MLP Regressor) · base do Grupo 1 (Daily Delhi Climate)

> Documento gerado por `src/build_docs_delhi.py` a partir dos arquivos em `results/`. Todos os números vêm da execução real do pipeline.
> Escopo desta entrega: **somente a base do Grupo 1**. HTML/PDF do relatório, apresentação oral e registro de demandas **não** fazem parte deste pacote.

## 1. Identificação e escopo
| Item | Valor |
|---|---|
| Grupo / modelo de especialização | 4 / MLP Regressor (`sklearn.neural_network.MLPRegressor`) |
| Modelos comuns | SARIMAX, Holt-Winters (referência univariada), Random Forest |
| Base | Daily Delhi Climate, arquivo congelado `datasets/DailyDelhiClimateTest.csv` (não modificado) |
| Período / frequência | 2017-01-01 a 2017-04-24, diária, 114 observações |
| Alvo | `meantemp` (°C); horizonte h = 1 dia |
| Exógenas | `humidity`, `wind_speed`, `meanpressure` (sempre defasadas) + calendário do dia-alvo |
| Combinações executadas | 4 modelos × 1 base (as outras 4 bases não fazem parte deste arquivo) |

## 2. Ambiente e reprodutibilidade
- Python 3.12; `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib` (ver `requirements.txt`). **PyTorch e statsmodels não foram usados** (indisponíveis no ambiente de execução).
- STL, Holt-Winters, SARIMAX, ACF, Ljung-Box e Diebold-Mariano foram **implementados no projeto** (`src/tsutils_delhi.py`, `src/ts_models_delhi.py`) e validados com séries sintéticas (`tests/`). A comparação numérica direta com `statsmodels` não é usada como critério: as implementações deste projeto usam CSS, inicialização explícita dos estados e convenção própria para a tendência inicial, enquanto `statsmodels` ajusta por verossimilhança e inicializa estados de outra forma. Assim, a comparação válida aqui é interna, com a mesma implementação e protocolo para todos os modelos; diferenças de nível devem ser interpretadas como diferenças de convenção, não como erro de reprodução.
- Semente global 42; MLP com sementes 42..46 no teste (5) e 42..44 na busca (3). Buscas salvas em `results/busca_*.csv` (apague-os para refazê-las).
- Reprodução: `pip install -r requirements.txt` → `python tests/test_core_delhi.py && python tests/test_leakage_delhi.py` → `python src/pipeline_delhi.py` → `python src/build_notebooks_delhi.py` → `python src/build_docs_delhi.py`.
- Protocolo: {"n_rows": 100, "n_test": 22, "n_inner": 14, "train_targets": "2017-01-15 a 2017-04-02", "inner_targets": "2017-03-20 a 2017-04-02", "test_targets": "2017-04-03 a 2017-04-24", "seed": 42, "mlp_seeds_inner": 3, "mlp_seeds_final": 5, "m_seasonal": 7}

## 3. Dados e qualidade
| checagem | resultado |
|---|---|
| observações | 114 |
| período | 2017-01-01 a 2017-04-24 |
| datas duplicadas | 0 |
| datas faltantes na grade diária | 0 |
| datas fora de ordem | 0 |
| ausentes em meantemp | 0 |
| ausentes em humidity | 0 |
| ausentes em wind_speed | 0 |
| ausentes em meanpressure | 0 |

Atípicos por escore z robusto (MAD > 3,5):

| variavel | base | data | valor | z_robusto |
|---|---|---|---|---|
| meantemp | diferença | 2017-01-26 | 16.179 | -3.500 |
| meanpressure | nível | 2017-01-01 | 59.000 | -149.978 |
| meanpressure | diferença | 2017-01-02 | 1018.278 | 531.142 |
| meanpressure | diferença | 2017-02-20 | 1005.375 | -4.036 |

Decisões de limpeza:
1. Pressão = 59,0 hPa em 01/01/2017 é impossível (demais ≈ 1000 hPa): substituída (NaN → valor do dia seguinte). Distorcia o desvio-padrão da pressão (≈ 89 hPa) e não havia sido detectada na entrega original.
2. Variação de temperatura em 26/01 e queda de pressão em 20/02: **mantidas** (variação meteorológica plausível).
3. Sem duplicidades, sem datas faltantes, sem ausentes. Sem transformação do alvo.

## 4. STL e sazonalidade (período 7; ns=13, nt=21)
- Força da sazonalidade (Hyndman & Athanasopoulos) = **0.124** (fraca); força da tendência = **0.941**.
- Variância de y: tendência 90.6%, sazonal 0.8%, resíduo 5.9%.
- Resíduo STL: ACF(1) = 0.52; Ljung-Box p(5) < 0,001 (não é ruído branco: episódios meteorológicos de poucos dias).
- Com 114 dias (< 1 ciclo anual) a **sazonalidade anual não é estimável**. Fases da tendência:

| inicio | fim | dias | fase | trend_inicio | trend_fim | variacao | inclinacao_media |
|---|---|---|---|---|---|---|---|
| 2017-01-01 | 2017-01-14 | 14 | queda | 18.32 | 13.69 | -4.62 | -0.36 |
| 2017-01-15 | 2017-01-25 | 11 | crescimento | 13.79 | 16.70 | 2.91 | 0.29 |
| 2017-01-26 | 2017-02-08 | 14 | estabilidade | 16.79 | 16.35 | -0.44 | -0.03 |
| 2017-02-09 | 2017-02-22 | 14 | crescimento | 16.43 | 20.57 | 4.14 | 0.32 |
| 2017-02-23 | 2017-03-11 | 17 | estabilidade | 20.71 | 20.03 | -0.68 | -0.04 |
| 2017-03-12 | 2017-04-24 | 44 | crescimento | 20.22 | 34.62 | 14.40 | 0.33 |

## 5. Feature Engineering
- Origem t → alvo y(t+1). Features (w=14, **30 colunas**): lags de y (0..w−1), médias de 3 e 7 dias, desvio de 7 dias, diferença de 1 dia; umidade, vento e pressão em t, t−1, t−2; variação da pressão; sen/cos do dia do ano **do dia-alvo**.
- **Nenhuma exógena de t+1 é usada** (não conhecida na origem). Excluídos: dia da semana (sem mecanismo físico; sazonalidade semanal fraca), feriados (irrelevantes).
- Ausentes dos lags/janelas: 13 primeiras origens descartadas (sem imputação); 100 origens utilizáveis para todo `w`.
- Auditoria de vazamento executável: `tests/test_leakage.py`. Padronização do MLP ajustada só no treino de cada origem.
- RF e MLP usam **exatamente as mesmas features** (w compartilhado, ver seção 7).

## 6. Protocolo walk-forward
Teste = últimos 22 alvos (**2017-04-03 a 2017-04-24**), janela expansiva com **reajuste a cada origem**, h=1, mesmas origens para todos os modelos e referências. Seleção de hiperparâmetros na validação interna (14 origens: 2017-03-20 a 2017-04-02); fixos no teste.

## 7. Modelos, espaços de busca e seleção
**Janela compartilhada (RF e MLP):** w* = menor média dos melhores MAE internos (decisão só com validação interna).

| w | melhor_inner_MAE_RF | melhor_inner_MAE_MLP | media |
|---|---|---|---|
| 3 | 1.2821 | 1.3313 | 1.3067 |
| 7 | 1.3101 | 1.2338 | 1.2719 |
| 14 | 1.3105 | 1.2288 | 1.2696 |

**Selecionados:**

| modelo | hiperparâmetros |
|---|---|
| Holt-Winters | {"trend": "add", "seasonal": null, "damped": false, "m": 7} |
| SARIMAX | {"order": [2, 1, 1], "seasonal_order": [0, 0, 0], "m": 7, "exog": true} |
| Random Forest | {"w": 14, "n_estimators": 100, "max_depth": 6, "min_samples_split": 5, "min_samples_leaf": 1, "max_features": "sqrt", "target": "delta"} |
| MLP Regressor | {"w": 14, "hidden": [32], "activation": "relu", "alpha": 10.0, "solver": "lbfgs", "lr": 0.01, "target": "delta"} |

Espaços: Holt-Winters (tendência {nenhuma, aditiva, amortecida} × sazonalidade {nenhuma, aditiva, multiplicativa}, m=7); SARIMAX ((p,d,q) ∈ {0,1,2}×{0,1}×{0,1}, (P,D,Q) ∈ {(0,0,0),(1,0,0),(0,0,1),(1,0,1),(0,1,1)}, m=7, com/sem exógenas; AIC/BIC por CSS); Random Forest (60 de 972 configurações); MLP (60 de 1.152 configurações; alvo nível ou Δ).
Resultados completos das buscas: `results/busca_*.csv`. Cinco melhores do MLP:

| w | hidden | activation | alpha | solver | lr | target | inner_MAE |
|---|---|---|---|---|---|---|---|
| 14 | (32,) | relu | 10.0000 | lbfgs | 0.0100 | delta | 1.2288 |
| 7 | (32,) | relu | 10.0000 | lbfgs | 0.0100 | delta | 1.2338 |
| 7 | (16,) | tanh | 0.0100 | adam | 0.0010 | delta | 1.2448 |
| 14 | (16,) | tanh | 10.0000 | adam | 0.0010 | delta | 1.2509 |
| 14 | (32,) | relu | 10.0000 | adam | 0.0010 | delta | 1.2692 |

Holt-Winters (todas as combinações):

| trend | seasonal | damped | inner_MAE | alpha | beta | gamma | phi | SSE_treino |
|---|---|---|---|---|---|---|---|---|
| add | nan | False | 1.2581 | 0.9005 | 0.0258 |  |  | 288.4968 |
| add | nan | True | 1.3851 | 0.8811 | 0.0010 |  | 0.8000 | 276.5907 |
| nan | nan | False | 1.3860 | 0.8811 |  |  |  | 276.8807 |
| nan | add | False | 1.6190 | 0.9207 |  | 0.1501 |  | 376.7564 |
| add | add | False | 1.6205 | 0.9399 | 0.0306 | 0.1477 |  | 393.7631 |
| add | add | True | 1.6239 | 0.9076 | 0.0010 | 0.1516 | 0.8000 | 372.2092 |
| add | mul | False | 1.6410 | 0.9170 | 0.0267 | 0.1290 |  | 381.7370 |
| nan | mul | False | 1.6641 | 0.8965 |  | 0.1383 |  | 367.1715 |
| add | mul | True | 1.6663 | 0.8808 | 0.0010 | 0.1386 | 0.8010 | 362.1069 |

## 8. Resultados (MAE fora da amostra, 22 previsões)
| modelo | MAE | RMSE | vies | DM_vs_persistencia | p_DM |
|---|---|---|---|---|---|
| Holt-Winters | 1.052 | 1.396 | -0.153 | 0.105 | 0.917 |
| SARIMAX | 1.101 | 1.489 | -0.282 | -0.275 | 0.786 |
| Random Forest | 1.172 | 1.489 | 0.356 | -1.255 | 0.223 |
| MLP Regressor | 1.104 | 1.450 | 0.155 | -0.420 | 0.679 |
| Persistência (ref.) | 1.059 | 1.380 | 0.102 |  |  |
| Média móvel 7d (ref.) | 1.794 | 2.120 | 0.508 |  |  |
| Sazonal ingênuo m=7 (ref.) | 2.864 | 3.301 | 1.027 |  |  |

Tempo de execução do walk-forward (s):

| modelo | segundos |
|---|---|
| Holt-Winters | 0.810 |
| SARIMAX | 2.300 |
| Random Forest | 7.230 |
| MLP Regressor | 85.650 |

- Menor MAE numérico: **Holt-Winters**. **Nenhuma diferença contra a persistência é significativa** (Diebold-Mariano, p ≥ 0.22).
- Ranking (1 base): 1º Holt-Winters, 2º SARIMAX, 3º MLP Regressor, 4º Random Forest. Vitórias: Holt-Winters = 1; demais = 0. Com uma única base, a posição média é a própria posição.
- O dia **07/04/2017** (queda de 31,2 → 27,0 °C) gera erro ≈ −4 °C em todos os modelos (≈ 0,2 °C do MAE de cada um); os resíduos dos modelos têm correlação 0,88-0,99 entre si.

## 9. Resíduos e Ljung-Box (erro = real − previsto)
| modelo | Q5 | p5 | Q10 | p10 | ACF1 |
|---|---|---|---|---|---|
| Holt-Winters | 1.827 | 0.873 | 5.482 | 0.857 | -0.007 |
| SARIMAX | 5.333 | 0.377 | 6.785 | 0.746 | 0.234 |
| Random Forest | 2.688 | 0.748 | 4.900 | 0.898 | -0.072 |
| MLP Regressor | 1.771 | 0.880 | 5.368 | 0.865 | 0.065 |

Nenhum p-valor abaixo de 0,05; limite da ACF ±0.42. Com n=22 o poder do teste é baixo (lag 10 é grande para n=22). Viés pequeno e não significativo em todos; RF com maior viés (subestima por não extrapolar).

## 10. Importância das features
**Permutação (MLP, grupos):**

| feature | delta_MAE | desvio | MAE_base |
|---|---|---|---|
| calendário (sen/cos anual) | 0.025 | 0.014 | 1.318 |
| pressão | 0.015 | 0.062 | 1.318 |
| umidade | -0.018 | 0.023 | 1.318 |
| temperatura (lags/médias/dif) | -0.024 | 0.039 | 1.318 |
| vento | -0.144 | 0.079 | 1.318 |

**Permutação (Random Forest, grupos):**

| feature | delta_MAE | desvio | MAE_base |
|---|---|---|---|
| pressão | 0.031 | 0.032 | 1.046 |
| vento | 0.025 | 0.026 | 1.046 |
| umidade | 0.008 | 0.011 | 1.046 |
| calendário (sen/cos anual) | 0.004 | 0.004 | 1.046 |
| temperatura (lags/médias/dif) | -0.002 | 0.014 | 1.046 |

**MLP, top 8 individuais:**

| feature | delta_MAE | desvio | MAE_base |
|---|---|---|---|
| cos_doy | 0.033 | 0.018 | 1.318 |
| wind_lag0 | 0.031 | 0.024 | 1.318 |
| y_lag11 | 0.014 | 0.012 | 1.318 |
| y_lag4 | 0.011 | 0.008 | 1.318 |
| pres_diff1 | 0.008 | 0.010 | 1.318 |
| y_lag6 | 0.006 | 0.004 | 1.318 |
| y_lag10 | 0.003 | 0.007 | 1.318 |
| y_lag12 | 0.002 | 0.004 | 1.318 |

**Random Forest, top 8 individuais (com importância por impureza):**

| feature | delta_MAE | desvio | MAE_base | impureza_MDI |
|---|---|---|---|---|
| pres_lag1 | 0.027 | 0.019 | 1.046 | 0.057 |
| wind_lag0 | 0.025 | 0.016 | 1.046 | 0.028 |
| hum_lag1 | 0.009 | 0.006 | 1.046 | 0.029 |
| wind_lag1 | 0.007 | 0.026 | 1.046 | 0.060 |
| wind_lag2 | 0.005 | 0.006 | 1.046 | 0.027 |
| y_lag0 | 0.004 | 0.003 | 1.046 | 0.035 |
| y_mean3 | 0.003 | 0.004 | 1.046 | 0.026 |
| sin_doy | 0.003 | 0.003 | 1.046 | 0.030 |

**SARIMAX (coeficientes; erros-padrão aproximados via Gauss-Newton/CSS):**

| termo | coef | erro_padrao | z | p_valor |
|---|---|---|---|---|
| humidity_lag1 | 0.043 | 0.031 | 1.370 | 0.171 |
| wind_speed_lag1 | -0.109 | 0.054 | -2.009 | 0.045 |
| meanpressure_lag1 | 0.038 | 0.108 | 0.352 | 0.725 |
| sin_doy | -13.743 | 17.220 | -0.798 | 0.425 |
| cos_doy | -26.900 | 16.144 | -1.666 | 0.096 |
| ar.L1 | 0.334 | 0.573 | 0.583 | 0.560 |
| ar.L2 | -0.071 | 0.155 | -0.457 | 0.648 |
| ma.L1 | -0.416 | 0.578 | -0.719 | 0.472 |

Holt-Winters: sem importância de features (nível, tendência e ausência de sazonalidade na seção 7). Importâncias do MLP são **instáveis** (grupo "vento" negativo × `wind_lag0` positivo). O vento do dia é a única exógena consistente entre RF, MLP (individual) e SARIMAX, com efeito pequeno.

## 11. Estudo do MLP
- Arquitetura final: 30 entradas → [32] (relu) → 1; α(L2)=10.0; solver lbfgs; alvo Δ. **1.025 parâmetros para 78 amostras** (≈ 13×): sem L2 forte haveria memorização.
- Sementes (MAE de teste): 1.115, 1.107, 1.109, 1.107, 1.091 (desvio 0.009).
- Curva de aprendizado (Adam, diagnóstico): melhor MAE de validação interna 1.414 na época 6; na época 300 treino 1.162 × validação 1.539 (sobreajuste leve).
- **Pós-hoc (não alterou a seleção)** sensibilidade ao α:

| alpha | MAE_validacao_interna | MAE_teste_posthoc |
|---|---|---|
| 1.000 | 1.455 | 1.296 |
| 3.000 | 1.419 | 1.218 |
| 10.000 | 1.229 | 1.108 |
| 30.000 | 1.376 | 1.074 |
| 100.000 | 1.291 | 1.048 |

- **Pós-hoc** ablação por grupo (3 sementes):

| configuracao | MAE_teste |
|---|---|
| completo | 1.108 |
| sem umidade | 1.110 |
| sem vento | 1.069 |
| sem pressão | 1.126 |
| sem calendário (sen/cos anual) | 1.081 |

Explicação didática completa (funcionamento, hipóteses, hiperparâmetros): `notebooks/04_estudo_MLP_e_importancia.ipynb`.

## 12. Auditoria da entrega original e o que foi corrigido
| Problema na entrega original | Correção |
|---|---|
| Saídas idênticas (MAE 6,4696, RMSE 1,3456) nos 4 notebooks, inclusive logs do MLP dentro do SARIMAX/RF/HW | Notebooks regerados por execução real (`src/nbexec.py`); resultados distintos por modelo |
| "legalzinho vs baseline": RMSE padronizado (1,3456) comparado a °C (1,38) | Tudo em °C; comparação com persistência + Diebold-Mariano + bootstrap |
| MAE 6,47 °C (≈ 6× a persistência) por prever o nível sob deslocamento de distribuição | Alvo Δ escolhido por validação interna: MAE ≈ 1.10 °C |
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
