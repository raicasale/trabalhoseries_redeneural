from trabalhoseries_redeneural.src.nbexec_delhi import md, code

CELLS = [
md("""
# 02 · Feature Engineering, protocolo walk-forward e otimização de hiperparâmetros
**Grupo 4 · MLP Regressor · base do Grupo 1 (Daily Delhi Climate)**

Este notebook documenta (i) as features e por que nenhuma usa informação indisponível na data da previsão, (ii) o protocolo de validação e
(iii) a busca de hiperparâmetros dos 4 modelos. A busca pesada foi executada por `python src/pipeline.py`; aqui os resultados são lidos de `results/`.
"""),
code("""
import sys, json, warnings
sys.path.insert(0, '../src')
import numpy as np, pandas as pd
import trabalhoseries_redeneural.src.tsutils_delhi as tu
import trabalhoseries_redeneural.src.plots_delhi as pl
warnings.filterwarnings('ignore')
D = tu.clean(tu.load_raw())
R = '../results/'
meta = json.load(open(R + 'protocolo.json')); sel = json.load(open(R + 'hiperparametros_selecionados.json'))
display(pd.Series(meta).rename('valor').to_frame())
"""),
md("""
## 1. Features e disponibilidade no momento da previsão
Regra de ouro: na **origem t** só se conhece o que foi observado até o dia t. O alvo é y(t+1). Portanto, **umidade, vento e pressão de t+1 nunca são usados**
(seriam "temperatura observada no futuro", o exemplo de vazamento do enunciado); as exógenas entram **defasadas** (t, t−1, t−2).
Somente o **calendário do dia-alvo** é conhecido com antecedência.
"""),
code("""
disp = pd.DataFrame([
 ['y_lag0 ... y_lag{w-1}',      'temperatura em t, t-1, ...',                      'alvo defasado',        'Disponível (observado até t)'],
 ['y_mean3, y_mean7, y_std7',   'média (3 e 7 dias) e desvio (7 dias) até t',      'janela móvel',         'Disponível (janela termina em t)'],
 ['y_diff1',                    'y(t) - y(t-1)',                                   'diferença',            'Disponível'],
 ['hum_lag0..2',                'umidade em t, t-1, t-2',                          'exógena defasada',     'Disponível; a de t+1 NÃO é usada (vazamento)'],
 ['wind_lag0..2',               'vento em t, t-1, t-2',                            'exógena defasada',     'Disponível; a de t+1 NÃO é usada (vazamento)'],
 ['pres_lag0..2, pres_diff1',   'pressão em t, t-1, t-2 e sua variação',           'exógena defasada',     'Disponível; a de t+1 NÃO é usada (vazamento)'],
 ['sin_doy, cos_doy',           'dia do ano do DIA-ALVO (codificação cíclica)',    'calendário',           'Conhecido antecipadamente'],
], columns=['feature','definição','tipo','disponibilidade'])
display(disp)
excl = pd.DataFrame([
 ['dia da semana (sen/cos 7)', 'Excluída: não há mecanismo físico de ciclo semanal na temperatura, e a STL mostra força sazonal semanal ≈ 0,12 (notebook 01). Seria ruído.'],
 ['feriados', 'Não aplicável: feriados não afetam a temperatura média diária.'],
 ['umidade/vento/pressão em t+1', 'Excluídas: não conhecidas na origem (vazamento). Uma previsão meteorológica disponível em t poderia ser usada, mas a base não a contém.'],
], columns=['item','decisão'])
display(excl)
"""),
code("""
X, ytgt, ycur, tdate = tu.build_features(D, w=14)
print('Features (w=14):', X.shape[1], '| origens utilizáveis:', len(X))
print(list(X.columns))
# Valores ausentes produzidos por lags/janelas: antes do descarte
Xfull = tu.build_features(D, w=14, max_w=1)[0]
na_rows = int(Xfull.isna().any(axis=1).sum())
print(f'Linhas com NaN gerados por lags/janelas: {na_rows} (primeiros dias, sem histórico) -> descartadas, sem imputação.')
print(f'Após o descarte: {len(X)} origens, NaN restantes = {int(X.isna().sum().sum())}')
"""),
md("""
**Tratamento dos ausentes dos lags/janelas.** O maior lag (14 dias) e a janela de 7 dias exigem histórico; as primeiras origens são **descartadas** (não imputadas).
Todos os valores de `w` testados (3, 7, 14) usam **exatamente as mesmas 100 origens**, para que a comparação entre `w` seja justa.

### Auditoria de vazamento (teste executável)
Para três origens, o futuro (t+1 em diante) de alvo e exógenas é substituído por lixo (valores embaralhados + 1000).
Se alguma feature da origem t dependesse do futuro, ela mudaria.
"""),
code("""
rng = np.random.default_rng(0)
X0 = tu.build_features(D, w=14)[0]
for pos in (30, 60, 90):
    D2 = D.copy(); fut = D2.index > pos
    for c in [tu.TARGET] + tu.EXOG:
        D2.loc[fut, c] = rng.permutation(D2.loc[fut, c].values) + 1000.0
    X1 = tu.build_features(D2, w=14)[0]
    ok_row = np.allclose(X0.loc[pos].values, X1.loc[pos].values)
    ok_past = np.allclose(X0.loc[X0.index <= pos].values, X1.loc[X1.index <= pos].values)
    print(f'origem em {D.loc[pos,"date"].date()}: features inalteradas = {ok_row} | todas as origens anteriores inalteradas = {ok_past}')
"""),
md("""
O mesmo teste roda em `tests/test_leakage.py`. Outro ponto de vazamento comum, a **padronização**, é tratado assim: o `StandardScaler` do MLP é ajustado **somente
no treino de cada origem** (dentro de `MLPModel.fit`), nunca sobre toda a série.

## 2. Protocolo de validação walk-forward
- **Horizonte h = 1 dia.** A origem t prevê o dia t+1.
- **Teste:** últimos 22 alvos (03/04/2017 a 24/04/2017). **Walk-forward expansivo**: em cada origem o modelo é **reajustado** com todo o histórico disponível, prevê 1 passo, o tempo avança e o valor real é incorporado.
- **Mesmas origens, horizonte e conjunto de teste** para SARIMAX, Holt-Winters, Random Forest, MLP e as referências.
- **Hiperparâmetros** escolhidos *antes* do teste, por walk-forward interno nas **14 origens imediatamente anteriores ao teste** (região de treino). Durante o teste, ficam **fixos**.
"""),
code("""
n_train = meta['n_rows'] - meta['n_test']
pl.protocol(meta, None, n_train, meta['n_inner'], meta['n_test'])
display(pd.DataFrame({'bloco': ['alvos de treino (inclui validação interna)', 'validação interna (seleção)', 'teste walk-forward'],
                      'alvos': [meta['train_targets'], meta['inner_targets'], meta['test_targets']]}))
"""),
md("""
**Referências obrigatórias.** Por haver ACF(1) ≈ 0,95, a comparação só é informativa com a **persistência** (ŷ(t+1)=y(t)).
Incluem-se também o sazonal ingênuo (m=7) e a média móvel de 7 dias. **Não** são contados como "modelos" no ranking de 4.

## 3. Otimização de hiperparâmetros
**Método.** Busca em grade (Holt-Winters, SARIMAX; espaços pequenos) e **busca aleatória reprodutível** (RF e MLP, 60 configurações cada, semente 42).
**Critério:** MAE médio do walk-forward interno (14 origens, com reajuste a cada origem). O teste **não participa** de nenhuma decisão.
Para o MLP, cada configuração é treinada com 3 sementes e as previsões são promediadas (reduz a variância de inicialização).

| Modelo | Espaço de busca | Procedimento |
|---|---|---|
| Holt-Winters | tendência {nenhuma, aditiva, aditiva amortecida} × sazonalidade {nenhuma, aditiva, multiplicativa}; m=7; α, β, γ, φ estimados por mínimos quadrados a cada reajuste | grade (9 combinações) |
| SARIMAX | (p,d,q) ∈ {0,1,2}×{0,1}×{0,1}; (P,D,Q) ∈ {(0,0,0),(1,0,0),(0,0,1),(1,0,1),(0,1,1)}; m=7; com/sem exógenas | grade (120 combinações); AIC/BIC(CSS) registrados como apoio |
| Random Forest | w ∈ {3,7,14}; n_estimators ∈ {100,200}; max_depth ∈ {None,3,6}; min_samples_split ∈ {2,5,10}; min_samples_leaf ∈ {1,3,5}; max_features ∈ {sqrt, 0,5, 1,0}; alvo ∈ {nível, Δ} | aleatória, 60 de 972 |
| MLP Regressor | w ∈ {3,7,14}; camadas ocultas ∈ {(16),(32),(64),(32,16)}; ativação ∈ {relu, tanh}; α(L2) ∈ {1e-3,1e-2,1e-1,1,3,10}; solver ∈ {lbfgs, adam}; taxa (adam) ∈ {1e-3,1e-2}; alvo ∈ {nível, Δ} | aleatória, 60 de 1.152 |

`alvo = Δ` significa que o modelo prevê y(t+1) − y(t) e a previsão final soma y(t) de volta (informação disponível na origem).
"""),
code("""
hw = pd.read_csv(R + 'busca_holt_winters.csv'); sx = pd.read_csv(R + 'busca_sarimax.csv')
rf = pd.read_csv(R + 'busca_random_forest.csv'); ml = pd.read_csv(R + 'busca_mlp.csv')
print('Holt-Winters (todas as 9 combinações, ordenadas por MAE interno):'); display(hw.round(4))
print('SARIMAX (10 melhores de', len(sx), 'combinações):'); display(sx.head(10).round(4))
"""),
code("""
print('Random Forest (10 melhores de', len(rf), 'configurações):'); display(rf.drop(columns='seg').head(10).round(4))
print('MLP Regressor (10 melhores de', len(ml), 'configurações):'); display(ml.drop(columns='seg').head(10).round(4))
"""),
md("""
### 3.1 Janela `w` compartilhada (RF e MLP usam as mesmas features)
O enunciado (5.3) exige o mesmo conjunto de features no Random Forest e no modelo de especialização. Em uma primeira rodada deste trabalho a busca deixava `w` livre por modelo e a
seleção resultou em `w` diferentes (RF=3, MLP=14), o que **violaria** a exigência. A regra adotada passou a ser: escolher **um único `w`** que minimize a média dos melhores MAE internos
de RF e MLP; depois, cada modelo escolhe seus demais hiperparâmetros **dentro de `w*`**. A decisão usa somente a validação interna (o teste não entrou).
"""),
code("""
jw = pd.read_csv(R + 'selecao_w_conjunto.csv'); display(jw.round(4))
print('w* =', int(jw.sort_values('media').iloc[0].w), '(w=7 fica praticamente empatado: diferença de', round(float(jw.set_index('w').media[7]-jw.set_index('w').media[14]), 4), '°C de MAE interno)')
"""),
md("""
### 3.2 Efeito dos hiperparâmetros (MAE interno médio por valor; todas as configurações avaliadas)
"""),
code("""
def marg(df, cols):
    out = []
    for c in cols:
        g = df.groupby(c)['inner_MAE'].agg(['count', 'mean', 'min']).reset_index().rename(columns={c: 'valor'})
        g.insert(0, 'hiperparâmetro', c); out.append(g)
    return pd.concat(out, ignore_index=True)
print('MLP Regressor'); display(marg(ml, ['target', 'w', 'hidden', 'activation', 'alpha', 'solver']).round(3))
print('Random Forest'); display(marg(rf, ['target', 'w', 'max_depth', 'min_samples_leaf', 'max_features']).round(3))
"""),
md("""
**Cuidado com a leitura das médias acima.** A busca aleatória não é balanceada: configurações com `alvo = nível` (ruins) se misturam às demais e distorcem as médias
por valor (por exemplo, α=10 aparece com a *pior* média global, porque várias de suas configurações usam alvo nível). Por isso a comparação justa de cada hiperparâmetro é feita **dentro de `alvo = Δ`**:
"""),
code("""
dl = ml[ml.target == 'delta']
print('MLP, apenas configurações com alvo = Δ'); display(marg(dl, ['alpha', 'solver', 'activation', 'hidden', 'w']).round(3))
sx['d'] = sx['order'].str.extract(r'\\(\\d, (\\d),')[0]
print('SARIMAX - resumo por escolha'); display(pd.concat([
    sx.groupby('exog')['inner_MAE'].agg(['count', 'min', 'median']).reset_index().rename(columns={'exog': 'valor'}).assign(grupo='usa exógenas'),
    sx.groupby('d')['inner_MAE'].agg(['count', 'min', 'median']).reset_index().rename(columns={'d': 'valor'}).assign(grupo='d (diferenciação)'),
    sx.groupby('seasonal_order')['inner_MAE'].agg(['count', 'min', 'median']).reset_index().rename(columns={'seasonal_order': 'valor'}).assign(grupo='(P,D,Q) m=7'),
], ignore_index=True)[['grupo', 'valor', 'count', 'min', 'median']].round(3))
"""),
md("""
**Leitura.**
- **`alvo = Δ` é o hiperparâmetro de maior efeito** em ambos os modelos (MLP: média interna ≈ 1,47 contra 2,13 com alvo nível; RF: ≈ 1,34 contra 2,79). Prever a *variação diária* e somar y(t) de volta
  corrige a falha central da entrega original (prever o nível com uma rede que não extrapola, sob forte tendência).
- No MLP, **α (L2) alto ajuda** dentro de `alvo = Δ` (α=10: média ≈ 1,28; α ≤ 1: ≈ 1,50-1,65): com ~80 amostras e 30 features, penalizar pesos grandes reduz a memorização e aproxima o modelo da persistência. Com poucas configurações por célula (1 a 7), isto é uma tendência, não uma prova.
- **Ativação e solver não diferem** (relu 1,470 × tanh 1,467; adam 1,481 × lbfgs 1,451); arquitetura e `w` mostram diferenças pequenas e ruidosas. Com 14 origens de validação, diferenças de poucos centésimos de °C **não são significativas**.
- **SARIMAX:** usar exógenas reduz o melhor MAE interno de ≈ 1,37 para ≈ 1,09 e `d=1` é o melhor; a sazonalidade m=7 não traz ganho (as ordens com (P,D,Q) ≠ (0,1,1) empatam; (0,1,1) piora), coerente com a STL. As exógenas do SARIMAX incluem o calendário sen/cos, que sob `d=1` atua como termo de tendência determinística; o ganho não deve ser atribuído apenas a umidade/vento/pressão (o notebook 04 separa os efeitos).
- **Holt-Winters:** a melhor combinação tem **tendência aditiva e nenhuma sazonalidade**, também coerente com a STL (sazonalidade fraca).
"""),
code("""
rows = []
for k, v in sel.items():
    rows.append((k, json.dumps(v, ensure_ascii=False)))
display(pd.DataFrame(rows, columns=['modelo', 'hiperparâmetros selecionados (fixos no teste)']))
"""),
md("""
### 3.3 Justificativa das escolhas e limites do procedimento
- A seleção é **por desempenho fora da amostra em janelas expansivas** (walk-forward interno), nunca por ajuste no treino ou no teste.
- A validação interna tem **14 pontos**: é pouca informação; por isso a busca é moderada (60 configurações), as sementes do MLP são promediadas e as conclusões sobre hiperparâmetros individuais são cautelosas.
- **Borda do espaço (MLP):** o α escolhido (10) é o maior valor da grade. Em vez de alterar a seleção depois de conhecer o teste, foi feita uma **análise de sensibilidade pós-hoc**, claramente rotulada (notebook 04), mostrando que a validação interna **não** prefere α maiores (30 e 100 pioram o MAE interno).
- Os códigos de busca estão em `src/pipeline.py` (`search_hw`, `search_sarimax`, `search_rf`, `search_mlp`).
"""),
]
