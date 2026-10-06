"""Pipeline completo e reprodutível (Grupo 4 - MLP | base: Daily Delhi Climate).

Uso:  python src/pipeline.py          (gera tudo em results/)
Protocolo (idêntico para os 4 modelos e para as referências):
  * horizonte h=1 dia; origem t usa somente informação até t; alvo = temperatura média de t+1;
  * teste = últimos 22 alvos (03/04/2017 a 24/04/2017), walk-forward expansivo com reajuste a cada origem;
  * hiperparâmetros escolhidos ANTES do teste, por walk-forward interno nas 14 origens imediatamente
    anteriores ao teste (região de treino). O teste nunca participa de decisões de seleção.
"""
import sys, json, time, itertools, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
import trabalhoseries_redeneural.src.tsutils_delhi as tu
from trabalhoseries_redeneural.src.ts_models_delhi import HoltWinters, SARIMAXLite

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RES = tu.ROOT / "results"
RES.mkdir(exist_ok=True)
SEED = 42
N_TEST, N_INNER = 22, 14
MLP_SEEDS_INNER, MLP_SEEDS_FINAL = 3, 5
M_SEAS = 7


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ modelos tabulares
class RFModel:
    def __init__(self, **hp):
        self.hp = hp

    def fit(self, X, y):
        self.m = RandomForestRegressor(random_state=SEED, n_jobs=1, **self.hp).fit(X, y)
        return self

    def predict_all(self, X):
        return self.m.predict(X)[None, :]

    def predict(self, X):
        return self.predict_all(X).mean(0)


class MLPModel:
    """sklearn MLPRegressor com padronização de X e y (ajustada só no treino) e conjunto de sementes."""

    def __init__(self, hidden, activation, alpha, solver, lr, n_seeds):
        self.hidden, self.activation, self.alpha, self.solver, self.lr, self.n_seeds = hidden, activation, alpha, solver, lr, n_seeds

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.sx = StandardScaler().fit(X)
        self.ym, self.ys = y.mean(), y.std() or 1.0
        Xs, ys = self.sx.transform(X), (y - self.ym) / self.ys
        self.models = []
        for s in range(self.n_seeds):
            kw = dict(hidden_layer_sizes=self.hidden, activation=self.activation, alpha=self.alpha, solver=self.solver,
                      random_state=SEED + s, early_stopping=False)
            if self.solver == "adam":
                kw.update(learning_rate_init=self.lr, max_iter=300, batch_size=min(32, len(X)))
            else:
                kw.update(max_iter=500)
            self.models.append(MLPRegressor(**kw).fit(Xs, ys))
        return self

    def predict_all(self, X):
        Xs = self.sx.transform(np.asarray(X, float))
        return np.vstack([m.predict(Xs) * self.ys + self.ym for m in self.models])

    def predict(self, X):
        return self.predict_all(X).mean(0)


# ------------------------------------------------------------------ dados / alinhamento
raw = tu.load_raw()
D = tu.clean(raw)
Yfull = D[tu.TARGET].values
FEAT = {w: tu.build_features(D, w=w) for w in (3, 7, 14)}
_, Y_LEVEL, Y_CUR, T_DATE = FEAT[7]
N_ROWS = len(Y_LEVEL)
TEST_ROWS = list(range(N_ROWS - N_TEST, N_ROWS))
INNER_ROWS = list(range(N_ROWS - N_TEST - N_INNER, N_ROWS - N_TEST))
ORIGIN_IDX = FEAT[7][0].index.values  # posição do dia-origem t em D
assert len(TEST_ROWS) == N_TEST


def feature_frame(w, drop_groups=()):
    X = FEAT[w][0]
    if drop_groups:
        keep = [c for c in X.columns if not any(tu.GROUPS[g](c) for g in drop_groups)]
        X = X[keep]
    return X


def wf_tabular(make_model, w, rows, delta, drop_groups=()):
    """Walk-forward expansivo. Retorna (previsões nível [n], previsões por semente [s,n], tempo)."""
    X = feature_frame(w, drop_groups)
    yl, yc = Y_LEVEL.values, Y_CUR.values
    out, per_seed, t0 = [], [], time.time()
    for i in rows:
        tr = slice(0, i)                        # origens < i  => alvos conhecidos até a origem i
        ytr = yl[tr] - (yc[tr] if delta else 0.0)
        mdl = make_model().fit(X.iloc[tr].values, ytr)
        p = mdl.predict_all(X.iloc[[i]].values)[:, 0] + (yc[i] if delta else 0.0)
        per_seed.append(p); out.append(p.mean())
    return np.array(out), np.array(per_seed).T, time.time() - t0


# exógenas do SARIMAX alinhadas ao dia-alvo (nenhuma usa o futuro): umidade/vento/pressão do dia anterior ao alvo
# + calendário do dia-alvo
def sarimax_exog(use_exog):
    if not use_exog:
        return None
    E = pd.DataFrame({"humidity_lag1": D["humidity"].shift(1), "wind_speed_lag1": D["wind_speed"].shift(1),
                      "meanpressure_lag1": D["meanpressure"].shift(1)})
    doy = D["date"].dt.dayofyear
    E["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    E["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return E


def wf_sarimax(order, sorder, use_exog, rows):
    E = sarimax_exog(use_exog)
    preds, t0 = [], time.time()
    for i in rows:
        t = ORIGIN_IDX[i]                      # origem; alvo em t+1
        y_tr = Yfull[1:t + 1]
        X_tr = E.values[1:t + 1] if use_exog else None
        mdl = SARIMAXLite(order, sorder, M_SEAS).fit(y_tr, X_tr)
        preds.append(mdl.forecast1(E.values[t + 1] if use_exog else None))
    return np.array(preds), time.time() - t0


def wf_hw(trend, seasonal, damped, rows):
    preds, t0 = [], time.time()
    for i in rows:
        t = ORIGIN_IDX[i]
        preds.append(HoltWinters(trend, seasonal, damped, M_SEAS).fit(Yfull[:t + 1]).forecast1())
    return np.array(preds), time.time() - t0


def rows_mae(pred, rows):
    return tu.mae(Y_LEVEL.values[rows], pred)


# ------------------------------------------------------------------ buscas de hiperparâmetros
def search_hw():
    out = []
    for trend, damped in [(None, False), ("add", False), ("add", True)]:
        for seasonal in (None, "add", "mul"):
            try:
                p, _ = wf_hw(trend, seasonal, damped, INNER_ROWS)
                full = HoltWinters(trend, seasonal, damped, M_SEAS).fit(Yfull[:ORIGIN_IDX[TEST_ROWS[0]] + 1])
                out.append(dict(trend=trend, seasonal=seasonal, damped=damped, inner_MAE=rows_mae(p, INNER_ROWS),
                                alpha=full.params_["alpha"], beta=full.params_["beta"], gamma=full.params_["gamma"],
                                phi=full.params_["phi"], SSE_treino=full.sse_))
            except Exception as ex:
                log("HW falhou", trend, seasonal, damped, ex)
    return pd.DataFrame(out).sort_values("inner_MAE").reset_index(drop=True)


def search_sarimax():
    out = []
    t_tr_end = ORIGIN_IDX[TEST_ROWS[0]]  # última origem do treino completo = origem anterior ao 1º teste
    orders = list(itertools.product([0, 1, 2], [0, 1], [0, 1]))
    sorders = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
    for use_exog in (True, False):
        E = sarimax_exog(use_exog)
        for o in orders:
            for so in sorders:
                try:
                    p, _ = wf_sarimax(o, so, use_exog, INNER_ROWS)
                    full = SARIMAXLite(o, so, M_SEAS).fit(Yfull[1:t_tr_end + 1], E.values[1:t_tr_end + 1] if use_exog else None)
                    out.append(dict(order=o, seasonal_order=so, m=M_SEAS, exog=use_exog, inner_MAE=rows_mae(p, INNER_ROWS),
                                    AIC_css=full.aic_, BIC_css=full.bic_))
                except Exception as ex:
                    pass
    return pd.DataFrame(out).sort_values("inner_MAE").reset_index(drop=True)


def sample_space(space, n, seed):
    rng = np.random.default_rng(seed)
    keys = list(space)
    total = int(np.prod([len(space[k]) for k in keys]))
    seen, cfgs = set(), []
    while len(cfgs) < min(n, total):
        c = tuple(int(rng.integers(len(space[k]))) for k in keys)
        if c in seen:
            continue
        seen.add(c); cfgs.append({k: space[k][i] for k, i in zip(keys, c)})
    return cfgs


RF_SPACE = dict(w=[3, 7, 14], n_estimators=[100, 200], max_depth=[None, 3, 6], min_samples_split=[2, 5, 10],
                min_samples_leaf=[1, 3, 5], max_features=["sqrt", 0.5, 1.0], target=["level", "delta"])
MLP_SPACE = dict(w=[3, 7, 14], hidden=[(16,), (32,), (64,), (32, 16)], activation=["relu", "tanh"],
                 alpha=[1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0], solver=["lbfgs", "adam"], lr=[1e-3, 1e-2],
                 target=["level", "delta"])


def search_rf(n_iter=60):
    out = []
    for k, c in enumerate(sample_space(RF_SPACE, n_iter, SEED)):
        hp = {a: c[a] for a in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features")}
        p, _, t = wf_tabular(lambda: RFModel(**hp), c["w"], INNER_ROWS, c["target"] == "delta")
        out.append({**c, "inner_MAE": rows_mae(p, INNER_ROWS), "seg": t})
        if (k + 1) % 10 == 0:
            log(f"RF busca {k + 1}/{n_iter}")
    return pd.DataFrame(out).sort_values("inner_MAE").reset_index(drop=True)


def search_mlp(n_iter=60):
    out = []
    for k, c in enumerate(sample_space(MLP_SPACE, n_iter, SEED)):
        p, _, t = wf_tabular(lambda: MLPModel(c["hidden"], c["activation"], c["alpha"], c["solver"], c["lr"], MLP_SEEDS_INNER),
                             c["w"], INNER_ROWS, c["target"] == "delta")
        out.append({**c, "hidden": str(c["hidden"]), "inner_MAE": rows_mae(p, INNER_ROWS), "seg": t})
        if (k + 1) % 5 == 0:
            log(f"MLP busca {k + 1}/{n_iter}")
    return pd.DataFrame(out).sort_values("inner_MAE").reset_index(drop=True)


# ------------------------------------------------------------------ importância por permutação
def permutation_importance(models_predict, X, yl, yc, delta, n_repeats=100, seed=SEED, groups=None):
    """ΔMAE (nível) ao embaralhar cada coluna (ou grupo) no conjunto de teste. `models_predict(Xarr)->pred_delta/level`."""
    rng = np.random.default_rng(seed)
    base = tu.mae(yl, models_predict(X.values) + (yc if delta else 0.0))
    cols = list(X.columns)
    units = {c: [c] for c in cols} if groups is None else groups
    rows = []
    for name, cs in units.items():
        idx = [cols.index(c) for c in cs]
        d = []
        for _ in range(n_repeats):
            Xp = X.values.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xp[perm][:, idx]
            d.append(tu.mae(yl, models_predict(Xp) + (yc if delta else 0.0)) - base)
        rows.append((name, np.mean(d), np.std(d), base))
    return pd.DataFrame(rows, columns=["feature", "delta_MAE", "desvio", "MAE_base"]).sort_values("delta_MAE", ascending=False)


# ------------------------------------------------------------------ main
def cached(name, fn):
    """Reaproveita results/<name>.csv se existir (apague o arquivo para refazer a busca)."""
    f = RES / name
    if f.exists():
        log(f"{name}: reaproveitando busca salva")
        return pd.read_csv(f)
    df = fn(); df.to_csv(f, index=False); return df


def main():
    T0 = time.time()
    log(f"origens={N_ROWS} | treino completo: {ORIGIN_IDX[0]}..{ORIGIN_IDX[TEST_ROWS[0]-1]} | teste: {T_DATE.iloc[TEST_ROWS[0]].date()}..{T_DATE.iloc[-1].date()}")
    meta = dict(n_rows=N_ROWS, n_test=N_TEST, n_inner=N_INNER,
                train_targets=f"{T_DATE.iloc[0].date()} a {T_DATE.iloc[TEST_ROWS[0]-1].date()}",
                inner_targets=f"{T_DATE.iloc[INNER_ROWS[0]].date()} a {T_DATE.iloc[INNER_ROWS[-1]].date()}",
                test_targets=f"{T_DATE.iloc[TEST_ROWS[0]].date()} a {T_DATE.iloc[-1].date()}",
                seed=SEED, mlp_seeds_inner=MLP_SEEDS_INNER, mlp_seeds_final=MLP_SEEDS_FINAL, m_seasonal=M_SEAS)
    json.dump(meta, open(RES / "protocolo.json", "w"), indent=2, ensure_ascii=False)

    # ---- buscas (somente região de treino)
    sel = {}
    s_hw = cached("busca_holt_winters.csv", search_hw)
    r = s_hw.iloc[0]
    sel["Holt-Winters"] = dict(trend=r.trend if isinstance(r.trend, str) else None,
                               seasonal=r.seasonal if isinstance(r.seasonal, str) else None, damped=bool(r.damped), m=M_SEAS)
    s_sx = cached("busca_sarimax.csv", search_sarimax)
    r = s_sx.iloc[0]
    o = r.order if not isinstance(r.order, str) else eval(r.order)
    so = r.seasonal_order if not isinstance(r.seasonal_order, str) else eval(r.seasonal_order)
    sel["SARIMAX"] = dict(order=list(o), seasonal_order=list(so), m=M_SEAS, exog=bool(r.exog))
    s_rf = cached("busca_random_forest.csv", search_rf)
    s_ml = cached("busca_mlp.csv", search_mlp)
    # janela w COMPARTILHADA (enunciado 5.3: RF e modelo de especialização usam as mesmas features):
    # w* minimiza a média dos melhores MAE internos de RF e MLP. Decisão só com a validação interna.
    jw = pd.DataFrame({"w": [3, 7, 14]})
    jw["melhor_inner_MAE_RF"] = [s_rf[s_rf.w == w].inner_MAE.min() for w in jw.w]
    jw["melhor_inner_MAE_MLP"] = [s_ml[s_ml.w == w].inner_MAE.min() for w in jw.w]
    jw["media"] = jw[["melhor_inner_MAE_RF", "melhor_inner_MAE_MLP"]].mean(axis=1)
    jw.to_csv(RES / "selecao_w_conjunto.csv", index=False)
    W_STAR = int(jw.sort_values("media").iloc[0].w)
    log("w compartilhado =", W_STAR)
    r = s_rf[s_rf.w == W_STAR].iloc[0]
    sel["Random Forest"] = {k: (None if (isinstance(r[k], float) and np.isnan(r[k])) else r[k]) for k in RF_SPACE}
    for k in ("w", "n_estimators", "min_samples_split", "min_samples_leaf"):
        sel["Random Forest"][k] = int(sel["Random Forest"][k])
    if sel["Random Forest"]["max_depth"] is not None:
        sel["Random Forest"]["max_depth"] = int(sel["Random Forest"]["max_depth"])
    mf = sel["Random Forest"]["max_features"]
    if not (isinstance(mf, str) and mf == "sqrt"):
        sel["Random Forest"]["max_features"] = float(mf)
    r = s_ml[s_ml.w == W_STAR].iloc[0]
    sel["MLP Regressor"] = dict(w=int(r.w), hidden=list(eval(r.hidden)), activation=r.activation, alpha=float(r.alpha),
                                solver=r.solver, lr=float(r.lr), target=r.target)
    json.dump(sel, open(RES / "hiperparametros_selecionados.json", "w"), indent=2, ensure_ascii=False, default=str)
    log("selecionados:", sel)

    # ---- teste final walk-forward
    y_true = Y_LEVEL.values[TEST_ROWS]
    preds, times = {}, {}
    h = sel["Holt-Winters"]; preds["Holt-Winters"], times["Holt-Winters"] = wf_hw(h["trend"], h["seasonal"], h["damped"], TEST_ROWS)
    s = sel["SARIMAX"]; preds["SARIMAX"], times["SARIMAX"] = wf_sarimax(tuple(s["order"]), tuple(s["seasonal_order"]), s["exog"], TEST_ROWS)
    f = sel["Random Forest"]
    rf_hp = {k: f[k] for k in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features")}
    preds["Random Forest"], _, times["Random Forest"] = wf_tabular(lambda: RFModel(**rf_hp), f["w"], TEST_ROWS, f["target"] == "delta")
    m = sel["MLP Regressor"]
    mk = lambda ns: (lambda: MLPModel(tuple(m["hidden"]), m["activation"], m["alpha"], m["solver"], m["lr"], ns))
    preds["MLP Regressor"], mlp_seed_preds, times["MLP Regressor"] = wf_tabular(mk(MLP_SEEDS_FINAL), m["w"], TEST_ROWS, m["target"] == "delta")
    log("teste final ok")

    yc_t = Y_CUR.values[TEST_ROWS]
    base = pd.DataFrame({"data": T_DATE.iloc[TEST_ROWS].values, "real": y_true,
                         "Persistência (ref.)": yc_t,
                         "Sazonal ingênuo m=7 (ref.)": feature_frame(7).iloc[TEST_ROWS]["y_lag6"].values,
                         "Média móvel 7d (ref.)": feature_frame(7).iloc[TEST_ROWS]["y_mean7"].values})
    for k, v in preds.items():
        base[k] = v
    base.to_csv(RES / "previsoes_walkforward.csv", index=False)
    pd.DataFrame(mlp_seed_preds.T, columns=[f"seed_{i}" for i in range(MLP_SEEDS_FINAL)]).assign(data=T_DATE.iloc[TEST_ROWS].values, real=y_true) \
        .to_csv(RES / "previsoes_mlp_por_semente.csv", index=False)
    pd.Series(times).rename("segundos").to_csv(RES / "tempos_execucao.csv")

    # ---- importâncias (modelo final treinado em todo o treino; avaliação no teste, apenas interpretação)
    Xtr_i = range(0, TEST_ROWS[0])
    # RF
    Xrf = feature_frame(f["w"]); delta = f["target"] == "delta"
    ytr = Y_LEVEL.values[:TEST_ROWS[0]] - (Y_CUR.values[:TEST_ROWS[0]] if delta else 0.0)
    rf = RFModel(**rf_hp).fit(Xrf.iloc[:TEST_ROWS[0]].values, ytr)
    Xte = Xrf.iloc[TEST_ROWS]
    grp = lambda X: {g: [c for c in X.columns if fn(c)] for g, fn in tu.GROUPS.items() if any(fn(c) for c in X.columns)}
    pi = permutation_importance(lambda A: rf.predict(A), Xte, y_true, yc_t, delta)
    pi["impureza_MDI"] = pi["feature"].map(dict(zip(Xrf.columns, rf.m.feature_importances_)))
    pi.to_csv(RES / "importancia_rf.csv", index=False)
    permutation_importance(lambda A: rf.predict(A), Xte, y_true, yc_t, delta, groups=grp(Xte)).to_csv(RES / "importancia_rf_grupos.csv", index=False)
    # MLP
    Xml = feature_frame(m["w"]); delta_m = m["target"] == "delta"
    ytr = Y_LEVEL.values[:TEST_ROWS[0]] - (Y_CUR.values[:TEST_ROWS[0]] if delta_m else 0.0)
    mlp = mk(MLP_SEEDS_FINAL)().fit(Xml.iloc[:TEST_ROWS[0]].values, ytr)
    Xte_m = Xml.iloc[TEST_ROWS]
    permutation_importance(lambda A: mlp.predict(A), Xte_m, y_true, yc_t, delta_m).to_csv(RES / "importancia_mlp.csv", index=False)
    permutation_importance(lambda A: mlp.predict(A), Xte_m, y_true, yc_t, delta_m, groups=grp(Xte_m)).to_csv(RES / "importancia_mlp_grupos.csv", index=False)
    log("importâncias ok")

    # ---- SARIMAX: coeficientes finais e HW: estados
    E = sarimax_exog(s["exog"]); t_end = ORIGIN_IDX[TEST_ROWS[0]]
    sx = SARIMAXLite(tuple(s["order"]), tuple(s["seasonal_order"]), M_SEAS).fit(Yfull[1:t_end + 1], E.values[1:t_end + 1] if s["exog"] else None)
    sx.coef_table(list(E.columns) if s["exog"] else []).assign(AIC_css=sx.aic_, BIC_css=sx.bic_).to_csv(RES / "coeficientes_sarimax.csv", index=False)
    hw = HoltWinters(h["trend"], h["seasonal"], h["damped"], M_SEAS).fit(Yfull[:t_end + 1])
    json.dump(dict(params=hw.params_, nivel_final=float(hw.l_), tendencia_final=float(hw.b_),
                   sazonais_finais=[float(v) for v in (hw.s_[-M_SEAS:] if hw.seasonal else [])]),
              open(RES / "estados_holt_winters.json", "w"), indent=2)

    # ---- estudo do MLP: sensibilidade à semente, ablação por grupo, curva de aprendizado
    pd.DataFrame({"semente": range(MLP_SEEDS_FINAL),
                  "MAE_teste": [tu.mae(y_true, mlp_seed_preds[k]) for k in range(MLP_SEEDS_FINAL)]}).to_csv(RES / "mlp_sensibilidade_semente.csv", index=False)
    abl = []
    for g in [()] + [(g,) for g in tu.GROUPS if g != "temperatura (lags/médias/dif)"]:
        p, _, _ = wf_tabular(mk(3), m["w"], TEST_ROWS, delta_m, drop_groups=g)
        abl.append(("completo" if not g else "sem " + g[0], tu.mae(y_true, p)))
    pd.DataFrame(abl, columns=["configuracao", "MAE_teste"]).to_csv(RES / "mlp_ablacao.csv", index=False)
    log("ablação ok")
    # sensibilidade PÓS-HOC ao alpha (a escolha oficial caiu na borda do espaço de busca; nada abaixo altera a seleção)
    sens = []
    for a in [1.0, 3.0, 10.0, 30.0, 100.0]:
        mka = lambda ns, a=a: (lambda: MLPModel(tuple(m["hidden"]), m["activation"], a, m["solver"], m["lr"], ns))
        pin, _, _ = wf_tabular(mka(MLP_SEEDS_INNER), m["w"], INNER_ROWS, delta_m)
        pte, _, _ = wf_tabular(mka(3), m["w"], TEST_ROWS, delta_m)
        sens.append((a, rows_mae(pin, INNER_ROWS), tu.mae(y_true, pte)))
    pd.DataFrame(sens, columns=["alpha", "MAE_validacao_interna", "MAE_teste_posthoc"]).to_csv(RES / "mlp_sensibilidade_alpha_posthoc.csv", index=False)
    log("sensibilidade alpha ok")
    # curva de aprendizado (Adam, arquitetura selecionada) treinando só em linhas < início da validação interna e medindo na validação interna
    X14 = feature_frame(m["w"]); tr_n = INNER_ROWS[0]
    ytr_c = Y_LEVEL.values[:tr_n] - (Y_CUR.values[:tr_n] if delta_m else 0.0)
    sx_ = StandardScaler().fit(X14.iloc[:tr_n].values); ym, ysd = ytr_c.mean(), ytr_c.std()
    Xa, Xv = sx_.transform(X14.iloc[:tr_n].values), sx_.transform(X14.iloc[INNER_ROWS].values)
    mm = MLPRegressor(hidden_layer_sizes=tuple(m["hidden"]), activation=m["activation"], alpha=m["alpha"], solver="adam",
                      learning_rate_init=m["lr"] if m["solver"] == "adam" else 1e-3, batch_size=32, random_state=SEED)
    curve = []
    for ep in range(1, 301):
        mm.partial_fit(Xa, (ytr_c - ym) / ysd)
        ptr = mm.predict(Xa) * ysd + ym + (Y_CUR.values[:tr_n] if delta_m else 0.0)
        pv = mm.predict(Xv) * ysd + ym + (Y_CUR.values[INNER_ROWS] if delta_m else 0.0)
        curve.append((ep, tu.mae(Y_LEVEL.values[:tr_n], ptr), tu.mae(Y_LEVEL.values[INNER_ROWS], pv)))
    pd.DataFrame(curve, columns=["epoca", "MAE_treino", "MAE_validacao_interna"]).to_csv(RES / "mlp_curva_aprendizado.csv", index=False)
    log(f"FIM em {(time.time() - T0) / 60:.1f} min")


if __name__ == "__main__":
    main()
