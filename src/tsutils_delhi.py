"""Utilitários do trabalho de Séries Temporais (Grupo 4 - MLP | base: Daily Delhi Climate).

Contém: leitura/limpeza da base, engenharia de features sem vazamento, STL (implementação
própria, Cleveland et al. 1990), ACF, Ljung-Box, métricas e teste de Diebold-Mariano.
Dependências: numpy, pandas, scipy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "datasets" / "DailyDelhiClimateTest.csv"
TARGET = "meantemp"
EXOG = ["humidity", "wind_speed", "meanpressure"]


# ----------------------------------------------------------------------------- dados
def load_raw(path=DATA_FILE):
    return pd.read_csv(path, parse_dates=["date"])


def quality_report(df):
    """Tabela com checagens de qualidade (ausentes, duplicidades, irregularidade)."""
    rows = []
    full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    rows.append(("observações", len(df)))
    rows.append(("período", f"{df['date'].min().date()} a {df['date'].max().date()}"))
    rows.append(("datas duplicadas", int(df["date"].duplicated().sum())))
    rows.append(("datas faltantes na grade diária", int(len(full.difference(df["date"])))))
    rows.append(("datas fora de ordem", int((df["date"].diff().dropna() <= pd.Timedelta(0)).sum())))
    for c in [TARGET] + EXOG:
        rows.append((f"ausentes em {c}", int(df[c].isna().sum())))
    return pd.DataFrame(rows, columns=["checagem", "resultado"])


def outlier_table(df, cols=None, k=3.5):
    """Atípicos por escore z robusto (MAD) sobre o nível e sobre a primeira diferença."""
    cols = cols or [TARGET] + EXOG
    out = []
    for c in cols:
        for kind, s in (("nível", df[c]), ("diferença", df[c].diff())):
            s2 = s.dropna()
            med = s2.median()
            mad = np.median(np.abs(s2 - med)) or 1e-9
            z = 0.6745 * (s - med) / mad
            for i in z.index[np.abs(z) > k]:
                out.append((c, kind, df.loc[i, "date"].date(), float(df.loc[i, c]), float(z[i])))
    return pd.DataFrame(out, columns=["variavel", "base", "data", "valor", "z_robusto"])


def clean(df, pressure_floor=900.0):
    """Pressão < 900 hPa é fisicamente impossível ao nível do solo (59,0 em 2017-01-01):
    vira NaN e é preenchida (interpolação; na 1ª linha, com o valor do dia seguinte)."""
    d = df.sort_values("date").reset_index(drop=True).copy()
    bad = d["meanpressure"] < pressure_floor
    d["pressure_corrected"] = bad
    d.loc[bad, "meanpressure"] = np.nan
    d["meanpressure"] = d["meanpressure"].interpolate(limit_direction="both")
    return d


# ----------------------------------------------------------------------------- features
def build_features(d, w=7, max_w=14):
    """Matriz tabular para prever y(t+1) na ORIGEM t usando apenas informação até t.

    - lags de y (0..w-1), médias/desvio móvel terminando em t, diferença de 1 dia;
    - exógenas (umidade, vento, pressão) observadas em t, t-1, t-2 (a exógena de t+1 NÃO é conhecida);
    - calendário do dia-alvo t+1 (conhecido antecipadamente), codificação cíclica anual.
    Linhas sem histórico (t < max_w-1) são descartadas para que todo w use as mesmas origens.
    Retorna (X, alvo_nivel, y_atual, data_alvo)."""
    y = d[TARGET]
    F = pd.DataFrame(index=d.index)
    for k in range(w):
        F[f"y_lag{k}"] = y.shift(k)
    F["y_mean3"] = y.rolling(3).mean()
    F["y_mean7"] = y.rolling(7).mean()
    F["y_std7"] = y.rolling(7).std()
    F["y_diff1"] = y.diff(1)
    for c, short in zip(EXOG, ["hum", "wind", "pres"]):
        for k in range(3):
            F[f"{short}_lag{k}"] = d[c].shift(k)
    F["pres_diff1"] = d["meanpressure"].diff(1)
    tgt_date = d["date"].shift(-1)
    doy = tgt_date.dt.dayofyear
    F["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    F["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    target = y.shift(-1)
    keep = (np.arange(len(d)) >= max_w - 1) & target.notna().values
    return F.loc[keep], target.loc[keep], y.loc[keep], tgt_date.loc[keep]


GROUPS = {
    "temperatura (lags/médias/dif)": lambda c: c.startswith("y_"),
    "umidade": lambda c: c.startswith("hum_"),
    "vento": lambda c: c.startswith("wind_"),
    "pressão": lambda c: c.startswith("pres_"),
    "calendário (sen/cos anual)": lambda c: c in ("sin_doy", "cos_doy"),
}


# ----------------------------------------------------------------------------- métricas / diagnóstico
def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    den = np.sum(x ** 2)
    return np.array([1.0] + [np.sum(x[k:] * x[:-k]) / den for k in range(1, nlags + 1)])


def ljung_box(res, lags, model_df=0):
    """Q de Ljung-Box e p-valor (qui-quadrado, gl = lag - model_df)."""
    res = np.asarray(res, float)
    n = len(res)
    r = acf(res, max(lags))
    out = []
    for h in lags:
        q = n * (n + 2) * np.sum(r[1:h + 1] ** 2 / (n - np.arange(1, h + 1)))
        out.append((h, q, 1 - stats.chi2.cdf(q, max(h - model_df, 1))))
    return pd.DataFrame(out, columns=["lag", "Q", "p_valor"])


def diebold_mariano(e1, e2):
    """DM (h=1), perda absoluta, correção HLN. H0: mesma acurácia. estat>0 => modelo 2 melhor."""
    e1, e2 = np.abs(np.asarray(e1)), np.abs(np.asarray(e2))
    d = e1 - e2
    n = len(d)
    var = np.var(d, ddof=1) / n
    if var == 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var) * np.sqrt((n - 1) / n)
    return float(dm), float(2 * (1 - stats.t.cdf(abs(dm), n - 1)))


# ----------------------------------------------------------------------------- STL
def _tricube(u):
    u = np.abs(u)
    return np.where(u < 1, (1 - u ** 3) ** 3, 0.0)


def loess(x, y, xe, span, degree=1, w=None):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    w = np.ones(n) if w is None else np.asarray(w, float)
    out = np.empty(len(xe))
    q = min(int(span), n)
    for k, x0 in enumerate(xe):
        d = np.abs(x - x0)
        h = np.sort(d)[q - 1]
        if span > n:
            h = h * span / n
        h = max(h, 1e-12) * 1.000001
        wt = _tricube(d / h) * w
        sw = wt.sum()
        if sw <= 1e-12:
            out[k] = np.mean(y); continue
        if degree == 0 or q < 3:
            out[k] = np.sum(wt * y) / sw; continue
        xm = np.sum(wt * x) / sw
        sxx = np.sum(wt * (x - xm) ** 2)
        if sxx < 1e-12:
            out[k] = np.sum(wt * y) / sw; continue
        b = np.sum(wt * (x - xm) * y) / sxx
        out[k] = np.sum(wt * y) / sw + b * (x0 - xm)
    return out


def _odd(v):
    v = int(np.ceil(v))
    return v if v % 2 == 1 else v + 1


def _ma(x, k):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def stl(y, period, seasonal=13, trend=None, low_pass=None, inner=2, robust=False):
    """STL de Cleveland et al. (1990), aditiva, loess local-linear. Retorna trend, seasonal, resid."""
    y = np.asarray(y, float); n = len(y); P = period
    ns = _odd(seasonal)
    nl = low_pass or _odd(P)
    nt = trend or _odd(1.5 * P / (1 - 1.5 / ns))
    T = np.zeros(n); rw = np.ones(n); S = np.zeros(n)
    n_outer = 15 if robust else 1
    for o in range(n_outer):
        for _ in range(inner):
            detr = y - T
            C = np.zeros(n + 2 * P)
            for j in range(P):
                idx = np.arange(j, n, P)
                if len(idx) == 0:
                    continue
                m = len(idx)
                sm = loess(np.arange(1, m + 1), detr[idx], np.arange(0, m + 2), ns, 1, rw[idx])
                C[j + np.arange(0, m + 2) * P] = sm
            L = _ma(_ma(_ma(C, P), P), 3)
            L = loess(np.arange(n), L, np.arange(n), nl, 1)
            S = C[P:P + n] - L
            T = loess(np.arange(n), y - S, np.arange(n), nt, 1, rw)
        if robust and o < n_outer - 1:
            R = y - T - S
            h = 6 * np.median(np.abs(R))
            u = np.abs(R) / max(h, 1e-12)
            rw = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
    return {"trend": T, "seasonal": S, "resid": y - T - S,
            "params": dict(period=P, seasonal=ns, trend=nt, low_pass=nl)}


def strength(dec):
    """Força (Hyndman & Athanasopoulos): F = max(0, 1 - Var(R)/Var(X+R)); X = sazonal ou tendência."""
    R = dec["resid"]
    fs = max(0.0, 1 - np.var(R) / np.var(dec["seasonal"] + R))
    ft = max(0.0, 1 - np.var(R) / np.var(dec["trend"] + R))
    return fs, ft


def trend_phases(dates, trend, thr=0.10, min_len=7):
    """Segmenta a tendência STL em crescimento / queda / estabilidade pela inclinação (°C/dia, média móvel
    centrada de 7 dias). Segmentos menores que `min_len` dias são absorvidos pelo vizinho anterior."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    tr = np.asarray(trend, float)
    slope = pd.Series(np.gradient(tr)).rolling(7, center=True, min_periods=3).mean().values
    lab = np.where(slope > thr, "crescimento", np.where(slope < -thr, "queda", "estabilidade"))
    segs = []
    s = 0
    for i in range(1, len(lab) + 1):
        if i == len(lab) or lab[i] != lab[s]:
            segs.append([s, i - 1, lab[s]]); s = i
    merged = []
    for a, b, l in segs:
        if merged and (b - a + 1) < min_len:
            merged[-1][1] = b
        else:
            merged.append([a, b, l])
    cls = lambda a, b: (lambda sl: "crescimento" if sl > thr else ("queda" if sl < -thr else "estabilidade"))((tr[b] - tr[a]) / max(b - a, 1))
    fused = []
    for a, b, _ in merged:                                   # funde vizinhos que ficaram com a mesma fase
        if fused and cls(*fused[-1][:2]) == cls(a, b):
            fused[-1][1] = b
        else:
            fused.append([a, b, None])
    out = []
    for a, b, _ in fused:
        sl = (tr[b] - tr[a]) / max(b - a, 1)
        fase = cls(a, b)
        out.append(dict(inicio=dates[a], fim=dates[b], dias=b - a + 1, fase=fase,
                        trend_inicio=tr[a], trend_fim=tr[b], variacao=tr[b] - tr[a], inclinacao_media=sl))
    return pd.DataFrame(out)
