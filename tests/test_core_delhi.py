"""Testes de sanidade das implementações próprias (séries sintéticas com parâmetros conhecidos)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import trabalhoseries_redeneural.src.tsutils_delhi as tu
from trabalhoseries_redeneural.src.ts_models_delhi import HoltWinters, SARIMAXLite


def test_stl_recupera_componentes():
    rng = np.random.default_rng(0)
    n, P = 140, 7
    t = np.arange(n)
    trend = 0.05 * t + 2 * np.sin(2 * np.pi * t / 140)
    seas = 1.5 * np.sin(2 * np.pi * t / P)
    y = trend + seas + rng.normal(0, 0.15, n)
    dec = tu.stl(y, P, seasonal=13, trend=21)
    assert np.allclose(dec["trend"] + dec["seasonal"] + dec["resid"], y)
    assert np.corrcoef(dec["seasonal"], seas)[0, 1] > 0.97
    assert np.corrcoef(dec["trend"], trend)[0, 1] > 0.99
    fs, ft = tu.strength(dec)
    assert fs > 0.9 and ft > 0.9
    rng2 = np.random.default_rng(1)
    ruido = rng2.normal(0, 1, 140)
    fs0, _ = tu.strength(tu.stl(ruido, P, seasonal=13, trend=21))
    assert fs0 < 0.5


def test_ljung_box_ruido_branco_vs_ar():
    rng = np.random.default_rng(2)
    wn = rng.normal(size=400)
    assert tu.ljung_box(wn, [10]).p_valor[0] > 0.05
    ar = np.zeros(400)
    for i in range(1, 400):
        ar[i] = 0.7 * ar[i - 1] + rng.normal()
    assert tu.ljung_box(ar, [10]).p_valor[0] < 1e-6


def test_holt_winters_sazonal_aditivo():
    rng = np.random.default_rng(3)
    t = np.arange(120)
    y = 20 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.2, 120)
    hw = HoltWinters(trend="add", seasonal="add", m=7).fit(y[:119])
    assert abs(hw.forecast1() - y[119]) < 1.0
    naive_err = abs(y[118] - y[119])
    assert abs(hw.forecast1() - y[119]) < naive_err + 0.5


def test_sarimax_recupera_coeficientes():
    rng = np.random.default_rng(4)
    n = 600
    x = rng.normal(size=(n, 1))
    eps = rng.normal(0, 1, n)
    eta = np.zeros(n)
    for i in range(1, n):
        eta[i] = 0.6 * eta[i - 1] + eps[i]
    y = 2.0 * x[:, 0] + 5.0 + eta
    m = SARIMAXLite(order=(1, 0, 0)).fit(y, x)
    tab = m.coef_table(["x"])
    b = tab.set_index("termo")["coef"]
    assert abs(b["x"] - 2.0) < 0.15 and abs(b["ar.L1"] - 0.6) < 0.08 and abs(b["const"] - 5.0) < 0.5


def test_sarimax_previsao_1passo_ar1():
    rng = np.random.default_rng(5)
    y = np.zeros(800)
    for i in range(1, 800):
        y[i] = 0.8 * y[i - 1] + rng.normal()
    m = SARIMAXLite(order=(1, 0, 0)).fit(y[:700], None)
    errs = []
    for t in range(700, 799):
        mm = SARIMAXLite(order=(1, 0, 0)).fit(y[:t], None)
        errs.append(abs(mm.forecast1() - y[t]))
    persist = np.mean(np.abs(np.diff(y[699:800])))
    assert np.mean(errs) < persist


def test_sarimax_diferenciado_passeio_aleatorio():
    rng = np.random.default_rng(6)
    y = np.cumsum(rng.normal(size=300))
    m = SARIMAXLite(order=(0, 1, 0)).fit(y[:-1], None)
    assert abs(m.forecast1() - y[-2]) < 1e-9  # passeio aleatório puro: previsão = último valor


def test_sarimax_constante_sem_exogena():
    rng = np.random.default_rng(7)
    y = np.zeros(900)
    for i in range(1, 900):
        y[i] = 10 + 0.5 * y[i - 1] + rng.normal()   # média = 20
    errs = [abs(SARIMAXLite(order=(1, 0, 0)).fit(y[:t], None).forecast1() - y[t]) for t in range(800, 899)]
    assert np.mean(errs) < 1.0, np.mean(errs)       # sem a constante o erro seria ~10


def test_sarimax_exogena_com_diferenciacao():
    rng = np.random.default_rng(8)
    n = 400
    x = rng.normal(size=(n, 1))
    y = np.cumsum(rng.normal(0, 0.3, n)) + 3.0 * x[:, 0]
    errs, naive = [], []
    for t in range(300, 399):
        m = SARIMAXLite(order=(0, 1, 1)).fit(y[:t], x[:t])
        errs.append(abs(m.forecast1(x[t]) - y[t])); naive.append(abs(y[t - 1] - y[t]))
    assert np.mean(errs) < 0.7 * np.mean(naive), (np.mean(errs), np.mean(naive))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK ", name)
