"""Holt-Winters e SARIMAX implementados com numpy/scipy (statsmodels indisponível no ambiente de execução).

HoltWinters : nível + tendência (aditiva, opcionalmente amortecida) + sazonalidade (aditiva/multiplicativa),
              parâmetros de suavização por mínimos quadrados dos erros de 1 passo (L-BFGS-B),
              estados iniciais heurísticos (como a opção 'heuristic' do statsmodels).
SARIMAXLite : regressão com erros SARIMA(p,d,q)(P,D,Q)_m estimada por Soma de Quadrados Condicional (CSS).
              AIC/BIC são os de CSS (n*log(sigma2)+penalização), úteis para comparar ordens, não idênticos
              aos de máxima verossimilhança exata do statsmodels.
"""
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy import stats


# ============================================================================ Holt-Winters
class HoltWinters:
    def __init__(self, trend=None, seasonal=None, damped=False, m=7):
        assert trend in (None, "add") and seasonal in (None, "add", "mul")
        self.trend, self.seasonal, self.damped, self.m = trend, seasonal, damped and trend is not None, m

    def _init_states(self, y):
        m = self.m
        if self.seasonal:
            l0 = y[:m].mean()
            b0 = (y[m:2 * m].mean() - y[:m].mean()) / m if self.trend else 0.0
            s0 = (y[:m] - l0) if self.seasonal == "add" else (y[:m] / l0)
            return l0, b0, s0
        l0 = y[0]
        b0 = np.mean(np.diff(y[:min(len(y), 6)])) if self.trend else 0.0
        return l0, b0, np.zeros(0)

    def _run(self, y, p):
        a, b, g, phi = p
        m = self.m
        l, bb, s = self._init_states(y)
        s = list(s)
        fitted = np.empty(len(y))
        start = m if self.seasonal else 1
        # (re)inicia a partir do início da série; o 1º ciclo serve de inicialização
        for t in range(len(y)):
            base = l + (phi * bb if self.trend else 0.0)
            if self.seasonal:
                st = s[t % m] if t < m else s[-m]
                yhat = base + st if self.seasonal == "add" else base * st
            else:
                st = 0.0; yhat = base
            fitted[t] = yhat
            if t < start:
                if self.seasonal and t < m:  # primeiro ciclo: estados já inicializados
                    pass
                else:
                    l_new = a * y[t] + (1 - a) * base
                    bb = (b * (l_new - l) + (1 - b) * phi * bb) if self.trend else 0.0
                    l = l_new
                continue
            if self.seasonal == "add":
                l_new = a * (y[t] - st) + (1 - a) * base
                s_new = g * (y[t] - base) + (1 - g) * st
            elif self.seasonal == "mul":
                l_new = a * (y[t] / st) + (1 - a) * base
                s_new = g * (y[t] / base) + (1 - g) * st
            else:
                l_new = a * y[t] + (1 - a) * base
            if self.trend:
                bb = b * (l_new - l) + (1 - b) * phi * bb
            l = l_new
            if self.seasonal:
                s.append(s_new)
        return fitted, l, bb, s

    def fit(self, y):
        y = np.asarray(y, float)
        if self.seasonal and len(y) < 2 * self.m + 1:
            raise ValueError("série curta demais")
        start = self.m if self.seasonal else 1

        def sse(p):
            f, *_ = self._run(y, p)
            return float(np.sum((y[start:] - f[start:]) ** 2))

        bounds = [(0.01, 0.99), (0.001, 0.99) if self.trend else (0, 0.0001),
                  (0.001, 0.99) if self.seasonal else (0, 0.0001), (0.8, 0.98) if self.damped else (1.0, 1.0001)]
        starts = [(0.3, 0.1, 0.1, 0.95 if self.damped else 1.0), (0.7, 0.05, 0.3, 0.9 if self.damped else 1.0)]
        best = None
        for s0 in starts:
            r = minimize(sse, s0, bounds=bounds, method="L-BFGS-B")
            if best is None or r.fun < best.fun:
                best = r
        p = best.x
        if not self.damped:
            p[3] = 1.0
        self.params_ = dict(alpha=p[0], beta=p[1] if self.trend else None,
                            gamma=p[2] if self.seasonal else None, phi=p[3] if self.damped else None)
        self._p = p
        self.fittedvalues_, self.l_, self.b_, self.s_ = self._run(y, p)
        self.n_ = len(y)
        self.sse_ = best.fun
        return self

    def forecast1(self):
        """Previsão 1 passo à frente."""
        phi = self._p[3]
        base = self.l_ + (phi * self.b_ if self.trend else 0.0)
        if self.seasonal:
            st = self.s_[-self.m]
            return base + st if self.seasonal == "add" else base * st
        return base


# ============================================================================ SARIMAX
def _diff_poly(d, D, m):
    c = np.array([1.0])
    for _ in range(d):
        c = np.convolve(c, [1.0, -1.0])
    for _ in range(D):
        k = np.zeros(m + 1); k[0], k[m] = 1.0, -1.0
        c = np.convolve(c, k)
    return c


def _seasonal_poly(coefs, m, sign):
    """polinômio 1 + sign*c1 B^m + ... """
    k = np.zeros(len(coefs) * m + 1); k[0] = 1.0
    for i, cf in enumerate(coefs, 1):
        k[i * m] = sign * cf
    return k


def _lin_poly(coefs, sign):
    return np.concatenate([[1.0], sign * np.asarray(coefs, float)])


def _apply_poly(c, arr):
    """aplica operador (c0 + c1 B + ...) a arr (1D ou 2D), descartando as len(c)-1 primeiras linhas."""
    k = len(c)
    if k == 1:
        return arr.copy()
    n = arr.shape[0]
    out = np.zeros_like(arr[k - 1:], dtype=float)
    for i, ci in enumerate(c):
        out += ci * arr[k - 1 - i: n - i]
    return out


class SARIMAXLite:
    def __init__(self, order=(1, 0, 0), seasonal_order=(0, 0, 0), m=7):
        self.p, self.d, self.q = order
        self.P, self.D, self.Q = seasonal_order
        self.m = m

    # --- helpers
    def _polys(self, ar, sar, ma, sma):
        A = np.convolve(_lin_poly(ar, -1.0), _seasonal_poly(sar, self.m, -1.0))
        M = np.convolve(_lin_poly(ma, +1.0), _seasonal_poly(sma, self.m, +1.0))
        return A, M

    def _split(self, th):
        k = self.k_
        beta = th[:k]
        i = k
        ar = th[i:i + self.p]; i += self.p
        sar = th[i:i + self.P]; i += self.P
        ma = th[i:i + self.q]; i += self.q
        sma = th[i:i + self.Q]
        return beta, ar, sar, ma, sma

    def _arma_resid(self, z, A, M):
        n = len(z); pA = len(A) - 1; qM = len(M) - 1
        e = np.zeros(n)
        for t in range(pA, n):
            v = z[t]
            for i in range(1, pA + 1):
                if A[i] != 0.0:
                    v += A[i] * z[t - i]
            for j in range(1, qM + 1):
                if M[j] != 0.0 and t - j >= 0:
                    v -= M[j] * e[t - j]
            e[t] = v
        return e

    def _design(self, X, n):
        cols = []
        if X is not None and X.shape[1] > 0:
            cols.append(np.asarray(X, float))
        if self.d + self.D == 0:
            cols.append(np.ones((n, 1)))
        return np.hstack(cols) if cols else np.zeros((n, 0))

    def fit(self, y, X=None):
        y = np.asarray(y, float); n = len(y)
        self.kx_ = 0 if X is None else np.asarray(X).shape[1]
        Z = self._design(X, n)
        self.k_ = Z.shape[1]
        self.c_ = _diff_poly(self.d, self.D, self.m)
        yd = _apply_poly(self.c_, y)
        Zd = _apply_poly(self.c_, Z) if self.k_ else np.zeros((len(yd), 0))
        npar_arma = self.p + self.P + self.q + self.Q
        pA = self.p + self.P * self.m
        self.pA_ = pA

        beta0 = np.linalg.lstsq(Zd, yd, rcond=None)[0] if self.k_ else np.zeros(0)
        th0 = np.concatenate([beta0, np.zeros(npar_arma)])

        def resid(th):
            beta, ar, sar, ma, sma = self._split(th)
            z = yd - (Zd @ beta if self.k_ else 0.0)
            A, M = self._polys(ar, sar, ma, sma)
            return self._arma_resid(z, A, M)[pA:]

        lo = np.concatenate([np.full(self.k_, -np.inf), np.full(npar_arma, -0.98)])
        hi = np.concatenate([np.full(self.k_, np.inf), np.full(npar_arma, 0.98)])
        if len(th0) == 0:
            self.theta_ = th0
            r = resid(th0); self.sse_ = float(r @ r); self.cov_ = np.zeros((0, 0))
        else:
            res = least_squares(resid, th0, bounds=(lo, hi), method="trf")
            self.theta_ = res.x
            r = res.fun; self.sse_ = float(r @ r)
            neff = len(r)
            s2 = self.sse_ / max(neff - len(th0), 1)
            try:
                self.cov_ = s2 * np.linalg.inv(res.jac.T @ res.jac)
            except np.linalg.LinAlgError:
                self.cov_ = np.full((len(th0), len(th0)), np.nan)
        self.neff_ = len(yd) - pA
        self.sigma2_ = self.sse_ / self.neff_
        kk = len(self.theta_) + 1
        self.aic_ = self.neff_ * np.log(self.sigma2_) + 2 * kk
        self.bic_ = self.neff_ * np.log(self.sigma2_) + kk * np.log(self.neff_)
        self._y, self._Z = y, Z
        self._yd, self._Zd = yd, Zd
        return self

    def forecast1(self, x_next=None):
        """Previsão 1 passo; x_next = linha de exógenas do instante T+1 (já conhecida/alinhada)."""
        beta, ar, sar, ma, sma = self._split(self.theta_)
        A, M = self._polys(ar, sar, ma, sma)
        z = self._yd - (self._Zd @ beta if self.k_ else 0.0)
        e = self._arma_resid(z, A, M)
        n = len(z)
        zhat = 0.0
        for i in range(1, len(A)):
            if A[i] != 0.0 and n - i >= 0:
                zhat -= A[i] * z[n - i]
        for j in range(1, len(M)):
            if M[j] != 0.0 and n - j >= 0:
                zhat += M[j] * e[n - j]
        # linha de regressores em T+1: exógenas conhecidas + constante (quando não há diferenciação)
        parts = []
        if self.kx_:
            parts.append(np.asarray(x_next, float))
        if self.d + self.D == 0:
            parts.append(np.array([1.0]))
        znext = np.concatenate(parts) if parts else np.zeros(0)
        # diferença de Z em T+1: sum_k c_k Z_{T+1-k}
        c = self.c_
        if self.k_:
            Zext = np.vstack([self._Z, znext[None, :]])
            dz = np.zeros(self.k_)
            for k, ck in enumerate(c):
                dz += ck * Zext[-1 - k]
            dyhat = zhat + dz @ beta
        else:
            dyhat = zhat
        yhat = dyhat
        for k in range(1, len(c)):
            yhat -= c[k] * self._y[-k]
        return float(yhat)

    def coef_table(self, exog_names):
        names = list(exog_names) + (["const"] if self.d + self.D == 0 else [])
        names += [f"ar.L{i}" for i in range(1, self.p + 1)]
        names += [f"sar.L{i * self.m}" for i in range(1, self.P + 1)]
        names += [f"ma.L{i}" for i in range(1, self.q + 1)]
        names += [f"sma.L{i * self.m}" for i in range(1, self.Q + 1)]
        se = np.sqrt(np.clip(np.diag(self.cov_), 0, None)) if self.cov_.size else np.zeros(0)
        z = self.theta_ / np.where(se > 0, se, np.nan)
        p = 2 * (1 - stats.norm.cdf(np.abs(z)))
        import pandas as pd
        return pd.DataFrame({"termo": names, "coef": self.theta_, "erro_padrao": se, "z": z, "p_valor": p})
