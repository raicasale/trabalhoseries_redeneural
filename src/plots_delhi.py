"""Funções de gráfico usadas pelos notebooks (matplotlib)."""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import trabalhoseries_redeneural.src.tsutils_delhi as tu

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
COLORS = {"Holt-Winters": "#8c564b", "SARIMAX": "#1f77b4", "Random Forest": "#2ca02c", "MLP Regressor": "#d62728",
          "Persistência (ref.)": "#7f7f7f"}
MODELS = ["Holt-Winters", "SARIMAX", "Random Forest", "MLP Regressor"]


def _save(fig, name):
    from pathlib import Path
    out = tu.ROOT / "figures"; out.mkdir(exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=130, bbox_inches="tight")


def series_overview(D, outliers_date=None):
    cols = [("meantemp", "Temperatura média (°C) - ALVO"), ("humidity", "Umidade (%)"),
            ("wind_speed", "Vento (km/h)"), ("meanpressure", "Pressão (hPa) - após correção")]
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for a, (c, t) in zip(ax, cols):
        a.plot(D["date"], D[c], lw=1.4, color="#d62728" if c == "meantemp" else "#1f77b4")
        a.set_ylabel(t, fontsize=8)
    if D["pressure_corrected"].any():
        r = D[D["pressure_corrected"]].iloc[0]
        ax[3].scatter([r["date"]], [r["meanpressure"]], color="k", zorder=5, s=25)
        ax[3].annotate("valor original 59 hPa\n(impossível) substituído", (r["date"], r["meanpressure"]),
                       xytext=(10, -28), textcoords="offset points", fontsize=8)
    fig.tight_layout(); _save(fig, "01_series_visao_geral"); return fig


def raw_pressure(raw):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].plot(raw["date"], raw["meanpressure"], color="#7f7f7f"); ax[0].set_title("Pressão BRUTA (escala distorcida pelo 59 hPa)")
    ax[1].boxplot([raw["meantemp"], raw["humidity"], raw["wind_speed"]], labels=["temp", "umidade", "vento"]); ax[1].set_title("Dispersão do alvo e exógenas")
    fig.tight_layout(); _save(fig, "02_pressao_bruta_e_boxplot"); return fig


def stl_panel(dates, y, dec, title):
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for a, (v, t) in zip(ax, [(y, "Observado"), (dec["trend"], "Tendência"), (dec["seasonal"], f"Sazonal (período {dec['params']['period']})"), (dec["resid"], "Resíduo")]):
        a.plot(dates, v, lw=1.3); a.set_ylabel(t)
    ax[2].axhline(0, color="k", lw=0.6); ax[3].axhline(0, color="k", lw=0.6)
    ax[0].set_title(title); fig.tight_layout(); _save(fig, "03_stl"); return fig


def trend_phases(dates, trend, phases):
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(dates, trend, color="k", lw=1.6)
    cmap = {"crescimento": "#d62728", "queda": "#1f77b4", "estabilidade": "#bbbbbb"}
    for _, r in phases.iterrows():
        ax.axvspan(r["inicio"], r["fim"], color=cmap[r["fase"]], alpha=0.25)
    ax.set_title("Tendência STL com fases (vermelho: crescimento | azul: queda | cinza: estabilidade)")
    ax.set_ylabel("°C"); fig.tight_layout(); _save(fig, "04_tendencia_fases"); return fig


def protocol(meta, dates_target, n_train, n_inner, n_test):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.barh(0, n_train - n_inner, left=0, color="#9ecae1", label=f"treino puro ({n_train - n_inner} origens)")
    ax.barh(0, n_inner, left=n_train - n_inner, color="#fdae6b", label=f"validação interna ({n_inner}) - escolha de hiperparâmetros")
    ax.barh(0, n_test, left=n_train, color="#d62728", alpha=0.8, label=f"teste walk-forward ({n_test}) - uso único")
    ax.set_yticks([]); ax.set_xlabel("índice da origem de previsão (cada origem prevê o dia seguinte)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=3, fontsize=8, frameon=False)
    ax.set_title("Protocolo: janela expansiva, reajuste a cada origem, h = 1 dia"); fig.tight_layout(); _save(fig, "05_protocolo_walkforward"); return fig


def forecasts(df):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["data"], df["real"], "k-o", ms=3.5, lw=1.8, label="Real")
    for m in MODELS + ["Persistência (ref.)"]:
        ax.plot(df["data"], df[m], lw=1.2, ls="--" if "ref" in m else "-", color=COLORS[m], label=m)
    ax.set_ylabel("°C"); ax.legend(ncol=3, fontsize=8); ax.set_title("Previsões 1 dia à frente - teste walk-forward")
    fig.autofmt_xdate(); fig.tight_layout(); _save(fig, "06_previsoes_teste"); return fig


def mae_bars(tab):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = [COLORS.get(m, "#bbbbbb") for m in tab["modelo"]]
    ax.barh(tab["modelo"][::-1], tab["MAE"][::-1], color=cols[::-1])
    for i, v in enumerate(tab["MAE"][::-1]):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("MAE (°C) - menor é melhor"); ax.set_title("MAE fora da amostra (22 previsões)")
    fig.tight_layout(); _save(fig, "07_mae_barras"); return fig


def residual_panels(df):
    fig, ax = plt.subplots(len(MODELS), 3, figsize=(13, 2.6 * len(MODELS)))
    for i, m in enumerate(MODELS):
        e = (df["real"] - df[m]).values
        ax[i, 0].bar(df["data"], e, color=COLORS[m], width=0.8); ax[i, 0].axhline(0, color="k", lw=0.7)
        ax[i, 0].set_title(f"{m}: resíduo no tempo"); ax[i, 0].tick_params(axis="x", labelrotation=45, labelsize=7)
        r = tu.acf(e, 8); n = len(e)
        ax[i, 1].bar(range(1, 9), r[1:], color=COLORS[m]); ax[i, 1].axhline(0, color="k", lw=0.7)
        for s in (1, -1):
            ax[i, 1].axhline(s * 1.96 / np.sqrt(n), color="r", ls="--", lw=0.8)
        ax[i, 1].set_title("ACF dos resíduos (IC 95% ±1,96/√n)"); ax[i, 1].set_ylim(-1, 1)
        ax[i, 2].hist(e, bins=8, color=COLORS[m], alpha=0.8); ax[i, 2].axvline(0, color="k", lw=0.7)
        ax[i, 2].set_title(f"Histograma (média={e.mean():.2f}, dp={e.std(ddof=1):.2f})")
    fig.tight_layout(); _save(fig, "08_residuos_acf"); return fig


def importance_bars(df, title, name, top=12):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    ax.barh(d["feature"], d["delta_MAE"], xerr=d["desvio"], color="#4c72b0")
    ax.axvline(0, color="k", lw=0.7); ax.set_xlabel("aumento do MAE (°C) ao embaralhar a feature")
    ax.set_title(title, fontsize=10); fig.tight_layout(); _save(fig, name); return fig


def mlp_curve(curve):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(curve["epoca"], curve["MAE_treino"], label="MAE treino"); ax.plot(curve["epoca"], curve["MAE_validacao_interna"], label="MAE validação interna")
    ax.set_xlabel("época (Adam, arquitetura selecionada)"); ax.set_ylabel("°C"); ax.legend(); ax.set_title("Curva de aprendizado do MLP")
    fig.tight_layout(); _save(fig, "11_mlp_curva_aprendizado"); return fig
