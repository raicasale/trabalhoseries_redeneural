"""Gera o relatório HTML autocontido a partir de results/ e figures/."""
import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trabalhoseries_redeneural.src import tsutils_delhi as tu

ROOT = tu.ROOT
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
OUT = ROOT / "relatorios" / "relatorio_tecnico_delhi_temp.html"


def table(frame):
    return frame.to_html(index=False, classes="data", border=0, justify="left")


def image(name):
    path = FIGURES / f"{name}.png"
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(name)}"><figcaption>{html.escape(name.replace("_", " ").title())}</figcaption></figure>'


def main():
    predictions = pd.read_csv(RESULTS / "previsoes_walkforward.csv", parse_dates=["data"])
    models = ["Holt-Winters", "SARIMAX", "Random Forest", "MLP Regressor"]
    metrics = pd.DataFrame(
        {
            "Modelo": models + ["Persistência (referência)"],
            "MAE (°C)": [tu.mae(predictions.real, predictions[m]) for m in models + ["Persistência (ref.)"]],
            "RMSE (°C)": [tu.rmse(predictions.real, predictions[m]) for m in models + ["Persistência (ref.)"]],
            "Viés (°C)": [(predictions.real - predictions[m]).mean() for m in models + ["Persistência (ref.)"]],
        }
    ).round(3)
    meta = json.loads((RESULTS / "protocolo.json").read_text(encoding="utf-8"))
    quality = tu.quality_report(tu.load_raw())
    sections = [
        ("1. Identificação e escopo", "<p>Grupo 4, especialização em MLP Regressor, usando a base Daily Delhi Climate. O alvo é <code>meantemp</code>, com horizonte de um dia.</p>"),
        ("2. Dados e qualidade", table(quality) + image("01_series_visao_geral") + "<p>A pressão de 59 hPa foi corrigida por interpolação; não há datas duplicadas, ausentes ou fora de ordem.</p>"),
        ("3. Decomposição STL e sazonalidade", image("03_stl") + "<p>A decomposição semanal é descritiva. Com 114 dias, a sazonalidade anual não é estimável.</p>"),
        ("4. Engenharia de atributos", "<p>São usados lags, janelas móveis, diferenças, exógenas defasadas e calendário do dia-alvo. Nenhuma feature usa informação de t+1.</p>"),
        ("5. Protocolo walk-forward", image("05_protocolo_walkforward") + f"<p>{html.escape(json.dumps(meta, ensure_ascii=False))}</p>"),
        ("6. Modelos comparados", "<p>Holt-Winters, SARIMAX, Random Forest e MLP Regressor são comparados à persistência, à média móvel e ao ingênuo sazonal.</p>"),
        ("7. Otimização e seleção", "<p>A seleção ocorre exclusivamente nas 14 origens de validação interna; os hiperparâmetros ficam fixos durante as 22 previsões do teste.</p>" + image("07_mae_barras")),
        ("8. Resultados fora da amostra", table(metrics) + image("06_previsoes_teste")),
        ("9. Resíduos e Ljung-Box", image("08_residuos_acf") + "<p>Os testes são reportados com cautela: n=22 dá baixo poder estatístico, e nenhum p-valor deve ser interpretado como prova de ausência de dependência.</p>"),
        ("10. Importância das features", image("09_importancia_mlp") + image("10_importancia_rf") + "<p>As importâncias são interpretativas e foram calculadas após a avaliação; não foram usadas para selecionar o modelo.</p>"),
        ("11. Estudo do MLP", image("11_mlp_curva_aprendizado") + "<p>A sensibilidade a sementes, alpha e ablação por grupos está em <code>results/</code> e na documentação completa.</p>"),
        ("12. Limitações e interpretação", "<p>Há uma única base, 22 previsões e modelos próprios de Holt-Winters/SARIMAX. O ranking é instável e não demonstra superioridade estatística sobre a persistência.</p>"),
        ("13. Reprodução e referências", "<p>Execute <code>pip install -r requirements.txt</code>, os testes, <code>src/pipeline_delhi.py</code>, <code>src/build_notebooks_delhi.py</code>, <code>src/build_docs_delhi.py</code> e este gerador. Consulte <code>docs/DOCUMENTACAO_TREINAMENTO.md</code> para tabelas completas e referências.</p>"),
    ]
    body = "\n".join(f"<section><h2>{title}</h2>{content}</section>" for title, content in sections)
    document = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório técnico — Grupo 4 — Daily Delhi Climate</title>
<style>
:root{{font-family:Arial,sans-serif;color:#172033;background:#f5f7fb;line-height:1.5}}
body{{margin:0}} main{{max-width:1100px;margin:auto;background:#fff;padding:32px 48px}}
h1{{font-size:2.2rem;margin-bottom:.3rem}} h2{{color:#243b80;border-bottom:2px solid #dbe3f5;padding-bottom:.3rem}}
section{{break-inside:avoid;margin:28px 0}} code{{background:#eef2f8;padding:2px 5px;border-radius:4px}}
table.data{{border-collapse:collapse;width:100%;font-size:.9rem}} .data th,.data td{{border:1px solid #d8dfeb;padding:7px}} .data th{{background:#edf2fb}}
figure{{margin:18px 0;text-align:center}} figure img{{max-width:100%;height:auto;border:1px solid #dde3ed}} figcaption{{font-size:.8rem;color:#5e6b80}}
.print{{position:fixed;right:20px;top:20px;padding:10px 14px;border:0;border-radius:5px;background:#243b80;color:#fff;cursor:pointer}}
@media print{{body{{background:#fff}} main{{padding:0;max-width:none}} .print{{display:none}} section{{break-inside:avoid}}}}
</style></head><body><button class="print" onclick="window.print()">Imprimir / Salvar PDF</button>
<main><h1>Relatório técnico executivo</h1><p><strong>Grupo 4 · MLP Regressor · Daily Delhi Climate</strong></p>{body}</main></body></html>"""
    OUT.write_text(document, encoding="utf-8")
    print("ok", OUT, len(document))


if __name__ == "__main__":
    main()
