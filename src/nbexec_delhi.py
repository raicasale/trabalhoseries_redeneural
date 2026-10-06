"""Mini-executor de notebooks (sem Jupyter): executa as células em um namespace único, captura stdout,
`display(...)` e figuras do matplotlib e grava um .ipynb (nbformat 4.4) com as saídas REAIS.
Os notebooks gerados abrem normalmente no Jupyter/VS Code e podem ser reexecutados lá."""
import ast, base64, io, json, os, sys, traceback, contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(True)}


def _lines(s):
    return s.splitlines(True)


def run(cells, out_path, cwd):
    os.chdir(cwd)
    ns = {"__name__": "__main__"}
    count = 0
    for idx, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        count += 1
        outs = []

        def display(*objs):
            for o in objs:
                if isinstance(o, matplotlib.figure.Figure):
                    continue
                if isinstance(o, pd.DataFrame):
                    outs.append({"output_type": "display_data", "metadata": {},
                                 "data": {"text/html": _lines(o.to_html(border=0, float_format=lambda v: f"{v:.4g}")),
                                          "text/plain": _lines(o.to_string())}})
                elif isinstance(o, pd.Series):
                    display(o.to_frame())
                else:
                    outs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": _lines(repr(o))}})

        ns["display"] = display
        src = "".join(cell["source"])
        buf = io.StringIO()
        try:
            tree = ast.parse(src)
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {idx}>", "exec"), ns)
                if last is not None:
                    v = eval(compile(last, f"<cell {idx}>", "eval"), ns)
                    if v is not None:
                        display(v)
        except Exception:
            print(src); traceback.print_exc()
            raise SystemExit(f"Falha na célula {idx} de {out_path}")
        text = buf.getvalue()
        if text:
            outs.insert(0, {"output_type": "stream", "name": "stdout", "text": _lines(text)})
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO(); fig.savefig(b, format="png", dpi=100, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {},
                         "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": ["<Figure>"]}})
        plt.close("all")
        cell["outputs"] = outs
        cell["execution_count"] = count
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
