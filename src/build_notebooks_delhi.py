"""Gera e executa os quatro notebooks Delhi, com saídas reais embutidas."""
import sys, importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "nb"))
import trabalhoseries_redeneural.src.nbexec_delhi as nbexec

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"01": "01_dados_e_STL_Delhi.ipynb", "02": "02_features_protocolo_otimizacao_Delhi.ipynb",
         "03": "03_resultados_residuos_Delhi.ipynb", "04": "04_estudo_MLP_e_importancia_Delhi.ipynb"}
sel = sys.argv[1:] or list(NAMES)
for k in sel:
    mod = importlib.import_module(f"nb{k}_delhi")
    nbexec.run(mod.CELLS, str(ROOT / "notebooks" / NAMES[k]), str(ROOT / "notebooks"))
    print("ok", NAMES[k])
