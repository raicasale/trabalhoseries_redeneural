"""Auditoria de vazamento: alterar o FUTURO de uma origem t não pode mudar as features dessa origem."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
import trabalhoseries_redeneural.src.tsutils_delhi as tu


def test_features_nao_dependem_do_futuro():
    D = tu.clean(tu.load_raw())
    X0, _, _, _ = tu.build_features(D, w=14)
    rng = np.random.default_rng(0)
    for origem_pos in (30, 60, 90):                    # posição da origem t em D
        D2 = D.copy()
        fut = D2.index > origem_pos
        for c in [tu.TARGET] + tu.EXOG:                # embaralha o futuro (> t) de alvo e exógenas
            D2.loc[fut, c] = rng.permutation(D2.loc[fut, c].values) + 1000.0
        X1, _, _, _ = tu.build_features(D2, w=14)
        assert np.allclose(X0.loc[origem_pos].values, X1.loc[origem_pos].values), origem_pos
        antes = X0.index <= origem_pos
        assert np.allclose(X0.loc[antes].values, X1.loc[antes].values)


def test_mesmas_origens_para_todo_w():
    D = tu.clean(tu.load_raw())
    idx = [tu.build_features(D, w=w)[0].index.tolist() for w in (3, 7, 14)]
    assert idx[0] == idx[1] == idx[2]


def test_alvo_e_o_dia_seguinte():
    D = tu.clean(tu.load_raw())
    X, y, ycur, tdate = tu.build_features(D, w=7)
    for i in (0, 10, 50, len(X) - 1):
        t = X.index[i]
        assert y.iloc[i] == D.loc[t + 1, tu.TARGET] and ycur.iloc[i] == D.loc[t, tu.TARGET]
        assert tdate.iloc[i] == D.loc[t + 1, "date"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK ", n)
