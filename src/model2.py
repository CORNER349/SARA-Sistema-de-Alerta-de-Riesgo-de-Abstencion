"""Predicción territorial de abstención con validación temporal.

Ejecutar desde la raíz del repositorio: python src/model2.py
Requiere los archivos agg_2009.csv ... agg_2024.csv definidos en prep.py.

Las características de cada elección t provienen únicamente de t-1 y t-2.
El nivel nacional N[t] se usa SOLO para definir la variable residual observada
al evaluar retrospectivamente. Para pronósticos futuros, final.py fija un
escenario nacional explícito (NAT27).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prep import YRS, load, nat

fr = load()
N = nat(fr)
MID = {2009, 2015, 2021, 2027}

for year in YRS:
    frame = fr[year]
    frame["r"] = frame["abst"] - N[year]
    frame["mkey"] = (
        frame["EDOCVE"].astype(str).str.zfill(2)
        + "-"
        + frame["MPIOCVE"].astype(str).str.zfill(3)
    )
    frame["nsr"] = frame["NS"] / frame["LN"].clip(lower=1)


def munr(year):
    """Desviación municipal histórica, ponderada por población elegible."""
    grouped = fr[year].groupby("mkey").agg(nv=("NV", "sum"), b=("base", "sum"))
    return (grouped["nv"] / grouped["b"].replace(0, np.nan) - N[year]).rename("mr")


M = {year: munr(year) for year in YRS}


def build(t, tgt=True):
    """Construye características con información anterior a la elección t.

    tgt se conserva por compatibilidad. Cuando t es histórico, incluye las
    etiquetas observadas r y abst para evaluación, pero NO se usan en X().
    """
    if t in YRS:
        idx = YRS.index(t)
    elif t > YRS[-1]:
        idx = len(YRS)
    else:
        raise ValueError(f"Año no soportado: {t}")
    if idx < 2:
        raise ValueError("Se necesitan dos elecciones anteriores")

    p, q = YRS[idx - 1], YRS[idx - 2]
    # El universo territorial es el de t para evaluar retrospectivamente;
    # para el futuro es el de la última elección disponible. La presencia
    # de una sección nueva no equivale a conocer sus datos futuros.
    current = fr[t] if t in YRS else fr[YRS[-1]]
    d = current[["key", "mkey", "EDOCVE", "MPIOCVE"]].copy()
    if t in YRS:
        d["r"] = current["r"].to_numpy()
        d["abst"] = current["abst"].to_numpy()
        d["base_eval"] = current["base"].to_numpy()
        d["nsr_eval"] = current["nsr"].to_numpy()

    # Variables explicativas SOLO de la elección inmediatamente anterior.
    lag_cols = ["key", "LN", "base", "NS", "nsr", "jov", "old", "fem", "TIPOSEC"]
    d = d.merge(fr[p][lag_cols], on="key", how="left", validate="one_to_one")
    # Secciones nuevas: imputación conservadora a partir del último municipio.
    municipal = fr[p].groupby("mkey").agg(
        LN_m=("LN", "sum"), base_m=("base", "sum"),
        NS_m=("NS", "sum"), jov_m=("LN_18_29", "sum"),
        old_m=("LN_60mas", "sum"), fem_m=("LN_M", "sum"),
    )
    municipal["jov_m"] /= municipal["LN_m"].clip(lower=1)
    municipal["old_m"] /= municipal["LN_m"].clip(lower=1)
    municipal["fem_m"] /= municipal["LN_m"].clip(lower=1)
    municipal["nsr_m"] = municipal["NS_m"] / municipal["LN_m"].clip(lower=1)
    for col in ("jov", "old", "fem", "nsr"):
        d[col] = d[col].fillna(d["mkey"].map(municipal[f"{col}_m"]))
        d[col] = d[col].fillna(fr[p][col].median())
    # Para una sección sin historia no hay estimación defendible de su LN;
    # el valor del último municipio no es un sustituto de LN seccional.
    # En evaluación se conserva base real únicamente como peso; en 2027
    # se mantiene la base 2024 para secciones que sí existen en 2024.
    d["base"] = d["base"].fillna(0)
    d["LN"] = d["LN"].fillna(0)
    d["NS"] = d["NS"].fillna(0)
    d["TIPOSEC"] = d["TIPOSEC"].fillna("?")

    for suffix, year in (("1", p), ("2", q)):
        lag = fr[year][["key", "r"]].rename(columns={"r": "rs" + suffix})
        d = d.merge(lag, on="key", how="left", validate="one_to_one")
        d["rm" + suffix] = d["mkey"].map(M[year])
        d["has" + suffix] = d["rs" + suffix].notna().astype(int)
        d["rm" + suffix] = d["rm" + suffix].fillna(0.0)
        d["rs" + suffix] = d["rs" + suffix].fillna(d["rm" + suffix])

    d["mid"] = int(t in MID)
    d["t"] = t
    return d


def X(d):
    """Matriz de variables sin columnas del año objetivo."""
    return pd.DataFrame({
        "rs1": d["rs1"], "rs2": d["rs2"],
        "rm1": d["rm1"], "rm2": d["rm2"],
        "jov": d["jov"], "old": d["old"], "fem": d["fem"],
        "mid": d["mid"], "mid_jov": d["mid"] * d["jov"],
        "mid_rm1": d["mid"] * d["rm1"],
        "mid_rs1": d["mid"] * d["rs1"],
        "has1": d["has1"], "has2": d["has2"],
        "rural": d["TIPOSEC"].eq("R").astype(int),
        "mixta": d["TIPOSEC"].eq("M").astype(int),
    }, index=d.index).replace([np.inf, -np.inf], np.nan).fillna(0)


TR = {t: build(t) for t in [2015, 2018, 2021, 2024]}


def ok(d):
    """Calidad de la etiqueta observada; nunca filtrar el futuro con datos futuros."""
    if "base_eval" in d:
        return (d["nsr_eval"] < 0.5) & (d["base_eval"] >= 30) & d["r"].notna()
    return (d["nsr"] < 0.5) & (d["base"] >= 30)


def fit(years):
    if not years or any(y not in TR for y in years):
        raise ValueError("Años de entrenamiento inválidos")
    D = pd.concat([TR[t] for t in years], ignore_index=True)
    D = D.loc[ok(D)].copy()
    if D.empty:
        raise ValueError("No hay observaciones válidas para entrenar")
    features, target = X(D), D["r"]
    weights = D["base_eval"]
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(features, target, ridge__sample_weight=weights)
    gb = HistGradientBoostingRegressor(
        max_depth=4, max_iter=250, learning_rate=0.06,
        min_samples_leaf=100, early_stopping=False, random_state=0,
    )
    gb.fit(features, target, sample_weight=weights)
    return ridge, gb


def agg(d, column):
    """Media municipal ponderada por base electoral de evaluación."""
    weight_col = "base_eval" if "base_eval" in d else "base"
    valid = d.loc[d[column].notna() & d[weight_col].gt(0)].copy()
    valid["weighted"] = valid[weight_col] * valid[column]
    grouped = valid.groupby("mkey").agg(
        weighted=("weighted", "sum"), total=(weight_col, "sum")
    )
    return grouped["weighted"] / grouped["total"], grouped["total"]


def cls(values):
    return pd.cut(values, [-np.inf, 0.40, 0.50, np.inf], labels=["V", "A", "R"])


if __name__ == "__main__":
    for test, train in [(2021, [2015, 2018]), (2024, [2015, 2018, 2021])]:
        ridge, gb = fit(train)
        d = TR[test].loc[ok(TR[test])].copy()
        if d.empty:
            print(f"Sin observaciones válidas para {test}")
            continue
        d["p_ridge"] = ridge.predict(X(d))
        d["p_gb"] = gb.predict(X(d))
        d["p_pers"] = d["rs1"]
        d["p_mun1"] = d["rm1"]
        d["p_mun12"] = (d["rm1"] + d["rm2"]) / 2
        print(f"\n== Evaluación {test}: {len(d):,} secciones válidas ==")
        actual, municipal_weight = agg(d, "r")
        for col in ["p_pers", "p_mun1", "p_mun12", "p_ridge", "p_gb"]:
            pred, _ = agg(d, col)
            common = actual.index.intersection(pred.index)
            a, p, w = actual.loc[common], pred.loc[common], municipal_weight.loc[common]
            mae_sec = np.average(np.abs(d[col] - d["r"]), weights=d["base_eval"])
            mae_mun = np.average(np.abs(p - a), weights=w)
            corr = p.corr(a) if len(common) > 1 and p.std() > 0 and a.std() > 0 else np.nan
            agreement = np.average((cls(p + N[test]) == cls(a + N[test])).astype(float), weights=w)
            print(f"{col:8s} | MAE sección {mae_sec * 100:.2f} pp "
                  f"| MAE municipio {mae_mun * 100:.2f} pp "
                  f"| correlación {corr:.3f} | acuerdo semáforo {agreement:.3f}")
