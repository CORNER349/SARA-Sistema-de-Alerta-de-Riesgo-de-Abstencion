"""Agrega los CSV de Conteos Censales (un CSV por entidad) a una fila por seccion.

Uso:
    python agregar_ccpc.py CARPETA_CON_CSV SALIDA.csv
Ejemplo:
    python agregar_ccpc.py DatosAbiertos_DECEyEC_ConteosCensales2021 agg_2021.csv
"""
import os
import sys
import glob
import pandas as pd

carpeta, salida = sys.argv[1], sys.argv[2]
archivos = sorted(glob.glob(os.path.join(carpeta, "**", "*.csv"), recursive=True))
if not archivos:
    sys.exit("No encontre CSV en " + carpeta)

GRUPOS = {"18_29": (18, 29), "30_44": (30, 44), "45_59": (45, 59), "60mas": (60, 200)}
llaves = ["AELEC", "EDOCVE", "EDONOM", "MPIOCVE", "MPIONOM", "SECCION", "TIPOSEC", "DEF", "DEL"]
partes = []

for f in archivos:
    d = pd.read_csv(f, encoding="utf-8-sig")
    d.columns = [c.strip().upper() for c in d.columns]
    llaves_ok = [k for k in llaves if k in d.columns]
    for c in ("LN", "SV", "NV", "NS"):
        d[c] = d[c].fillna(0)
    out = d.groupby(llaves_ok, as_index=False)[["LN", "SV", "NV", "NS"]].sum()
    # desgloses por grupo de edad y por sexo (LN y NV, para tasas de abstencion)
    for nombre, (a, b) in GRUPOS.items():
        m = d["EDAD"].between(a, b)
        g = d[m].groupby(llaves_ok)[["LN", "NV", "NS"]].sum().add_suffix("_" + nombre)
        out = out.merge(g, on=llaves_ok, how="left")
    for nombre, v in (("H", 0), ("M", 1)):
        g = d[d["SEXO"] == v].groupby(llaves_ok)[["LN", "NV", "NS"]].sum().add_suffix("_" + nombre)
        out = out.merge(g, on=llaves_ok, how="left")
    partes.append(out.fillna(0))
    print("ok", os.path.basename(f), len(out), "secciones")

res = pd.concat(partes, ignore_index=True)
# abstencion = NV / (LN - NS); participacion = SV / (LN - NS)
base = (res["LN"] - res["NS"]).clip(lower=1)
res["ABSTENCION"] = res["NV"] / base
res["PARTICIPACION"] = res["SV"] / base
res.to_csv(salida, index=False, encoding="utf-8")
print("Listo:", salida, res.shape)