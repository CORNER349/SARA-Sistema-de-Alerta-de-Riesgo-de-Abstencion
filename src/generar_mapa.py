"""Genera el mapa de escenarios (sin vs con intervencion) a partir del semaforo municipal.

Requisitos:  pip install geopandas folium pyogrio pandas

Uso:
    python generar_mapa.py municipios_inegi.geojson semaforo_municipal_2027.csv SALIDA

- Capa municipal con columnas CVE_ENT, CVE_MUN, NOMGEO: puede ser municipios_inegi.geojson
  (incluido, Marco Geoestadistico del INEGI mayo 2021, 2,436 municipios) o el 00mun.shp mas
  reciente de inegi.org.mx -> Geografia -> Marco Geoestadistico (incluye municipios nuevos).
- Cruza por entidad + nombre del municipio (los codigos de municipio del INE y del
  INEGI NO siempre coinciden). Lo que no cruza se reporta en SALIDA/sin_cruce.csv.

Salida en SALIDA/: mapa_escenarios.geojson, mapa_escenarios.gpkg, mapa_escenarios.html
"""
import sys, os, re, unicodedata, difflib
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import DualMap


shp, csv, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\bgral\b\.?", "general", s)
    s = re.sub(r"\bdr\b\.?", "doctor", s)
    return re.sub(r"[^a-z0-9]", "", s)

g = gpd.read_file(shp)
g = g.to_crs(4326)
cols = {c.upper(): c for c in g.columns}
ent, nom = cols["CVE_ENT"], cols.get("NOMGEO") or cols["NOM_MUN"]
mun = cols.get("CVE_MUN")
g["ent"] = g[ent].astype(int)
g["nk"] = g[nom].map(norm)
g["_id"] = range(len(g))

d = pd.read_csv(csv)
d["ent"] = d.EDOCVE.astype(int)
d["nk"] = d.MPIONOM.map(norm)
# nombres que cambiaron entre el INE y el Marco Geoestadistico (agrega los que necesites)
ALIAS = {(23, "solidaridad"): "playadelcarmen"}
d["nk"] = [ALIAS.get((e, k), k) for e, k in zip(d.ent, d.nk)]

# 1) cruce exacto por entidad + nombre normalizado
m = d.merge(g[["_id", "ent", "nk"]].drop_duplicates(["ent", "nk"]), on=["ent", "nk"], how="left")
# 2) lo que falta: nombre muy parecido dentro de la misma entidad
libres = set(g._id) - set(m._id.dropna().astype(int))
for i in m.index[m._id.isna()]:
    cand = g[(g.ent == m.at[i, "ent"]) & g._id.isin(libres)]
    r = difflib.get_close_matches(m.at[i, "nk"], list(cand.nk), n=1, cutoff=0.82)
    if not r:   # un nombre contiene al otro (p. ej. "Cintalapa" / "Cintalapa de Figueroa"), si es unico
        r = [k for k in cand.nk if len(m.at[i, "nk"]) >= 6 and (k.startswith(m.at[i, "nk"]) or m.at[i, "nk"].startswith(k))]
        r = r if len(r) == 1 else []
    if r:
        j = int(cand[cand.nk == r[0]]._id.iloc[0]); m.at[i, "_id"] = j; libres.discard(j)

sin = m[m._id.isna()]
sin[["EDONOM", "MPIONOM", "LN", "semaforo_sin"]].sort_values("LN", ascending=False).to_csv(
    os.path.join(out, "sin_cruce.csv"), index=False, encoding="utf-8-sig")
print(f"Municipios del modelo: {len(m)} | con poligono: {m._id.notna().sum()} | "
      f"lista nominal cubierta: {100*m.LN[m._id.notna()].sum()/m.LN.sum():.1f}%")
print(f"Sin cruce: {len(sin)} (ver sin_cruce.csv). Revisa los de mayor LN y corrige nombres si hace falta.")

m = m[m._id.notna()].copy(); m["_id"] = m._id.astype(int)
keep = ["EDONOM", "MPIONOM", "LN", "abst_sin", "semaforo_sin", "abst_con", "semaforo_con", "votos_extra", "_id"]
geo = g[["_id", "geometry"]].merge(m[keep], on="_id").drop(columns="_id")
geo["geometry"] = geo.geometry.simplify(0.003, preserve_topology=True)   # archivo mas ligero
for c in ("abst_sin", "abst_con"): geo[c] = geo[c].round(4)
geo["votos_extra"] = geo.votos_extra.round(0)
geo.to_file(os.path.join(out, "mapa_escenarios.geojson"), driver="GeoJSON")
geo.to_file(os.path.join(out, "mapa_escenarios.gpkg"), driver="GPKG")

COL = {"Verde": "#2e9e5b", "Amarillo": "#f2c230", "Rojo": "#d64545"}
def capa(mapa, campo, abst, titulo):
    folium.GeoJson(
        geo.to_json(), name=titulo,
        style_function=lambda f, c=campo: {"fillColor": COL.get(f["properties"][c], "#999"),
                                           "color": "#555", "weight": 0.3, "fillOpacity": 0.8},
        tooltip=folium.GeoJsonTooltip(
            fields=["MPIONOM", "EDONOM", abst, campo, "votos_extra"],
            aliases=["Municipio", "Estado", "Abstencion proyectada 2027", "Semaforo", "Votantes adicionales"]),
    ).add_to(mapa)
dm = DualMap(location=[23.6, -102.5], zoom_start=5, tiles=None)
capa(dm.m1, "semaforo_sin", "abst_sin", "Sin intervencion")
capa(dm.m2, "semaforo_con", "abst_con", "Con intervencion")
leyenda = ('<div style="position:fixed;top:10px;left:50px;z-index:9999;background:white;padding:8px 12px;'
           'font:13px sans-serif;border-radius:6px">Izquierda: <b>sin intervencion</b> | Derecha: <b>con intervencion</b> '
           '(hipotetica)<br><span style="color:#2e9e5b">&#9632;</span> Verde &lt;45% &nbsp;'
           '<span style="color:#d4a017">&#9632;</span> Amarillo 45-55% &nbsp;'
           '<span style="color:#d64545">&#9632;</span> Rojo &gt;55%</div>')
dm.get_root().html.add_child(folium.Element(leyenda))
dm.save(os.path.join(out, "mapa_escenarios.html"))
print("Listo ->", out)
