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
import sys, os, re, unicodedata, difflib, html, json
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

# Cruce 1:1. Los nombres repetidos no se asignan arbitrariamente.
m = d.copy()
m["_id"] = pd.NA
m["metodo_cruce"] = "sin_cruce"
conteo_d = d.groupby(["ent", "nk"])["nk"].transform("size")
conteo_g = g.groupby(["ent", "nk"])["nk"].transform("size")
geo_unicos = g.loc[conteo_g.eq(1), ["_id", "ent", "nk"]]
mapa_unicos = geo_unicos.set_index(["ent", "nk"])["_id"].to_dict()
for i in m.index[conteo_d.eq(1)]:
    clave = (int(m.at[i, "ent"]), m.at[i, "nk"])
    if clave in mapa_unicos:
        m.at[i, "_id"] = int(mapa_unicos[clave])
        m.at[i, "metodo_cruce"] = "nombre_exacto"

# Para nombres homonimos, probar codigos SOLO cuando el nombre tambien coincide.
# No se supone que MPIOCVE y CVE_MUN siempre sean equivalentes.
if mun is not None and "MPIOCVE" in m.columns:
    m["codigo_ine"] = pd.to_numeric(m["MPIOCVE"], errors="coerce")
    g["codigo_inegi"] = pd.to_numeric(g[mun], errors="coerce")
    duplicados_d = conteo_d.gt(1)
    duplicados_g = conteo_g.gt(1)
    for i in m.index[duplicados_d & m["_id"].isna()]:
        codigo = m.at[i, "codigo_ine"]
        if pd.isna(codigo):
            continue
        posibles = g.loc[
            duplicados_g & g["ent"].eq(m.at[i, "ent"])
            & g["nk"].eq(m.at[i, "nk"])
            & g["codigo_inegi"].eq(codigo)
        ]
        if len(posibles) == 1:
            candidato = int(posibles.iloc[0]["_id"])
            # Asignar solo si no hay otro registro INE con la misma clave.
            coincidencias_ine = m.loc[
                m["ent"].eq(m.at[i, "ent"]) & m["nk"].eq(m.at[i, "nk"])
                & m["codigo_ine"].eq(codigo)
            ]
            if len(coincidencias_ine) == 1 and candidato not in set(m["_id"].dropna().astype(int)):
                m.at[i, "_id"] = candidato
                m.at[i, "metodo_cruce"] = "codigo_y_nombre_homonimo"

# Cruce aproximado solo para nombres no ambiguos, con un candidato inequívoco.
libres = set(g["_id"]) - set(m["_id"].dropna().astype(int))
claves_ambiguas_d = set(zip(d.loc[conteo_d.gt(1), "ent"], d.loc[conteo_d.gt(1), "nk"]))
claves_ambiguas_g = set(zip(g.loc[conteo_g.gt(1), "ent"], g.loc[conteo_g.gt(1), "nk"]))
for i in m.index[m["_id"].isna()]:
    clave = (int(m.at[i, "ent"]), m.at[i, "nk"])
    if clave in claves_ambiguas_d or clave in claves_ambiguas_g:
        m.at[i, "metodo_cruce"] = "nombre_ambiguo"
        continue
    cand = g.loc[g["ent"].eq(m.at[i, "ent"]) & g["_id"].isin(libres)].copy()
    if cand.empty:
        continue
    claves_candidatas = pd.MultiIndex.from_frame(cand[["ent", "nk"]])
    mascara_ambiguos = claves_candidatas.isin(claves_ambiguas_g)
    cand = cand.loc[~mascara_ambiguos]
    if cand.empty:
        continue
    objetivo = m.at[i, "nk"]
    cand["similitud"] = cand["nk"].map(lambda nombre: difflib.SequenceMatcher(None, objetivo, nombre).ratio())
    cand["compatible"] = cand["similitud"].ge(0.82) | cand["nk"].map(
        lambda nombre: len(objetivo) >= 6 and (nombre.startswith(objetivo) or objetivo.startswith(nombre))
    )
    opciones = cand.loc[cand["compatible"]].sort_values("similitud", ascending=False)
    # Si hay mas de una coincidencia plausible, no arriesgar una asignacion.
    if len(opciones) != 1:
        continue
    candidato = int(opciones.iloc[0]["_id"])
    # No aceptar que otro registro sin asignar sea un candidato igual de plausible.
    otros = m.loc[m.index != i]
    if any(
        int(row.ent) == int(m.at[i, "ent"]) and
        (difflib.SequenceMatcher(None, row.nk, opciones.iloc[0]["nk"]).ratio() >= 0.82)
        for row in otros.loc[otros["_id"].isna()].itertuples()
    ):
        continue
    m.at[i, "_id"] = candidato
    m.at[i, "metodo_cruce"] = "nombre_aproximado"
    libres.remove(candidato)

if m["_id"].dropna().duplicated().any():
    raise ValueError("Cruce invalido: un poligono fue asignado a varios municipios")

m.loc[m["_id"].isna(), ["EDONOM", "MPIONOM", "metodo_cruce"]].to_csv(
    os.path.join(out, "ambiguos_y_pendientes.csv"), index=False, encoding="utf-8-sig"
)

sin = m[m._id.isna()]
sin[["EDONOM", "MPIONOM", "LN", "semaforo_sin"]].sort_values("LN", ascending=False).to_csv(
    os.path.join(out, "sin_cruce.csv"), index=False, encoding="utf-8-sig")
print(f"Municipios del modelo: {len(m)} | con poligono: {m._id.notna().sum()} | "
      f"lista nominal cubierta: {100*m.LN[m._id.notna()].sum()/m.LN.sum():.1f}%")
print(f"Sin cruce: {len(sin)} (ver sin_cruce.csv). Revisa los de mayor LN y corrige nombres si hace falta.")


# Un solo cruce cartográfico para todos los escenarios.
base = m.loc[m['_id'].notna(), ['mkey','_id']].copy()
base['_id'] = base['_id'].astype(int)
geom = g[['_id', 'geometry']].merge(base, on='_id', validate='one_to_one')
geom['geometry'] = geom.geometry.simplify(0.003, preserve_topology=True)

escenarios = {}
for nombre in ('conservador', 'base', 'optimista'):
    ruta = os.path.join('out', 'escenarios', nombre, 'semaforo_municipal_2027.csv')
    # Se resuelve relativo al CSV base, cuyo padre habitual es out/.
    carpeta_out = os.path.dirname(os.path.abspath(csv))
    ruta = os.path.join(carpeta_out, 'escenarios', nombre, 'semaforo_municipal_2027.csv')
    if not os.path.exists(ruta):
        raise FileNotFoundError(f'No existe {ruta}. Ejecuta src/final.py primero.')
    df = pd.read_csv(ruta)
    if len(df) != len(d) or df['mkey'].duplicated().any() or set(df.mkey) != set(d.mkey):
        raise ValueError(f'Claves municipales diferentes en escenario {nombre}')
    escenarios[nombre] = df

# Guardar GIS del escenario base con porcentajes numéricos originales.
geo_base = geom.merge(escenarios['base'][['mkey','EDONOM','MPIONOM','LN','abst_sin','semaforo_sin',
                                       'abst_con','semaforo_con','votos_extra']], on='mkey', validate='one_to_one')
geo_base.to_file(os.path.join(out, 'mapa_escenarios.geojson'), driver='GeoJSON')
geo_base.to_file(os.path.join(out, 'mapa_escenarios.gpkg'), driver='GPKG')

COL = {'Verde': '#2e9e5b', 'Amarillo': '#f2c230', 'Rojo': '#d64545'}
dm = DualMap(location=[23.6, -102.5], zoom_start=5, tiles=None)
capas = {}
resumen = {}
for nombre, df in escenarios.items():
    geo = geom.merge(df[['mkey','EDONOM','MPIONOM','LN','abst_sin','semaforo_sin',
                         'abst_con','semaforo_con','votos_extra']], on='mkey', validate='one_to_one')
    geo['abst_sin_pct'] = (geo.abst_sin * 100).map(lambda v: f'{v:.2f} %')
    geo['abst_con_pct'] = (geo.abst_con * 100).map(lambda v: f'{v:.2f} %')
    geo['votos_extra_fmt'] = geo.votos_extra.fillna(0).astype(int).map(lambda v: f'{v:,}')
    resumen[nombre] = {
        'rojos_sin': int((df.semaforo_sin == 'Rojo').sum()),
        'rojos_con': int((df.semaforo_con == 'Rojo').sum()),
        'votos': int(df.votos_extra.fillna(0).sum()),
        'nacional': {'conservador':52.0,'base':48.2,'optimista':44.0}[nombre]
    }
    capas[nombre] = []
    for mapa, lado in ((dm.m1, 'sin'), (dm.m2, 'con')):
        campo = 'semaforo_' + lado
        capa = folium.GeoJson(
            geo.to_json(), name=f'{nombre}: {lado}',
            style_function=lambda f, c=campo: {
                'fillColor': COL.get(f['properties'][c], '#999999'),
                'color': '#555555', 'weight': .3, 'fillOpacity': .8},
            tooltip=folium.GeoJsonTooltip(
                fields=['MPIONOM','EDONOM','abst_'+lado+'_pct',campo,'votos_extra_fmt'],
                aliases=['Municipio','Estado','Abstención estimada 2027','Semáforo','Votos adicionales hipotéticos'],
                sticky=True, labels=True),
            show=(nombre == 'base')
        ).add_to(mapa)
        capas[nombre].append(capa.get_name())

# El HTML contiene los tres escenarios: no necesita servidor ni CSV externos.
mapas_js = [dm.m1.get_name(), dm.m2.get_name()]
js = '''
<script>
(function(){
 const maps = MAPAS;
 const layers = CAPAS;
 const stats = STATS;
 const fmt = n => Number(n).toLocaleString('es-MX');
 function update(){
   const chosen = document.getElementById('ine-escenario').value;
   Object.keys(layers).forEach(k => layers[k].forEach((layer, idx) => {
      if(k === chosen) { if(!maps[idx].hasLayer(layer)) maps[idx].addLayer(layer); }
      else { if(maps[idx].hasLayer(layer)) maps[idx].removeLayer(layer); }
   }));
   const s = stats[chosen];
   document.getElementById('ine-estadisticas').innerHTML =
     '<b>Abstención nacional supuesta:</b> '+s.nacional.toFixed(1)+' % &nbsp; | &nbsp; '+
     '<b>Municipios rojos:</b> '+fmt(s.rojos_sin)+' → '+fmt(s.rojos_con)+' &nbsp; | &nbsp; '+
     '<b>Votos adicionales hipotéticos:</b> '+fmt(s.votos);
 }
 document.getElementById('ine-escenario').addEventListener('change', update);
 update();
})();
</script>
'''
js = js.replace('MAPAS', '['+', '.join(mapas_js)+']')
js = js.replace('CAPAS', '{'+','.join(json.dumps(k)+':['+', '.join(v)+']' for k,v in capas.items())+'}')
js = js.replace('STATS', json.dumps(resumen, ensure_ascii=False))
# IMPORTANTE: NO insertar JS en root.script: Folium puede colocarlo antes
# de que existan las variables de Leaflet y las capas, dejando el mapa en blanco.
control = '''<div style="position:fixed;top:12px;left:50px;z-index:9999;background:white;
 padding:12px 16px;font:14px Arial,sans-serif;border:1px solid #bbb;border-radius:8px;
 box-shadow:0 2px 8px #7775;max-width:750px;">
 <b>INE · Escenarios territoriales 2027</b> &nbsp;
 <label for="ine-escenario">Escenario:</label>
 <select id="ine-escenario" style="padding:5px;font-size:14px;">
 <option value="conservador">Conservador</option><option value="base" selected>Base</option>
 <option value="optimista">Optimista</option></select>
 <div style="margin-top:7px"><b>Izquierda:</b> sin intervención &nbsp; | &nbsp;
 <b>Derecha:</b> con intervención hipotética</div>
 <div id="ine-estadisticas" style="margin-top:6px"></div>
 <div style="margin-top:7px"><span style="color:#2e9e5b">■</span> Verde &lt;45% &nbsp;
 <span style="color:#c8a000">■</span> Amarillo 45–55% &nbsp;
 <span style="color:#d64545">■</span> Rojo &gt;55%</div>
 <small>Simulación no causal. Cobertura cartográfica: COBERTURA de TOTAL municipios del modelo.</small>
 </div>'''.replace('COBERTURA',str(len(geom))).replace('TOTAL',str(len(d)))
dm.get_root().html.add_child(folium.Element(control))
# Folium coloca la inicialización de Leaflet DESPUÉS de </body>.
# Por eso el selector debe insertarse antes de </html>, no antes de </body>.
ruta_html = os.path.join(out, 'mapa_escenarios.html')
dm.save(ruta_html)
with open(ruta_html, 'r', encoding='utf-8') as archivo:
    contenido = archivo.read()
if '</html>' not in contenido:
    raise RuntimeError('No se encontró </html> en el HTML generado por Folium')
contenido = contenido.replace('</html>', js + '\n</html>', 1)
with open(ruta_html, 'w', encoding='utf-8') as archivo:
    archivo.write(contenido)
print('Mapa con selector generado:',os.path.join(out,'mapa_escenarios.html'))
print('Cobertura cartográfica:',len(geom),'de',len(d),'municipios')