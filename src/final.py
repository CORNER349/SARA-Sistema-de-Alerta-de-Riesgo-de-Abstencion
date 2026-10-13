"""Análisis de sensibilidad 2027. Ejecutar: python src/final.py

Tres hipótesis ilustrativas, no pronósticos ni impactos causales.
El escenario Base conserva las rutas CSV usadas por el mapa existente.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from model2 import fr, fit, build, X

OUT = Path('out')
OUT.mkdir(exist_ok=True)
SCENARIOS = {
    'Conservador': (0.52, 0.02),
    'Base': (0.482, 0.06),
    'Optimista': (0.44, 0.10),
}

ridge, gb = fit([2015, 2018, 2021, 2024])
d = build(2027)
d['rhat'] = (ridge.predict(X(d)) + gb.predict(X(d)) + (d['rm1'] + d['rm2']) / 2) / 3
if d['base'].sum() <= 0:
    raise ValueError('La base electoral total debe ser positiva')
d['rhat'] -= np.average(d['rhat'], weights=d['base'])
d = d.merge(fr[2024][['key', 'MPIONOM', 'EDONOM', 'DEF', 'LN_18_29']],
            on='key', how='left', validate='one_to_one')
if d[['DEF', 'LN_18_29']].isna().any().any():
    raise ValueError('Faltan datos territoriales o de población juvenil para 2024')


def semaforo(values):
    return pd.cut(values, [-np.inf, 0.45, 0.55, np.inf],
                  labels=['Verde', 'Amarillo', 'Rojo'])


def calibrar(nacional):
    """Ajusta la media ponderada exactamente al supuesto, incluso tras clip."""
    lo, hi = -2.0, 2.0
    for _ in range(65):
        mid = (lo + hi) / 2
        p = (d['rhat'] + mid).clip(0.02, 0.98)
        if np.average(p, weights=d['base']) < nacional:
            lo = mid
        else:
            hi = mid
    return (d['rhat'] + (lo + hi) / 2).clip(0.02, 0.98)


def tabla(datos, keys):
    t = datos.groupby(keys, dropna=False).agg(
        EDONOM=('EDONOM', 'first'), LN=('LN', 'sum'),
        base=('base', 'sum'), nv=('nv27', 'sum'),
        jov=('LN_18_29', 'sum'), MPIONOM=('MPIONOM', 'first'),
        DEFn=('DEF', 'first'), votos_extra=('votos_extra', 'sum')
    ).reset_index()
    t['abst_sin'] = t['nv'] / t['base'].replace(0, np.nan)
    t['semaforo_sin'] = semaforo(t['abst_sin'])
    t['abst_con'] = (t['nv'] - t['votos_extra']) / t['base'].replace(0, np.nan)
    t['semaforo_con'] = semaforo(t['abst_con'])
    return t


def ejecutar(nombre, nacional, delta):
    datos = d.copy()
    datos['abst27'] = calibrar(nacional)
    datos['nv27'] = datos['abst27'] * datos['base']
    municipios = datos.groupby(['EDOCVE', 'mkey'], dropna=False).agg(
        base=('base', 'sum'), nv=('nv27', 'sum'))
    municipios['intervenir'] = municipios['nv'] / municipios['base'].replace(0, np.nan) > 0.55
    datos['intervenir'] = pd.MultiIndex.from_frame(datos[['EDOCVE', 'mkey']]).map(
        municipios['intervenir']).fillna(False).to_numpy(dtype=bool)
    datos['votos_extra'] = np.minimum(
        np.where(datos['intervenir'], delta * datos['LN_18_29'], 0.0),
        datos['nv27'])
    mun = tabla(datos, ['EDOCVE', 'mkey']).sort_values('nv', ascending=False)
    distrito = tabla(datos, ['EDOCVE', 'DEF'])
    if not np.isclose(mun['votos_extra'].sum(), distrito['votos_extra'].sum()):
        raise AssertionError(f'Votos adicionales inconsistentes: {nombre}')
    if not np.isclose(mun['nv'].sum(), distrito['nv'].sum()):
        raise AssertionError(f'Abstenciones inconsistentes: {nombre}')

    folder = OUT / 'escenarios' / nombre.lower()
    folder.mkdir(parents=True, exist_ok=True)
    mun.to_csv(folder / 'semaforo_municipal_2027.csv', index=False, encoding='utf-8-sig')
    distrito.to_csv(folder / 'semaforo_distrito_federal_2027.csv', index=False, encoding='utf-8-sig')
    if nombre == 'Base':
        # Compatibilidad con src/generar_mapa.py y otros scripts existentes.
        mun.to_csv(OUT / 'semaforo_municipal_2027.csv', index=False, encoding='utf-8-sig')
        distrito.to_csv(OUT / 'semaforo_distrito_federal_2027.csv', index=False, encoding='utf-8-sig')
        datos[['key', 'EDOCVE', 'EDONOM', 'MPIOCVE', 'MPIONOM', 'DEF',
               'LN', 'jov', 'old', 'fem', 'TIPOSEC', 'abst27']].round(4).to_csv(
            OUT / 'riesgo_seccion_2027.csv', index=False, encoding='utf-8-sig')

    n_votos = datos['votos_extra'].sum()
    base_total = datos['base'].sum()
    resumen = {
        'escenario': nombre,
        'abst_nacional_supuesta': nacional,
        'delta_juvenil_supuesto': delta,
        'abst_sin': datos['nv27'].sum() / base_total,
        'abst_con': (datos['nv27'].sum() - n_votos) / base_total,
        'votos_extra': n_votos,
        'municipios_rojos_sin': int((mun['semaforo_sin'] == 'Rojo').sum()),
        'municipios_rojos_con': int((mun['semaforo_con'] == 'Rojo').sum()),
        'distritos_rojos_sin': int((distrito['semaforo_sin'] == 'Rojo').sum()),
        'distritos_rojos_con': int((distrito['semaforo_con'] == 'Rojo').sum()),
    }
    print(f"{nombre:12} | abstención {resumen['abst_sin']:.2%} → {resumen['abst_con']:.2%} "
          f"| votos extra {n_votos:,.0f} | municipios rojos "
          f"{resumen['municipios_rojos_sin']} → {resumen['municipios_rojos_con']} "
          f"| distritos rojos {resumen['distritos_rojos_sin']} → {resumen['distritos_rojos_con']}")
    return resumen


if __name__ == '__main__':
    print('\nANÁLISIS DE SENSIBILIDAD — SUPUESTOS HIPOTÉTICOS (NO CAUSALES)')
    summary = [ejecutar(name, nat, delta) for name, (nat, delta) in SCENARIOS.items()]
    pd.DataFrame(summary).to_csv(OUT / 'resumen_escenarios_2027.csv', index=False, encoding='utf-8-sig')
    print('\nResumen: out/resumen_escenarios_2027.csv')
