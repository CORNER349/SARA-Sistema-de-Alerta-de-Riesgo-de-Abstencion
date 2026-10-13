"""Preparación de conteos censales INE para model2.py y final.py.

Lee data/processed/agg_YYYY.csv desde la raíz del repositorio, independientemente
 del directorio de trabajo. Agrupa filas con la misma sección antes de calcular tasas.
"""
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'data' / 'processed'
YRS = [2009, 2012, 2015, 2018, 2021, 2024]
NUM = ['LN', 'SV', 'NV', 'NS'] + [
    f'{c}_{g}' for g in ['18_29', '30_44', '45_59', '60mas', 'H', 'M']
    for c in ['LN', 'NV', 'NS']
]
META = ['key', 'EDOCVE', 'EDONOM', 'MPIOCVE', 'MPIONOM', 'TIPOSEC', 'DEF', 'DEL']


def load():
    out = {}
    for y in YRS:
        ruta = A / f'agg_{y}.csv'
        if not ruta.is_file():
            raise FileNotFoundError(f'Falta {ruta}. Genera primero el agregado del año {y}.')
        d = pd.read_csv(ruta)
        faltantes = sorted(set(NUM + META[1:] + ['SECCION']) - set(d.columns))
        if faltantes:
            raise ValueError(f'{ruta.name}: faltan columnas: {faltantes}')
        if d[NUM].isna().any().any():
            raise ValueError(f'{ruta.name}: hay valores nulos en los conteos.')
        if (d[NUM] < 0).any().any():
            raise ValueError(f'{ruta.name}: hay conteos negativos.')
        if not (d.LN == d.SV + d.NV + d.NS).all():
            raise ValueError(f'{ruta.name}: LN no coincide con SV+NV+NS.')

        d['key'] = (d.EDOCVE.astype(int).astype(str).str.zfill(2)
                    + '-' + d.SECCION.astype(int).astype(str).str.zfill(4))
        # Mantener una sola fila por sección; sumar conteos antes de obtener tasas.
        repetidas = int(d.duplicated('key').sum())
        # Las etiquetas territoriales de una sección deben ser consistentes.
        conflict = d.groupby('key')[['EDOCVE', 'MPIOCVE']].nunique(dropna=False)
        nconflict = int((conflict > 1).any(axis=1).sum())
        if nconflict:
            raise ValueError(f'{ruta.name}: {nconflict} secciones tienen municipio/entidad contradictorios.')
        meta = d.sort_values('LN', ascending=False).drop_duplicates('key')[META]
        s = d.groupby('key', as_index=False)[NUM].sum().merge(meta, on='key', validate='one_to_one')
        if s.key.duplicated().any():
            raise ValueError(f'{ruta.name}: quedaron claves de sección duplicadas.')
        base = s.LN - s.NS
        invalidas = int((base <= 0).sum())
        # No existe tasa de abstención cuando la base electoral efectiva es cero.
        # Estas secciones se excluyen del modelado; no se inventa base=1.
        s = s.loc[base > 0].copy()
        s['base'] = s.LN - s.NS
        s['abst'] = s.NV / s.base
        s['jov'] = s.LN_18_29 / s.LN
        s['old'] = s.LN_60mas / s.LN
        s['fem'] = s.LN_M / s.LN
        if not s['abst'].between(0, 1).all():
            raise ValueError(f'{ruta.name}: abstención fuera de [0, 1].')
        print(f'{y}: {len(d)} filas originales; {repetidas} claves repetidas; '
              f'{invalidas} secciones sin base efectiva; {len(s)} secciones válidas')
        out[y] = s
    return out


def nat(fr):
    """Abstención nacional ponderada por la base electoral efectiva."""
    return {y: fr[y].NV.sum() / fr[y]['base'].sum() for y in YRS}