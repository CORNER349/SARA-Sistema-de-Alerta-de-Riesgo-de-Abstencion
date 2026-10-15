"""Genera cuatro graficas del proyecto INE 2026 desde los CSV existentes.
Ejecutar desde la raiz: ./.venv/bin/python src/generar_graficas.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'out'
DEST = ROOT / 'reports' / 'graficas'
DEST.mkdir(parents=True, exist_ok=True)
SCENARIOS = [('Conservador', 'conservador'), ('Base', 'base'), ('Optimista', 'optimista')]
COL = {'Conservador': '#c17d35', 'Base': '#3978a6', 'Optimista': '#38876d'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 125})

def guardar(nombre):
    plt.tight_layout()
    destino = DEST / nombre
    plt.savefig(destino, dpi=200, bbox_inches='tight')
    plt.close()
    print('Generada:', destino.relative_to(ROOT))

def leer_escenario(slug):
    ruta = OUT / 'escenarios' / slug / 'semaforo_municipal_2027.csv'
    if slug == 'base' and not ruta.exists():
        ruta = OUT / 'semaforo_municipal_2027.csv'
    if not ruta.exists():
        raise FileNotFoundError(f'No existe {ruta}. Primero ejecuta src/final.py para generar escenarios.')
    d = pd.read_csv(ruta)
    necesarias = {'base','nv','votos_extra','semaforo_sin','semaforo_con'}
    faltantes = necesarias - set(d.columns)
    if faltantes:
        raise ValueError(f'Faltan columnas {faltantes} en {ruta}')
    return d

# 1) MAE retrospectivo registrado en las evaluaciones previas del proyecto.
# Valores en puntos porcentuales; actualizar si se vuelven a evaluar los modelos.
mae = pd.DataFrame({
    'Año': [2018, 2021, 2024],
    'Persistencia': [9.30, 6.03, 6.45],
    'Ridge': [7.20, 6.73, 4.66],
    'Gradient Boosting': [8.06, 7.06, 4.30],
    'Combinado': [6.42, 6.74, 4.62],
}).set_index('Año')
ax = mae.plot(kind='bar', figsize=(10, 5.5), width=.78)
ax.set(title='Error de predicción por modelo y elección', xlabel='Año de evaluación', ylabel='MAE (puntos porcentuales)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=.2)
ax.legend(title='Modelo', ncol=2)
guardar('01_comparacion_modelos.png')

resumen = []
for nombre, slug in SCENARIOS:
    d = leer_escenario(slug)
    total_base = pd.to_numeric(d['base'], errors='raise').sum()
    total_nv = pd.to_numeric(d['nv'], errors='raise').sum()
    extras = pd.to_numeric(d['votos_extra'], errors='raise').sum()
    if total_base <= 0:
        raise ValueError(f'Base no positiva en {slug}')
    resumen.append({'Escenario': nombre, 'Sin': 100 * total_nv / total_base,
                    'Con': 100 * (total_nv - extras) / total_base,
                    'Votos extra': extras, 'datos': d})

# 2) Tasas agregadas calculadas desde los CSV, no promedios simples de municipios.
x = np.arange(len(resumen))
fig, ax = plt.subplots(figsize=(9, 5.5))
a = ax.bar(x-.19, [r['Sin'] for r in resumen], .38, label='Sin intervención', color='#777f89')
b = ax.bar(x+.19, [r['Con'] for r in resumen], .38, label='Con intervención hipotética', color='#368f79')
ax.bar_label(a, fmt='%.2f%%', padding=3)
ax.bar_label(b, fmt='%.2f%%', padding=3)
ax.set_xticks(x, [r['Escenario'] for r in resumen])
ax.set(ylabel='Abstención nacional (%)', title='Abstención estimada en 2027 por escenario')
ax.set_ylim(0, max(r['Sin'] for r in resumen)*1.18)
ax.legend()
ax.grid(axis='y', alpha=.2)
guardar('02_abstencion_escenarios.png')

# 3) Votos adicionales estimados.
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar([r['Escenario'] for r in resumen], [r['Votos extra'] for r in resumen],
              color=[COL[r['Escenario']] for r in resumen], width=.55)
ax.bar_label(bars, labels=[f"{r['Votos extra']:,.0f}" for r in resumen], padding=5)
ax.set(title='Votos adicionales hipotéticos por escenario', ylabel='Votos adicionales estimados')
ax.set_ylim(0, max(r['Votos extra'] for r in resumen)*1.18)
ax.grid(axis='y', alpha=.2)
guardar('03_votos_adicionales.png')

# 4) Comparación del semáforo municipal en los tres escenarios.
fig, axs = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
orden = ['Verde', 'Amarillo', 'Rojo']
colores = ['#2e9e5b', '#f2c230', '#d64545']
for ax, r in zip(axs, resumen):
    d = r['datos']
    antes = d['semaforo_sin'].value_counts().reindex(orden, fill_value=0)
    despues = d['semaforo_con'].value_counts().reindex(orden, fill_value=0)
    xx = np.arange(3)
    ax.bar(xx-.2, antes.values, .4, label='Sin', color=colores, alpha=.55)
    ax.bar(xx+.2, despues.values, .4, label='Con', color=colores, alpha=1)
    ax.set_xticks(xx, orden)
    ax.set_title(r['Escenario'])
    ax.grid(axis='y', alpha=.15)
axs[0].set_ylabel('Número de municipios')
axs[0].legend(title='Intervención')
fig.suptitle('Municipios por nivel de riesgo — escenarios 2027', fontsize=14)
guardar('04_semaforo_municipal.png')

pd.DataFrame([{k: v for k, v in r.items() if k != 'datos'} for r in resumen]).to_csv(
    DEST / 'resumen_para_graficas.csv', index=False, encoding='utf-8-sig')
print('Listo: cuatro graficas y resumen CSV.')
