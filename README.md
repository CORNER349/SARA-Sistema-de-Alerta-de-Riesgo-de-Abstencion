# Hackathon INE 2026 — Reto 2

## Predicción y escenarios de riesgo de abstención electoral

**Equipo:** Tlacuvoz

**Integrantes:**
- Oscar Corona Neria
- Gustavo Angel Nava Bautista
- Ilse Habibi Lara Tehozol

## Descripción

Proyecto desarrollado para el Reto 2 del Hackathon INE 2026.

El objetivo es analizar patrones históricos de abstención electoral, identificar territorios con mayor riesgo y explorar escenarios hipotéticos de participación para 2027.

Se utilizan datos electorales agregados y modelos de aprendizaje automático. No se utilizan datos personales de electores.

## Datos utilizados

Se emplean datos electorales correspondientes a:

- 2009
- 2012
- 2015
- 2018
- 2021
- 2024

Los archivos procesados están disponibles en `data/processed/`.

Los datos originales no se incluyen debido a su tamaño.

## Modelos

Se implementaron y evaluaron los siguientes métodos:

- Persistencia histórica.
- Promedios municipales históricos.
- Regresión Ridge.
- HistGradientBoostingRegressor.
- Estimación combinada para el análisis de escenarios.

Las variables incluyen información histórica de abstención, características agregadas de la lista nominal y características electorales.

## Evaluación

El desempeño se evalúa mediante el error absoluto medio (MAE), expresado en puntos porcentuales.

| Modelo | MAE 2021 | MAE 2024 |
|---|---:|---:|
| Persistencia | 6.03 | 6.45 |
| Promedio municipal anterior | 8.00 | 7.86 |
| Promedio de dos elecciones municipales | 7.94 | 6.58 |
| Ridge | 6.73 | 4.66 |
| Gradient Boosting | 7.06 | 4.30 |

Gradient Boosting obtuvo el menor MAE en la evaluación de 2024. En 2021, el mejor resultado correspondió a persistencia.

## Escenarios hipotéticos para 2027

Se analizan tres escenarios:

| Escenario | Abstención inicial supuesta | Abstención simulada | Votos adicionales hipotéticos |
|---|---:|---:|---:|
| Conservador | 52.00% | 51.77% | 206,441 |
| Base | 48.20% | 47.85% | 312,407 |
| Optimista | 44.00% | 43.73% | 243,082 |

**Importante:** las tasas nacionales iniciales son supuestos de escenario. Los votos adicionales representan resultados de una simulación, no efectos causales demostrados ni predicciones garantizadas.

## Resultados y visualizaciones

### Gráficas

Las cuatro gráficas se encuentran en `reports/graficas/`:

1. Comparación de modelos.
2. Abstención por escenario.
3. Votos adicionales hipotéticos.
4. Semáforo municipal.

### Mapa interactivo

Archivo principal:

`reports/mapa_tres_escenarios/mapa_escenarios.html`

El mapa permite comparar dos vistas sincronizadas y seleccionar los escenarios conservador, base y optimista.

La cartografía incluida corresponde al Marco Geoestadístico del INEGI de mayo de 2021.

El mapa representa 2,417 de los 2,465 municipios analizados. Los 48 municipios sin correspondencia cartográfica permanecen en los resultados tabulares y del modelo. La cobertura cartográfica alcanza aproximadamente el 99.4% de la lista nominal considerada.

## Instalación

Se recomienda Python 3.12.

Clonar el repositorio:

```bash
git clone https://github.com/CORNER349/INE-Hakaton-2026.git
cd INE-Hakaton-2026
```

Crear y activar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Los comandos siguientes se ejecutan desde la raíz del proyecto, con el entorno virtual activado.

### 1. Evaluar los modelos

```bash
python src/model2.py
```

### 2. Generar los escenarios

```bash
python src/final.py
```

### 3. Generar las gráficas

```bash
python src/generar_graficas.py
```

### 4. Generar el mapa

```bash
python src/generar_mapa.py \
  src/municipios_inegi.geojson \
  out/semaforo_municipal_2027.csv \
  reports/mapa_tres_escenarios
```

### 5. Visualizar el mapa

Abrir en un navegador:

`reports/mapa_tres_escenarios/mapa_escenarios.html`

## Limitaciones

- Los escenarios de 2027 son ejercicios de sensibilidad sujetos a supuestos.
- No se demuestra una relación causal entre las intervenciones hipotéticas y la participación electoral.
- El desempeño de los modelos cambia entre elecciones.
- La cartografía disponible no permite representar todos los municipios de los resultados tabulares.
- Los resultados deben interpretarse como herramientas de análisis territorial y no como certezas electorales.

## Bibliotecas utilizadas

| Biblioteca | Versión | Documentación |
|---|---|---|
| pandas | 3.0.6 | https://pandas.pydata.org/docs/ |
| NumPy | 2.5.4 | https://numpy.org/doc/ |
| scikit-learn | 1.9.1 | https://scikit-learn.org/stable/ |
| Matplotlib | 3.11.2 | https://matplotlib.org/stable/ |
| GeoPandas | 1.2.0 | https://geopandas.org/en/stable/ |
| Folium | 0.20.0 | https://python-visualization.github.io/folium/ |

Las versiones corresponden a las dependencias registradas para el entorno del proyecto.

## Fuentes

- Instituto Nacional Electoral (INE): https://www.ine.mx/
- Instituto Nacional de Estadística y Geografía (INEGI): https://www.inegi.org.mx/

## Repositorio

https://github.com/CORNER349/INE-Hakaton-2026
