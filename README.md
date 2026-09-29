# Vaso de Leche – Análisis de Cobertura

## Descripción

Proyecto desarrollado durante la Datathon de Supervisión para analizar la cobertura del Programa Vaso de Leche.

La aplicación permite explorar la distribución de beneficiarios por caserío, visualizar información mediante mapas y gráficos, y realizar proyecciones utilizando modelos de regresión.

## Objetivo

Facilitar el análisis de la información de los beneficiarios del Programa Vaso de Leche mediante una herramienta interactiva que permita visualizar los datos y obtener estimaciones para apoyar la toma de decisiones.

## Funcionalidades

- Visualización de beneficiarios por caserío.
- Mapas interactivos para explorar la distribución geográfica.
- Rankings de beneficiarios.
- Gráficos para el análisis de los datos.
- Predicción de beneficiarios mediante modelos de regresión lineal.
- Interfaz interactiva para consultar la información.

## Tecnologías utilizadas

- Python
- Streamlit
- Pandas
- Scikit-learn
- Folium
- Altair
- Joblib

## Modelo de predicción

Se utilizaron modelos de Regresión Lineal (`LinearRegression`) para realizar estimaciones relacionadas con las categorías de beneficiarios.

Los modelos fueron entrenados a partir de los datos disponibles y posteriormente utilizados para generar predicciones dentro de la aplicación.

## Visualización

El proyecto utiliza:

- Folium para mapas interactivos.
- Altair para gráficos.
- Streamlit para construir la interfaz web y presentar los resultados.

## Instalación

Clonar el repositorio:

```bash
git clone URL_DEL_REPOSITORIO
