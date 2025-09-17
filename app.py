import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import altair as alt
import joblib

# ====================
# Configuración general de Streamlit
# ====================
st.set_page_config(page_title="Programa Vaso de Leche - Cobertura", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
    .stApp { font-family: 'Segoe UI', Tahoma, sans-serif;  }
    h1, h2, h3, h4 { color: #003366; }
    .card { 
        border-radius: 8px; 
        padding: 15px; 
        margin-bottom: 12px; 
        
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .stDataFrame { border: 1px solid #ddd; border-radius: 6px; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ====================
# Cargar dataset con limpieza fuerte
# ====================
df = pd.read_csv("beneficiarios_limpio.csv", skip_blank_lines=True)

# 1. Eliminar columnas basura tipo 'Unnamed'
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# 2. Quitar espacios invisibles en los nombres de columnas
df.columns = df.columns.str.strip()

# 3. Eliminar filas totalmente vacías
df = df.dropna(how="all")

# 4. Filtrar caseríos vacíos o nulos
if "CASERIO" in df.columns:
    df = df[df["CASERIO"].astype(str).str.strip() != ""]
    df = df[df["CASERIO"].notna()]

# 5. Resetear índice limpio
df = df.reset_index(drop=True)

# ====================
# Cargar modelos individuales por categoría
# ====================
modelos = {
    "BENEFICIARIOS DE 0-6 AÑOS": joblib.load("modelo_beneficiarios_de_0-6_anos.pkl"),
    "BENEFICIARIOS GESTANTES": joblib.load("modelo_beneficiarios_gestantes.pkl"),
    "BENEFICIARIOS LACTANTES": joblib.load("modelo_beneficiarios_lactantes.pkl"),
    "BENEFICIARIOS DISCAPACITADOS": joblib.load("modelo_beneficiarios_discapacitados.pkl"),
    "BENEFICIARIOS ANCIANOS": joblib.load("modelo_beneficiarios_ancianos.pkl"),
}


# Diccionario de coordenadas fijas
coords_dict = {
    "ARHUAY": (-9.134, -77.745),
    "BELLA VISTA": (-9.140, -77.740),
    "LA FLORIDA": (-9.145, -77.750),
    "COCHAPAMPA": (-9.120, -77.730),
    "CAJAPAMPA": (-9.150, -77.720),
    "ENCAYOC": (-9.130, -77.760),
    "RANRAHIRCA": (-9.138, -77.743),
    "MONTEBELLO": (-9.135, -77.755),
    "APACHICO": (-9.142, -77.735),
    "UCHUCOTO": (-9.125, -77.748),
}

geolocator = Nominatim(user_agent="vaso_de_leche")

def get_coords(lugar):
    if lugar in coords_dict:
        return coords_dict[lugar]
    try:
        location = geolocator.geocode(f"{lugar}, Ranrahirca, Yungay, Áncash, Perú")
        if location:
            return location.latitude, location.longitude
    except:
        return None, None
    return None, None

if "lat" not in df.columns or "lon" not in df.columns:
    df["lat"], df["lon"] = zip(*df["CASERIO"].apply(get_coords))

# ====================
# Layout principal con Tabs
# ====================
st.title("Programa Vaso de Leche - Análisis de Cobertura")

tab_global, tab_caserio = st.tabs(["Proyección Global", "Información por Caserío"])

# -------- TAB 1: Proyección global --------
with tab_global:
    st.subheader("Mapa de Calor - Proyección de Beneficiarios")

    growth = st.slider("Seleccione el porcentaje de crecimiento proyectado", -20, 50, 10)

    df["TOTAL_PROYECTADO"] = (df["TOTAL DE BENEFICIARIOS"] * (1 + growth/100)).astype(int)

    map_col, info_col = st.columns([3, 2])

    with map_col:
        from folium.plugins import HeatMap
        m2 = folium.Map(location=[-9.138, -77.743], zoom_start=13)
        heat_data = [
            [row["lat"], row["lon"], row["TOTAL_PROYECTADO"]] 
            for _, row in df.iterrows() if not pd.isna(row["lat"])
        ]
        HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(m2)
        st_folium(m2, width="100%", height=500)

    with info_col:
        st.subheader("Ranking de Caseríos (Proyección)")
        ranking = df[["CASERIO", "TOTAL DE BENEFICIARIOS", "TOTAL_PROYECTADO"]].sort_values(
            by="TOTAL_PROYECTADO", ascending=False
        )
        st.dataframe(ranking.reset_index(drop=True), height=500)

# -------- TAB 2: Exploración por caserío --------
with tab_caserio:
    st.subheader("Información Detallada del Caserío")

    map_col, info_col = st.columns([3, 2])

    with map_col:
        m = folium.Map(location=[-9.138, -77.743], zoom_start=13)
        for _, row in df.iterrows():
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    tooltip=row["NOMBRE DE ORGANIZACIÓN"],
                    icon=folium.Icon(color="darkblue", icon="info-sign")
                ).add_to(m)
        st_map = st_folium(m, width="100%", height=500)

    with info_col:
        selected_org = st_map.get("last_object_clicked_tooltip") if st_map else None

        if selected_org:
            org_data = df[df["NOMBRE DE ORGANIZACIÓN"] == selected_org].iloc[0]

            st.markdown(f"""
                <div class='card'>
                    <h3>{org_data['NOMBRE DE ORGANIZACIÓN']}</h3>
                    <p><b>Caserío:</b> {org_data['CASERIO']}<br>
                    <b>Distrito:</b> {org_data['DISTRITO']}<br>
                    <b>Provincia:</b> {org_data['PROVINCIA']}<br>
                    <b>Departamento:</b> {org_data['DEPARTAMENTO']}</p>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(f"**Total de beneficiarios:** {org_data['TOTAL DE BENEFICIARIOS']}")

            tab1, tab2 = st.tabs(["Datos actuales", "Predicciones"])

            with tab1:
                col1, col2 = st.columns([1,1])
                data = {
                    "Niños 0-6": org_data['BENEFICIARIOS DE 0-6 AÑOS'],
                    "Gestantes": org_data['BENEFICIARIOS GESTANTES'],
                    "Lactantes": org_data['BENEFICIARIOS LACTANTES'],
                    "Discapacitados": org_data['BENEFICIARIOS DISCAPACITADOS'],
                    "Ancianos": org_data['BENEFICIARIOS ANCIANOS']
                }
                chart_df = pd.DataFrame(list(data.items()), columns=["Categoría", "Cantidad"])
                chart = alt.Chart(chart_df).mark_arc(innerRadius=50).encode(
                    theta="Cantidad",
                    color=alt.Color("Categoría", scale=alt.Scale(scheme="set2")),
                    tooltip=["Categoría", "Cantidad"]
                ).properties(width=280, height=280)
                col1.altair_chart(chart, use_container_width=True)
                col2.dataframe(chart_df.set_index("Categoría"), height=250)

            with tab2:
                col3, col4 = st.columns([1, 1])
                features = ["BENEFICIARIOS DE 0-6 AÑOS","BENEFICIARIOS GESTANTES",
                            "BENEFICIARIOS LACTANTES","BENEFICIARIOS DISCAPACITADOS",
                            "BENEFICIARIOS ANCIANOS"]
                short_names = {
                    "BENEFICIARIOS DE 0-6 AÑOS": "0-6",
                    "BENEFICIARIOS GESTANTES": "Gestantes",
                    "BENEFICIARIOS LACTANTES": "Lactantes",
                    "BENEFICIARIOS DISCAPACITADOS": "Discapacitados",
                    "BENEFICIARIOS ANCIANOS": "Ancianos"
                }

                datos_pred = org_data[features].to_frame().T
                predicciones = {}
                for cat, mod in modelos.items():
                    X_pred = datos_pred[[c for c in features if c != cat]]
                    try:
                        pred = mod.predict(X_pred)[0]
                        predicciones[short_names[cat]] = int(pred)
                    except:
                        predicciones[short_names[cat]] = "N/A"

                pred_table_df = pd.DataFrame(list(predicciones.items()), columns=["Categoría", "Predicción"])
                col3.dataframe(pred_table_df.set_index("Categoría"), height=250)

                pred_chart_df = pred_table_df[pred_table_df["Predicción"].apply(lambda x: isinstance(x, int))]
                bar_chart = alt.Chart(pred_chart_df).mark_bar(
                    cornerRadiusTopLeft=3, cornerRadiusTopRight=3
                ).encode(
                    x=alt.X("Categoría", sort=None),
                    y="Predicción",
                    color=alt.Color("Categoría", scale=alt.Scale(scheme="set2"), legend=None),
                    tooltip=["Categoría", "Predicción"]
                ).properties(width=220, height=280)

                col4.altair_chart(bar_chart, use_container_width=True)
