import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="📊 Dashboard Ejecutivo de Producción", layout="wide")

st.title("📊 Dashboard Ejecutivo de Producción")
st.caption("Interactivo, visual y actualizado en tiempo real")

# COLORES CORPORATIVOS (ajusta a los de tu empresa si quieres)
CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36"
]

# Configura ID de tu sheet y nombre de la hoja (tab)
SHEET_ID = "1U3DwxRVqQFwuPUs0-zvmitgz_LWdhScy-3fu-awBOHU"
SHEET_NAME = "Produccion"  # Cambia esto si tu hoja/tab tiene otro nombre

@st.cache_data(show_spinner="Cargando datos de Google Sheets...")
def cargar_datos(sheet_id, sheet_name):
    # Leer credencial desde secrets
    sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    # Mostrar nombres de hojas para debug
    sheet_names = [ws.title for ws in sh.worksheets()]
    if sheet_name not in sheet_names:
        raise ValueError(f"La hoja/tab '{sheet_name}' no existe. Hojas disponibles: {sheet_names}")
    ws = sh.worksheet(sheet_name)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df

try:
    df = cargar_datos(SHEET_ID, SHEET_NAME)
    st.success("✅ Datos cargados correctamente.")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    # Ejemplo de gráfico (ajusta a tus columnas reales)
    if not df.empty:
        # Elige columnas numéricas para graficar
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 1:
            col = numeric_cols[0]
            fig = px.histogram(df, x=col, color_discrete_sequence=CORPORATE_COLORS)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se encontraron columnas numéricas para graficar.")
    else:
        st.warning("La hoja/tab está vacía.")

except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
