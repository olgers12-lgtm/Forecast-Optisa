import streamlit as st
import pandas as pd
import plotly.express as px
import gspread
from gspread_dataframe import get_as_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import json

CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36"
]

st.set_page_config(
    page_title="Dashboard Ejecutivo de Producción",
    layout="wide",
    initial_sidebar_state="expanded"
)

SHEET_ID = "1U3DwxRVqQFwuPUs0-zvmitgz_LWdhScy-3fu-awBOHU"     # <-- Tu ID de Google Sheet
SHEET_NAME = "Produccion"                                     # <-- Nombre de la hoja/tab

def cargar_gsheet(sheet_id, sheet_name):
    service_account_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(service_account_info, scope)
    gc = gspread.authorize(credentials)
    sh = gc.open_by_key(sheet_id)
    worksheet = sh.worksheet(sheet_name)
    df = get_as_dataframe(worksheet, evaluate_formulas=True)
    df.columns = df.columns.astype(str)
    df = df[df['Indicador'].notna()]
    return df

st.title("📊 Dashboard Ejecutivo de Producción")
st.subheader("Interactivo, visual y actualizado en tiempo real")

try:
    df = cargar_gsheet(SHEET_ID, SHEET_NAME)

    def es_fecha_valida(fecha_str):
        try:
            fecha_str = fecha_str.lower().replace("setiembre", "sep").replace("septiembre", "sep")
            datetime.strptime(fecha_str, "%d-%b-%y")
            return True
        except:
            return False

    fechas = [col for col in df.columns if es_fecha_valida(col)]
    fechas_filtradas = [
        f for f in fechas
        if datetime.strptime(f.lower().replace("setiembre", "sep").replace("septiembre", "sep"), "%d-%b-%y") >= datetime.strptime("01-sep-25", "%d-%b-%y")
    ]
    indicadores = df['Indicador'].dropna().unique().tolist()

    st.sidebar.header("Filtros de visualización")
    fechas_dt = [
        datetime.strptime(f.lower().replace("setiembre", "sep").replace("septiembre", "sep"), "%d-%b-%y") for f in fechas_filtradas
    ]
    semana_map = {f: dt.isocalendar()[1] for f, dt in zip(fechas_filtradas, fechas_dt)}
    mes_map = {f: dt.strftime("%B") for f, dt in zip(fechas_filtradas, fechas_dt)}

    semana_unicas = sorted(set(semana_map.values()))
    mes_unicos = sorted(set(mes_map.values()))

    semana_seleccionada = st.sidebar.multiselect(
        "Semana del año", semana_unicas, default=semana_unicas
    )
    mes_seleccionado = st.sidebar.multiselect(
        "Mes", mes_unicos, default=mes_unicos
    )
    fecha_seleccionada = [
        f for f in fechas_filtradas
        if semana_map[f] in semana_seleccionada and mes_map[f] in mes_seleccionado
    ]
    indicador_seleccionado = st.sidebar.multiselect(
        "Indicador", indicadores, default=indicadores
    )

    df_filtrada = df[df['Indicador'].isin(indicador_seleccionado)][['Indicador'] + fecha_seleccionada]
    df_melt = df_filtrada.melt(id_vars="Indicador", var_name="Fecha", value_name="Valor")

    st.markdown("### KPIs Ejecutivos")
    col1, col2, col3, col4 = st.columns(4)
    try:
        entrada_real = df[df['Indicador'] == 'Entrada Real'][fecha_seleccionada].sum().sum()
        salida_real = df[df['Indicador'] == 'Salida Real'][fecha_seleccionada].sum().sum()
        wip_real = df[df['Indicador'] == 'WIP REAL (9AM)'][fecha_seleccionada].sum().sum()
        gap_salida = df[df['Indicador'] == 'GAP Salida'][fecha_seleccionada].sum().sum()
    except Exception:
        entrada_real = salida_real = wip_real = gap_salida = 0

    col1.metric("🔵 Entrada Real", f"{entrada_real:,}")
    col2.metric("🟢 Salida Real", f"{salida_real:,}")
    col3.metric("🟡 WIP REAL (9AM)", f"{wip_real:,}")
    col4.metric("🔴 GAP Salida", f"{gap_salida:,}")

    st.markdown("### Gráficos Ejecutivos")
    fig_entrada = px.line(
        df_melt[df_melt['Indicador'].str.contains('Entrada')],
        x="Fecha", y="Valor", color="Indicador",
        title="Entradas Proyectadas vs Reales",
        color_discrete_sequence=CORPORATE_COLORS
    )
    st.plotly_chart(fig_entrada, use_container_width=True)

    fig_salida = px.line(
        df_melt[df_melt['Indicador'].str.contains('Salida')],
        x="Fecha", y="Valor", color="Indicador",
        title="Salidas Proyectadas vs Reales",
        color_discrete_sequence=CORPORATE_COLORS
    )
    st.plotly_chart(fig_salida, use_container_width=True)

    fig_wip = px.bar(
        df_melt[df_melt['Indicador'].str.contains('WIP')],
        x="Fecha", y="Valor", color="Indicador",
        title="WIP Proyectado y Real",
        color_discrete_sequence=CORPORATE_COLORS
    )
    st.plotly_chart(fig_wip, use_container_width=True)

    fig_gap = px.area(
        df_melt[df_melt['Indicador'].str.contains('GAP')],
        x="Fecha", y="Valor", color="Indicador",
        title="GAP Salida y WIP",
        color_discrete_sequence=CORPORATE_COLORS
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    st.markdown("#### Filtro avanzado por día, semana y mes")
    filtro_tipo = st.radio("Visualizar por:", ["Día", "Semana", "Mes"], horizontal=True)
    if filtro_tipo == "Día":
        st.dataframe(df_filtrada, use_container_width=True)
    elif filtro_tipo == "Semana":
        semana_df = df_melt.copy()
        semana_df["Semana"] = semana_df["Fecha"].map(semana_map)
        st.dataframe(
            semana_df.groupby(["Indicador", "Semana"])["Valor"].sum().unstack(),
            use_container_width=True
        )
    elif filtro_tipo == "Mes":
        mes_df = df_melt.copy()
        mes_df["Mes"] = mes_df["Fecha"].map(mes_map)
        st.dataframe(
            mes_df.groupby(["Indicador", "Mes"])["Valor"].sum().unstack(),
            use_container_width=True
        )

    st.success("Dashboard actualizado con datos filtrados.")
except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")

st.markdown("""
<style>
footer {visibility: hidden;}
</style>
<div style='text-align: right; color: #1F2A56; font-size: 14px;'>
    <b>© 2025 Dashboard Ejecutivo | Industria 4.0 | Powered by Streamlit</b>
</div>
""", unsafe_allow_html=True)
st.write(st.secrets)  # Esto te muestra todos los secrets disponibles
