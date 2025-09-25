import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# Configuración de colores corporativos
CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36"
]

st.set_page_config(
    page_title="Dashboard Ejecutivo de Producción",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Autenticación simple para uploader/resumen ---
ADMIN_KEY = "admin123"  # Cambia por tu clave secreta personal

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.sidebar.header("Acceso administrador")
    user_key = st.sidebar.text_input("Ingresa tu clave secreta:", type="password")
    if st.sidebar.button("Acceder"):
        if user_key == ADMIN_KEY:
            st.session_state["authenticated"] = True
            st.sidebar.success("Acceso concedido.")
        else:
            st.sidebar.error("Clave incorrecta.")

if st.session_state["authenticated"]:
    st.sidebar.header("Subida de datos")
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu archivo Excel de producción", type=["xlsx", "xls"]
    )
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        st.sidebar.success("Archivo cargado correctamente.")
    st.sidebar.markdown("---")

# --- Mostrar dashboard a todos ---
st.title("📊 Dashboard Ejecutivo de Producción")
st.subheader("Interactivo - Supervisión en tiempo real")

# --- Carga de datos solo si existen ---
if "uploaded_file" in st.session_state and st.session_state["uploaded_file"]:
    df = pd.read_excel(st.session_state["uploaded_file"], header=0)
    df.columns = df.columns.astype(str)

    # --- Filtrar fechas desde 01-sept-2025 en adelante ---
    fechas = [col for col in df.columns if '-' in col]
    fechas_filtradas = [
        f for f in fechas
        if datetime.strptime(f, "%d-%b") >= datetime.strptime("01-sep", "%d-%b")
    ]
    indicadores = df['Indicador'].dropna().unique().tolist()

    # --- Mostrar filtros avanzados ---
    st.sidebar.header("Filtros de visualización")
    # Generar mapeo de fechas a semana y mes
    fechas_dt = [
        datetime.strptime(f + "-2025", "%d-%b-%Y") for f in fechas_filtradas
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

    # --- Data filtrada ---
    df_filtrada = df[df['Indicador'].isin(indicador_seleccionado)]
    df_filtrada = df_filtrada[['Indicador'] + fecha_seleccionada]
    df_melt = df_filtrada.melt(id_vars="Indicador", var_name="Fecha", value_name="Valor")

    # --- Entradas a Surf (75% de entradas respectivas) ---
    entradas = df_melt[df_melt['Indicador'].str.contains('Entrada')]
    entradas_surf = entradas.copy()
    entradas_surf['Valor'] = entradas_surf['Valor'] * 0.75
    entradas_surf['Indicador'] = "Entradas a Surf (75%)"

    # --- KPIs admin (solo tú los ves) ---
    if st.session_state["authenticated"]:
        st.markdown("### KPIs Resumidos (solo admin)")
        col1, col2 = st.columns(2)
        entrada_real = df[df['Indicador'] == 'Entrada Real'][fecha_seleccionada].sum().sum()
        salida_real = df[df['Indicador'] == 'Salida Real'][fecha_seleccionada].sum().sum()
        col1.metric("🔵 Entrada Real", f"{entrada_real:,}")
        col2.metric("🟢 Salida Real", f"{salida_real:,}")

    # --- Gráficos cool ---
    st.markdown("### Gráficos Ejecutivos")
    fig_entrada = px.line(
        pd.concat([entradas, entradas_surf]),
        x="Fecha", y="Valor", color="Indicador",
        title="Entradas Reales y Entradas a Surf (75%)",
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

    # --- Filtros por día/semana/mes ---
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
else:
    st.info("El dashboard estará disponible cuando el admin suba el archivo de datos.")

# --- Footer corporativo ---
st.markdown("""
<style>
footer {visibility: hidden;}
</style>
<div style='text-align: right; color: #1F2A56; font-size: 14px;'>
    <b>© 2025 Dashboard Ejecutivo | Industria 4.0 | Powered by Streamlit</b>
</div>
""", unsafe_allow_html=True)
