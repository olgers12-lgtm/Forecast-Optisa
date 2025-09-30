import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import re

st.set_page_config(page_title="🚀 Dashboard Ejecutivo de Producción", layout="wide", initial_sidebar_state="expanded")

CORPORATE_COLORS = ["#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36", "#A3FFAE"]
WIP_THRESHOLDS = {"Alerta": 1200, "Crítico": 1500}
EFFICIENCY_GOAL = 95

SHEET_ID = "1U3DwxRVqQFwuPUs0-zvmitgz_LWdhScy-3fu-awBOHU"
SHEET_NAME = "Produccion"

@st.cache_data(ttl=600)
def cargar_datos(sheet_id, sheet_name):
    sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(sa_info, scope)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(sheet_name)
    data = ws.get_all_records()
    df = pd.DataFrame(data)
    return df

def agregar_ano(col):
    col = col.strip().lower()
    match = re.match(r"(\d{1,2})-(sept|oct)", col)
    if match:
        num, mes = match.groups()
        year = "2025"
        mes_map = {"sept": "09", "oct": "10"}
        return f"{year}-{mes_map[mes]}-{int(num):02d}"
    if "-" in col and col.count("-") == 2:
        return col
    return None

# --- Datos base ---
df = cargar_datos(SHEET_ID, SHEET_NAME)
col_indicador = next((c for c in df.columns if "indicador" in c.lower()), None)
if not col_indicador:
    st.error("No se encontró columna 'Indicador'. Las columnas son: " + str(df.columns.tolist()))
    st.stop()
df = df[df[col_indicador].notnull() & (df[col_indicador] != '')]

# --- Fechas robustas ---
fechas = [c for c in df.columns if c != col_indicador]
fechas_dt = [agregar_ano(f) for f in fechas]
fechas_validas = [fechas[i] for i in range(len(fechas)) if fechas_dt[i] is not None]
fechas_dt_validas = [f for f in fechas_dt if f is not None]

# --- Melt largo ---
indicadores = df[col_indicador].unique().tolist()
indicador_sel = st.multiselect("Selecciona indicadores:", indicadores, default=indicadores)
df_melt = df[df[col_indicador].isin(indicador_sel)].melt(
    id_vars=[col_indicador],
    value_vars=fechas_validas,
    var_name='Fecha',
    value_name='Valor'
)
df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"].apply(agregar_ano), format="%Y-%m-%d", errors="coerce")
df_melt["Valor"] = pd.to_numeric(df_melt["Valor"], errors="coerce")
df_melt = df_melt.dropna(subset=["Fecha_dt"])

# --- Panel de filtros moderno y estético ---
with st.sidebar:
    st.header("📅 Filtros de fechas")
    filtro_tipo = st.radio(
        "Agrupar por:",
        [
            "🗓️ Día",
            "📆 Semana",
            "🗓️ Mes",
            "🎯 Rango personalizado"
        ],
        horizontal=False
    )
    if "Día" in filtro_tipo:
        fechas_disponibles = sorted(df_melt["Fecha_dt"].dt.date.unique())
        fechas_sel = st.date_input(
            "Selecciona uno o más días:",
            value=fechas_disponibles[-1],
            min_value=min(fechas_disponibles),
            max_value=max(fechas_disponibles),
            format="DD/MM/YYYY"
        )
        if isinstance(fechas_sel, list):
            mask_fecha = df_melt["Fecha_dt"].dt.date.isin(fechas_sel)
        else:
            mask_fecha = df_melt["Fecha_dt"].dt.date == fechas_sel
    elif "Semana" in filtro_tipo:
        # Semana ISO y año
        df_melt["SemanaISO"] = df_melt["Fecha_dt"].dt.isocalendar().week
        df_melt["AñoISO"] = df_melt["Fecha_dt"].dt.isocalendar().year
        semanas_unicas = sorted(df_melt[["AñoISO", "SemanaISO"]].drop_duplicates().values.tolist())
        # Etiquetas visuales
        semana_labels = []
        for year, week in semanas_unicas:
            semana_df = df_melt[(df_melt["AñoISO"] == year) & (df_melt["SemanaISO"] == week)]
            fecha_ini = semana_df["Fecha_dt"].min().strftime("%d-%b")
            fecha_fin = semana_df["Fecha_dt"].max().strftime("%d-%b")
            semana_labels.append(f"Semana {week} ({fecha_ini} - {fecha_fin})")
        semana_sel_idx = st.selectbox(
            "Selecciona semana:",
            options=range(len(semanas_unicas)),
            format_func=lambda i: semana_labels[i]
        )
        year_sel, week_sel = semanas_unicas[semana_sel_idx]
        mask_fecha = (df_melt["AñoISO"] == year_sel) & (df_melt["SemanaISO"] == week_sel)
    elif "Mes" in filtro_tipo:
        meses_disponibles = sorted(df_melt["Fecha_dt"].dt.strftime("%B %Y").unique())
        mes_sel = st.selectbox("Selecciona mes:", options=meses_disponibles)
        mask_fecha = df_melt["Fecha_dt"].dt.strftime("%B %Y") == mes_sel
    else:
        fecha_min, fecha_max = df_melt["Fecha_dt"].min(), df_melt["Fecha_dt"].max()
        fecha_inicio, fecha_fin = st.date_input(
            "Selecciona el rango de fechas:",
            value=(fecha_max - pd.Timedelta(days=7), fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
            format="DD/MM/YYYY"
        )
        mask_fecha = (df_melt["Fecha_dt"] >= pd.to_datetime(fecha_inicio)) & (df_melt["Fecha_dt"] <= pd.to_datetime(fecha_fin))

df_filtrado_fecha = df_melt[mask_fecha]

# --- KPIs productivos y Delta vs periodo anterior ---
st.subheader("🧮 KPIs Industriales")
col1, col2, col3, col4, col5 = st.columns(5)
try:
    entrada_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
    salida_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
    salida_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida proyectada")]['Valor'].sum()
    eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None
    wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("wip")]['Valor']
    wip_delta = wip.iloc[-1] - wip.iloc[0] if len(wip) > 1 else 0

    # Deltas vs periodo anterior
    prev_mask = df_melt["Fecha_dt"] < df_filtrado_fecha["Fecha_dt"].min()
    prev_df = df_melt[prev_mask]
    entrada_real_prev = prev_df[prev_df[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
    salida_real_prev = prev_df[prev_df[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
    eficiencia_prev = salida_real_prev / salida_proj * 100 if salida_proj > 0 else None

    delta_entrada = entrada_real - entrada_real_prev
    delta_salida = salida_real - salida_real_prev
    delta_ef = eficiencia - eficiencia_prev if eficiencia and eficiencia_prev else None

    col1.metric("Entrada Real", f"{int(entrada_real)}", f"{delta_entrada:+}")
    col2.metric("Salida Real", f"{int(salida_real)}", f"{delta_salida:+}")
    col3.metric("Eficiencia (%)", f"{eficiencia:.1f}%" if eficiencia else "-", f"{delta_ef:+.1f}%" if delta_ef else "-")
    if not wip.empty and pd.notnull(wip.mean()):
        col4.metric("WIP Promedio", f"{wip.mean():.1f}")
        col5.metric("WIP Delta", f"{wip_delta:+}")
    else:
        col4.metric("WIP Promedio", "-", "Sin datos")
        col5.metric("WIP Delta", "-", "Sin datos")
    if eficiencia and eficiencia < EFFICIENCY_GOAL:
        st.error(f"⚠️ Eficiencia debajo del objetivo ({EFFICIENCY_GOAL}%)")
    if not wip.empty and wip.max() > WIP_THRESHOLDS["Crítico"]:
        st.error(f"🚨 WIP crítico: {int(wip.max())} (lim. {WIP_THRESHOLDS['Crítico']})")
    elif not wip.empty and wip.max() > WIP_THRESHOLDS["Alerta"]:
        st.warning(f"⚠️ WIP en alerta: {int(wip.max())} (lim. {WIP_THRESHOLDS['Alerta']})")
except Exception as e:
    st.warning(f"No se pueden calcular KPIs: {e}")

# --- Visualización avanzada ---
st.subheader("📊 Evolución y Heatmap WIP")
fig = go.Figure()
for i, ind in enumerate(indicador_sel):
    data = df_filtrado_fecha[df_filtrado_fecha[col_indicador] == ind].sort_values("Fecha_dt")
    if not data.empty:
        fig.add_trace(go.Scatter(
            x=data["Fecha_dt"], y=data["Valor"],
            mode='lines+markers',
            name=ind,
            line=dict(color=CORPORATE_COLORS[i % len(CORPORATE_COLORS)], width=3),
            marker=dict(size=8),
            hovertemplate=f'Indicador: {ind}<br>Fecha: %{{x}}<br>Valor: %{{y}}<extra></extra>'
        ))
fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Valor",
    legend_title="Indicador",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

wip_inds = [w for w in indicador_sel if "wip" in w.lower()]
if wip_inds and not df_filtrado_fecha.empty:
    df_wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].isin(wip_inds)].copy()
    df_wip = df_wip.sort_values("Fecha_dt")
    df_wip_pivot = df_wip.pivot(index=col_indicador, columns="Fecha_dt", values="Valor")
    df_wip_pivot = df_wip_pivot.sort_index(axis=1)
    colorscale = [
        [0, "#A3FFAE"], [0.6, "#F6AE2D"], [0.9, "#F74B36"], [1, "#8B0000"]
    ]
    vmax = max(WIP_THRESHOLDS["Crítico"] + 200, float(df_wip_pivot.max().max() if not df_wip_pivot.empty else 0))
    fig_hm = px.imshow(
        df_wip_pivot,
        aspect="auto",
        color_continuous_scale=colorscale,
        zmin=0,
        zmax=vmax,
        labels=dict(color="WIP"),
        text_auto=True
    )
    fig_hm.update_xaxes(
        tickvals=list(df_wip_pivot.columns),
        ticktext=[d.strftime("%d-%b") for d in df_wip_pivot.columns]
    )
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Selecciona un indicador WIP y rango de fechas válido para ver el heatmap.")

if not df_filtrado_fecha.empty:
    csv = df_filtrado_fecha.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv'
    )

with st.expander("🗂️ Mostrar/ocultar hoja original de Google Sheets"):
    if st.button("Mostrar hoja original"):
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Haz clic en el botón para mostrar la hoja completa sólo si la necesitas.")
