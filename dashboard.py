import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import re
from datetime import datetime

# --- CONFIGURACIÓN MODERNA ---
st.set_page_config(
    page_title="🚀 Dashboard Ejecutivo de Producción",
    layout="wide",
    initial_sidebar_state="expanded"
)

CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36", "#A3FFAE"
]
WIP_THRESHOLDS = {"Alerta": 1200, "Crítico": 1500}
EFFICIENCY_GOAL = 95

SHEET_ID = "1U3DwxRVqQFwuPUs0-zvmitgz_LWdhScy-3fu-awBOHU"
SHEET_NAME = "Produccion"

@st.cache_data(ttl=600)
def cargar_datos(sheet_id, sheet_name):
    sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
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

# --- DATA PREP ---
df = cargar_datos(SHEET_ID, SHEET_NAME)
col_indicador = next((c for c in df.columns if "indicador" in c.lower()), None)
if not col_indicador:
    st.error("No se encontró columna 'Indicador'. Las columnas son: " + str(df.columns.tolist()))
    st.stop()
df = df[df[col_indicador].notnull() & (df[col_indicador] != '')]

fechas = [c for c in df.columns if c != col_indicador]
fechas_dt = [agregar_ano(f) for f in fechas]
fechas_validas = [fechas[i] for i in range(len(fechas)) if fechas_dt[i] is not None]
fechas_dt_validas = [f for f in fechas_dt if f is not None]

indicadores = df[col_indicador].unique().tolist()
indicador_sel = st.multiselect(
    "Selecciona indicadores:",
    indicadores,
    default=[i for i in indicadores if "real" in i.lower() or "proyectada" in i.lower() or "wip" in i.lower()]
)

df_melt = df[df[col_indicador].isin(indicador_sel)].melt(
    id_vars=[col_indicador],
    value_vars=fechas_validas,
    var_name='Fecha',
    value_name='Valor'
)
df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"].apply(agregar_ano), format="%Y-%m-%d", errors="coerce")
df_melt["Valor"] = pd.to_numeric(df_melt["Valor"], errors="coerce")
df_melt = df_melt.dropna(subset=["Fecha_dt"])

# --- SIDEBAR FILTROS MODERNOS ---
with st.sidebar:
    st.markdown("<h2 style='color:#0D8ABC'>📅 Filtros de Fechas</h2>", unsafe_allow_html=True)
    filtro_tipo = st.radio(
        "Agrupar por:",
        ["🗓️ Día", "📆 Semana", "🗓️ Mes", "🎯 Rango personalizado"],
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
        df_melt["SemanaISO"] = df_melt["Fecha_dt"].dt.isocalendar().week
        df_melt["AñoISO"] = df_melt["Fecha_dt"].dt.isocalendar().year
        semanas_unicas = sorted(df_melt[["AñoISO", "SemanaISO"]].drop_duplicates().values.tolist())
        semana_labels = [
            f"<span style='color:#0D8ABC;font-weight:bold;'>Semana {week}</span><span style='color:#555;'> ({df_melt[(df_melt['AñoISO'] == year) & (df_melt['SemanaISO'] == week)]['Fecha_dt'].min().strftime('%d-%b')} - {df_melt[(df_melt['AñoISO'] == year) & (df_melt['SemanaISO'] == week)]['Fecha_dt'].max().strftime('%d-%b')})</span>"
            for year, week in semanas_unicas
        ]
        semana_sel_idx = st.selectbox(
            "Selecciona semana:", options=range(len(semanas_unicas)),
            format_func=lambda i: f"Semana {semanas_unicas[i][1]} ({df_melt[(df_melt['AñoISO'] == semanas_unicas[i][0]) & (df_melt['SemanaISO'] == semanas_unicas[i][1])]['Fecha_dt'].min().strftime('%d-%b')} - {df_melt[(df_melt['AñoISO'] == semanas_unicas[i][0]) & (df_melt['SemanaISO'] == semanas_unicas[i][1])]['Fecha_dt'].max().strftime('%d-%b')})"
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

# --- KPIs INDUSTRIALES ESTÉTICOS ---
st.markdown("<h2 style='color:#F6AE2D'>🧮 KPIs Industriales</h2>", unsafe_allow_html=True)
kpi_cols = st.columns(4)

entrada_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
entrada_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada-proyectada")]['Valor'].sum()
salida_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
salida_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida proyectada")]['Valor'].sum()
wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("wip")]['Valor']

# Delta vs Proyectado
delta_entrada = entrada_real - entrada_proj if entrada_proj else None
delta_salida = salida_real - salida_proj if salida_proj else None
eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None

delta_entrada_str = f"{delta_entrada:.1f}" if delta_entrada is not None else "-"
delta_salida_str = f"{delta_salida:.1f}" if delta_salida is not None else "-"
eficiencia_str = f"{eficiencia:.1f}" if eficiencia is not None else "-"
wip_prom_str = f"{wip.mean():.1f}" if not wip.empty and pd.notnull(wip.mean()) else "-"

color_ent = '#F74B36' if delta_entrada is not None and delta_entrada < 0 else '#61C0BF'
color_sal = '#F74B36' if delta_salida is not None and delta_salida < 0 else '#61C0BF'

kpi_cols[0].markdown(
    f"""
    <div style='background:#fff;border-radius:12px;padding:20px 10px;box-shadow:0 2px 8px #eee;text-align:center'>
      <div style='font-size:20px;font-weight:bold;'>Entrada Real</div>
      <div style='font-size:34px;font-weight:bold;color:#1F2A56'>{int(entrada_real):,}</div>
      <span style='font-size:20px;color:{color_ent};font-weight:bold;'>{delta_entrada_str}</span>
    </div>
    """, unsafe_allow_html=True
)
kpi_cols[1].markdown(
    f"""
    <div style='background:#fff;border-radius:12px;padding:20px 10px;box-shadow:0 2px 8px #eee;text-align:center'>
      <div style='font-size:20px;font-weight:bold;'>Salida Real</div>
      <div style='font-size:34px;font-weight:bold;color:#1F2A56'>{int(salida_real):,}</div>
      <span style='font-size:20px;color:{color_sal};font-weight:bold;'>{delta_salida_str}</span>
    </div>
    """, unsafe_allow_html=True
)
kpi_cols[2].markdown(
    f"""
    <div style='background:#fff;border-radius:12px;padding:20px 10px;box-shadow:0 2px 8px #eee;text-align:center'>
      <div style='font-size:20px;font-weight:bold;'>Eficiencia (%)</div>
      <div style='font-size:34px;font-weight:bold;color:#1F2A56'>{eficiencia_str}</div>
    </div>
    """, unsafe_allow_html=True
)
kpi_cols[3].markdown(
    f"""
    <div style='background:#fff;border-radius:12px;padding:20px 10px;box-shadow:0 2px 8px #eee;text-align:center'>
      <div style='font-size:20px;font-weight:bold;'>WIP Promedio</div>
      <div style='font-size:34px;font-weight:bold;color:#1F2A56'>{wip_prom_str}</div>
    </div>
    """, unsafe_allow_html=True
)

# --- ALERTA WIP SOLO HOY ---
hoy = datetime.today().date()
df_hoy = df_filtrado_fecha[df_filtrado_fecha["Fecha_dt"].dt.date == hoy]
wip_hoy = df_hoy[df_hoy[col_indicador].str.lower().str.contains("wip")]['Valor']

if not wip_hoy.empty:
    wip_max_hoy = wip_hoy.max()
    if wip_max_hoy > WIP_THRESHOLDS["Crítico"]:
        st.markdown(
            f"<div style='background:#ffeaea;border-radius:8px;padding:10px;color:#F74B36;font-weight:bold;'>"
            f"🚨 WIP crítico: {int(wip_max_hoy)} (lim. {WIP_THRESHOLDS['Crítico']})"
            f"</div>", unsafe_allow_html=True
        )
    elif wip_max_hoy > WIP_THRESHOLDS["Alerta"]:
        st.markdown(
            f"<div style='background:#fffbe6;border-radius:8px;padding:10px;color:#F6AE2D;font-weight:bold;'>"
            f"⚠️ WIP en alerta: {int(wip_max_hoy)} (lim. {WIP_THRESHOLDS['Alerta']})"
            f"</div>", unsafe_allow_html=True
        )

# --- GRÁFICO TEMPORAL COOL ---
st.markdown("<h2 style='color:#3EC0ED'>📊 Evolución de Indicadores</h2>", unsafe_allow_html=True)
fig = go.Figure()
for i, ind in enumerate(indicador_sel):
    data = df_filtrado_fecha[df_filtrado_fecha[col_indicador] == ind].sort_values("Fecha_dt")
    if not data.empty:
        fig.add_trace(go.Scatter(
            x=data["Fecha_dt"].dt.strftime("%d-%b"),
            y=data["Valor"],
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
    template="plotly_white",
    font=dict(family="Segoe UI,Roboto,Arial", size=15),
    margin=dict(l=30, r=30, t=40, b=40)
)
st.plotly_chart(fig, use_container_width=True)

# --- HEATMAP WIP COOL ---
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
    fig_hm.update_layout(
        font=dict(family="Segoe UI,Roboto,Arial", size=15),
        margin=dict(l=30, r=30, t=40, b=40)
    )
    st.markdown("<h2 style='color:#61C0BF'>🌡️ Heatmap WIP</h2>", unsafe_allow_html=True)
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Selecciona un indicador WIP y rango de fechas válido para ver el heatmap.")

# --- DESCARGA Y HOJA ORIGINAL ---
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
