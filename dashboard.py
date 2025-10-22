"""
Dashboard de Producción — versión limpia y visualmente mejorada

- Se eliminaron todas las secciones ML/IA.
- Sidebar rediseñado con "cards" (CSS ligero), botones Select All / Clear, multiselect con scroll.
- Parsing robusto de encabezados (agregar_ano) para detectar columnas fecha/futuras.
- Filtros Día / Semana / Mes / Rango y selectores creados dinámicamente.
- KPIs, gráficos históricos, heatmap WIP y descarga CSV incluidos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import re
from datetime import datetime, timedelta

# ---------- CONFIG ----------
st.set_page_config(
    page_title="🚀 Dashboard de Producción",
    layout="wide",
    initial_sidebar_state="expanded"
)

CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36", "#A3FFAE"
]
WIP_THRESHOLDS = {"Alerta": 1200, "Crítico": 1500}

SHEET_ID = "1U3DwxRVqQFwuPUs0-zvmitgz_LWdhScy-3fu-awBOHU"
SHEET_NAME = "Produccion"

# ---------- UTIL: Cargar datos ----------
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

# ---------- UTIL: parsing robusto de encabezados a fecha ----------
def agregar_ano(col):
    """
    Intenta convertir el texto de la cabecera de columna a 'YYYY-MM-DD'.
    Maneja formatos ISO, 'día mes [año]', abreviaturas español/inglés, etc.
    Devuelve string 'YYYY-MM-DD' o None si no pudo parsear.
    """
    col_orig = str(col).strip()
    if not col_orig:
        return None

    # 1) Intentar parse directo con pandas
    try:
        dt = pd.to_datetime(col_orig, dayfirst=True, yearfirst=False, errors="coerce")
        if not pd.isna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    # 2) Mapa de meses (español/inglés abrevs y completos)
    mes_map = {
        "ene":"01","ene.":"01","enero":"01",
        "feb":"02","feb.":"02","febrero":"02",
        "mar":"03","mar.":"03","marzo":"03",
        "abr":"04","abr.":"04","abril":"04",
        "may":"05","mayo":"05",
        "jun":"06","jun.":"06","junio":"06",
        "jul":"07","jul.":"07","julio":"07",
        "ago":"08","ago.":"08","agosto":"08",
        "sep":"09","sep.":"09","sept":"09","sept.":"09","septiembre":"09",
        "oct":"10","oct.":"10","octubre":"10",
        "nov":"11","nov.":"11","noviembre":"11",
        "dic":"12","dic.":"12","diciembre":"12","dec":"12","dec.":"12"
    }

    # 3) Regex: día + mes (posible año opcional)
    m = re.match(r"^\s*(\d{1,2})\s*[-/\\\s]?\s*([A-Za-zñÑ\.]+)(?:[-/\\\s]?(\d{2,4}))?\s*$", col_orig)
    if m:
        dia = int(m.group(1))
        mes_txt = m.group(2).lower().strip().replace(".", "")
        año_txt = m.group(3)
        mes_key = mes_txt[:4] if len(mes_txt) >= 3 else mes_txt
        # buscar clave completa en mes_map
        mes_num = None
        if mes_txt in mes_map:
            mes_num = mes_map[mes_txt]
        else:
            for k in mes_map.keys():
                if k.startswith(mes_key):
                    mes_num = mes_map[k]
                    break
        if not mes_num:
            return None
        # determinar año
        if año_txt:
            año = int(año_txt)
            if año < 100:
                año += 2000
        else:
            año = datetime.today().year  # por defecto año actual
        try:
            dt = pd.Timestamp(year=año, month=int(mes_num), day=dia)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    # 4) Regex ISO explícito y otros formatos numéricos
    m2 = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", col_orig)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            dt = pd.Timestamp(year=y, month=mo, day=d)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    # No se pudo parsear
    return None

# ---------- Cargar y preparar datos ----------
df = cargar_datos(SHEET_ID, SHEET_NAME)
col_indicador = next((c for c in df.columns if "indicador" in c.lower()), None)
if not col_indicador:
    st.error("No se encontró columna 'Indicador'. Las columnas son: " + str(df.columns.tolist()))
    st.stop()
df = df[df[col_indicador].notnull() & (df[col_indicador] != '')]

# ---------- SIDEBAR: styled improved ----------
# Inyectar CSS ligero para mejorar visual
st.markdown(
    """
    <style>
    /* Card look for sidebar groups */
    .sidebar-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(19, 38, 63, 0.04);
        border: 1px solid rgba(30, 41, 59, 0.04);
    }
    .sidebar-card h4 {
        margin: 0 0 8px 0;
        font-size: 14px;
        color: #0D8ABC;
    }
    .sidebar-smallnote {
        color: #6b7280; font-size:12px; margin-top:8px;
    }
    /* Force multiselect box to have max height and internal scroll */
    .stMultiSelect > div[role="listbox"] {
        max-height: 170px !important;
        overflow: auto !important;
    }
    /* Buttons small style */
    .side-btns .stButton>button {
        padding: 6px 10px;
        border-radius: 8px;
        background: #ffffff;
        border: 1px solid rgba(30,41,59,0.08);
    }
    .sidebar-compact { gap: 8px; display:flex; flex-direction:column; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<div class='sidebar-card sidebar-compact'>", unsafe_allow_html=True)
    st.markdown("<h4>📅 Controles / Filtros</h4>", unsafe_allow_html=True)

    # compact checkbox + label inline
    col_a, col_b = st.columns([0.12, 0.88])
    with col_a:
        include_future = st.checkbox("", value=True, key="chk_include_future")
    with col_b:
        st.markdown("<div style='font-size:13px;margin-top:4px;'>Incluir fechas<br><span style='color:#6b7280;font-size:11px;'>proyectadas / futuras</span></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # INDICADORES card
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<h4>🧾 Indicadores</h4>", unsafe_allow_html=True)

    indicadores = df[col_indicador].unique().tolist()

    # Quick action buttons
    st.markdown("<div class='side-btns'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1,1])
    if col1.button("Seleccionar todo", key="btn_select_all_ind"):
        st.session_state["sidebar_indicadores"] = indicadores
    if col2.button("Borrar selección", key="btn_clear_ind"):
        st.session_state["sidebar_indicadores"] = []
    st.markdown("</div>", unsafe_allow_html=True)

    indicador_sel = st.multiselect(
        "Selecciona indicadores",
        options=indicadores,
        default=st.session_state.get("sidebar_indicadores", [i for i in indicadores if "real" in i.lower() or "proyect" in i.lower() or "wip" in i.lower()]),
        key="sidebar_indicadores"
    )

    st.markdown("<div class='sidebar-smallnote'>Usa la búsqueda para filtrar. Usa los botones para seleccionar o limpiar.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # FECHA card
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<h4>📅 Filtro de fechas</h4>", unsafe_allow_html=True)

    filtro_tipo = st.radio(
        "Agrupar por:",
        ["🗓️ Día", "📆 Semana", "🗓️ Mes", "🎯 Rango personalizado"],
        index=0,
        key="sidebar_filtro_tipo"
    )

    st.markdown("<div class='sidebar-smallnote'>Selecciona cómo quieres agrupar y después el rango/fecha.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Construir lista de columnas válidas (fechas) ----------
fechas_cols = [c for c in df.columns if c != col_indicador]
fechas_dt_parsed = [agregar_ano(c) for c in fechas_cols]
today_ts = pd.Timestamp(datetime.today().date())

fechas_validas = []
for col, parsed in zip(fechas_cols, fechas_dt_parsed):
    if parsed is None:
        continue
    try:
        parsed_ts = pd.Timestamp(parsed)
    except Exception:
        continue
    if not include_future and parsed_ts > today_ts:
        continue
    fechas_validas.append(col)

# ---------- Melt para análisis (solo columnas válidas) ----------
if not st.session_state.get("sidebar_indicadores"):
    indicador_sel = [i for i in df[col_indicador].unique().tolist() if "real" in i.lower() or "proyect" in i.lower() or "wip" in i.lower()]
else:
    indicador_sel = st.session_state.get("sidebar_indicadores", [])

df_melt = df[df[col_indicador].isin(indicador_sel)].melt(
    id_vars=[col_indicador],
    value_vars=fechas_validas,
    var_name='Fecha',
    value_name='Valor'
)
df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"].apply(agregar_ano), format="%Y-%m-%d", errors="coerce")
df_melt["Valor"] = pd.to_numeric(df_melt["Valor"], errors="coerce")
df_melt = df_melt.dropna(subset=["Fecha_dt"])

# ---------- SIDEBAR: crear selectores de fecha ahora que df_melt existe ----------
with st.sidebar:
    if filtro_tipo == "🗓️ Día":
        fechas_disponibles = sorted(df_melt["Fecha_dt"].dt.date.unique()) if not df_melt.empty else []
        default_dia = datetime.today().date() if datetime.today().date() in fechas_disponibles else (fechas_disponibles[-1] if fechas_disponibles else datetime.today().date())
        fechas_sel = st.date_input(
            "Selecciona uno o más días:",
            value=default_dia,
            min_value=min(fechas_disponibles) if fechas_disponibles else None,
            max_value=max(fechas_disponibles) if fechas_disponibles else None,
            format="DD/MM/YYYY",
            key="sidebar_fechas_sel"
        )
    elif filtro_tipo == "📆 Semana":
        if not df_melt.empty:
            df_melt["SemanaISO"] = df_melt["Fecha_dt"].dt.isocalendar().week
            df_melt["AñoISO"] = df_melt["Fecha_dt"].dt.isocalendar().year
            semanas_unicas = sorted(df_melt[["AñoISO", "SemanaISO"]].drop_duplicates().values.tolist())
        else:
            semanas_unicas = []
        if semanas_unicas:
            try:
                semana_default_idx = next(i for i, (year, week) in enumerate(semanas_unicas) if year == datetime.today().isocalendar().year and week == datetime.today().isocalendar().week)
            except StopIteration:
                semana_default_idx = 0
            semana_labels = [
                f"Semana {week} ({df_melt[(df_melt['AñoISO'] == year) & (df_melt['SemanaISO'] == week)]['Fecha_dt'].min().strftime('%d-%b')} - {df_melt[(df_melt['AñoISO'] == year) & (df_melt['SemanaISO'] == week)]['Fecha_dt'].max().strftime('%d-%b')})"
                for year, week in semanas_unicas
            ]
            semana_sel_idx = st.selectbox(
                "Selecciona semana:", options=range(len(semanas_unicas)),
                format_func=lambda i: semana_labels[i],
                index=semana_default_idx,
                key="sidebar_semana_sel"
            )
        else:
            semana_sel_idx = None
    elif filtro_tipo == "🗓️ Mes":
        meses_disponibles = sorted(df_melt["Fecha_dt"].dt.strftime("%B %Y").unique()) if not df_melt.empty else []
        if meses_disponibles:
            mes_sel = st.selectbox("Selecciona mes:", options=meses_disponibles, key="sidebar_mes_sel")
        else:
            mes_sel = None
    else:
        fecha_min = df_melt["Fecha_dt"].min() if not df_melt.empty else pd.Timestamp(datetime.today().date())
        fecha_max = df_melt["Fecha_dt"].max() if not df_melt.empty else pd.Timestamp(datetime.today().date())
        fecha_min_date = fecha_min.date()
        fecha_max_date = fecha_max.date()
        try:
            value_default = (datetime.today().date(), datetime.today().date()) if fecha_min_date <= datetime.today().date() <= fecha_max_date else (fecha_max_date - timedelta(days=7), fecha_max_date)
        except Exception:
            value_default = (fecha_max_date - timedelta(days=7), fecha_max_date)
        fecha_rango = st.date_input(
            "Selecciona el rango de fechas:",
            value=value_default,
            min_value=fecha_min_date,
            max_value=fecha_max_date,
            format="DD/MM/YYYY",
            key="sidebar_fecha_rango"
        )

# ---------- Construir mask_fecha según selects ----------
if filtro_tipo == "🗓️ Día":
    if isinstance(fechas_sel, list):
        mask_fecha = df_melt["Fecha_dt"].dt.date.isin(fechas_sel)
    else:
        mask_fecha = df_melt["Fecha_dt"].dt.date == fechas_sel
    agrupador = "dia"
elif filtro_tipo == "📆 Semana":
    if 'semanas_unicas' in locals() and semanas_unicas:
        year_sel, week_sel = semanas_unicas[semana_sel_idx]
        mask_fecha = (df_melt["AñoISO"] == year_sel) & (df_melt["SemanaISO"] == week_sel)
    else:
        mask_fecha = pd.Series([False]*len(df_melt))
    agrupador = "semana"
elif filtro_tipo == "🗓️ Mes":
    if 'mes_sel' in locals() and mes_sel:
        mask_fecha = df_melt["Fecha_dt"].dt.strftime("%B %Y") == mes_sel
    else:
        mask_fecha = pd.Series([False]*len(df_melt))
    agrupador = "mes"
else:
    try:
        fecha_inicio, fecha_fin = [pd.Timestamp(f) for f in fecha_rango]
        mask_fecha = (df_melt["Fecha_dt"] >= fecha_inicio) & (df_melt["Fecha_dt"] <= fecha_fin)
    except Exception:
        mask_fecha = pd.Series([False]*len(df_melt))
    agrupador = "rango"

df_filtrado_fecha = df_melt[mask_fecha]

# --------- KPIs INDUSTRIALES DINÁMICOS ACUMULADO AL CORTE ---------
st.markdown("<h2 style='color:#F6AE2D'>🧮 KPIs Optisa</h2>", unsafe_allow_html=True)
kpi_cols = st.columns(4)

if not df_filtrado_fecha.empty:
    fecha_corte = df_filtrado_fecha["Fecha_dt"].max()
    df_corte = df_filtrado_fecha[df_filtrado_fecha["Fecha_dt"] <= fecha_corte]
    entrada_real = df_corte[df_corte[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
    entrada_proj = df_corte[df_corte[col_indicador].str.lower().str.contains("entrada-proyectada|entrada proyectada")]['Valor'].sum()
    salida_real = df_corte[df_corte[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
    salida_proj = df_corte[df_corte[col_indicador].str.lower().str.contains("salida proyectada|salida-proyectada")]['Valor'].sum()
    wip = df_corte[df_corte[col_indicador].str.lower().str.contains("wip")]['Valor']
    delta_entrada = entrada_real - entrada_proj if entrada_proj else None
    delta_salida = salida_real - salida_proj if salida_proj else None
    eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None
else:
    fecha_corte = pd.Timestamp(datetime.today().date())
    entrada_real = entrada_proj = salida_real = salida_proj = 0
    delta_entrada = delta_salida = eficiencia = None
    wip = pd.Series(dtype=float)

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

# --- Alertas WIP solo hoy ---
df_hoy = df_filtrado_fecha[df_filtrado_fecha["Fecha_dt"].dt.date == datetime.today().date()]
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

# --- Entradas/Salidas Proyectadas ACUMULADAS abajo (siempre visible) ---
st.markdown("""
<div style='margin-bottom:20px;padding:12px 16px;background:#f7f7f7;border-radius:12px;box-shadow:0 2px 8px #eee;display:flex;gap:36px;justify-content:center;'>
    <div style='font-size:20px;color:#0D8ABC;'><strong>Entrada Proyectada:</strong> {}</div>
    <div style='font-size:20px;color:#F6AE2D;'><strong>Salida Proyectada:</strong> {}</div>
    <div style='font-size:20px;color:#61C0BF;'><strong>Eficiencia Proyectada (%):</strong> {}</div>
</div>
""".format(
    int(entrada_proj) if pd.notnull(entrada_proj) else "-",
    int(salida_proj) if pd.notnull(salida_proj) else "-",
    f"{(salida_proj/entrada_proj*100):.1f}" if entrada_proj and pd.notnull(salida_proj) and entrada_proj>0 else "-"
), unsafe_allow_html=True)

# ---------- GRÁFICOS HISTÓRICOS ----------
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

# ---------- HEATMAP WIP ----------
wip_inds = [w for w in indicador_sel if "wip" in w.lower()]
if wip_inds and not df_filtrado_fecha.empty:
    df_wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].isin(wip_inds)].copy()
    df_wip = df_wip.sort_values("Fecha_dt")
    df_wip_pivot = df_wip.pivot(index=col_indicador, columns="Fecha_dt", values="Valor")
    df_wip_pivot = df_wip_pivot.sort_index(axis=1)
    colorscale = [
        [0.0, "#A3FFAE"],
        [0.6, "#B6FF6B"],
        [0.67, "#F6F658"],
        [0.8, "#F6AE2D"],
        [0.92, "#F74B36"],
        [1.0, "#8B0000"]
    ]
    vmax = max(1500, float(df_wip_pivot.max().max() if not df_wip_pivot.empty else 0))
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
    fig_hm.update_layout(font=dict(family="Segoe UI,Roboto,Arial", size=15), margin=dict(l=30, r=30, t=40, b=40))
    st.markdown("<h2 style='color:#61C0BF'>🌡️ Heatmap WIP</h2>", unsafe_allow_html=True)
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("Selecciona un indicador WIP y rango de fechas válido para ver el heatmap.")

# ---------- DESCARGA ----------
if not df_filtrado_fecha.empty:
    csv = df_filtrado_fecha.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv',
        key="download_filtered_csv"
    )

# ---------- EXPANDER: Mostrar hoja original (checkbox con key único) ----------
with st.expander("🗂️ Mostrar/ocultar hoja original de Google Sheets"):
    if st.checkbox("Mostrar hoja original", key="chk_show_sheet_main"):
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Haz clic en 'Mostrar hoja original' para ver la hoja completa sólo si la necesitas.")
