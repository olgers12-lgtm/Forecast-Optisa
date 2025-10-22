"""
Dashboard de Producción — versión corregida

Cambios principales aplicados en esta versión:
- Corregido AttributeError: ya no se accede a st.sidebar.session_state; se usa st.session_state correctamente.
- Los selectores de fecha (Día/Semana/Mes/Rango) están arriba en el sidebar, antes del bloque ML.
- La sección ML está oculta por defecto y solo se muestra si el usuario introduce la clave correcta (st.secrets["ML_PASSWORD"]).
- Eliminado por completo el panel "Resto del año" (no se renderiza).
- Evité colisiones de widgets (keys únicas) y dejé el resto del dashboard funcional como antes.
- Mantengo el pipeline ML (Prophet si está disponible; fallback con GradientBoosting).
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
    col_orig = str(col).strip()
    if not col_orig:
        return None
    try:
        dt = pd.to_datetime(col_orig, dayfirst=True, yearfirst=False, errors="coerce")
        if not pd.isna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
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
    m = re.match(r"^\s*(\d{1,2})\s*[-/\\\s]?\s*([A-Za-zñÑ\.]+)(?:[-/\\\s]?(\d{2,4}))?\s*$", col_orig)
    if m:
        dia = int(m.group(1))
        mes_txt = m.group(2).lower().strip().replace(".", "")
        año_txt = m.group(3)
        mes_key = mes_txt[:4] if len(mes_txt) >= 3 else mes_txt
        mes_num = None
        if mes_txt in mes_map:
            mes_num = mes_map[mes_txt]
        else:
            for k, v in mes_map.items():
                if k.startswith(mes_key):
                    mes_num = v
                    break
        if not mes_num:
            return None
        if año_txt:
            año = int(año_txt)
            if año < 100:
                año += 2000
        else:
            año = datetime.today().year
        try:
            dt = pd.Timestamp(year=año, month=int(mes_num), day=dia)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
    m2 = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$", col_orig)
    if m2:
        y, mo, d = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        try:
            dt = pd.Timestamp(year=y, month=mo, day=d)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
    return None

# ---------- Cargar datos ----------
df = cargar_datos(SHEET_ID, SHEET_NAME)
col_indicador = next((c for c in df.columns if "indicador" in c.lower()), None)
if not col_indicador:
    st.error("No se encontró columna 'Indicador'. Las columnas son: " + str(df.columns.tolist()))
    st.stop()
df = df[df[col_indicador].notnull() & (df[col_indicador] != '')]

# ---------- SIDEBAR: controles (ordenados) ----------
with st.sidebar:
    st.markdown("<h2 style='color:#0D8ABC'>📅 Controles / Filtros</h2>", unsafe_allow_html=True)

    # include future columns
    include_future = st.checkbox("Incluir fechas proyectadas/futuras (columnas nuevas)", value=True, key="chk_include_future")

    # Indicadores selection (moved to sidebar)
    indicadores = df[col_indicador].unique().tolist()
    indicador_sel = st.multiselect(
        "Selecciona indicadores:",
        indicadores,
        default=[i for i in indicadores if "real" in i.lower() or "proyect" in i.lower() or "wip" in i.lower()],
        key="sidebar_indicadores"
    )

    st.markdown("---")
    st.markdown("<b>Filtro de fechas</b>", unsafe_allow_html=True)
    filtro_tipo = st.radio(
        "Agrupar por:",
        ["🗓️ Día", "📆 Semana", "🗓️ Mes", "🎯 Rango personalizado"],
        horizontal=False,
        key="sidebar_filtro_tipo"
    )

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
if not indicador_sel:
    indicador_sel = [i for i in df[col_indicador].unique().tolist() if "real" in i.lower() or "proyect" in i.lower() or "wip" in i.lower()]

df_melt = df[df[col_indicador].isin(indicador_sel)].melt(
    id_vars=[col_indicador],
    value_vars=fechas_validas,
    var_name='Fecha',
    value_name='Valor'
)
df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"].apply(agregar_ano), format="%Y-%m-%d", errors="coerce")
df_melt["Valor"] = pd.to_numeric(df_melt["Valor"], errors="coerce")
df_melt = df_melt.dropna(subset=["Fecha_dt"])

# ---------- SIDEBAR: create date selectors (now that df_melt exists) ----------
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

    # --- ML unlock block moved AFTER date selectors so selectors are higher in sidebar
    st.markdown("---")
    st.markdown("<b>ML / IA (privado)</b>", unsafe_allow_html=True)
    # Initialize ml_unlocked in session state if missing
    if "ml_unlocked" not in st.session_state:
        st.session_state["ml_unlocked"] = False

    ml_password_input = st.text_input("Clave ML (solo admins)", type="password", key="ml_pass_input")
    secret_ml = st.secrets.get("ML_PASSWORD") if hasattr(st, "secrets") else None
    if ml_password_input:
        if secret_ml and ml_password_input == secret_ml:
            st.session_state["ml_unlocked"] = True
            st.success("Sección ML desbloqueada")
        elif not secret_ml:
            st.warning("ML_PASSWORD no encontrado en st.secrets. Añade la clave en secrets para protección real.", icon="⚠️")
        else:
            st.error("Clave ML incorrecta", icon="❌")

    # ML controls (visible only if unlocked) — we still render controls but they are inert until unlocked
    ml_param = st.selectbox("Indicador ML:", ["Salida Real", "Entrada Real"], key="ml_param_sidebar")
    ml_horizon = st.slider("Horizonte ML (días)", min_value=3, max_value=30, value=7, key="ml_horizon_sidebar")
    ml_lookback = st.number_input("Histórico ML (días)", min_value=30, max_value=365, value=90, step=30, key="ml_lookback_sidebar")

    if st.session_state["ml_unlocked"]:
        if st.button("Cerrar sesión ML", key="btn_ml_logout"):
            st.session_state["ml_unlocked"] = False
            st.success("Sesión ML cerrada")

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

# ---------- SECCIÓN ML (privada) ----------
# Read ML controls from session_state (they were created in the sidebar)
param_ml = st.session_state.get("ml_param_sidebar", "Salida Real")
horizon = st.session_state.get("ml_horizon_sidebar", 7)
lookback_days = st.session_state.get("ml_lookback_sidebar", 90)

if st.session_state.get("ml_unlocked", False):
    st.markdown("<hr>")
    st.markdown("<h2 style='color:#0D8ABC'>🤖 Predicción Inteligente (ML & IA) - Privado</h2>", unsafe_allow_html=True)

    # Reuse ML pipeline (Prophet if available, else fallback)
    def prepare_daily_series(df_melt_local, indicador_keyword, lookback_days_local):
        dfs = df_melt_local[df_melt_local[col_indicador].str.lower().str.contains(indicador_keyword)].copy()
        dfs = dfs.dropna(subset=["Fecha_dt", "Valor"])
        if dfs.empty:
            return pd.DataFrame()
        daily = dfs.groupby("Fecha_dt")["Valor"].sum().sort_index().to_frame(name="y")
        max_date = daily.index.max()
        min_date = max_date - pd.Timedelta(days=lookback_days_local)
        daily = daily[daily.index >= min_date]
        if daily.empty:
            return pd.DataFrame()
        q1 = daily["y"].quantile(0.25)
        q3 = daily["y"].quantile(0.75)
        iqr = q3 - q1 if q3 >= q1 else 0
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        daily["y_clipped"] = daily["y"].clip(lower=lower, upper=upper)
        return daily

    def build_features_from_series(series, lags=(1,2,3,7,14)):
        df_local = series.to_frame().copy()
        for lag in lags:
            df_local[f"lag_{lag}"] = df_local["y_clipped"].shift(lag)
        df_local["rolling_3"] = df_local["y_clipped"].rolling(3).mean()
        df_local["rolling_7"] = df_local["y_clipped"].rolling(7).mean()
        df_local["dow"] = df_local.index.dayofweek
        df_local["is_weekend"] = (df_local["dow"] >= 5).astype(int)
        df_local = df_local.dropna()
        return df_local

    ind_keyword = "salida real" if param_ml == "Salida Real" else "entrada real"
    ind_proj_keyword = "salida proyectada" if "salida" in ind_keyword else "entrada-proyectada"

    daily_ml = prepare_daily_series(df_melt, ind_keyword, lookback_days)

    if daily_ml.empty or daily_ml["y_clipped"].sum() == 0:
        st.warning("No hay suficientes datos limpios para entrenar/predicción ML. Ajusta filtros o selecciona otro indicador.")
    else:
        use_prophet = False
        try:
            from prophet import Prophet  # type: ignore
            use_prophet = True
        except Exception:
            use_prophet = False
            st.info("Prophet no disponible, usando fallback ML.", icon="ℹ️")

        if use_prophet:
            try:
                df_prophet = daily_ml.reset_index().rename(columns={daily_ml.reset_index().columns[0]: "ds", "y_clipped": "y"})
                df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
                df_prophet["is_weekend"] = df_prophet["ds"].dt.dayofweek >= 5
                df_prophet["dow"] = df_prophet["ds"].dt.dayofweek

                m = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.01,
                    seasonality_prior_scale=5,
                )
                m.add_regressor("is_weekend")
                m.add_regressor("dow")
                m.fit(df_prophet)

                future = m.make_future_dataframe(periods=horizon, freq="D")
                future["is_weekend"] = future["ds"].dt.dayofweek >= 5
                future["dow"] = future["ds"].dt.dayofweek
                forecast = m.predict(future)

                hist_min = df_prophet["y"].min()
                hist_max = df_prophet["y"].max()
                lower_clip = max(hist_min, 0)
                upper_clip = hist_max * 1.2 + 1
                forecast["yhat"] = forecast["yhat"].clip(lower=lower_clip, upper=upper_clip)
                forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=lower_clip, upper=upper_clip)
                forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=lower_clip, upper=upper_clip)

                fig_ml = go.Figure()
                fig_ml.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"], mode="lines+markers", name="Histórico filtrado", line=dict(color='#0D8ABC')))
                fig_ml.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicción ML (Prophet)", line=dict(color='#F6AE2D', dash='dash')))
                fig_ml.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], line=dict(color='rgba(246,174,45,0.2)'), showlegend=False))
                fig_ml.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill='tonexty', fillcolor='rgba(246,174,45,0.18)', line=dict(color='rgba(246,174,45,0.2)'), showlegend=False))

                df_proj = df_melt[df_melt[col_indicador].str.lower().str.contains(ind_proj_keyword)].copy()
                if not df_proj.empty:
                    proj_daily = df_proj.groupby("Fecha_dt")["Valor"].sum().sort_index()
                    fig_ml.add_trace(go.Scatter(x=proj_daily.index, y=proj_daily.values, mode="lines+markers", name="Proyección Área (Excel)", line=dict(color="#61C0BF", dash="dot")))

                fig_ml.update_layout(title=f"Predicción Prophet: {param_ml}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")
                st.plotly_chart(fig_ml, use_container_width=True)

            except Exception as e:
                st.warning("Prophet falló en tiempo de ejecución — usando fallback ML. Detalle: " + str(e))
                use_prophet = False

        if not use_prophet:
            try:
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.model_selection import TimeSeriesSplit
                from sklearn.metrics import mean_absolute_error
            except Exception as e:
                st.error("No se pudo importar scikit-learn. Instálalo (pip install scikit-learn) o arregla Prophet. Error: " + str(e))
                st.stop()

            series = daily_ml["y_clipped"].copy()
            df_feats = build_features_from_series(series, lags=(1,2,3,7,14))
            if df_feats.shape[0] < 10:
                st.warning("Pocas filas tras crear features. Aumenta lookback o mejora datos.")
            else:
                X = df_feats.drop(columns=["y", "y_clipped"], errors='ignore')
                y = df_feats["y"] if "y" in df_feats.columns else df_feats["y_clipped"]

                tscv = TimeSeriesSplit(n_splits=3)
                val_scores = []
                residuals = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    model_cv = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3)
                    model_cv.fit(X_train, y_train)
                    preds = model_cv.predict(X_val)
                    val_scores.append(mean_absolute_error(y_val, preds))
                    residuals.extend(list(y_val - preds))

                model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3)
                model.fit(X, y)

                last_known = series.copy()
                preds_dates = []
                preds_values = []
                for i in range(horizon):
                    next_date = last_known.index.max() + pd.Timedelta(days=1)
                    feats = {}
                    for lag in (1,2,3,7,14):
                        idx = next_date - pd.Timedelta(days=lag)
                        feats[f"lag_{lag}"] = last_known.get(idx, last_known.iloc[-1])
                    feats["rolling_3"] = last_known[-3:].mean() if len(last_known) >= 3 else last_known.mean()
                    feats["rolling_7"] = last_known[-7:].mean() if len(last_known) >= 7 else last_known.mean()
                    feats["dow"] = next_date.dayofweek
                    feats["is_weekend"] = int(feats["dow"] >= 5)
                    feat_df = pd.DataFrame([feats])
                    feat_df = feat_df.reindex(columns=X.columns, fill_value=0)
                    pred = model.predict(feat_df)[0]
                    hist_min = series.min()
                    hist_max = series.max()
                    pred = float(np.clip(pred, max(hist_min, 0), hist_max * 1.2 + 1))
                    preds_dates.append(next_date)
                    preds_values.append(pred)
                    last_known.loc[next_date] = pred

                resid_std = np.std(residuals) if residuals else np.std(y - model.predict(X))
                ci_upper = np.array(preds_values) + 1.96 * resid_std
                ci_lower = np.array(preds_values) - 1.96 * resid_std
                ci_lower = np.clip(ci_lower, 0, None)

                fig_fb = go.Figure()
                fig_fb.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines+markers", name="Histórico filtrado", line=dict(color="#0D8ABC")))
                fig_fb.add_trace(go.Scatter(x=df_feats.index, y=df_feats["y_clipped"].values, mode="lines", name="Histórico usado (features)", line=dict(color="#5BC0EB", dash="dot")))
                fig_fb.add_trace(go.Scatter(x=preds_dates, y=preds_values, mode="lines+markers", name="Predicción ML (GBM fallback)", line=dict(color="#F6AE2D", dash="dash")))
                fig_fb.add_trace(go.Scatter(x=preds_dates, y=ci_upper, line=dict(color="rgba(246,174,45,0.2)"), showlegend=False))
                fig_fb.add_trace(go.Scatter(x=preds_dates, y=ci_lower, fill='tonexty', fillcolor='rgba(246,174,45,0.18)', line=dict(color='rgba(246,174,45,0.2)'), showlegend=False))

                df_proj = df_melt[df_melt[col_indicador].str.lower().str.contains(ind_proj_keyword)].copy()
                if not df_proj.empty:
                    proj_daily = df_proj.groupby("Fecha_dt")["Valor"].sum().sort_index()
                    fig_fb.add_trace(go.Scatter(x=proj_daily.index, y=proj_daily.values, mode="lines+markers", name="Proyección Área (Excel)", line=dict(color="#61C0BF", dash="dot")))

                fig_fb.update_layout(title=f"Predicción ML fallback: {param_ml}", xaxis_title="Fecha", yaxis_title="Valor", template="plotly_white")
                st.plotly_chart(fig_fb, use_container_width=True)
                st.success("Predicción generada con fallback ML (GradientBoosting).")
                st.write(f"MAE CV estimado: {np.mean(val_scores):.1f}  —  Residual std approx: {resid_std:.1f}")

# ---------- EXPANDER: Mostrar hoja original (checkbox con key único) ----------
with st.expander("🗂️ Mostrar/ocultar hoja original de Google Sheets"):
    if st.checkbox("Mostrar hoja original", key="chk_show_sheet_main"):
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Haz clic en 'Mostrar hoja original' para ver la hoja completa sólo si la necesitas.")

# ---------- FOOTER ----------

