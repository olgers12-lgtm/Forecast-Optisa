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
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="🚀 Dashboard Ejecutivo de Producción", layout="wide", initial_sidebar_state="expanded")

# --- Paleta y thresholds ---
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

def linear_trend(x, y, future_days=3):
    # x: fechas como ordinal, y: valores
    if len(x) < 3:
        return None, None  # No hay suficiente data
    x_ord = np.array([d.toordinal() for d in x]).reshape(-1, 1)
    y_np = np.array(y)
    reg = LinearRegression()
    reg.fit(x_ord, y_np)
    x_pred = np.array([x[-1].toordinal() + i for i in range(1, future_days+1)]).reshape(-1, 1)
    y_pred = reg.predict(x_pred)
    fechas_pred = [x[-1] + pd.Timedelta(days=i) for i in range(1, future_days+1)]
    return fechas_pred, y_pred

# --- Sidebar industrial (navegación KPIs) ---
with st.sidebar:
    st.header("🔎 Navegación KPIs")
    kpi_panel = st.radio("Panel:", ["KPIs Productivos", "Visualización", "Recomendaciones", "Ayuda"])

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

# --- Filtros avanzados ---
agrupamiento = st.radio("Agrupar por:", ["Día", "Semana (L-D)", "Mes", "Rango personalizado"], horizontal=True)
mask_fecha = pd.Series([True] * len(df_melt))
if agrupamiento == "Día":
    fechas_unicas = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").dropna().unique().tolist()
    fechas_unicas = sorted(fechas_unicas, key=lambda x: pd.to_datetime(x, format="%d-%b-%Y"))
    fechas_sel = st.multiselect("Selecciona días:", fechas_unicas, default=fechas_unicas[-7:])
    mask_fecha = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").isin(fechas_sel)
elif agrupamiento == "Semana (L-D)":
    semanas_disponibles = sorted(df_melt["Fecha_dt"].dt.isocalendar().week.dropna().unique())
    semana_sel = st.select_slider("Selecciona semana:", options=semanas_disponibles, value=semanas_disponibles[-1] if semanas_disponibles else None)
    mask_fecha = df_melt["Fecha_dt"].dt.isocalendar().week == semana_sel
elif agrupamiento == "Mes":
    meses_disponibles = sorted(df_melt["Fecha_dt"].dt.strftime("%B %Y").dropna().unique())
    mes_sel = st.selectbox("Selecciona mes:", options=meses_disponibles, index=len(meses_disponibles)-1 if meses_disponibles else 0)
    mask_fecha = df_melt["Fecha_dt"].dt.strftime("%B %Y") == mes_sel
else:
    fecha_min, fecha_max = df_melt["Fecha_dt"].min(), df_melt["Fecha_dt"].max()
    fecha_inicio, fecha_fin = st.date_input("Rango de fechas:", [fecha_max - pd.Timedelta(days=14), fecha_max])
    mask_fecha = (df_melt["Fecha_dt"] >= pd.to_datetime(fecha_inicio)) & (df_melt["Fecha_dt"] <= pd.to_datetime(fecha_fin))

df_filtrado_fecha = df_melt[mask_fecha]

# --- KPIs productivos y Delta vs periodo anterior ---
if kpi_panel == "KPIs Productivos":
    st.subheader("🧮 KPIs Industriales y Delta")
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

        # KPIs visuales
        col1.metric("Entrada Real", f"{int(entrada_real)}", f"{delta_entrada:+}")
        col2.metric("Salida Real", f"{int(salida_real)}", f"{delta_salida:+}")
        col3.metric("Eficiencia (%)", f"{eficiencia:.1f}%" if eficiencia else "-", f"{delta_ef:+.1f}%" if delta_ef else "-")
        if not wip.empty and pd.notnull(wip.mean()):
            col4.metric("WIP Promedio", f"{wip.mean():.1f}")
            col5.metric("WIP Delta", f"{wip_delta:+}")
        else:
            col4.metric("WIP Promedio", "-", "Sin datos")
            col5.metric("WIP Delta", "-", "Sin datos")
        # Alertas automáticas
        if eficiencia and eficiencia < EFFICIENCY_GOAL:
            st.error(f"⚠️ Eficiencia está debajo del objetivo ({EFFICIENCY_GOAL}%)")
        if not wip.empty and wip.max() > WIP_THRESHOLDS["Crítico"]:
            st.error(f"🚨 WIP crítico: {int(wip.max())} (lim. {WIP_THRESHOLDS['Crítico']})")
        elif not wip.empty and wip.max() > WIP_THRESHOLDS["Alerta"]:
            st.warning(f"⚠️ WIP en alerta: {int(wip.max())} (lim. {WIP_THRESHOLDS['Alerta']})")
    except Exception as e:
        st.warning(f"No se pueden calcular KPIs: {e}")

# --- Visualización avanzada ---
if kpi_panel == "Visualización":
    st.subheader("📊 Evolución y Heatmap WIP")
    # Gráfico temporal interactivo
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
                hovertemplate=f'Indicador: {ind}<br>Fecha: %{x}<br>Valor: %{y}<extra></extra>'
            ))
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor",
        legend_title="Indicador",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap WIP por fecha
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

    # --- Predicción simple de WIP ---
    st.subheader("🧮 Predicción WIP (Regresión lineal)")
    if not df_filtrado_fecha.empty and wip_inds:
        wip_pred_data = df_filtrado_fecha[df_filtrado_fecha[col_indicador].isin(wip_inds)].sort_values("Fecha_dt")
        x = list(wip_pred_data["Fecha_dt"])
        y = list(wip_pred_data["Valor"])
        fechas_pred, y_pred = linear_trend(x, y, future_days=3)
        if fechas_pred is not None:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=x, y=y, mode='lines+markers', name="WIP Real", line=dict(color="#0D8ABC")
            ))
            fig_pred.add_trace(go.Scatter(
                x=fechas_pred, y=y_pred, mode='markers+lines', name="WIP Predicho", line=dict(dash='dash', color="#F74B36"), marker=dict(size=10)
            ))
            fig_pred.update_layout(
                xaxis_title="Fecha",
                yaxis_title="WIP",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            max_pred = max(y_pred)
            if max_pred > WIP_THRESHOLDS["Crítico"]:
                st.error(f"🚨 Predicción: WIP podría superar el crítico ({max_pred:.0f}) en {fechas_pred[np.argmax(y_pred)].strftime('%d-%b')}")
            elif max_pred > WIP_THRESHOLDS["Alerta"]:
                st.warning(f"⚠️ Predicción: WIP en alerta ({max_pred:.0f})")
        else:
            st.info("No hay suficientes datos para predecir WIP.")
    else:
        st.info("Selecciona indicador WIP y rango válido para ver predicción.")

# --- Panel de recomendaciones automáticas ---
if kpi_panel == "Recomendaciones":
    st.subheader("💡 Recomendaciones automáticas")
    feedback = []
    # KPIs
    entrada_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
    salida_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
    salida_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida proyectada")]['Valor'].sum()
    eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None
    wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("wip")]['Valor']

    if eficiencia and eficiencia < EFFICIENCY_GOAL:
        feedback.append("✔️ Revisa causas de baja eficiencia: ¿problemas en procesos críticos, paros, cuellos de botella?")
    if not wip.empty and wip.max() > WIP_THRESHOLDS["Crítico"]:
        feedback.append("🚨 WIP crítico, considera liberar capacidad, ajustar lotes o priorizar despachos.")
    elif not wip.empty and wip.max() > WIP_THRESHOLDS["Alerta"]:
        feedback.append("⚠️ WIP alto, monitorea acumulación y revisa sincronización de entrada/salida.")
    if salida_real < salida_proj:
        feedback.append("🔺 Salida por debajo del plan, revisa horas hombre y disponibilidad de equipo.")
    if not feedback:
        feedback.append("✅ Todos los KPIs dentro de parámetros industriales. Mantener monitoreo.")
    for f in feedback:
        st.info(f)

# --- Panel de ayuda y onboarding ---
if kpi_panel == "Ayuda":
    st.header("🆘 Ayuda y onboarding")
    st.markdown("""
    - **Gráficos interactivos**: Haz hover sobre los puntos para ver tooltip con interpretación industrial.
    - **Predicción**: El dashboard estima el WIP para los próximos 3 días usando ML simple.
    - **KPIs con delta**: Todos los KPIs muestran variación vs periodo anterior.
    - **Alertas**: Colores y mensajes automáticos según thresholds industriales.
    - **Panel lateral**: Navega entre KPIs, visuales y recomendaciones.
    - **Descarga y compartir**: Exporta datos y comparte el dashboard con tu equipo.
    """)
    st.video("https://www.youtube.com/watch?v=JvS2triCgOY")  # Video demo genérico de dashboard industrial
    st.markdown("¿Dudas o soporte TI? [Contactar equipo industrial](mailto:soporte@tufabrica.com)")

# --- Panel para mostrar la hoja original ---
with st.expander("🗂️ Mostrar/ocultar hoja original de Google Sheets"):
    if st.button("Mostrar hoja original"):
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Haz clic en el botón para mostrar la hoja completa sólo si la necesitas.")

# --- Descarga de datos filtrados ---
if not df_filtrado_fecha.empty:
    csv = df_filtrado_fecha.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv'
    )
