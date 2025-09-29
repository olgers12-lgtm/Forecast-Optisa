import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="📊 Dashboard Ejecutivo de Producción", layout="wide")
st.title("📊 Dashboard Ejecutivo de Producción")
st.caption("Interactivo, visual y actualizado en tiempo real")

CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36"
]

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

try:
    df = cargar_datos(SHEET_ID, SHEET_NAME)
    st.success("✅ Datos cargados correctamente.")
    st.dataframe(df.head(), use_container_width=True)

    # Detecta columnas no-fecha, no-'Indicador'
    cols = df.columns.tolist()
    fechas = [c for c in cols if c not in ("Indicador",) and not any(x in str(c).lower() for x in ["indicador"])]
    posibles_id_vars = [c for c in cols if c not in fechas]
    # Asegura 'Indicador' está presente
    id_vars = [c for c in posibles_id_vars if "indicador" in c.lower()]
    # Si hay otra columna extra, la añade
    otras = [c for c in posibles_id_vars if c not in id_vars]
    id_vars = id_vars + otras

    st.write("Columnas del DataFrame:", cols)
    st.write("id_vars usados para melt:", id_vars)
    st.write("Columnas de fechas:", fechas)

    indicadores = df[id_vars[0]].unique().tolist()
    indicador_sel = st.multiselect("Selecciona uno o más indicadores para analizar:", indicadores, default=indicadores)
    rango_fechas = st.slider(
        "Selecciona el rango de fechas:",
        min_value=0, max_value=len(fechas)-1,
        value=(0, len(fechas)-1),
        format="%d"
    )
    fechas_sel = fechas[rango_fechas[0]:rango_fechas[1]+1]

    # FILTRADO por indicador
    df_filtrado = df[df[id_vars[0]].isin(indicador_sel)]
    df_melt = df_filtrado.melt(
        id_vars=id_vars,
        value_vars=fechas_sel,
        var_name='Fecha',
        value_name='Valor'
    )

    # --- KPIs ---
    st.subheader("🔎 KPIs rápidos")
    kpi_cols = st.columns(len(indicador_sel))
    for i, ind in enumerate(indicador_sel):
        data = df_melt[df_melt[id_vars[0]] == ind]['Valor']
        # Intenta convertir a numérico
        data = pd.to_numeric(data, errors="coerce").dropna()
        if not data.empty:
            kpi_cols[i].metric(
                f"{ind}",
                f"Total: {int(data.sum())}",
                f"Prom: {data.mean():.1f} | Max: {int(data.max())} | Min: {int(data.min())}"
            )
        else:
            kpi_cols[i].metric(f"{ind}", "Sin datos numéricos", "")

    # --- GRAFICO INTERACTIVO ---
    st.subheader("📈 Evolución temporal")
    fig = go.Figure()
    for i, ind in enumerate(indicador_sel):
        datos = df_melt[df_melt[id_vars[0]] == ind]
        y = pd.to_numeric(datos['Valor'], errors="coerce")
        fig.add_trace(go.Scatter(
            x=datos['Fecha'], y=y,
            mode='lines+markers',
            name=ind,
            line=dict(color=CORPORATE_COLORS[i % len(CORPORATE_COLORS)], width=3),
            marker=dict(size=8)
        ))
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title="Valor",
        legend_title="Indicador",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- HEATMAP ---
    st.subheader("🌡️ Heatmap de valores")
    df_heatmap = df_filtrado.set_index(id_vars[0])[fechas_sel]
    import plotly.express as px
    df_heatmap = df_heatmap.apply(pd.to_numeric, errors="coerce")
    fig_hm = px.imshow(
        df_heatmap,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels=dict(color="Valor"),
        text_auto=True
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # --- DESCARGA ---
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Descargar datos filtrados (CSV)",
        data=csv,
        file_name='datos_filtrados.csv',
        mime='text/csv'
    )

    # --- AYUDA CONTEXTUAL ---
    with st.expander("ℹ️ ¿Cómo leer este dashboard?"):
        st.markdown("""
        - **Selecciona los indicadores** para comparar tendencias clave.
        - **Ajusta el rango de fechas** para análisis de periodos críticos.
        - **KPIs** resumen rápidamente el desempeño y extremos.
        - **El gráfico** muestra la evolución temporal de los indicadores seleccionados.
        - **El heatmap** permite identificar días críticos de un vistazo.
        - **Descarga** los datos para compartir o análisis avanzado.
        """)

except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
