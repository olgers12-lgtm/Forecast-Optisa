import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="📊 Dashboard Ejecutivo de Producción", layout="wide")
st.title("📊 Dashboard Ejecutivo de Producción")
st.caption("Interactivo, visual y actualizado en tiempo real")

CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36"
]

# === Aquí pones tu código de carga de datos desde Google Sheets (df) ===
# df = ... (ya lo tienes cargado)
# Para ejemplo, df = pd.DataFrame(...)
# Supongo que ya tienes el DataFrame df con la estructura de tu imagen

# --- INTERACTIVIDAD Y KPIs ---
indicadores = df['Indicador'].unique().tolist()
fechas = df.columns[2:]

indicador_sel = st.multiselect("Selecciona uno o más indicadores para analizar:", indicadores, default=indicadores)
rango_fechas = st.slider(
    "Selecciona el rango de fechas:",
    min_value=0, max_value=len(fechas)-1,
    value=(0, len(fechas)-1),
    format="%d"
)
fechas_sel = fechas[rango_fechas[0]:rango_fechas[1]+1]

# --- FILTRADO ---
df_filtrado = df[df['Indicador'].isin(indicador_sel)]
df_melt = df_filtrado.melt(id_vars=['todo en Jobs', 'Indicador'], value_vars=fechas_sel, var_name='Fecha', value_name='Valor')

# --- KPIs (Ejemplo: suma total, promedio, máximo, mínimo por indicador en el periodo filtrado) ---
st.subheader("🔎 KPIs rápidos")
kpi_cols = st.columns(len(indicador_sel))
for i, ind in enumerate(indicador_sel):
    data = df_melt[df_melt['Indicador'] == ind]['Valor']
    kpi_cols[i].metric(
        f"{ind}",
        f"Total: {int(data.sum())}",
        f"Prom: {data.mean():.1f} | Max: {int(data.max())} | Min: {int(data.min())}"
    )

# --- GRAFICO INTERACTIVO ---
st.subheader("📈 Evolución temporal")
fig = go.Figure()
for i, ind in enumerate(indicador_sel):
    datos = df_melt[df_melt['Indicador'] == ind]
    fig.add_trace(go.Scatter(
        x=datos['Fecha'], y=datos['Valor'],
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
df_heatmap = df_filtrado.set_index('Indicador')[fechas_sel]
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
