import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(page_title="📊 Dashboard Ejecutivo de Producción", layout="wide")
st.title("📊 Dashboard Ejecutivo de Producción")
st.caption("Interactivo, visual y actualizado en tiempo real. Diseño Industrial & Senior.")

CORPORATE_COLORS = [
    "#1F2A56", "#0D8ABC", "#3EC0ED", "#61C0BF", "#F6AE2D", "#F74B36"
]
WIP_THRESHOLD = 1200

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
    # --- Mostramos columnas para depuración ---
    st.write("Columnas detectadas:", df.columns.tolist())
    st.write(df.head())

    # --- Detecta columna clave con limpieza de espacios ---
    col_indicador = None
    for c in df.columns:
        if "indicador" in c.lower():
            col_indicador = c
            break
    if not col_indicador:
        st.error("No se encontró columna 'Indicador'. Las columnas son: " + str(df.columns.tolist()))
        st.stop()
    # Limpia espacios en el nombre si los hubiera
    col_indicador = col_indicador.strip()

    # --- Elimina filas vacías en columna clave ---
    df = df[df[col_indicador].notnull() & (df[col_indicador] != '')]

    # --- Detecta columnas de fechas ---
    fechas = [c for c in df.columns if c != col_indicador]
    # Solo usa columnas que parecen fechas válidas
    fechas_dt = pd.to_datetime(fechas, format="%d-%b-%y", errors="coerce")
    fechas_validas = [fechas[i] for i in range(len(fechas_dt)) if not pd.isnull(fechas_dt[i])]
    if not fechas_validas:
        st.warning("No hay columnas de fechas válidas. Revisar encabezados.")
        st.stop()

    # --- Selección de indicadores ---
    indicadores = df[col_indicador].unique().tolist()
    indicador_sel = st.multiselect("Selecciona uno o más indicadores para analizar:", indicadores, default=indicadores)

    # --- Transforma a formato largo (melt) ---
    df_melt = df[df[col_indicador].isin(indicador_sel)].melt(
        id_vars=[col_indicador],
        value_vars=fechas_validas,
        var_name='Fecha',
        value_name='Valor'
    )
    df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"], format="%d-%b-%y", errors="coerce")
    df_melt["Valor"] = pd.to_numeric(df_melt["Valor"], errors="coerce")
    df_melt = df_melt.dropna(subset=["Fecha_dt"])

    # --- Filtros de fechas robustos ---
    agrupamiento = st.radio("Agrupar por:", ["Día", "Semana (L-D)", "Mes", "Rango personalizado"], horizontal=True)

    mask_fecha = pd.Series([True] * len(df_melt))
    if agrupamiento == "Día":
        fechas_unicas = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").dropna().unique().tolist()
        if fechas_unicas:
            fechas_sel = st.multiselect("Selecciona días:", fechas_unicas, default=fechas_unicas[-7:])
            mask_fecha = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").isin(fechas_sel)
        else:
            st.warning("No hay días válidos para mostrar.")
            mask_fecha = pd.Series([False] * len(df_melt))
    elif agrupamiento == "Semana (L-D)":
        semanas_disponibles = sorted(df_melt["Fecha_dt"].dt.isocalendar().week.dropna().unique())
        if semanas_disponibles:
            semana_sel = st.select_slider("Selecciona semana:", options=semanas_disponibles, value=semanas_disponibles[-1])
            mask_fecha = df_melt["Fecha_dt"].dt.isocalendar().week == semana_sel
        else:
            st.warning("No hay semanas válidas para mostrar.")
            mask_fecha = pd.Series([False] * len(df_melt))
    elif agrupamiento == "Mes":
        meses_disponibles = sorted(df_melt["Fecha_dt"].dt.strftime("%B %Y").dropna().unique())
        if meses_disponibles:
            mes_sel = st.selectbox("Selecciona mes:", options=meses_disponibles, index=len(meses_disponibles)-1)
            mask_fecha = df_melt["Fecha_dt"].dt.strftime("%B %Y") == mes_sel
        else:
            st.warning("No hay meses válidos para mostrar.")
            mask_fecha = pd.Series([False] * len(df_melt))
    else:  # Rango personalizado
        fecha_min, fecha_max = df_melt["Fecha_dt"].min(), df_melt["Fecha_dt"].max()
        fecha_inicio, fecha_fin = st.date_input("Rango de fechas:", [fecha_max - pd.Timedelta(days=14), fecha_max])
        mask_fecha = (df_melt["Fecha_dt"] >= pd.to_datetime(fecha_inicio)) & (df_melt["Fecha_dt"] <= pd.to_datetime(fecha_fin))

    df_filtrado_fecha = df_melt[mask_fecha]

    # --- KPIs industriales avanzados ---
    st.subheader("🔢 KPIs industriales")
    col1, col2, col3, col4 = st.columns(4)
    try:
        entrada_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada-proyectada")]['Valor'].sum()
        entrada_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
        salida_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida proyectada")]['Valor'].sum()
        salida_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
        wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("wip")]['Valor']
        eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None

        col1.metric("Total Entrada Real", f"{int(entrada_real)}")
        col2.metric("Total Salida Real", f"{int(salida_real)}")
        col3.metric("Eficiencia Salida (%)", f"{eficiencia:.1f}%" if eficiencia else "-")
        if not wip.empty:
            wip_prom = wip.mean()
            wip_max = wip.max()
            col4.metric("WIP Promedio", f"{wip_prom:.1f}", f"Max: {int(wip_max)}", delta_color="inverse" if wip_max > WIP_THRESHOLD else "normal")
        else:
            col4.metric("WIP Promedio", "-", "Sin datos")
    except Exception as e:
        st.warning(f"KPIs industriales no disponibles. Error: {e}")

    # --- Gráfico interactivo ---
    st.subheader("📈 Evolución temporal")
    try:
        fig = go.Figure()
        for i, ind in enumerate(indicador_sel):
            data = df_filtrado_fecha[df_filtrado_fecha[col_indicador] == ind].sort_values("Fecha_dt")
            fig.add_trace(go.Scatter(
                x=data["Fecha_dt"], y=data["Valor"],
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
    except Exception as e:
        st.warning(f"No se pudo graficar evolución temporal. Error: {e}")

    # --- Heatmap específico para WIP ---
    st.subheader("🌡️ Heatmap WIP (Rojo si > 1200)")
    wip_inds = [w for w in indicador_sel if "wip" in w.lower()]
    if wip_inds and not df_filtrado_fecha.empty:
        try:
            df_wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].isin(wip_inds)].pivot(index=col_indicador, columns="Fecha", values="Valor")
            df_wip = df_wip.apply(pd.to_numeric, errors="coerce")
            colorscale = [
                [0, "#61C0BF"],      # verde
                [0.75, "#F6AE2D"],   # amarillo
                [0.9, "#F74B36"],    # rojo fuerte
                [1, "#8B0000"]
            ]
            vmax = max(WIP_THRESHOLD + 300, float(df_wip.max().max() if not df_wip.empty else 0))
            fig_hm = px.imshow(
                df_wip,
                aspect="auto",
                color_continuous_scale=colorscale,
                zmin=0,
                zmax=vmax,
                labels=dict(color="WIP"),
                text_auto=True
            )
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo graficar heatmap WIP. Error: {e}")
    else:
        st.info("Selecciona un indicador WIP y rango de fechas válido para ver el heatmap.")

    # --- Descarga de datos filtrados ---
    if not df_filtrado_fecha.empty:
        csv = df_filtrado_fecha.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Descargar datos filtrados (CSV)",
            data=csv,
            file_name='datos_filtrados.csv',
            mime='text/csv'
        )

    # --- Botón para mostrar/ocultar la hoja original ---
    with st.expander("🗂️ Mostrar/ocultar hoja original de Google Sheets"):
        if st.button("Mostrar hoja original"):
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Haz clic en el botón para mostrar la hoja completa sólo si la necesitas.")

    # --- Ayuda contextual ---
    with st.expander("ℹ️ ¿Cómo usar este dashboard?"):
        st.markdown("""
        - **KPIs**: resumen dinámico de totales, promedios y extremos para cada indicador.
        - **Agrupamiento dinámico**: cambia de día, semana, mes o rango personalizado de fechas.
        - **Gráfico interactivo**: líneas por indicador, permite comparación visual.
        - **Heatmap WIP**: zonas rojas alertan sobre exceso de trabajos en proceso (>1200).
        - **Descarga**: exporta los datos filtrados para análisis externo.
        - **Hoja original**: puedes mostrar/ocultar la hoja completa si la necesitas para análisis profundo.
        """)

except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
