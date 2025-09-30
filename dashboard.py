import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
from datetime import datetime

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

def agregar_ano(col):
    col = col.strip().lower()
    if "-" in col and col.count("-") == 2:
        return col
    if "sept" in col or "oct" in col:
        year = "2025"
    else:
        year = "2025"
    return f"{col}-{year}"

try:
    df = cargar_datos(SHEET_ID, SHEET_NAME)

    col_indicador = None
    for c in df.columns:
        if "indicador" in c.lower():
            col_indicador = c.strip()
            break
    if not col_indicador:
        st.error("No se encontró columna 'Indicador'. Las columnas son: " + str(df.columns.tolist()))
        st.stop()

    df = df[df[col_indicador].notnull() & (df[col_indicador] != '')]

    fechas = [c for c in df.columns if c != col_indicador]
    fechas_pares = [(f, agregar_ano(f)) for f in fechas]
    fechas_dt = [pd.to_datetime(f_con_ano, format="%d-%b-%Y", errors="coerce") for f, f_con_ano in fechas_pares]
    fechas_validas = [fechas_pares[i][0] for i in range(len(fechas_pares)) if not pd.isnull(fechas_dt[i])]
    fechas_con_ano_validas = [fechas_pares[i][1] for i in range(len(fechas_pares)) if not pd.isnull(fechas_dt[i])]
    if not fechas_validas:
        st.warning("No hay columnas de fechas válidas. Revisar encabezados y formato de fechas.")
        st.stop()

    indicadores = df[col_indicador].unique().tolist()
    indicador_sel = st.multiselect("Selecciona uno o más indicadores para analizar:", indicadores, default=indicadores)

    df_melt = df[df[col_indicador].isin(indicador_sel)].melt(
        id_vars=[col_indicador],
        value_vars=fechas_validas,
        var_name='Fecha',
        value_name='Valor'
    )
    df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"].apply(agregar_ano), format="%d-%b-%Y", errors="coerce")
    df_melt["Valor"] = pd.to_numeric(df_melt["Valor"], errors="coerce")
    df_melt = df_melt.dropna(subset=["Fecha_dt"])

    agrupamiento = st.radio("Agrupar por:", ["Día", "Semana (L-D)", "Mes", "Rango personalizado"], horizontal=True)
    mask_fecha = pd.Series([True] * len(df_melt))
    if agrupamiento == "Día":
        fechas_unicas = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").dropna().unique().tolist()
        fechas_unicas = sorted(fechas_unicas, key=lambda x: pd.to_datetime(x, format="%d-%b-%Y"))
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
    else:
        fecha_min, fecha_max = df_melt["Fecha_dt"].min(), df_melt["Fecha_dt"].max()
        fecha_inicio, fecha_fin = st.date_input("Rango de fechas:", [fecha_max - pd.Timedelta(days=14), fecha_max])
        mask_fecha = (df_melt["Fecha_dt"] >= pd.to_datetime(fecha_inicio)) & (df_melt["Fecha_dt"] <= pd.to_datetime(fecha_fin))

    df_filtrado_fecha = df_melt[mask_fecha]

    st.subheader("🔢 KPIs industriales")
    col1, col2, col3, col4 = st.columns(4)
    try:
        if not df_filtrado_fecha.empty:
            entrada_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada-proyectada")]['Valor'].sum()
            entrada_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("entrada real")]['Valor'].sum()
            salida_proj = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida proyectada")]['Valor'].sum()
            salida_real = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("salida real")]['Valor'].sum()
            wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].str.lower().str.contains("wip")]['Valor']
            eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None

            col1.metric("Total Entrada Real", f"{int(entrada_real) if pd.notnull(entrada_real) else '-'}")
            col2.metric("Total Salida Real", f"{int(salida_real) if pd.notnull(salida_real) else '-'}")
            col3.metric("Eficiencia Salida (%)", f"{eficiencia:.1f}%" if eficiencia else "-")
            if not wip.empty and pd.notnull(wip.mean()):
                wip_prom = wip.mean()
                wip_max = wip.max()
                col4.metric("WIP Promedio", f"{wip_prom:.1f}", f"Max: {int(wip_max)}", delta_color="inverse" if wip_max > WIP_THRESHOLD else "normal")
            else:
                col4.metric("WIP Promedio", "-", "Sin datos")
        else:
            st.warning("No hay datos filtrados para mostrar KPIs.")
    except Exception as e:
        st.warning(f"KPIs industriales no disponibles. Error: {e}")

    st.subheader("📈 Evolución temporal")
    try:
        if not df_filtrado_fecha.empty:
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

    st.subheader("🌡️ Heatmap WIP (Rojo si > 1200)")
    wip_inds = [w for w in indicador_sel if "wip" in w.lower()]
    if wip_inds and not df_filtrado_fecha.empty:
        try:
            df_wip = df_filtrado_fecha[df_filtrado_fecha[col_indicador].isin(wip_inds)].copy()
            df_wip["Fecha_dt"] = pd.to_datetime(df_wip["Fecha"].apply(agregar_ano), format="%d-%b-%Y", errors="coerce")
            df_wip = df_wip.sort_values("Fecha_dt")
            df_wip_pivot = df_wip.pivot(index=col_indicador, columns="Fecha_dt", values="Valor")
            df_wip_pivot = df_wip_pivot.sort_index(axis=1)
            colorscale = [
                [0, "#61C0BF"], [0.75, "#F6AE2D"], [0.9, "#F74B36"], [1, "#8B0000"]
            ]
            vmax = max(WIP_THRESHOLD + 300, float(df_wip_pivot.max().max() if not df_wip_pivot.empty else 0))
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
        except Exception as e:
            st.warning(f"No se pudo graficar heatmap WIP. Error: {e}")
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

    with st.expander("ℹ️ ¿Cómo usar este dashboard?"):
        st.markdown("""
        - **KPIs**: resumen dinámico de totales, promedios y extremos para cada indicador.
        - **Agrupamiento dinámico**: cambia de día, semana, mes o rango personalizado de fechas.
        - **Gráfico interactivo**: líneas por indicador, permite comparación visual.
        - **Heatmap WIP**: zonas rojas alertan sobre exceso de trabajos en proceso (>1200), ordenado cronológicamente.
        - **Descarga**: exporta los datos filtrados para análisis externo.
        - **Hoja original**: puedes mostrar/ocultar la hoja completa si la necesitas para análisis profundo.
        """)

except Exception as e:
    st.error(f"No se pudieron cargar los datos: {e}")
