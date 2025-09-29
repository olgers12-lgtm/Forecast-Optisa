# Detectar el usuario logueado (puedes obtenerlo de secrets, de cookies, o hardcodear)
st.session_state["user_login"] = "olgers12-lgtm"  # Cambia esto a tu lógica real si es necesario
import streamlit as st
import pandas as pd

# df_melt debe tener columnas: Indicador, Fecha, Valor, Fecha_dt

# Conversión de fechas
df_melt["Fecha_dt"] = pd.to_datetime(df_melt["Fecha"], format="%d-%b-%y", errors="coerce")
df_melt = df_melt.dropna(subset=["Fecha_dt"])

# Filtros avanzados de fecha
agrupamiento = st.radio("Agrupar por:", ["Día", "Semana (L-D)", "Mes", "Rango personalizado"], horizontal=True)

if agrupamiento == "Día":
    fechas_unicas = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").unique().tolist()
    fechas_sel = st.multiselect("Selecciona días:", fechas_unicas, default=fechas_unicas[-7:])
    mask_fecha = df_melt["Fecha_dt"].dt.strftime("%d-%b-%Y").isin(fechas_sel)
elif agrupamiento == "Semana (L-D)":
    df_melt["Semana"] = df_melt["Fecha_dt"].dt.isocalendar().week
    semanas_disponibles = sorted(df_melt["Semana"].unique())
    semana_sel = st.select_slider("Selecciona semana:", options=semanas_disponibles, value=semanas_disponibles[-1])
    mask_fecha = df_melt["Semana"] == semana_sel
elif agrupamiento == "Mes":
    df_melt["Mes"] = df_melt["Fecha_dt"].dt.strftime("%B %Y")
    meses_disponibles = sorted(df_melt["Mes"].unique())
    mes_sel = st.selectbox("Selecciona mes:", options=meses_disponibles, index=len(meses_disponibles)-1)
    mask_fecha = df_melt["Mes"] == mes_sel
else:  # Rango personalizado
    fecha_min, fecha_max = df_melt["Fecha_dt"].min(), df_melt["Fecha_dt"].max()
    fecha_inicio, fecha_fin = st.date_input("Rango de fechas:", [fecha_max - pd.Timedelta(days=14), fecha_max])
    mask_fecha = (df_melt["Fecha_dt"] >= pd.to_datetime(fecha_inicio)) & (df_melt["Fecha_dt"] <= pd.to_datetime(fecha_fin))

# Aplicar filtro
df_filtrado_fecha = df_melt[mask_fecha]

# KPIs industriales avanzados
col1, col2, col3, col4 = st.columns(4)

try:
    entrada_proj = df_filtrado_fecha[df_filtrado_fecha['Indicador'].str.lower().str.contains("entrada-proyectada")]['Valor'].sum()
    entrada_real = df_filtrado_fecha[df_filtrado_fecha['Indicador'].str.lower().str.contains("entrada real")]['Valor'].sum()
    salida_proj = df_filtrado_fecha[df_filtrado_fecha['Indicador'].str.lower().str.contains("salida proyectada")]['Valor'].sum()
    salida_real = df_filtrado_fecha[df_filtrado_fecha['Indicador'].str.lower().str.contains("salida real")]['Valor'].sum()
    wip = df_filtrado_fecha[df_filtrado_fecha['Indicador'].str.lower().str.contains("wip")]['Valor']
    eficiencia = salida_real / salida_proj * 100 if salida_proj > 0 else None

    col1.metric("Total Entrada Real", f"{int(entrada_real)}")
    col2.metric("Total Salida Real", f"{int(salida_real)}")
    col3.metric("Eficiencia Salida (%)", f"{eficiencia:.1f}%" if eficiencia else "-")
    if not wip.empty:
        wip_prom = wip.mean()
        wip_max = wip.max()
        col4.metric("WIP Promedio", f"{wip_prom:.1f}", f"Max: {int(wip_max)}", delta_color="inverse" if wip_max > 1200 else "normal")
    else:
        col4.metric("WIP Promedio", "-", "Sin datos")
except Exception as e:
    st.warning(f"KPIs industriales no disponibles. Error: {e}")

# Solo muestra la tabla a ti (olgers12-lgtm)
if st.session_state.get("user_login", "") == "olgers12-lgtm":
    st.dataframe(df_filtrado_fecha)
