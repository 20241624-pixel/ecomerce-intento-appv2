import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Sistema de Predicción de Intención de Compra",
    layout="wide"
)

st.title("Sistema de Predicción de Intención de Compra en Tiempo Real")
st.caption("Modelo Predictivo y Prescriptivo para Optimización de Conversión en E-Commerce")

# ==============================================================================
# 2. CARGA DEL MODELO Y COLUMNAS
# ==============================================================================
@st.cache_resource
def cargar_recursos():
    try:
        model = joblib.load('modelo_ecommerce.joblib')
        cols = joblib.load('columnas_modelo.joblib')
        return model, cols
    except Exception as e:
        st.error(f"Error al cargar los componentes del modelo: {e}")
        return None, None

model, columnas_modelo = cargar_recursos()

# ==============================================================================
# 3. BARRA LATERAL (ENTRADAS DE NAVEGACIÓN CON VALORES EXACTOS)
# ==============================================================================
st.sidebar.header("Parámetros de Entrada")

# Indicadores de comportamiento (Entradas numéricas directas)
page_values = st.sidebar.number_input(
    "Valor de Página (PageValues)", 
    min_value=0.0, 
    max_value=400.0, 
    value=8.5, 
    step=0.1, 
    format="%.2f"
)

exit_rates = st.sidebar.number_input(
    "Tasa de Salida (ExitRates)", 
    min_value=0.000, 
    max_value=1.000, 
    value=0.035, 
    step=0.001, 
    format="%.3f"
)

bounce_rates = st.sidebar.number_input(
    "Tasa de Rebote (BounceRates)", 
    min_value=0.000, 
    max_value=1.000, 
    value=0.010, 
    step=0.001, 
    format="%.3f"
)

st.sidebar.subheader("Métricas de Navegación")
product_related = st.sidebar.number_input(
    "Páginas de Producto Visitadas", 
    min_value=0, 
    max_value=1000, 
    value=22, 
    step=1
)

product_duration = st.sidebar.number_input(
    "Tiempo en Páginas de Producto (seg)", 
    min_value=0.0, 
    max_value=50000.0, 
    value=750.0, 
    step=1.0, 
    format="%.1f"
)

st.sidebar.subheader("Contexto de la Sesión")
visitor_type = st.sidebar.selectbox("Tipo de Visitante", ["Returning_Visitor", "New_Visitor", "Other"])
month = st.sidebar.selectbox("Mes", ["Jan", "Feb", "Mar", "May", "June", "Jul", "Aug", "Oct", "Nov", "Dec"], index=7)
weekend = st.sidebar.radio("¿Fin de Semana?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# ==============================================================================
# 4. PREPARACIÓN DE DATOS E INFERENCIA
# ==============================================================================
datos_entrada = {
    'Administrative': 0, 
    'Administrative_Duration': 0.0,
    'Informational': 0, 
    'Informational_Duration': 0.0,
    'ProductRelated': product_related, 
    'ProductRelated_Duration': product_duration,
    'BounceRates': bounce_rates, 
    'ExitRates': exit_rates,
    'PageValues': page_values, 
    'SpecialDay': 0.0,
    'OperatingSystems': 2, 
    'Browser': 2, 
    'Region': 1, 
    'TrafficType': 2,
    'Weekend': weekend, 
    'Month': month, 
    'VisitorType': visitor_type
}

df_input = pd.DataFrame([datos_entrada])
df_input_encoded = pd.get_dummies(df_input, columns=['Month', 'VisitorType'])

if columnas_modelo is not None:
    df_input_encoded = df_input_encoded.reindex(columns=columnas_modelo, fill_value=0)

# ==============================================================================
# 5. DIAGNÓSTICO Y RECOMENDACIÓN PRESCRIPTIVA
# ==============================================================================
st.subheader("Resultado de Evaluación")

if model is not None and columnas_modelo is not None:
    probabilidad = model.predict_proba(df_input_encoded)[0][1] * 100
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label="Probabilidad de Conversión",
            value=f"{probabilidad:.1f}%"
        )
        
    with col2:
        if probabilidad >= 66.0:
            st.success("ALTA INTENCIÓN DE COMPRA")
            st.markdown("**Acción recomendada:** Habilitar flujo de Checkout Express prioritario y omitir promociones interactivas.")
        elif 40.0 <= probabilidad < 66.0:
            st.warning("ZONA DE INDECISIÓN")
            st.markdown("**Acción recomendada:** Activar incentivo dinámico mediante cupón de descuento por tiempo limitado o envío gratuito.")
        else:
            st.info("BAJA INTENCIÓN DE COMPRA")
            st.markdown("**Acción recomendada:** Mantener navegación estándar sin ofrecer descuentos monetarios directos.")