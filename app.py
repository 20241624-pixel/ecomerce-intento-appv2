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
# 3. FUNCIONES DE SINCRONIZACIÓN (SLIDER + INPUT EXACTO)
# ==============================================================================
def sync_input(key_num, key_slice):
    st.session_state[key_slice] = st.session_state[key_num]

def sync_slider(key_num, key_slice):
    st.session_state[key_num] = st.session_state[key_slice]

# Inicializar valores por defecto en session_state si no existen
if 'pv_val' not in st.session_state: st.session_state['pv_val'] = 8.5
if 'pv_slider' not in st.session_state: st.session_state['pv_slider'] = 8.5

if 'er_val' not in st.session_state: st.session_state['er_val'] = 0.035
if 'er_slider' not in st.session_state: st.session_state['er_slider'] = 0.035

if 'br_val' not in st.session_state: st.session_state['br_val'] = 0.010
if 'br_slider' not in st.session_state: st.session_state['br_slider'] = 0.010

# ==============================================================================
# 4. BARRA LATERAL (ENTRADAS HÍBRIDAS: SLIDER + CASILLA NUMÉRICA)
# ==============================================================================
st.sidebar.header("Parámetros de Entrada")

# 1. PageValues
st.sidebar.markdown("**Valor de Página (PageValues)**")
col_s1, col_n1 = st.sidebar.columns([2, 1])
with col_s1:
    st.slider("pv_s", 0.0, 360.0, key="pv_slider", on_change=sync_slider, args=('pv_val', 'pv_slider'), label_visibility="collapsed")
with col_n1:
    st.number_input("pv_n", 0.0, 360.0, key="pv_val", step=0.1, format="%.2f", on_change=sync_input, args=('pv_val', 'pv_slider'), label_visibility="collapsed")
page_values = st.session_state['pv_val']

# 2. ExitRates
st.sidebar.markdown("**Tasa de Salida (ExitRates)**")
col_s2, col_n2 = st.sidebar.columns([2, 1])
with col_s2:
    st.slider("er_s", 0.000, 0.200, key="er_slider", on_change=sync_slider, args=('er_val', 'er_slider'), label_visibility="collapsed")
with col_n2:
    st.number_input("er_n", 0.000, 1.000, key="er_val", step=0.001, format="%.3f", on_change=sync_input, args=('er_val', 'er_slider'), label_visibility="collapsed")
exit_rates = st.session_state['er_val']

# 3. BounceRates
st.sidebar.markdown("**Tasa de Rebote (BounceRates)**")
col_s3, col_n3 = st.sidebar.columns([2, 1])
with col_s3:
    st.slider("br_s", 0.000, 0.200, key="br_slider", on_change=sync_slider, args=('br_val', 'br_slider'), label_visibility="collapsed")
with col_n3:
    st.number_input("br_n", 0.000, 1.000, key="br_val", step=0.001, format="%.3f", on_change=sync_input, args=('br_val', 'br_slider'), label_visibility="collapsed")
bounce_rates = st.session_state['br_val']

st.sidebar.subheader("Métricas de Navegación")
product_related = st.sidebar.number_input("Páginas de Producto Visitadas", min_value=0, max_value=1000, value=22, step=1)
product_duration = st.sidebar.number_input("Tiempo en Páginas de Producto (seg)", min_value=0.0, max_value=50000.0, value=750.0, step=10.0, format="%.1f")

st.sidebar.subheader("Contexto de la Sesión")
visitor_type = st.sidebar.selectbox("Tipo de Visitante", ["Returning_Visitor", "New_Visitor", "Other"])
month = st.sidebar.selectbox("Mes", ["Jan", "Feb", "Mar", "May", "June", "Jul", "Aug", "Oct", "Nov", "Dec"], index=7)
weekend = st.sidebar.radio("¿Fin de Semana?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

# ==============================================================================
# 5. PREPARACIÓN DE DATOS E INFERENCIA
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
# 6. DIAGNÓSTICO Y RECOMENDACIÓN PRESCRIPTIVA
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