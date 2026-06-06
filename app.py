
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="DDC Predictor",
    page_icon="🦴",
    layout="wide"
)

# =========================
# ESTILO VISUAL AZUL
# =========================

st.markdown("""
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0057B8, #003B7A);
}

[data-testid="stSidebar"] * {
    color: white;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    color: #0B1F3A;
}

.subtitle {
    font-size: 18px;
    color: #3B4A60;
}

.info-box {
    background-color: #E8F2FF;
    color: #0057B8;
    padding: 16px;
    border-radius: 10px;
    font-weight: 600;
}

.card {
    background-color: white;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #D6E4F5;
    box-shadow: 0px 3px 12px rgba(0, 87, 184, 0.08);
    text-align: center;
}

.card-title {
    color: #3B4A60;
    font-size: 16px;
}

.card-number {
    color: #003B7A;
    font-size: 34px;
    font-weight: 800;
}

.section-title {
    color: #003B7A;
    font-size: 28px;
    font-weight: 800;
    margin-top: 30px;
}

.footer {
    text-align: center;
    color: #0057B8;
    background-color: #E8F2FF;
    padding: 18px;
    border-radius: 10px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.markdown("# 🦴 DDC Predictor")
st.sidebar.markdown("### Displasia del Desarrollo de Cadera")
menu = st.sidebar.radio(
    "Menú",
    [
        "Inicio",
        "Exploración de Datos",
        "Visualización",
        "Modelo Predictivo",
        "Predicción Individual",
        "Acerca del Proyecto"
    ]
)

# =========================
# CARGA DE DATOS
# =========================

df = pd.read_csv("dataset_sintetico_ddc.csv")
objetivo = "ddc_diagnostico"

# =========================
# INICIO
# =========================

if menu == "Inicio":

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            '<div class="main-title">Sistema Inteligente para Apoyo al Diagnóstico de Displasia del Desarrollo de Cadera (DDC)</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="subtitle">Aplicación educativa en Python y Streamlit para analizar factores de riesgo, visualizar datos y entrenar un modelo predictivo en Ortopedia Pediátrica.</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="info-box">Proyecto académico. Los resultados no reemplazan el criterio médico profesional.</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### 👦 👦  Ortopedia Pediátrica")
        st.image(
        "pelvis_ddc.png",
        caption="Radiografía ilustrativa de pelvis infantil",
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Pacientes</div>
            <div class="card-number">{df.shape[0]}</div>
            <div>Total en dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Variables</div>
            <div class="card-number">{df.shape[1]}</div>
            <div>Características</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">Objetivo</div>
            <div class="card-number">DDC</div>
            <div>Sí / No</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Distribución del Diagnóstico</div>', unsafe_allow_html=True)

    conteo = df[objetivo].value_counts()

    fig, ax = plt.subplots()
    conteo.plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    ax.set_title("DDC Diagnóstico")
    st.pyplot(fig)

# =========================
# EXPLORACIÓN
# =========================

elif menu == "Exploración de Datos":

    st.markdown('<div class="section-title">Exploración inicial del dataset</div>', unsafe_allow_html=True)

    resumen = pd.DataFrame({
        "Indicador": ["Pacientes", "Variables", "Datos faltantes"],
        "Valor": [df.shape[0], df.shape[1], int(df.isnull().sum().sum())]
    })

    st.table(resumen)

    st.subheader("Primeras filas")
    st.table(df.head(10))

    st.subheader("Tipos de variables")
    tipos = pd.DataFrame(df.dtypes.astype(str), columns=["Tipo"])
    st.table(tipos)

    st.subheader("Resumen estadístico")
    st.table(df.describe(include="all").fillna("").astype(str))

# =========================
# VISUALIZACIÓN
# =========================

elif menu == "Visualización":

    st.markdown('<div class="section-title">Visualización de variables</div>', unsafe_allow_html=True)

    columnas_visualizar = [c for c in df.columns if c != "id_paciente"]

    columna = st.selectbox("Seleccione una variable", columnas_visualizar)

    fig, ax = plt.subplots()

    if pd.api.types.is_numeric_dtype(df[columna]):
        ax.hist(df[columna].dropna())
        ax.set_title("Histograma de " + columna)
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")
    else:
        df[columna].astype(str).value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Distribución de " + columna)
        ax.set_xlabel(columna)
        ax.set_ylabel("Frecuencia")

    st.pyplot(fig)

# =========================
# MODELO
# =========================

elif menu == "Modelo Predictivo":

    st.markdown('<div class="section-title">Modelo predictivo</div>', unsafe_allow_html=True)

    X = df.drop(columns=[objetivo])
    y = df[objetivo]

    columnas_categoricas = X.select_dtypes(include="object").columns.tolist()
    columnas_numericas = X.select_dtypes(exclude="object").columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", "passthrough", columnas_numericas)
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", RandomForestClassifier(random_state=42))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    resultados = pd.DataFrame({
        "Métrica": ["Accuracy", "Precision", "Recall", "F1-score"],
        "Valor": [
            round(accuracy_score(y_test, y_pred), 3),
            round(precision_score(y_test, y_pred, pos_label="Si", zero_division=0), 3),
            round(recall_score(y_test, y_pred, pos_label="Si", zero_division=0), 3),
            round(f1_score(y_test, y_pred, pos_label="Si", zero_division=0), 3)
        ]
    })

    st.subheader("Resultados del modelo Random Forest")
    st.table(resultados)

    st.subheader("Matriz de confusión")
    matriz = confusion_matrix(y_test, y_pred)
    st.table(pd.DataFrame(matriz))

# =========================
# PREDICCIÓN INDIVIDUAL
# =========================

elif menu == "Predicción Individual":

    st.markdown('<div class="section-title">Predicción individual</div>', unsafe_allow_html=True)

    X = df.drop(columns=[objetivo])
    y = df[objetivo]

    columnas_categoricas = X.select_dtypes(include="object").columns.tolist()
    columnas_numericas = X.select_dtypes(exclude="object").columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", "passthrough", columnas_numericas)
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("clasificador", RandomForestClassifier(random_state=42))
        ]
    )

    modelo.fit(X, y)

    entrada = {}

    for col in X.columns:

        if col == "id_paciente":
            entrada[col] = "PACIENTE_NUEVO"
            continue

        columna_numerica = pd.to_numeric(X[col], errors="coerce")

        if pd.api.types.is_numeric_dtype(X[col]):
            entrada[col] = st.number_input(
                col,
                value=float(columna_numerica.mean())
            )
        else:
            opciones = sorted(X[col].astype(str).dropna().unique())
            entrada[col] = st.selectbox(col, opciones)

    nuevo = pd.DataFrame([entrada])

    if st.button("Predecir DDC"):
        pred = modelo.predict(nuevo)[0]
        prob = modelo.predict_proba(nuevo)[0].max()

        st.subheader("Resultado")
        st.write("Clasificación:", pred)
        st.write("Probabilidad estimada:", round(prob, 3))

        if pred == "Si":
            st.warning("El modelo clasifica al paciente como riesgo compatible con DDC.")
        else:
            st.success("El modelo clasifica al paciente como bajo riesgo de DDC.")

# =========================
# ACERCA DEL PROYECTO
# =========================

elif menu == "Acerca del Proyecto":

    st.markdown('<div class="section-title">Acerca del proyecto</div>', unsafe_allow_html=True)

    st.write("""
    Este proyecto final desarrolla una aplicación en Streamlit para explorar,
    analizar, visualizar y modelar un dataset sintético de Displasia del Desarrollo
    de Cadera. Está orientado a la especialidad de Ortopedia Pediátrica.
    """)

    st.write("""
    Herramientas utilizadas:
    - Python
    - Pandas
    - Scikit-learn
    - Matplotlib
    - Streamlit
    """)

    st.warning("El dataset es sintético y tiene fines exclusivamente académicos.")

st.markdown(
    '<div class="footer">Universidad Casa Grande - Maestría en Inteligencia Artificial y Ciencia de Datos</div>',
    unsafe_allow_html=True
)
