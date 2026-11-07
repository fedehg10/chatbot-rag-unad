import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    "retrieval_k": 3,
    "pdf_path": "documentacion_tecnica.pdf"
}

# --- 2. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Técnico de Datos", page_icon="🤖", layout="centered")

# --- 3. INICIALIZACIÓN DEL SISTEMA RAG LOCAL ---
@st.cache_resource
def init_rag_system_local():
    # Carga y segmentación del documento PDF de forma local
    if os.path.exists(CONFIG["pdf_path"]):
        loader = PyPDFLoader(CONFIG["pdf_path"])
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        
        # Base vectorial FAISS con embeddings locales (sin llamadas a API externas)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG["retrieval_k"]})
    else:
        retriever = None
        
    return retriever

# Cargamos el sistema de forma segura
try:
    retriever = init_rag_system_local()
except Exception as e:
    st.error(f"Error al inicializar el sistema RAG: {e}")
    retriever = None

# --- 4. BARRA LATERAL (INTERFAZ) ---
st.sidebar.title("⚙️ Configuración y Ayuda")
st.sidebar.info(
    "**Acerca del Asistente:**\n"
    "Prototipo RAG experto en herramientas de ingeniería de datos (Apache Airflow y dbt).\n\n"
    "**Funcionamiento:**\n"
    "Busca y extrae de forma inteligente los fragmentos exactos de la documentación técnica oficial."
)

st.sidebar.markdown("### 💡 Preguntas Recomendadas")
st.sidebar.markdown("- ¿Cómo configuro un DAG básico en Airflow?")
st.sidebar.markdown("- ¿Cuáles son los comandos básicos para ejecutar un proyecto en dbt?")

# --- 5. INTERFAZ PRINCIPAL Y CHAT ---
st.markdown("### Asistente Técnico RAG - Ingeniería de Datos")
st.markdown("Pregúntame sobre configuración, buenas prácticas o resolución de errores en herramientas open-source.")

# Capturamos la entrada del usuario
prompt = st.chat_input("Escribe tu duda técnica aquí...")

if prompt:
    # Mostramos la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generamos y mostramos la respuesta basada en el documento
    with st.chat_message("assistant"):
        if retriever is None:
            response = "Error: El sistema no se ha inicializado correctamente. Revisa que el archivo PDF esté en el repositorio."
        else:
            try:
                # Buscamos fragmentos relevantes en el PDF usando la base vectorial local
                docs = retriever.invoke(prompt)
                
                if not docs or len(docs) == 0:
                    response = "No dispongo de información suficiente en la documentación provista para responder esta pregunta."
                else:
                    # Construimos la respuesta formal extrayendo y sintetizando el contenido técnico encontrado
                    context_text = "\n\n---\n\n".join([d.page_content for d in docs])
                    response = (
                        f"### Respuesta basada en la Documentación Técnica:\n\n"
                        f"{context_text}\n\n"
                        f"*Fuente extraída de la base de conocimiento vectorial (FAISS).* "
                    )
            except Exception as e:
                response = f"Ocurrió un error al procesar la consulta: {e}"

        st.markdown(response)
