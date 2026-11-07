import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

# --- 1. CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "temperature": 0.1,
    "max_tokens": 512,
    "retrieval_k": 3,
    "pdf_path": "documentacion_tecnica.pdf"
}

# --- 2. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Técnico de Datos", page_icon="🤖", layout="centered")

# --- 3. INICIALIZACIÓN DEL SISTEMA RAG (CON CACHÉ) ---
@st.cache_resource
def init_rag_system():
    # Inicializamos el modelo de lenguaje
    llm = HuggingFaceEndpoint(
        repo_id=CONFIG["model_name"],
        task="text-generation",
        max_new_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    )
    
    # Carga y segmentación del documento PDF
    if os.path.exists(CONFIG["pdf_path"]):
        loader = PyPDFLoader(CONFIG["pdf_path"])
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        
        # Base vectorial FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG["retrieval_k"]})
    else:
        retriever = None
        
    return llm, retriever

# Cargamos el sistema
try:
    llm, retriever = init_rag_system()
except Exception as e:
    st.error(f"Error al inicializar el sistema RAG: {e}")
    llm, retriever = None, None

# --- 4. BARRA LATERAL (INTERFAZ) ---
st.sidebar.title("⚙️ Configuración y Ayuda")
st.sidebar.info(
    "**Acerca del Asistente:**\n"
    "Prototipo RAG experto en herramientas de ingeniería de datos (Apache Airflow y dbt).\n\n"
    "**Limitaciones:**\n"
    "El modelo puede cometer errores o alucinar datos fuera de la documentación oficial."
)

st.sidebar.markdown("### 💡 Preguntas Recomendadas")
st.sidebar.markdown("- ¿Cómo configuro un DAG básico en Airflow?")
st.sidebar.markdown("- ¿Cuáles son los comandos básicos para ejecutar un proyecto en dbt?")

# --- 5. INTERFAZ PRINCIPAL Y CHAT ---
st.markdown("### Asistente Técnico RAG - Ingeniería de Datos")
st.markdown("Pregúntame sobre configuración, buenas prácticas o resolución de errores en herramientas open-source.")

# Capturamos la entrada del usuario de forma limpia abajo del todo
prompt = st.chat_input("Escribe tu duda técnica aquí...")

if prompt:
    # Mostramos la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generamos y mostramos la respuesta del asistente
    with st.chat_message("assistant"):
        if retriever is None or llm is None:
            response = "Error: El sistema no se ha inicializado correctamente. Revisa los secretos o el archivo PDF."
        else:
            try:
                # Buscamos fragmentos relevantes en el PDF
                docs = retriever.invoke(prompt)
                context = "\n\n".join([d.page_content for d in docs])

                if len(context.strip()) == 0:
                    response = "No dispongo de información suficiente en la documentación provista para responder esta pregunta."
                else:
                    input_text = f"Utiliza exclusivamente el siguiente contexto técnico para responder a la pregunta de manera clara y profesional.\n\nContexto:\n{context}\n\nPregunta: {prompt}\nRespuesta:"
                    raw_response = llm.invoke(input_text)
                    response = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            except Exception as e:
                response = f"Ocurrió un error al procesar la consulta con el modelo: {e}"

        st.markdown(response)
