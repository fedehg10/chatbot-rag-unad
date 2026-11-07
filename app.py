import streamlit as st
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint

# --- CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.1,
    "max_tokens": 512,
    "retrieval_k": 3,
    "pdf_path": "documentacion_tecnica.pdf"
}

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Asistente Técnico de Datos", page_icon="🤖", layout="centered")

# --- INICIALIZACIÓN DEL LLM Y LA BASE VECTORIAL (CON CACHÉ) ---
@st.cache_resource
def init_rag_system():
    # Inicializamos el modelo de lenguaje de Hugging Face
    llm = HuggingFaceEndpoint(
        repo_id=CONFIG["model_name"],
        task="text-generation",
        max_new_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        huggingfacehub_api_token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    )
    
    # Carga y segmentación del PDF de la base de conocimiento
    if os.path.exists(CONFIG["pdf_path"]):
        loader = PyPDFLoader(CONFIG["pdf_path"])
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(pages)
        
        # Generación de embeddings e índice FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": CONFIG["retrieval_k"]})
    else:
        retriever = None
        
    return llm, retriever

llm, retriever = init_rag_system()

# --- INTERFAZ: GUÍA DE USUARIO Y DESCARGO DE RESPONSABILIDAD ---
st.sidebar.title("⚙️ Configuración y Ayuda")
st.sidebar.info(
    "**Acerca del Asistente:**\n"
    "Este chatbot es un prototipo RAG experto en herramientas de ingeniería de datos de código abierto (Apache Airflow y dbt).\n\n"
    "**Limitaciones y Descargo:**\n"
    "El modelo puede cometer errores o alucinar información que no esté en la documentación oficial. Verifique siempre los comandos críticos."
)

st.sidebar.markdown("### 💡 Preguntas Recomendadas")
st.sidebar.markdown("- ¿Cómo configuro un DAG básico en Airflow?")
st.sidebar.markdown("- ¿Cuáles son los comandos básicos para ejecutar un proyecto en dbt?")

# --- TÍTULO PRINCIPAL ---
st.markdown("### Asistente Técnico RAG - Ingeniería de Datos")
st.markdown("Pregúntame sobre configuración, buenas prácticas o resolución de errores en herramientas open-source.")

# --- ENTRADA DEL CHAT ---
prompt = st.chat_input("Escribe tu duda técnica aquí...")

if prompt:
    # Muestra la pregunta del usuario
    with st.chat_message("user"):
        st.markdown(prompt)

    # Procesa y muestra la respuesta del asistente
    with st.chat_message("assistant"):
        if retriever is None:
            response = "Error: No se encontró el archivo PDF de la documentación técnica en el repositorio."
        else:
            docs = retriever.invoke(prompt)
            context = "\n\n".join([d.page_content for d in docs])

            if len(context.strip()) == 0:
                response = "No dispongo de información suficiente en la documentación provista para responder esta pregunta."
            else:
                input_text = f"Utiliza exclusivamente el siguiente contexto técnico para responder a la pregunta de manera clara y profesional.\n\nContexto:\n{context}\n\nPregunta: {prompt}\nRespuesta:"
                raw_response = llm.invoke({"inputs": input_text})
                response = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        st.markdown(response)
