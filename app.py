import streamlit as st
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURACIÓN CENTRALIZADA ---
CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "temperature": 0.1,
    "max_tokens": 512,
    "retrieval_k": 3,
    "pdf_path": "documentacion_tecnica.pdf"
}

st.set_page_config(page_title="Asistente Técnico de Datos", page_icon="🤖", layout="centered")

# --- INTERFAZ: GUÍA DE USUARIO Y DESCARGO DE RESPONSABILIDAD ---
st.sidebar.title("🛠️ Configuración y Ayuda")
st.sidebar.info(
    "**Acerca del Asistente:**\n"
    "Este chatbot es un prototipo RAG experto en herramientas de ingeniería de datos de código abierto.\n\n"
    "**Limitaciones y Descargo:**\n"
    "El modelo puede cometer errores o alucinar información que no esté en la documentación oficial. Verifique siempre los comandos críticos."
)

st.sidebar.markdown("### 💡 Preguntas Recomendadas")
st.sidebar.code("¿Cómo configuro un DAG en Airflow?")
st.sidebar.code("¿Cuáles son los comandos básicos de dbt?")

st.title("🤖 Asistente Técnico de Soporte (Data Engineering)")
st.write("Pregúntame sobre configuración, buenas prácticas o resolución de errores en herramientas open-source.")

# --- INICIALIZACIÓN DEL RAG Y MEMORIA ---
@st.cache_resource
def init_rag_system():
    if os.path.exists(CONFIG["pdf_path"]):
        loader = PyPDFLoader(CONFIG["pdf_path"])
        raw_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(raw_documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store.as_retriever(search_kwargs={"k": CONFIG["retrieval_k"]})
    return None

retriever = init_rag_system()

# Historial de chat en Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- LÓGICA DE INTERACCIÓN ---
if prompt := st.chat_input("Escribe tu duda técnica aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if retriever is None:
            response = "⚠️ Advertencia: No se encontró el archivo PDF de documentación en el entorno."
        else:
 docs = retriever.invoke(prompt)
context = "\n\n".join([d.page_content for d in docs])

if len(context.strip()) == 0:
    response = "No dispongo de información suficiente en la documentación provista para responder esta pregunta."
else:
    # Creamos el prompt que une el contexto del PDF y la pregunta del usuario
    input_text = f"Utiliza exclusivamente el siguiente contexto técnico para responder a la pregunta de manera clara y profesional.\n\nContexto:\n{context}\n\nPregunta: {prompt}\nRespuesta:"
    
    # Invocamos al modelo LLM (si es un ChatModel, usa .content; si es un LLM tradicional, devuelto como texto plano)
    raw_response = llm.invoke(input_text)
    response = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
