# --- ENTRADA DEL CHAT ---
prompt = st.chat_input("Escribe tu duda técnica aquí...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

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
                # Corrección: pasamos el input como texto plano (str) directamente
                raw_response = llm.invoke(str(input_text))
                response = raw_response.content if hasattr(raw_response, "content") else str(raw_response)

        st.markdown(response)
