from rag_pipline import answer_query, retrieve_docs, llm_model
import streamlit as st
import os
from vector_database import create_faiss_db  # new function to build FAISS from a file

# Step1: Setup Upload PDF functionality
upload_file = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=False)

# Step2: Chatbot Skeleton (Question & Answer)
user_query = st.text_area("Enter your prompt:", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:
    if upload_file:
        # Save uploaded file
        os.makedirs("pdfs", exist_ok=True)
        file_path = os.path.join("pdfs", upload_file.name)
        with open(file_path, "wb") as f:
            f.write(upload_file.getbuffer())

        # Create FAISS DB from this uploaded file
        faiss_db = create_faiss_db(file_path)

        st.chat_message("user").write(user_query)

        # RAG Pipeline using FAISS from uploaded file
        retrieved_docs = retrieve_docs(user_query, faiss_db)
        response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
        st.chat_message("AI Lawyer").write(response)

    else:
        st.error("Kindly upload a valid PDF file first!")
