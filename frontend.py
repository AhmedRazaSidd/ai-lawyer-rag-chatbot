#Step1: Setup Upload PDF functionality
import streamlit as st

upload_file =st.file_uploader("Upload PDF",type="pdf",accept_multiple_files=False)

#Step2: Chatbot Skeleton (Question & Answer)

user_query=st.text_area("Enter your prompt:",height=150,placeholder="Ask Anything!")

ask_question = st.button("Ask AI Lawyer")

if ask_question:
    
    if upload_file:
        st.chat_message("user").write(user_query)
        
        #RAG Pipeline
        fixed_response = "Hi, this is a fixed response!"
        st.chat_message("AI Lawyer").write(fixed_response)
    
    else:
        st.error("Kindly upload a valid PDF file first!")