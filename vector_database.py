from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#Step1: Upload & Load raw PDF(s)

pdfs_directory="pdfs/"

def upload_pdf(file):
    with open(pdfs_directory+file.name,"wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader=PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents 



# file_path = "pdfs/universal_human_rights.pdf"
# documents=load_pdf(file_path) 
# print(len(documents))