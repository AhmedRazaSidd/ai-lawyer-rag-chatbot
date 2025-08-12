from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

pdfs_directory = "pdfs/"

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    file_path = os.path.join(pdfs_directory, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def get_embedding_model(model_name):
    return OllamaEmbeddings(model=model_name)

# Function to create FAISS DB from any file
def create_faiss_db(file_path, model_name="deepseek-r1:1.5b"):
    documents = load_pdf(file_path)
    text_chunks = create_chunks(documents)
    return FAISS.from_documents(text_chunks, get_embedding_model(model_name))

# Default DB (only loads if default PDF exists)
default_file_path = os.path.join(pdfs_directory, "universal_human_rights.pdf")
faiss_db = None
if os.path.exists(default_file_path):
    faiss_db = create_faiss_db(default_file_path)
