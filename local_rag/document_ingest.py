import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_document(pdf_path, persist_directory="./local_rag/chroma_db"):
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} pages. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    
    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Storing embeddings in ChromaDB...")
    # Chroma uses persist_directory to save locally
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=persist_directory
    )
    vector_db.persist()
    print("Ingestion complete. Database persisted.")

if __name__ == "__main__":
    pdf_file = "local_rag/complex_ma_agreement.pdf"
    if not os.path.exists(pdf_file):
        print(f"Error: {pdf_file} not found. Please generate or provide it.")
    else:
        ingest_document(pdf_file)
