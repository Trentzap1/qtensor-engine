from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

class RAGPipeline:
    def __init__(self, persist_directory="./local_rag/chroma_db"):
        if not os.path.exists(persist_directory):
            raise ValueError(f"Chroma DB directory {persist_directory} does not exist. Run document_ingest.py first.")
            
        print("Loading local embeddings and ChromaDB...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        # Configure retriever to fetch top 3 most relevant chunks
        self.retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})

    def retrieve_context(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context

    def format_prompt(self, query: str, context: str) -> str:
        template = f"""<|system|>
You are a highly precise legal and financial assistant. You MUST answer the user's question using ONLY the provided context. If the context does not contain the answer, say "I do not know based on the provided context." Do not use outside knowledge. Do not hallucinate.

Context:
{context}

<|user|>
{query}

<|assistant|>
"""
        return template

if __name__ == "__main__":
    pipeline = RAGPipeline()
    query = "What is the break-up fee?"
    context = pipeline.retrieve_context(query)
    prompt = pipeline.format_prompt(query, context)
    print("--- Test Prompt ---")
    print(prompt)
