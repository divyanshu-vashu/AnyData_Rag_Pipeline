import os
import uuid
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum

import chromadb
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# --- Configuration ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
    print("Warning: GROQ_API_KEY not set. Please set it in your environment variables.")

DATA_DIRECTORY = "data"
VECTOR_STORE_DIRECTORY = os.path.join(DATA_DIRECTORY, "vector_store")
UPLOAD_DIRECTORY = os.path.join(DATA_DIRECTORY, "uploads")


app = FastAPI(
    title="AnyData RAG Pipeline API",
    description="Upload documents (PDF or TXT) and ask questions using a RAG pipeline.",
)


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        return self.model.encode(texts, show_progress_bar=True)

class VectorStore:
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = VECTOR_STORE_DIRECTORY):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._initialize_store()
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
        except Exception as e:
            raise
    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        ids = [f"doc_{uuid.uuid4().hex[:8]}_{i}" for i, _ in enumerate(documents)]
        metadatas = [doc.metadata for doc in documents]
        doc_texts = [doc.page_content for doc in documents]
        self.collection.add(ids=ids, embeddings=embeddings.tolist(), metadatas=metadatas, documents=doc_texts)

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        retrieved_docs = []
        if results.get('documents'):
            for i, (doc_text, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
                retrieved_docs.append({
                    'content': doc_text,
                    'metadata': metadata,
                    'similarity_score': 1 - distance,
                    'rank': i + 1
                })
        return retrieved_docs

embedding_manager = EmbeddingManager()
vector_store = VectorStore()
rag_retriever = RAGRetriever(vector_store, embedding_manager)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it", temperature=0.1, max_tokens=1024)

# --- Helper Functions for File Processing ---

def process_pdf(file_path: str):
    """Loads and processes a single PDF file."""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata['source_file'] = Path(file_path).name
            doc.metadata['file_type'] = 'pdf'
        return documents
    except Exception as e:
        print(f"Error processing PDF {file_path}: {e}")
        return []

def process_text(file_path: str):
    """Loads and processes a single text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        documents = [Document(
            page_content=text_content,
            metadata={
                'source_file': Path(file_path).name,
                'file_type': 'text'
            }
        )]
        return documents
    except Exception as e:
        print(f"Error processing text file {file_path}: {e}")
        return []

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def process_and_embed_file(file_path: str, file_type: str):
    """Processes a file, splits it, generates embeddings, and adds to the vector store."""
    if file_type == 'pdf':
        documents = process_pdf(file_path)
    elif file_type == 'text':
        documents = process_text(file_path)
    else:
        print(f"File type '{file_type}' not currently supported for processing.")
        return

    if documents:
        chunks = split_documents(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
        print(f"Successfully processed and embedded {file_path}")

# --- FastAPI Endpoints ---

class FileFormat(str, Enum):
    pdf = "pdf"
    text = "text"

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_type: str

@app.post("/upload/", response_model=UploadResponse)
async def upload_file_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_format: FileFormat = Form(...)
):
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    background_tasks.add_task(process_and_embed_file, file_path, file_format.value)

    return {"message": "File uploaded and processing started.", "filename": file.filename, "file_type": file_format.value}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/query/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    retrieved_docs = rag_retriever.retrieve(request.query, top_k=request.top_k)
    if not retrieved_docs:
        return {"answer": "No relevant information found.", "sources": []}

    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
    prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"

    try:
        response = llm.invoke(prompt)
        answer = response.content
        sources = [
            {
                "source_file": doc["metadata"].get("source_file", "Unknown"),
                "score": doc["similarity_score"],
            }
            for doc in retrieved_docs
        ]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
