import os
import uuid
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
if GROQ_API_KEY == "YOUR_GROQ_API_KEY":
    print("Warning: GROQ_API_KEY not set. Please set it in your environment variables.")


DATA_DIRECTORY = "data"
VECTOR_STORE_DIRECTORY = os.path.join(DATA_DIRECTORY, "vector_store")
UPLOAD_DIRECTORY = os.path.join(DATA_DIRECTORY, "uploads")


app = FastAPI(
    title="AnyData RAG Pipeline API",
    description="Upload documents and ask questions using a RAG pipeline.",
)



class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model not loaded")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = VECTOR_STORE_DIRECTORY):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        print(f"Adding {len(documents)} documents to vector store...")
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(results['ids'][0], results['documents'][0], results['metadatas'][0], results['distances'][0])):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []


embedding_manager = EmbeddingManager()
vector_store = VectorStore()
rag_retriever = RAGRetriever(vector_store, embedding_manager)
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it", temperature=0.1, max_tokens=1024)



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

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def process_and_embed_file(file_path: str, file_type: str):
    """Processes a file, splits it, generates embeddings, and adds to the vector store."""
    if file_type == 'pdf':
        documents = process_pdf(file_path)
    else:
        # Placeholder for other file types
        print(f"File type '{file_type}' not currently supported for processing.")
        return

    if documents:
        chunks = split_documents(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        vector_store.add_documents(chunks, embeddings)
        print(f"Successfully processed and embedded {file_path}")

# --- FastAPI Endpoints ---

class UploadResponse(BaseModel):
    message: str
    filename: str
    file_type: str

@app.post("/upload/", response_model=UploadResponse)
async def upload_file_and_process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    file_format: str = Form(...)
):
    """
    Uploads a file, saves it, and triggers background processing to create a vector DB.
    """
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # Add the processing to a background task so the API can respond immediately
    background_tasks.add_task(process_and_embed_file, file_path, file_format)

    return {"message": "File uploaded and processing started in the background.", "filename": file.filename, "file_type": file_format}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/query/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Takes a user query, retrieves relevant context from the vector DB,
    and generates an answer using an LLM.
    """
    retrieved_docs = rag_retriever.retrieve(request.query, top_k=request.top_k)
    if not retrieved_docs:
        return {"answer": "No relevant information found in the uploaded documents.", "sources": []}

    context = "\n\n".join([doc['content'] for doc in retrieved_docs])
    prompt = f"""
    You are a helpful AI assistant. Use the following context to answer the question accurately and concisely.

    Context:
    {context}

    Question: {request.query}

    Answer:
    """

    try:
        response = llm.invoke(prompt)
        answer = response.content
        sources = [
            {
                "source_file": doc["metadata"].get("source_file", "Unknown"),
                "page_number": doc["metadata"].get("page", "N/A"),
                "similarity_score": doc["similarity_score"],
            }
            for doc in retrieved_docs
        ]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response from LLM: {e}")

# --- Main Entry Point ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
