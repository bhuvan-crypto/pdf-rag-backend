from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 
import chromadb
# NEW IMPORTS FOR MISTRAL AI
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from fastapi.middleware.cors import CORSMiddleware

import os
import shutil

load_dotenv()

app = FastAPI(root_path="/api-python")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary global store (for demo)
pdf_vector_db = None

# --- Configuration ---
# MISTRAL API Key is used for both LLM and Embeddings
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 
if not MISTRAL_API_KEY:
    print("Warning: MISTRAL_API_KEY environment variable not found.")

# Chroma Cloud Configuration (Remaining the same)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "f7b07b69-188a-40de-a2c3-fbf74eaca987") 
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "Demo") 
COLLECTION_NAME = "pdf-rag-collection"

def get_chroma_client():
    """Initializes and returns the Chroma Cloud Client."""
    if not CHROMA_API_KEY:
        raise ValueError("CHROMA_API_KEY environment variable is not set.")
    
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    ) 

@app.on_event("startup")
async def load_vectorstore():
    global pdf_vector_db
    
    # 🌟 NEW EMBEDDING MODEL: Mistral AI Embeddings
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )
    
    try:
        chroma_client = get_chroma_client()
        # Connect to the remote Chroma collection
        pdf_vector_db = Chroma(
            client=chroma_client,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings
        )
        print(f"Connected to Chroma Cloud. Collection '{COLLECTION_NAME}' is ready.")
    except ValueError as ve:
        print(f"Error: {ve}")
        pdf_vector_db = None
    except Exception as e:
        print(f"Warning: Could not connect to Chroma Cloud or retrieve collection: {e}")
        pdf_vector_db = None

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_vector_db
    
    # 1. Initialize client and ensure clean slate in the cloud DB
    try:
        chroma_client = get_chroma_client()
        # Delete old collection
        try:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(f"Old Chroma Cloud collection '{COLLECTION_NAME}' deleted.")
        except Exception:
            pass # Collection may not exist

    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": f"Failed to connect to Chroma Cloud: {e}"}


    # ✅ read PDF in memory
    pdf_reader = PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # ✅ split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # ✅ MISTRAL EMBEDDINGS
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY
    )

    # ✅ create Chroma DB (sends data to the remote cloud server)
    pdf_vector_db = Chroma.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        client=chroma_client, 
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    return {"message": "PDF processed successfully", "chunks": len(chunks)}


# ... (rest of the code before prompt_template remains the same)

@app.post("/ask-basic/")
async def ask_basic(question: str):
    global pdf_vector_db
    if pdf_vector_db is None:
        return {"error": "No PDF uploaded yet or Chroma connection failed"}
    
    try:
        # Step 1: Get retriever from vector DB
        retriever = pdf_vector_db.as_retriever(search_kwargs={"k": 3})
        
        # LLM: ChatMistralAI
        llm = ChatMistralAI(
            model="mistral-tiny", # Fast, high-quality dev model
            mistral_api_key=MISTRAL_API_KEY,
            temperature=0.0, 
            max_tokens=256 # Reduced max tokens for shorter answers
        )
        
        # 🌟 UPDATED PROMPT FOR SHORT ANSWERS
        prompt_template = """
You are an expert Q&A assistant. Use ONLY the following context to answer the question briefly and concisely. 

Guidelines:
1.  Answer in a maximum of three sentences.
2.  If the answer is found in the context, provide a direct answer.
3.  If the context does not contain the answer, you MUST respond with: "I cannot find the answer in the provided documents."

Context:
{context}

Question: {question}

Answer:
"""

        PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
        )

        retrievalQA = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = retrievalQA.invoke({"query": question})
            
        return result

    except Exception as e:
        return {"error": str(e)}

# ... (rest of the code remains the same)

@app.get("/check-vectorstore/")
def check_vectorstore():
    global pdf_vector_db
    if pdf_vector_db is None:
        return {"exists": False, "message": "Vectorstore not initialized. Ensure Chroma keys are set and connections are successful."}
    
    try:
        # Check the remote Chroma Cloud collection count
        count = pdf_vector_db._collection.count()
        if count > 0:
            return {"exists": True, "message": f"Chroma Cloud collection '{COLLECTION_NAME}' is active and contains {count} documents."}
        else:
            return {"exists": False, "message": f"Chroma Cloud collection '{COLLECTION_NAME}' is active but empty. Upload a PDF."}
    except Exception as e:
        return {"exists": False, "message": f"Error interacting with Chroma Cloud: {e}"}