from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpointEmbeddings,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

import os
import shutil

load_dotenv()

app = FastAPI()

# Temporary global store (for demo)
pdf_vector_db = None

@app.on_event("startup")
async def load_vectorstore():
    global pdf_vector_db
    if os.path.exists("vectorstore"):
        # Re-initialize embeddings
        embeddings = HuggingFaceEndpointEmbeddings(
            huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Load FAISS from disk
        pdf_vector_db = FAISS.load_local("vectorstore", embeddings,    allow_dangerous_deserialization=True  )

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_vector_db
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")
        print("Old vectorstore deleted.")
    # ✅ read PDF in memory
    pdf_reader = PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # ✅ split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # ✅ HuggingFace embeddings
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
        model="sentence-transformers/all-MiniLM-L6-v2"  # small + fast
    )

    # ✅ create FAISS DB
    pdf_vector_db = FAISS.from_texts(chunks, embeddings)
    pdf_vector_db.save_local("vectorstore")  # stores index and metadata in folder "vectorstore"

    return {"message": "PDF processed successfully", "chunks": len(chunks)}
# Alternative: Simple endpoint that always works
@app.post("/ask-basic/")
async def ask_basic(question: str):
    global pdf_vector_db

    if pdf_vector_db is None:
        return {"error": "No PDF uploaded yet"}
    
    try:
        # Step 1: Get retriever from vector DB
        retriever = pdf_vector_db.as_retriever(search_kwargs={"k": 3})
        llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-Nemo-Base-2407",
                provider="novita",
                do_sample=False,
                huggingfacehub_api_token=os.getenv("HF_TOKEN")
            )
        prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
        2. If you find the answer, write the answer in a concise way with five sentences maximum.

        {context}

        Question: {question}

        Helpful Answer:
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
    

@app.get("/check-vectorstore/")
def check_vectorstore():
    if os.path.exists("vectorstore"):
        return {"exists": True, "message": "Vectorstore already exists."}
    else:
        return {"exists": False, "message": "No vectorstore found."}