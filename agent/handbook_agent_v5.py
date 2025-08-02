import os
import shutil
import pdfplumber
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA

# Load API keys from .env
load_dotenv()

# Set up embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Global for vectorstore directory
FAISS_DIR = "faiss_index"

# Extract and chunk text from uploaded file
def doc_splitter(uploaded_file):
    text = ""

    if uploaded_file.type == "application/pdf":
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return []

    elif uploaded_file.type.startswith("text"):
        try:
            text = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            print(f"Error reading text file: {e}")
            return []
    else:
        print("Unsupported file type. Only PDF and TXT allowed.")
        return []

    if not text.strip():
        print("Uploaded file contains no extractable text.")
        return []

    doc = Document(page_content=text, metadata={"source": uploaded_file.name})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents([doc])

# Add chunks to FAISS DB
def add_to_vectorstore(chunks):
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(FAISS_DIR)
    return db

# Load FAISS DB
def load_vectorstore():
    return FAISS.load_local(FAISS_DIR, embedding_model, allow_dangerous_deserialization=True)

# Set up LLM and QA Chain
def setup_qa_chain(db):
    api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.2)

    prompt_template = """
    You are a helpful assistant for school-related document questions.
    Only use information from the context. Do not hallucinate.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    retriever = db.as_retriever(search_kwargs={"k": 10})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                           return_source_documents=False,
                                           chain_type_kwargs={"prompt": PROMPT})
    return qa_chain

# Function to process documents from the data folder and setup RAG
def preprocess_and_setup_rag():
    """Process handbook and calendar documents and return QA chain"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    documents = []
    
    # Process all files in the data directory
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        if filename.endswith('.md'):
            # Process markdown files
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                doc = Document(page_content=text, metadata={"source": filename})
                documents.append(doc)
        
        elif filename.endswith('.pdf'):
            # Process PDF files
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    
                    if text.strip():
                        doc = Document(page_content=text, metadata={"source": filename})
                        documents.append(doc)
            except Exception as e:
                print(f"Error reading PDF {filename}: {e}")
    
    if not documents:
        raise Exception("No documents found in data directory")
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Create and save vectorstore
    if os.path.exists(FAISS_DIR):
        shutil.rmtree(FAISS_DIR)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(FAISS_DIR)
    
    # Return QA chain
    return setup_qa_chain(db)

# Function to answer questions using the QA chain
def answer_handbook_question(question, qa_chain):
    """Answer a question using the QA chain"""
    try:
        if qa_chain is None:
            return "Sorry, the system is not initialized yet. Please click 'Start Assistant' first."
        
        response = qa_chain.run(question)
        return response
    except Exception as e:
        return f"❌ Error: {e}"

# Function to process uploaded files (for file upload functionality)
def process_uploaded_file(uploaded_file):
    """Process an uploaded file and return QA chain"""
    chunks = doc_splitter(uploaded_file)
    if chunks:
        db = add_to_vectorstore(chunks)
        return setup_qa_chain(db)
    return None
