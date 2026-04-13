from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import re
from dotenv import load_dotenv

load_dotenv()

# ------------------- CONFIG -------------------

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "learning_langchain_v1"


# ------------------- CLEAN RESPONSE -------------------

def clean_response(text):
    # Remove <think>...</think> block
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ------------------- LOAD FILE -------------------

def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path=file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path=file_path)
    else:
        loader = TextLoader(file_path=file_path)

    return loader.load()


# ------------------- SPLIT -------------------

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


# ------------------- EMBEDDINGS -------------------

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ------------------- INGEST -------------------

def ingest_to_qdrant(docs):
    embeddings = get_embeddings()

    QdrantVectorStore.from_documents(
        documents=docs,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


# ------------------- LOAD VECTOR STORE -------------------

def load_vector_store():
    embeddings = get_embeddings()

    return QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


# ------------------- LLM -------------------

def get_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
    )


# ------------------- ASK QUESTION -------------------

def ask_question(query):
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a helpful assistant who responds based on available context.

    Context:
    {context}

    Answer ONLY from the context above.
    If you don't know, say "I don't know".
    If the question is not related to the context, say "The question is not related to the context".

    Question:
    {query}
    """

    llm = get_llm()
    response = llm.invoke(prompt)

    # ✅ Clean output
    return clean_response(response.content)


# ------------------- FULL PIPELINE -------------------

def process_file(file_path):
    docs = load_file(file_path)
    split_docs = split_documents(docs)

    ingest_to_qdrant(split_docs)

    return "Ingestion Complete"