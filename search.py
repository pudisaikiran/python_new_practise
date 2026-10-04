import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# ==========================================
# 1. SETUP EMBEDDINGS & QDRANT CONNECTION
# ==========================================
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# The exact same embedding model used to store the vectors
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")

embeddings_model = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
    azure_deployment=embedding_deployment
)

# Connect to the exact same Qdrant Cloud cluster
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY")
)
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Rag_project")

# Wrap the connection in the LangChain VectorStore
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=qdrant_collection_name,
    embedding=embeddings_model,
)

# Create a Retriever: This is the tool that embeds the question and runs the Cosine search.
# We tell it to return the top 3 closest chunks (k=3).
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# ==========================================
# 2. SETUP THE CHAT AI (LLM)
# ==========================================
# This reads MODEL_NAME from your .env which you set to "gpt-5.4-mini-1" (or similar)
chat_deployment = os.getenv("MODEL_NAME")

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version,
    azure_deployment=chat_deployment,
    temperature=0.2 # Keep it low so the AI is factual and doesn't hallucinate
)

# Give the AI strict instructions to only use our database chunks to answer
prompt_template = """You are a helpful assistant. Answer the user's question based ONLY on the following context retrieved from the database.
If the context doesn't contain the answer, just say "I don't know based on the provided documents."

Context (Database Chunks):
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(prompt_template)

# Helper function to join the 3 chunks together into one big string for the LLM to read
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the complete pipeline
# We export this so app.py can import and use it!
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
