import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rich.console import Console

# Import the chunked documents from our splitter script
from splitter import run_splitters

console = Console()

def generate_embeddings():
    # 1. Load environment variables from .env file
    load_dotenv()
    
    # Retrieve Azure OpenAI credentials
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    # Check for AZURE_OPENAI_EMBEDDING_DEPLOYMENT first, then AZURE_OPENAI_DEPLOYMENT
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    # Retrieve Qdrant credentials
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "my_documents")
    
    if not all([api_key, endpoint, api_version, deployment]):
        console.print("[bold red]✖ Missing Azure OpenAI credentials (or deployment name) in .env file![/bold red]")
        return
        
    if not all([qdrant_url, qdrant_api_key]):
        console.print("[bold red]✖ Missing Qdrant credentials (QDRANT_URL, QDRANT_API_KEY) in .env file![/bold red]")
        console.print("[yellow]Please update your .env file with QDRANT_URL and QDRANT_API_KEY.[/yellow]")
        return
        
    console.print("\n[bold cyan]Step 3: Initializing Azure OpenAI Embeddings...[/bold cyan]")
    
    # 2. Get the chunked documents from splitter.py
    chunks = run_splitters()  # 35 chunks
    
    if not chunks:
        console.print("[bold red]✖ No chunks received from splitter![/bold red]")
        return
        
    console.print(f"\n[bold green]✔ Received {len(chunks)} chunks for embedding.[/bold green]")
    
    # 3. Initialize Azure OpenAI Embeddings
    # NOTE: The deployment MUST be an embeddings model (like text-embedding-ada-002).
    # If AZURE_OPENAI_DEPLOYMENT points to an LLM like gpt-4o, this will fail.
    try:
        embeddings_model = AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            azure_deployment=deployment 
        )
        
        # 4. Embed the first chunk as a test
        first_chunk_text = chunks[0].page_content
        console.print(f"[yellow]Generating embedding vector for Chunk #1 (length: {len(first_chunk_text)} chars)...[/yellow]")
        
        # embed_query converts the text into a mathematical vector
        vector = embeddings_model.embed_query(first_chunk_text)
        
        console.print(f"[bold green]✔ Successfully generated embedding![/bold green]")
        console.print(f"Vector Dimensions: [bold white]{len(vector)}[/bold white]")
        console.print(f"Preview of first 5 numbers: {vector[:5]}")
        
        console.print("\n[bold cyan]Step 5: Connecting to Qdrant Database...[/bold cyan]")
        
        # 5. Connect to Qdrant Cloud
        qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key
        )
        
        # Check if collection exists, if not create it
        if not qdrant_client.collection_exists(collection_name=qdrant_collection_name):
            console.print(f"[yellow]Collection '{qdrant_collection_name}' not found. Creating it...[/yellow]")
            # We use len(vector) to automatically set the vector size based on the test embedding above
            qdrant_client.create_collection(
                collection_name=qdrant_collection_name,
                vectors_config=VectorParams(size=len(vector), distance=Distance.COSINE),
            )
            
        console.print(f"\n[bold cyan]Step 6: Storing {len(chunks)} chunks in Qdrant...[/bold cyan]")
        
        # 6. Store embeddings in Qdrant
        qdrant = QdrantVectorStore(
            client=qdrant_client,
            collection_name=qdrant_collection_name,
            embedding=embeddings_model,
        )
        
        # Add documents to Qdrant
        qdrant.add_documents(chunks)
        
        console.print(f"[bold green]✔ Successfully stored embeddings in Qdrant![/bold green]")
        
    except Exception as e:
        console.print(f"\n[bold red]✖ Error generating embeddings:[/bold red] {e}")
        console.print("[dim yellow]Note: Ensure your AZURE_OPENAI_DEPLOYMENT in .env points to an *Embedding* model (e.g. text-embedding-ada-002), NOT an LLM (like gpt-4o-mini). If it's an LLM, OpenAI will throw a model not found / unsupported error.[/dim yellow]")

if __name__ == "__main__":
    generate_embeddings()
