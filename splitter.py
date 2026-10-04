from document_loader import load_documents
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
# SemanticChunker requires an embedding model to evaluate the meaning of sentences
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

from rich.console import Console
from rich.panel import Panel

console = Console()

def run_splitters():
    # 1. Load the documents using our existing loader script
    console.print("\n[bold cyan]Step 1: Loading documents...[/bold cyan]")
    docs = load_documents("Data")
    # docs=documents found in the data_loader--8 docs
    
    if not docs:
        console.print("[bold red]No documents found to split![/bold red]")
        return
        
    console.print(f"[bold green]✔ Successfully loaded {len(docs)} documents/pages.[/bold green]\n")

    # ==========================================
    # DEFINING ALL AVAILABLE SPLITTERS
    # ==========================================
    
    # A. Fixed Character Text Splitter (splits strictly by a specific character, usually newline)
    fixed_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50
    )
    
    # B. Semantic Chunker (splits based on the semantic meaning/context using an AI embedding model)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        semantic_splitter = SemanticChunker(embeddings)
    except Exception as e:
        console.print(f"[yellow]Semantic splitter warning: {e}[/yellow]")
    
    # C. Recursive Character Text Splitter (The industry standard default)
    # It tries to split on paragraphs \n\n, then sentences \n, then spaces " "
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,       # Max size of each chunk
        chunk_overlap=100,    # Overlap between chunks so we don't cut contexts/sentences in half
        length_function=len,
        add_start_index=True, # Keeps track of where in the original doc this chunk came from
    )
    
    # ==========================================
    # PASS DOCUMENTS TO RECURSIVE SPLITTER
    # ==========================================
    console.print("[bold magenta]Step 2: Splitting documents using RecursiveCharacterTextSplitter...[/bold magenta]")
    
    # The split_documents method takes the list of LangChain Document objects and chunks them
    chunked_docs = recursive_splitter.split_documents(docs)
    print(chunked_docs)
    
    console.print(f"Original pages/elements count: [bold]{len(docs)}[/bold]")
    console.print(f"Total chunks created: [bold]{len(chunked_docs)}[/bold]\n")
    
    # Let's print a preview of up to 35 chunks to see how beautifully it split the text
    console.print("[bold cyan]Preview of up to 35 chunks:[/bold cyan]")
    for i in range(min(35, len(chunked_docs))):
        chunk = chunked_docs[i]
        
        # Format the metadata to be clearly visible
        meta_str = ", ".join([f"{k}: {v}" for k, v in chunk.metadata.items()])
        
        panel = Panel(
            chunk.page_content.strip(),
            title=f"[bold green]Chunk #{i + 1}[/bold green]",
            subtitle=f"[dim]Metadata: {meta_str}[/dim]",
            expand=False,
            border_style="cyan"
        )
        console.print(panel)
        
    return chunked_docs

if __name__ == "__main__":
    run_splitters()
