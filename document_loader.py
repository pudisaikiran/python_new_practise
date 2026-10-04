import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader
)
from rich.console import Console
from rich.panel import Panel

# Initialize the Rich console for colorful prints
console = Console()

def load_documents(data_dir: str = "Data"): 
    documents = []
    
    if not os.path.exists(data_dir): 
        console.print(f"[bold red]✖ Directory '{data_dir}' not found.[/bold red]")
        return documents

    console.print(f"\n[bold cyan]📂 Scanning directory:[/bold cyan] {data_dir}\n")

    for filename in os.listdir(data_dir): 
        filepath = os.path.join(data_dir, filename)
        
        # Skip if it's a directory or a hidden system file (like desktop.ini)
        if not os.path.isfile(filepath) or filename.startswith('.') or filename == "desktop.ini":
            continue
            
        console.print(f"[yellow]Processing:[/yellow] {filename} ...")
        
        try:
            if filename.lower().endswith(".pdf"):  
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                console.print(f"  [green]✔ Success! Loaded {len(docs)} pages.[/green]")
                
            elif filename.lower().endswith(".txt"):
                loader = TextLoader(filepath)
                docs = loader.load()
                documents.extend(docs)
                console.print(f"  [green]✔ Success! Loaded {len(docs)} document chunks.[/green]")
                
            elif filename.lower().endswith((".xlsx", ".xls")):
                loader = UnstructuredExcelLoader(filepath, mode="elements")
                docs = loader.load()
                documents.extend(docs)
                console.print(f"  [green]✔ Success! Loaded {len(docs)} excel elements.[/green]")
                
            else:
                console.print(f"  [dim]⚠ Skipped unsupported file type.[/dim]")
                
        except Exception as e:
            console.print(f"  [bold red]✖ Error loading {filename}:[/bold red] {str(e)}")
            
    return documents

if __name__ == "__main__":
    # Load all documents
    docs = load_documents()
    
    # Print a beautiful summary
    console.print("\n[bold magenta]📊 --- SUMMARY ---[/bold magenta]")
    console.print(f"Total documents (pages/elements) loaded: [bold white]{len(docs)}[/bold white]")
    
    # If we successfully loaded documents, show a pretty preview of the first one
    if docs:
        for idx, doc in enumerate(docs):
            # We don't truncate anymore, we print the FULL text
            full_text = doc.page_content.strip()
            
            console.print(f"\n[bold cyan]📄 Document #{idx + 1} (Full Text):[/bold cyan]")
            
            # Create a formatted panel box using Rich for EVERY page/document
            panel = Panel(
                full_text,
                title=f"[bold green]Source: {doc.metadata.get('source', 'Unknown')}[/bold green]",
                subtitle=f"[dim]Page/Element: {doc.metadata.get('page', 'N/A')}[/dim]",
                expand=False,
                border_style="cyan"
            )
            console.print(panel)
        console.print("\n[bold magenta]✔ Finished printing all data![/bold magenta]\n")

