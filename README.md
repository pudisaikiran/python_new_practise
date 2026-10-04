# Qdrant RAG Chat API

This project is a complete Retrieval-Augmented Generation (RAG) web application. It allows you to upload private documents (PDFs, Excel, Text) and ask an AI questions about those documents. The AI answers using the context retrieved from a Qdrant Vector Database.

The project features a **FastAPI** backend, **Azure OpenAI** for LLMs and Embeddings, **Qdrant** for vector storage, and a built-in user authentication system using **SQLite** and **JWT**.

---

## 🚀 Features

- **Document Ingestion:** Automatically parse PDFs, TXT, and Excel files.
- **Smart Text Chunking:** Uses LangChain's RecursiveTextSplitter to create optimal chunks for AI processing.
- **Vector Search:** Highly accurate semantic search powered by Qdrant and Azure OpenAI Embeddings.
- **AI Chat:** Interactive chat endpoint that restricts the AI to only answer based on the provided documents.
- **User Authentication:** Secure registration, login, and JWT token management.
- **Web UI:** Includes a built-in `index.html` frontend served by FastAPI.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following:

1. **Python 3.9+** installed on your machine.
2. **Azure OpenAI Account:** You need API keys and deployments for both an Embedding model and a Chat model.
3. **Qdrant Cloud Account:** A free cluster URL and API key from Qdrant.

---

## ⚙️ Setup and Installation

### 1. Clone or Download the Repository
Navigate to the project folder in your terminal:
```bash
cd python_new_practise
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
python -m venv env
# Activate on Windows:
.\env\Scripts\activate
# Activate on Mac/Linux:
source env/bin/activate
```

### 3. Install Dependencies
Install all required Python packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory (if it doesn't exist) and add your API keys:

```env
# Azure OpenAI Credentials
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
MODEL_NAME=your_chat_deployment_name

# Qdrant Credentials
QDRANT_URL=https://your-cluster-url.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=my_documents
```

---

## 🏃‍♂️ Running the Application

This project is split into two phases: **Data Ingestion** and **Running the API**.

### Phase 1: Ingesting your Documents
1. Place your private documents (PDFs, Excel files, Text files) inside the `Data/` folder.
2. Run the embeddings script to chunk the files, generate vectors, and store them in Qdrant:
```bash
python embeddings.py
```
*Wait for the script to finish. It will print a success message to the console once the vectors are stored.*

### Phase 2: Starting the Server
Once your database is populated, you can start the FastAPI web server.
```bash
python app.py
```
*The server will start on `http://localhost:8000`.*

---

## 🌐 API Endpoints

Once the server is running, you can access the following:

- **Web UI:** Open `http://localhost:8000/` in your browser.
- **Interactive Docs (Swagger UI):** `http://localhost:8000/docs`

**Core Endpoints:**
- `POST /chat`: Send a JSON payload `{"question": "your question"}` to get an AI response based on your documents.
- `POST /token`: Login endpoint to retrieve a JWT.
- `POST /register`: Create a new user account.

---

## 📚 Further Reading

For a deeper understanding of how the code works, the architecture, and file-by-file explanations, please read the [Technical Documentation](technical_documentation.md).
