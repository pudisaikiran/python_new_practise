# Technical Documentation: Qdrant RAG Chat API

## 1. Project Summary (Functional Overview)
This project is a **Retrieval-Augmented Generation (RAG) Chat Application**. 

**What does it do?**
Normally, AI models like ChatGPT only know what they were trained on. If you ask them about your company's private documents, they won't know the answer. This project solves that problem.
1. It takes a folder of private documents (PDFs, Excel files, Text files).
2. It breaks those documents down and stores them in a special "Vector Database".
3. When a user asks a question via a web interface, the system searches the database for the most relevant pieces of information.
4. It then sends both the question and the found information to an AI (Azure OpenAI) to generate a highly accurate, customized answer.

The project also includes a User Authentication system to ensure that only registered/logged-in users can use the API.

---

## 2. How the Project is Built (Technical Details & Architecture)

The architecture is built using modern Python frameworks and AI tools. Here are the core technical components and how they talk to each other:

### **Tech Stack**
- **Web Framework:** [FastAPI](https://fastapi.tiangolo.com/) - A modern, fast web framework for building APIs in Python.
- **AI / LLM Orchestration:** [LangChain](https://python.langchain.com/) - A framework designed to simplify the creation of applications using large language models (LLMs).
- **Vector Database:** [Qdrant](https://qdrant.tech/) - A specialized database that stores data as "vectors" (lists of numbers) instead of rows and columns, making semantic search extremely fast.
- **Relational Database (Auth):** SQLite with [SQLAlchemy](https://www.sqlalchemy.org/) ORM.
- **AI Models:** Azure OpenAI (Embeddings for text translation to vectors, and Chat Models for answering).
- **Security:** JWT (JSON Web Tokens) and bcrypt for secure password hashing.

### **The Data Flow (How data moves through the app)**
1. **Document Loading:** Files from the `Data/` folder are read and parsed into plain text.
2. **Text Splitting (Chunking):** The text is too large to process all at once, so it's split into smaller overlapping "chunks".
3. **Embedding Generation:** Each chunk is sent to an Azure OpenAI Embedding model. The model reads the text and turns the *meaning* of the text into a mathematical vector (a long list of numbers).
4. **Vector Storage:** These vectors, along with the original text, are saved into Qdrant.
5. **Searching / Querying:** When a user asks a question:
   - The question itself is turned into a vector.
   - The system performs a "Cosine Similarity Search" in Qdrant to find the database vectors that are mathematically closest to the question vector.
   - The original text chunks associated with those closest vectors are retrieved.
6. **AI Answering:** The retrieved text chunks and the user's question are combined into a prompt and sent to the Azure Chat LLM to generate a human-friendly response.

---

## 3. File-by-File Description (Codebase Guide)

Here is a detailed breakdown of what each Python script in the project does:

### `app.py`
**Purpose:** The main entry point of the application. This is the web server.
**Technical Details:** 
- Initializes the `FastAPI` application.
- Sets up CORS (Cross-Origin Resource Sharing) so a frontend (like `index.html`) can talk to the backend.
- Serves the frontend UI at the root `/` endpoint.
- Provides the `/chat` endpoint which receives the user's question, triggers the LangChain RAG pipeline (imported from `search.py`), and returns the AI's answer.
- Uses `uvicorn` to run the server on port 8000.

### `database.py`
**Purpose:** Sets up the traditional SQL database used for user authentication.
**Technical Details:**
- Uses SQLAlchemy to connect to a local SQLite database file (`sql_app.db`).
- Defines the "Schemas" (Database tables) as Python classes:
  - `User`: Stores user details, email, and a hashed password.
  - `RefreshToken`: Stores token data to keep users logged in securely.
- Provides a `get_db()` dependency function which safely opens and closes a database session for incoming API requests.

### `security.py`
**Purpose:** Handles all password encryption and token generation.
**Technical Details:**
- Uses `bcrypt` to securely hash user passwords. You should never save raw passwords in a database.
- Uses `jwt` to create JSON Web Tokens. When a user logs in, they get a JWT. They must send this token with every subsequent request to prove who they are.
- Contains helper functions to create access tokens and refresh tokens.

### `document_loader.py`
**Purpose:** Scans a directory and reads files into the Python environment.
**Technical Details:**
- Uses LangChain's community loaders (`PyPDFLoader`, `TextLoader`, `UnstructuredExcelLoader`).
- Iterates through a specified folder (default `Data/`), checks the file extension, and uses the appropriate tool to extract raw text from PDFs, Text files, and Excel spreadsheets.
- Uses the `rich` library to print beautiful, colored logs to the terminal.

### `splitter.py`
**Purpose:** Breaks large documents into smaller, manageable chunks.
**Technical Details:**
- Imports the loaded documents from `document_loader.py`.
- Uses LangChain's `RecursiveCharacterTextSplitter`. This is crucial because LLMs have a "context window" limit (how much text they can read at once).
- It splits text primarily by paragraphs (`\n\n`), then lines (`\n`), ensuring that sentences are not broken in half.
- Adds an "overlap" between chunks so context isn't lost between boundaries.

### `embeddings.py`
**Purpose:** The ingestion engine. Converts text chunks to vectors and uploads them to the Vector Database.
**Technical Details:**
- Loads environment variables (like API keys) using `dotenv`.
- Receives the chunked text from `splitter.py`.
- Initializes the `AzureOpenAIEmbeddings` model.
- Initializes the `QdrantClient` to connect to the cloud database.
- Uses `QdrantVectorStore.add_documents()` to process all chunks. Behind the scenes, this sends the text to Azure to get vectors, and then uploads both the text and the vectors to Qdrant.

### `search.py`
**Purpose:** The RAG (Retrieval-Augmented Generation) Engine.
**Technical Details:**
- Connects to the existing Qdrant database and Azure OpenAI models.
- Creates a `retriever` from the vector store, configuring it to fetch the top 3 closest matches (`k=3`) for any query.
- Defines a strict `ChatPromptTemplate` that forces the LLM to only answer based on the provided database context.
- Builds a LangChain pipeline (LCEL - LangChain Expression Language): `rag_chain = retriever -> prompt -> LLM -> Output`. 
- This `rag_chain` is imported and triggered by `app.py` when a user sends a chat message.

### `check_qdrant.py`
**Purpose:** A utility/debugging script.
**Technical Details:**
- Used to verify that data actually exists in the Qdrant database.
- It connects to Qdrant and pulls a single "point" (record), printing the ID, the payload (the original text), and the hidden mathematical vector to the console for learning and debugging purposes.

### `.env` (Not a Python file, but critical)
**Purpose:** Stores configuration and secrets.
**Technical Details:**
- Holds sensitive API keys (Azure, Qdrant) and configurations. These files are kept out of version control (like GitHub) to prevent hackers from stealing cloud credentials.

---

## 4. Learning Takeaways for Beginners
1. **Modularity is Key:** Notice how the code isn't just in one massive `app.py` file. Splitting logic into `database.py`, `security.py`, and `search.py` makes the code readable, maintainable, and easy for a team to collaborate on.
2. **Vectors are Magic:** Traditional databases find exact word matches. Vector databases find *meaning*. If you search "Puppy", a vector database knows that a document about a "Dog" is highly relevant, because their mathematical vectors are close together in multi-dimensional space.
3. **Prompt Engineering:** In `search.py`, look at how the prompt is constructed. By explicitly telling the AI: *"Answer the user's question based ONLY on the following context... If the context doesn't contain the answer, just say 'I don't know'"*, the system prevents the AI from "hallucinating" (making up fake answers).
