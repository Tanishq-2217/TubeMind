# TubeMind — YouTube Video RAG Pipeline

TubeMind is a Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **FAISS**, and **Google Gemini**. It extracts English transcripts from YouTube videos, splits them into semantic chunks, generates vector embeddings, and answers user questions grounded strictly in the video content.

## Features
- **Transcript Ingestion:** Fetches English captions directly from YouTube URLs using `YoutubeLoader` (`youtube-transcript-api`).
- **Semantic Chunking:** Splits long transcripts into overlapping segments (`chunk_size=1000`, `chunk_overlap=200`) via `RecursiveCharacterTextSplitter`.
- **Vector Search:** Embeds transcript chunks using `models/gemini-embedding-001` and indexes them locally in a **FAISS** vector store for similarity retrieval (`k=4`).
- **Context-Grounded Q&A:** Uses `ChatGoogleGenerativeAI` (`gemini-3.6-flash`) with a strict prompt template that prevents hallucination when the answer isn't present in the transcript.

## Tech Stack
- **Framework:** LangChain (`langchain-community`, `langchain-text-splitters`, `langchain-core`)
- **LLM & Embeddings:** Google Gemini (`langchain-google-genai`)
- **Vector Database:** FAISS (`faiss-cpu`)
- **Data Loader:** `youtube-transcript-api`
- **Environment Management:** `python-dotenv`

## Project Structure
```text
TubeMind/
├── Main.ipynb          # End-to-end RAG notebook (ingestion, embedding, retrieval, Q&A)
├── .env.example        # Environment variable template
├── .gitignore          # Excludes .env, venv/, and cache files
└── README.md           # Project documentation
```

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tanishq-2217/TubeMind.git
   cd TubeMind
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install langchain-community langchain-text-splitters langchain-core langchain-google-genai youtube-transcript-api faiss-cpu python-dotenv
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY="your-google-gemini-api-key"
   ```

## Usage
1. Open `Main.ipynb` in VS Code or Jupyter Notebook and select your `venv` Python kernel.
2. Run the environment setup cells to verify `GEMINI_API_KEY` loads properly.
3. Replace `link` with any YouTube video URL that has English transcripts enabled.
4. Run the embedding and retriever cells to build the FAISS index, then pass your query to `question` to generate a transcript-backed response.

## Roadmap
- [x] YouTube transcript extraction (`YoutubeLoader`)
- [x] Text splitting & chunking (`RecursiveCharacterTextSplitter`)
- [x] Gemini embeddings & FAISS vector store indexing
- [x] Similarity retriever & grounded Q&A prompt pipeline
