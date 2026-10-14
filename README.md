# The Himalayan Grand Hotel: LangChain RAG Chatbot

A beginner-friendly, terminal-based Retrieval-Augmented Generation (RAG) chatbot. It reads hotel information from PDF and Word documents, retrieves relevant passages using a Chroma vector database, and uses Google Gemini to answer guest questions.

## Features

- Loads PDF and DOCX files.
- Splits documents into smaller chunks with `RecursiveCharacterTextSplitter`.
- Creates embeddings with Google Gemini.
- Stores and searches vectors with Chroma.
- Retrieves the top 3 relevant chunks for each question.
- Uses a system prompt to keep answers grounded in the supplied hotel information.
- Runs in the Python terminal.

## RAG workflow

1. **Load documents:** LangChain document loaders read PDF and DOCX files from `hotel_docs/`.
2. **Split text:** `RecursiveCharacterTextSplitter` divides documents into chunks.
3. **Create embeddings:** `GoogleGenerativeAIEmbeddings` converts chunks into vectors.
4. **Store and retrieve:** Chroma stores vectors and finds chunks similar to the user's question.
5. **Generate an answer:** The retrieved context and question are sent to the Gemini chat model through a LangChain chain.

## Project structure

```text
HimalayanHotel_LangChain_RAG/
├── rag.py
├── requirements.txt
├── .env
├── hotel_docs/
│   └── hotel_information.docx
└── chroma_db/                 # Created automatically when the program runs
```

The project includes the hotel information document in `hotel_docs/`. Add additional PDF or DOCX hotel documents to the same folder.

## Requirements

- Python 3.10 or newer recommended
- A Google Gemini API key
- Internet access for Gemini API requests

## Installation

### 1. Open the project folder

Open a terminal in the folder containing `rag.py`.

### 2. Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install packages

```powershell
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

**Option A: PowerShell environment variable**

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

This applies to the current PowerShell session. Run the chatbot from the same terminal.

**Option B: `.env` file**

Create a file named `.env` beside `rag.py`:

```text
GEMINI_API_KEY=YOUR_API_KEY
```

Replace `YOUR_API_KEY` with your actual key. Do not upload `.env` to GitHub or share it publicly.

Get or manage a Gemini API key through [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Check model settings

In `rag.py`, the model settings are:

```python
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
```

The embedding model converts text into vectors. The chat model generates natural-language answers. Use model IDs that are available to your Gemini API project.

### 6. Run the chatbot

```powershell
python rag.py
```

Ask a question when the `You:` prompt appears. Type `exit` or `quit` to stop.

## Example questions

- What room options do you have?
- How much is a room for two people?
- What time is check-in?
- Can I check out later?
- What package would suit an anniversary?
- How do I book a room?
- What attractions are nearby?
- What should happen if a guest reports a broken lock?

Answers depend on the information available in the hotel documents.

## Main configuration

These values can be changed in `rag.py`:

| Setting | Purpose |
|---|---|
| `DOCUMENT_FOLDER` | Folder containing PDF and DOCX files |
| `CHROMA_FOLDER` | Local location for the Chroma database |
| `EMBEDDING_MODEL` | Gemini embedding model |
| `CHAT_MODEL` | Gemini chat model |
| `CHUNK_SIZE` | Maximum chunk size used by the splitter |
| `CHUNK_OVERLAP` | Text overlap between adjacent chunks |
| `TOP_K` | Number of relevant chunks retrieved per question |

## Troubleshooting

### `GEMINI_API_KEY is missing`

- Make sure the variable is named exactly `GEMINI_API_KEY`.
- If using PowerShell, set the variable and run `python rag.py` in that same terminal.
- If using `.env`, make sure it is beside `rag.py` and `python-dotenv` is installed.

### No PDF or DOCX documents found

Place your files inside `hotel_docs/`. Check that the file extension is `.pdf` or `.docx`, not an accidental extension such as `.docx.docx`.

### Model or API error

- Check that the model ID is available to your Gemini API project.
- Check your API key, network connection, API access, and usage limits.
- Keep the embedding model configured separately from the chat model.

### Scanned PDF has no text

`PyPDFLoader` extracts text from PDFs; it does not perform OCR. Scanned documents may need OCR before they can be used.

## Notes and limitations

- The program rebuilds the Chroma database and re-creates document embeddings each time it starts. This is simple for learning but can be slower and use more API calls as the document collection grows.
- The chatbot is instructed not to invent information, but generated answers should still be checked against the source documents for important hotel policies, prices, and bookings.
- The booking URL in the supplied hotel document appears to contain a placeholder. Replace it with the hotel's real booking link before using the chatbot with guests.
- Keep API keys private. Never commit `.env` or any file containing a real API key to version control.

## Technologies

- Python
- LangChain
- Google Gemini embeddings and chat model
- Chroma vector database
- `RecursiveCharacterTextSplitter`
- PyPDF and Docx2txt document loaders
- python-dotenv
