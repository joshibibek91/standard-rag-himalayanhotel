# Difference Between the Two Himalayan Hotel RAG Projects

This document compares the two versions built for the Himalayan Grand Hotel:

1. **Basic Python RAG**: uses plain Python functions, Gemini embeddings, NumPy cosine similarity, and a terminal loop.
2. **LangChain + Chroma RAG**: uses LangChain document loaders and chains, Gemini embeddings and chat, a text splitter, and Chroma as a vector database.

## 1. Quick comparison

| Area | Basic Python RAG | LangChain + Chroma RAG |
|---|---|---|
| Main approach | RAG steps are implemented manually in Python | Standard RAG components are connected through LangChain |
| Document loading | `pypdf` and `python-docx` | LangChain loaders: `PyPDFLoader` and `Docx2txtLoader` |
| Chunking | Manual character slicing | `RecursiveCharacterTextSplitter` |
| Embeddings | Gemini embedding API called directly | `GoogleGenerativeAIEmbeddings` integration |
| Vector storage | Embeddings remain in Python lists/dictionaries while the program runs | Chroma stores vectors in a local database directory |
| Similarity search | NumPy cosine similarity is calculated manually | Chroma retriever performs similarity search |
| Answer generation | Gemini SDK called directly with a prompt | `ChatGoogleGenerativeAI` is connected to a LangChain prompt and output parser |
| Prompt handling | Plain string prompt | `ChatPromptTemplate` with a system message and user question |
| Persistence | Document embeddings are recreated on every run and held in memory | Chroma database is written to disk, but the current sample rebuilds it on every run |
| Dependencies | Fewer libraries and fewer abstractions | More packages, integrations, and concepts |
| Best learning focus | Understand each RAG step from the ground up | Learn a common component-based RAG architecture |

## 2. Components used

### Project A: Basic Python RAG

- **`pypdf`**: reads text from PDF files.
- **`python-docx`**: reads text from Word documents.
- **Gemini embedding API**: converts document chunks and questions into vectors.
- **NumPy**: calculates cosine similarity between vectors.
- **Gemini generation API**: writes an answer from the retrieved context.
- **Python functions and terminal input**: connect the steps and let the user ask questions.

**Retrieval flow:**

```text
PDF / DOCX
    ↓
Extract text
    ↓
Manual chunking
    ↓
Gemini embeddings
    ↓
NumPy cosine similarity
    ↓
Top matching chunks
    ↓
Gemini answer
```

### Project B: LangChain + Chroma RAG

- **LangChain document loaders**: load PDF and DOCX files.
- **`RecursiveCharacterTextSplitter`**: creates overlapping text chunks, preferring natural separators such as paragraph breaks and spaces.
- **`GoogleGenerativeAIEmbeddings`**: creates vectors using Gemini.
- **Chroma**: stores and searches vectors.
- **`ChatGoogleGenerativeAI`**: calls Gemini for answer generation.
- **`ChatPromptTemplate`**: organizes the system prompt, context, and question.
- **LangChain runnables and output parser**: connect retrieval, prompt construction, model invocation, and text output.

**Retrieval flow:**

```text
PDF / DOCX
    ↓
LangChain document loaders
    ↓
RecursiveCharacterTextSplitter
    ↓
Gemini embeddings
    ↓
Chroma vector database
    ↓
Retriever (top 3 chunks)
    ↓
LangChain prompt + Gemini
    ↓
Answer
```

## 3. Uses and feasibility

| Need | More suitable version | Why |
|---|---|---|
| Learn the basic RAG concept | Basic Python RAG | Each step is visible and implemented directly |
| Teach cosine similarity and vector matching | Basic Python RAG | The similarity calculation is explicit |
| Build a small demo from a few documents | Either | Both can answer questions from a small document collection |
| Use standard RAG building blocks | LangChain + Chroma | Uses dedicated loaders, splitter, embedding integration, retriever, and vector store |
| Keep vectors in a local database | LangChain + Chroma | Chroma provides local vector storage |
| Expand to more documents or more advanced retrieval | LangChain + Chroma is a more convenient starting point | Its components can be replaced or extended without hand-writing every integration |
| Minimize dependencies and setup | Basic Python RAG | Fewer packages and abstractions |
| Integrate other LangChain-compatible tools later | LangChain + Chroma | Uses LangChain's component interfaces |

## 4. Which is better?

Neither is universally better. They optimize for different goals.

### Choose the basic Python version when:

- You are learning what happens inside a RAG pipeline.
- You want to understand embeddings, vectors, and cosine similarity directly.
- You prefer a small codebase with fewer dependencies.
- You are making a simple classroom demonstration.

### Choose the LangChain + Chroma version when:

- You want to learn the standard RAG component pattern used in many tutorials and prototypes.
- You want a dedicated vector database instead of implementing vector search yourself.
- You expect to experiment with other splitters, retrievers, prompts, or model integrations.
- You want a foundation that can be extended into a larger application.

**Practical recommendation:** Use the basic version first to understand the core ideas, then use LangChain + Chroma to learn how those ideas fit into a reusable RAG architecture.

## 5. Important limitation in the current LangChain + Chroma sample

Although Chroma persists data to a local folder, the current sample deletes and rebuilds that folder every time it starts. Therefore, it does **not** yet gain the faster startup benefit of reusing previously created embeddings. To reuse the database, the code would need to load the existing Chroma collection when the hotel documents have not changed and rebuild it only when necessary.

Both versions call Gemini APIs for embeddings and answer generation. Both require an API key, network access, and valid model IDs. Both can return incorrect answers, so hotel prices and policies should be checked against the source documents.

## 6. Summary

- **Basic Python RAG:** better for understanding the mechanics.
- **LangChain + Chroma RAG:** better for learning a modular RAG stack and extending the project.
- **Answer quality:** neither automatically guarantees better answers. Quality depends on document extraction, chunking, embedding and retrieval relevance, the prompt, and the model.
- **Speed and cost:** neither is automatically cheaper or faster. The current LangChain sample rebuilds its vector database on each run, so both versions repeat embedding work at startup.
