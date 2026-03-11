# InsightPDF – RAG-based PDF Question Answering System

InsightPDF is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about a PDF document using LLMs.

## Features

- PDF document ingestion
- Text chunking using RecursiveCharacterTextSplitter
- Semantic embeddings using Cohere
- Vector storage with ChromaDB
- Retrieval-based question answering

## Tech Stack

- Python
- LangChain
- Cohere Embeddings
- Chroma Vector Database

## Project Pipeline

PDF → Text Splitting → Embeddings → Chroma Vector DB → Retriever → LLM → Answer

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your API key in `.env`:

```
COHERE_API_KEY=your_api_key
```

Run ingestion:

```bash
python ingest.py
```

Ask questions:

```bash
python rag.py
```

## Example Question

```
When was Google founded?
```