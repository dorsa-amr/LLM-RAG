# Agentic RAG System for PubMed

A question-answering system that combines Retrieval-Augmented Generation (RAG) with multi-step reasoning agents to answer complex questions over PubMed articles.

## Features

- **Multi-step Reasoning**: Uses LangChain agents to reason over multiple documents
- **Semantic Retrieval**: Vector database-backed semantic search for relevant papers
- **PubMed Integration**: Direct access to PubMed articles via API
- **Optimized Pipeline**: Tunable retrieval parameters for improved response quality
- **Source Attribution**: Tracks and cites original sources

## Architecture

```
User Query
    ↓
Agent (LangChain)
    ↓
    ├→ Retrieval Tool (Vector DB + Semantic Search)
    ├→ Reasoning Tool (LLM-based reasoning)
    └→ Synthesis Tool (Answer generation)
    ↓
Final Answer with Citations
```

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and settings
   ```

3. **Prepare data**:
   ```bash
   python src/data_pipeline.py
   ```

4. **Run the system**:
   ```bash
   python src/main.py
   ```

## Project Structure

- `src/`: Core system components
  - `config.py`: Configuration management
  - `data_pipeline.py`: PubMed data fetching and embedding
  - `retriever.py`: Vector DB and semantic search
  - `agent.py`: LangChain agent setup
  - `main.py`: Application entry point
- `data/`: Document storage and vector DB
- `notebooks/`: Exploratory notebooks for development
- `tests/`: Unit and integration tests

## Usage

```python
from src.agent import create_qa_agent

agent = create_qa_agent()
response = agent.run("What are the latest treatments for alzheimer's disease?")
print(response)
```

## Performance Tuning

Key parameters in `.env`:
- `RETRIEVAL_TOP_K`: Number of documents to retrieve (default: 5)
- `CHUNK_SIZE`: Document chunk size in tokens (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `RETRIEVAL_SCORE_THRESHOLD`: Minimum similarity score

## Development Roadmap

- [ ] MVP: Basic retrieval + single-step QA
- [ ] Agent framework: Multi-step reasoning
- [ ] Optimization: Parameter tuning & evaluation
- [ ] Citations: Source attribution
- [ ] UI: Simple web interface
- [ ] Evaluation: Benchmark on standard datasets
