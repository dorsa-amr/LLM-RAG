# LLM-RAG

This project implements a Retrieval-Augmented Generation (RAG) system over PubMed articles. It uses LangChain, ChromaDB, and OpenAI APIs to provide accurate and reliable answers to user queries.

## Features
- Full-text ingestion with fallback to abstracts.
- Deterministic citation enforcement.
- Robust error handling and retry logic.

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Configure your `.env` file with the required API keys.

## Usage
Run the main script:
```bash
python src/main.py
```

## Contributing
Feel free to open issues or submit pull requests.

## License
MIT License
