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

### How to interact with the app
- When asked, choose `y` if you want to load sample PubMed articles.
- Enter a health topic (for example, `fatty liver`) when prompted.
- If you press Enter without typing a topic, the app uses `fatty liver` by default.
- After loading, ask your questions in plain language.
- Type `quit` anytime to exit.

## Contributing
Feel free to open issues or submit pull requests.

## License
MIT License
