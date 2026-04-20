# DiplomaAI

DiplomaAI is a prototype RAG system for generating draft official responses to citizen complaints. The project uses a local language model through Ollama, semantic search over a complaint-response dataset, and a simple anonymization step for incoming user text.

## Project Idea

The system follows a Retrieval-Augmented Generation pipeline:

1. Load a dataset of complaints and reference responses.
2. Convert complaint texts into embeddings with `sentence-transformers`.
3. Search for similar complaints with cosine similarity.
4. Build a prompt with the retrieved examples.
5. Generate a draft official response through a local Ollama model.
6. Restore anonymized values in the final response.

The dataset itself is Russian-language because the target domain is official responses to citizen complaints in Russia. The code, comments, prompts, and CLI messages are written in English.

## Project Structure

```text
.
|-- main.py                  # Interactive application entry point
|-- evaluate.py              # Baseline vs RAG evaluation script
|-- requirements.txt         # Python dependencies
|-- data/
|   |-- data.csv             # Complaint-response dataset
|   `-- knowledge_base.pkl   # Cached knowledge base archive
|-- results/
|   |-- evaluation_results.csv
|   `-- evaluation_summary.json
`-- src/
    |-- anonymizer.py        # Rule-based anonymization
    |-- config.py            # Project configuration
    |-- data_loader.py       # Dataset and knowledge base IO
    |-- generator.py         # Ollama prompt creation and generation
    |-- pipline.py           # End-to-end response pipeline
    |-- search_engine.py     # Similarity search
    `-- vectorizer.py        # SentenceTransformer embeddings
```

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install and run Ollama, then pull the configured model:

```bash
ollama pull llama3.2
ollama serve
```

The model name and API endpoint are configured in `src/config.py`.

## Run the Application

```bash
python3 main.py
```

On the first run, the project builds a knowledge base from `data/data.csv`. If the cache is missing or incompatible, it is rebuilt automatically.

## Run the Evaluation

The evaluation compares two modes on the same test set:

- `baseline`: the model receives only the complaint and general instructions.
- `rag`: the model receives the complaint, instructions, and similar examples retrieved from the knowledge base.

Run the full experiment:

```bash
python3 evaluate.py
```

Run a quick experiment on a smaller test subset:

```bash
python3 evaluate.py --limit 10
```

Outputs:

- `results/evaluation_results.csv`: per-example baseline and RAG responses, retrieved examples, ROUGE-L scores, and manual scoring columns.
- `results/evaluation_summary.json`: average metrics for the experiment.

## Evaluation Metrics

The script calculates `ROUGE-L F1` for each generated response against the reference response from the dataset. This metric estimates textual similarity, but it does not fully measure legal correctness or factual reliability.

For a diploma-quality evaluation, combine the automatic metric with manual scoring by criteria such as:

- relevance to the complaint;
- completeness of the answer;
- official business style;
- absence of invented facts;
- usefulness for a prosecutor-office employee.

## Notes and Limitations

This project is a prototype, not a production legal system. The anonymizer is rule-based and limited. Generated responses must be reviewed by a human specialist before any real use.
