# Therapeutic Companion Bot

A Python research project exploring multiple LLM/NLP approaches to mental health conversational support. There is no single unified application — each subdirectory under `scripts/` is an independent implementation.

## Project Structure

```
scripts/
├── GPT/            # GPT-2 fine-tuned chatbot + Streamlit UI
├── Llama/          # Llama-3 chatbot, intent taxonomy, evaluations
├── NLP/            # BERT notebook and NLP classification algorithm
├── preprocessing/  # Data ingestion and cleaning pipeline
└── misc_models/    # Additional models (HuggingFace, TensorFlow/Gradio)
```

## Running the Applications

Each script is standalone. Choose based on the model you want to run:

```bash
# Llama-3 8B chatbot (Gradio interface)
python scripts/misc_models/streamlit.py

# Advanced Streamlit UI (Llama-3 8B)
python scripts/misc_models/streamlitpro.py

# GPT-2 fine-tuned chatbot (Streamlit)
python scripts/GPT/app-streamlit.py

# TensorFlow intent-based chatbot (Gradio)
python scripts/misc_models/tf_model/app.py

# Jupyter notebooks (run with Jupyter)
scripts/GPT/gpt.ipynb
scripts/NLP/bert-model.ipynb
```

## Data Preprocessing

Raw data sources: Reddit mental health threads, WHO guidelines, HuggingFace Q&A datasets (CSV, Parquet, JSON, DOCX formats).

```bash
# Main preprocessing orchestration
python scripts/preprocessing/preprocess.py

# Generate Doc2Vec embeddings
python scripts/preprocessing/doc2vec.py
```

Helper utilities are in `scripts/preprocessing/utils.py` (CSV conversion, column operations, text cleaning).

## Key Files

| File | Purpose |
|------|---------|
| `scripts/Llama/intents.json` | 40+ intent patterns and responses for rule-based classification |
| `scripts/Llama/evaluations.py` | Model evaluation utilities (perplexity, BLEU, etc.) |
| `scripts/NLP/algorithm.py` | Doc2Vec-based classifier (knowledge vs. standard questions) |
| `scripts/misc_models/hffelladrin.py` | HuggingFace model integration |
| `scripts/preprocessing/utils.py` | Shared data manipulation helpers |

## Dependencies

There is no `requirements.txt`. Infer dependencies from imports in the target script. Common packages across the project:

- `transformers`, `torch` — HuggingFace models, Llama-3, BERT
- `tensorflow`, `tflearn` — TensorFlow chatbot
- `streamlit`, `gradio` — UI frameworks
- `pandas`, `numpy` — Data handling
- `gensim` — Doc2Vec embeddings
- `nltk`, `spacy` — Text preprocessing

## Testing

No formal test framework. Use `scripts/Llama/evaluations.py` for model-level evaluation. Notebooks serve as interactive experimentation environments.

## Notes

- Some scripts contain hardcoded paths (e.g., `/home/ubuntu/bot_test/`) that may need updating for your environment.
- `intents.json` drives the rule-based fallback layer used by the TensorFlow and Llama implementations.
- The `.gitignore` excludes data directories, model checkpoints, and Python cache — do not commit large binary files.
