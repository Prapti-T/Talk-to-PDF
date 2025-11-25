import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, BertForMaskedLM, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer

load_dotenv()

CONFIG = {
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "PINECONE_ENV": os.getenv("PINECONE_ENV"),
    "PINECONE_INDEX": os.getenv("PINECONE_INDEX", "rag-index"),

    "BERT_TOKENIZER": os.getenv("BERT_TOKENIZER", "bert-base-uncased"),
    "CHUNK_TOKENS": int(os.getenv("CHUNK_TOKENS", 512)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 64)),
    "CHUNK_MIN_TOKENS": int(os.getenv("CHUNK_MIN_TOKENS", 388)), 
    "EMBED_MODEL": os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),

    "REDIS_HOST": os.getenv("REDIS_HOST"),
    "REDIS_PORT": int(os.getenv("REDIS_PORT", 6379)),
    "REDIS_PASSWORD": os.getenv("REDIS_PASSWORD"),

    "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://localhost:11434"),

}

# -----------------------
# Shared tokenizer / models
# -----------------------
TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["BERT_TOKENIZER"], use_fast=True)
EMBEDDER = SentenceTransformer(CONFIG["EMBED_MODEL"])
MLM_MODEL = BertForMaskedLM.from_pretrained(CONFIG["BERT_TOKENIZER"])
REDIS_HOST = CONFIG["REDIS_HOST"]
REDIS_PORT = CONFIG["REDIS_PORT"]
REDIS_PASSWORD = CONFIG["REDIS_PASSWORD"]