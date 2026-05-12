"""Shared configuration for Lab 24."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Paths ===
ROOT = Path(__file__).parent
RAG_DATA_DIR = ROOT / "rag" / "data"
PHASE_A_DIR = ROOT / "phase-a"
PHASE_B_DIR = ROOT / "phase-b"
PHASE_C_DIR = ROOT / "phase-c"
PHASE_D_DIR = ROOT / "phase-d"

# === API keys ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# === Qdrant (inherit Day18) ===
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = "lab24_eval_guard"

# === Models ===
JUDGE_MODEL = "gpt-4o-mini"
GENERATOR_MODEL = "gpt-4o-mini"
SAFETY_MODEL = "gemini-1.5-flash"
EMBED_MODEL_OPENAI = "text-embedding-3-small"

# === RAG (inherited from Day 18) ===
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3
TEST_SET_PATH = ROOT / "phase-a" / "testset_v1.csv"

# === Phase A ===
TEST_SET_SIZE = 50
TEST_DISTRIBUTION = {"simple": 0.5, "reasoning": 0.25, "multi_context": 0.25}
RAGAS_TARGETS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.75,
}
RAGAS_MIN_OK = {
    "faithfulness": 0.75,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.65,
}

# === Phase B ===
PAIRWISE_SAMPLE_SIZE = 30
HUMAN_LABEL_SAMPLE = 10
ABSOLUTE_SAMPLE_SIZE = 30

# === Phase C ===
PII_LATENCY_P95_MAX_MS = 50
L3_LATENCY_P95_MAX_MS = 100
ADVERSARIAL_DETECTION_MIN = 0.70
PII_DETECTION_MIN = 0.80
OUTPUT_GUARD_DETECTION_MIN = 0.80
OUTPUT_GUARD_FP_MAX = 0.20
BENCHMARK_REQUESTS = 100
TOPIC_SIMILARITY_THRESHOLD = 0.4
ALLOWED_TOPICS = [
    "Nghị định 13 về bảo vệ dữ liệu cá nhân",
    "báo cáo tài chính doanh nghiệp",
    "luật pháp Việt Nam về quyền dữ liệu",
]

# === Budget ===
MAX_COST_USD = float(os.getenv("MAX_COST_USD", "15.0"))
COST_LOG = ROOT / "cost_log.jsonl"
