DEBUG = True

# Logging
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARN
LOG_STRUCTURED = False
LOG_TRACE_RETRIEVAL = True

# Retrieval tuning
DENSE_ALPHA = 0.7
RETRIEVE_K = 20
RERANK_K = 5
TOPK_TRACE = 5

# Context assembly
CONTEXT_MAX_TOKENS = 1500
CONTEXT_MAX_DOCS = 5
DEDUPLICATE_BY_FILE = True
COMPRESS_LOW_RELEVANCE = False
COMPRESS_THRESHOLD_RATIO = 0.35

# Query rewrite / intent
ENABLE_QUERY_REWRITE = True
ENABLE_INTENT_DETECTION = True

# Index / data
DATA_DIR = "data/goodwiki_markdown_sample"
STATE_PATH = "data/.rag_index_state.json"

# History
SAVE_QUERY_HISTORY = True
QUERY_HISTORY_PATH = "data/query_history.jsonl"
