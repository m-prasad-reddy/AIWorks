import json
from pathlib import Path
import hashlib
import datetime
from typing import Dict, Any

CACHE_FILE = "schema_cache.json"  # Simplified to just the filename
FEEDBACK_FILE = ".cache/feedback_log.json"
QUERY_CACHE_FILE = ".cache/query_cache.json"

def cache_schema(schema: Dict[str, Any], cache_key: str):
    """Stash the schema in a chill cache."""
    Path(".cache").mkdir(exist_ok=True)
    cache_path = Path(".cache") / f"{cache_key}_{CACHE_FILE}"  # Correct path construction
    with open(cache_path, "w") as f:
        json.dump(schema, f, indent=2)

def load_cached_schema(cache_key: str) -> Dict[str, Any]:
    """Grab the schema from cache if it’s there."""
    cache_path = Path(".cache") / f"{cache_key}_{CACHE_FILE}"
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def cache_query_result(prompt: str, sql: str):
    """Cache the query result for quick repeats."""
    query_cache = load_json(QUERY_CACHE_FILE)
    query_cache[hashlib.sha256(prompt.encode()).hexdigest()] = sql
    save_json(QUERY_CACHE_FILE, query_cache)

def get_cached_or_feedback_query(prompt: str) -> str:
    """Check cache or feedback for a pre-vibed query."""
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    query_cache = load_json(QUERY_CACHE_FILE)
    if prompt_hash in query_cache:
        return query_cache[prompt_hash]
    feedback_log = load_json(FEEDBACK_FILE)
    if prompt_hash in feedback_log:
        return feedback_log[prompt_hash]["corrected"]
    return None

def log_feedback(prompt: str, generated_sql: str, corrected_sql: str):
    """Log feedback when the SQL needs a tweak."""
    feedback_log = load_json(FEEDBACK_FILE)
    feedback_log[hashlib.sha256(prompt.encode()).hexdigest()] = {
        "generated": generated_sql,
        "corrected": corrected_sql,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    save_json(FEEDBACK_FILE, feedback_log)

def load_json(path: str) -> Dict:
    """Load JSON with a chill default."""
    if Path(path).exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: Dict):
    """Save JSON with a smooth touch."""
    Path(".cache").mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)