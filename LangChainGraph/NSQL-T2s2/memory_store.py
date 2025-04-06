from collections import defaultdict

query_cache = defaultdict(str)

def get_cached_sql(query: str) -> str:
    return query_cache.get(query.lower(), "")

def cache_sql(query: str, sql: str):
    query_cache[query.lower()] = sql
