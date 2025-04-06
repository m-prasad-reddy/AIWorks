from langchain_core.runnables import RunnableSequence
from app.schema_filter import get_all_table_info, select_relevant_tables, build_schema_prompt_fragment
from app.model_runner import run_nsql_model
from app.memory_store import get_cached_sql, cache_sql

def build_chain(conn_str: str):

    def _get_schema_fragment(input: dict) -> str:
        all_schema = get_all_table_info(conn_str)
        relevant = select_relevant_tables(input["query"], all_schema)
        return build_schema_prompt_fragment(relevant)

    def _build_prompt(input: dict, schema_fragment: str) -> str:
        return f"""{schema_fragment}
-- Using valid SQL Server syntax, answer the following question for the tables provided above.
-- {input['query']}
SELECT"""

    def _is_read_only(sql: str) -> bool:
        lowered = sql.lower()
        return lowered.strip().startswith("select") and all(x not in lowered for x in ["update", "delete", "insert", "alter", "drop"])

    def _nsql_infer(input: dict) -> str:
        cached = get_cached_sql(input["query"])
        if cached:
            return cached
        prompt = input["prompt"]
        raw_output = run_nsql_model(prompt)
        sql = raw_output.split("SELECT", 1)[-1]
        full_sql = "SELECT" + sql
        if not _is_read_only(full_sql):
            raise ValueError("Non-read-only query detected!")
        cache_sql(input["query"], full_sql)
        return full_sql

    return RunnableSequence(
        lambda input: {
            "query": input["query"],
            "schema": _get_schema_fragment(input)
        }
    ).map(lambda x: {
        "query": x["query"],
        "prompt": _build_prompt(x, x["schema"])
    }).map(_nsql_infer)
