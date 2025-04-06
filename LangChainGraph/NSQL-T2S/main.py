from database.connection import get_sqlserver_connection
from database.schema import extract_full_schema, identify_relevant_tables, format_schema
from nsql.model import NSQLLLM
from prompts.templates import get_sql_prompt_template
from utils.feedback import log_feedback
from utils.state import AgentState
from langchain_core.runnables import RunnableSequence

def execute_query(conn, sql_query):
    """Run the SQL and keep it read-only."""
    try:
        if not sql_query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries allowed, fam."
        cursor = conn.cursor()
        cursor.execute(sql_query)
        return cursor.fetchall()
    except Exception as e:
        return f"Query vibes off: {str(e)}"

def run_agent():
    """Launch the Text-to-SQL agent with a tight flow."""
    # Connect and load schema
    conn = get_sqlserver_connection()
    full_schema = extract_full_schema(conn)
    state = AgentState()

    # Set up the modern chain vibes
    prompt_template = get_sql_prompt_template()
    nsql_llm = NSQLLLM()
    # Use RunnableSequence instead of LLMChain
    sql_chain = RunnableSequence(prompt_template | nsql_llm)

    print("Text-to-SQL Agent is live! Drop your queries, SELECT-style only.")
    while True:
        user_query = input("Your query (or 'quit' to bounce): ").strip()
        if user_query.lower() == "quit":
            break

        # Check cache first
        cached_sql = state.get_response(user_query)
        if cached_sql:
            sql_query = cached_sql
            print(f"\nCached SQL (from memory, yo):\n{sql_query}")
        else:
            # Identify tables and generate SQL
            relevant_tables = identify_relevant_tables(full_schema, user_query)
            schema_str = format_schema(full_schema, relevant_tables)
            print(f"Chain ready: {sql_chain}")
            print(f"Input: schema={schema_str[:50]}..., query={user_query}")
            # Invoke the sequence
            result = sql_chain.invoke({"schema": schema_str, "query": user_query})
            print(f"Invoke output: {result}")
            # Handle output (should be a string from NSQLLLM)
            sql_query = result if isinstance(result, str) else "Error: No SQL generated"
            state.save_response(user_query, sql_query)
            print(f"\nGenerated SQL (tables: {', '.join(relevant_tables)}):\n{sql_query}")

        # Execute and show results
        results = execute_query(conn, sql_query)
        print(f"Results:\n{results}")

        # Feedback time
        feedback = input("Did it slap? (yes/no): ").strip().lower()
        is_correct = feedback == "yes"
        log_feedback(sql_query, is_correct)

    conn.close()
    print("Agent’s out. Feedback’s in 'feedback_log.txt'.")

if __name__ == "__main__":
    try:
        run_agent()
    except Exception as e:
        print(f"Big oof: {e}")