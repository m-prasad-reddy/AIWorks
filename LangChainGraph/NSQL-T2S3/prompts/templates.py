from langchain_core.prompts import PromptTemplate

def get_sql_prompt_template():
    """Craft a lean prompt for T-SQL vibes with [SCHEMA].[TABLE] and SELECT-only."""
    return PromptTemplate(
        input_variables=["schema", "query"],
        template="""{schema}

-- Using valid Transact-SQL (T-SQL), answer this (SELECT only):
-- Use [SCHEMA].[TABLE] format for table names (e.g., [sales].[orders]).
-- Example: SELECT [sales].[customers].city FROM [sales].[customers]
-- {query}
SELECT"""
    )