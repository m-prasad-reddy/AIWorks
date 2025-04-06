from langchain.prompts import PromptTemplate

def get_sql_prompt_template():
    """Craft a lean prompt for SQL generation."""
    return PromptTemplate(
        input_variables=["schema", "query"],
        template="{schema}\n\n-- Using valid SQL Server, answer this (SELECT only):\n-- {query}\n\nSELECT"
    )