# Load trained model
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

model = T5ForConditionalGeneration.from_pretrained("sql_model")
tokenizer = T5Tokenizer.from_pretrained("sql_model")

def generate_sql(query_text):
    input_text = f"translate English to SQL: {query_text}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Example
user_query = "Show me all sales orders created after 2007."
generated_sql = generate_sql(user_query)
print(generated_sql)
