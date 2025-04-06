from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B")
model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-6B")

def run_nsql_model(prompt: str, max_tokens: int = 300) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids, max_length=max_tokens)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
