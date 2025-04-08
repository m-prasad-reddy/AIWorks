import json
from pathlib import Path
from typing import Dict, List, Any  # Added missing import
from transformers import AutoTokenizer, AutoModelForCausalLM

class DynamicHintsGenerator:
    def __init__(self, model_name: str = "NumbersStation/nsql-6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hint_cache_path = Path(".cache") / "dynamic_table_hints.json"

    def _generate_hints_prompt(self, tables: List[str]) -> str:
        return f"""Given the database tables below, analyze relationships and categorize them by keywords (e.g., 'order', 'customer'). 
        Return JSON format: {{"keyword": ["schema.table1", "schema.table2"]}}

        Tables:
        {', '.join(tables)}

        Example Output:
        {{
            "order": ["sales.orders", "sales.order_items"],
            "customer": ["sales.customers"]
        }}

        Analysis:"""

    def generate_table_hints(self, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        tables = list(schema.keys())
        prompt = self._generate_hints_prompt(tables)
        
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, max_length=1024)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from model response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            print("⚠️ Failed to parse model response. Using fallback hints.")
            return self._get_fallback_hints(schema)

    def _get_fallback_hints(self, schema: Dict[str, Any]) -> Dict[str, List[str]]:
        """Fallback when model fails"""
        return {
            table.split('.')[-1].lower(): [table] for table in schema.keys()
        }

    def save_hints(self, hints: Dict[str, List[str]]):
        Path(".cache").mkdir(exist_ok=True)
        with open(self.hint_cache_path, "w") as f:
            json.dump(hints, f, indent=2)

    def load_hints(self) -> Dict[str, List[str]]:
        try:
            with open(self.hint_cache_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        