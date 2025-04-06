from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from pydantic import BaseModel

class NSQLLLM(LLM, BaseModel):
    """NSQL-6B with a SELECT-only vibe, now Pydantic-friendly."""
    # Pydantic config to allow arbitrary types (like tokenizer/model)
    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        """Set up the NSQL-6B essentials with a chill flow."""
        super().__init__()
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        """Lazy-load the tokenizer with smooth vibes."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("NumbersStation/nsql-6B")
        return self._tokenizer

    @property
    def model(self):
        """Lazy-load the model, keeping it fresh."""
        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained("NumbersStation/nsql-6B")
        return self._model

    def _call(self, prompt, stop=None):
        """Generate SQL, keeping it read-only."""
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            generated_ids = self.model.generate(input_ids, max_length=500)
            sql = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # Ensure it’s SELECT only
            if not sql.strip().upper().startswith("SELECT"):
                return "Error: Only SELECT queries allowed, fam."
            return sql
        except Exception as e:
            return f"Model vibes crashed: {str(e)}"

    def _identifying_params(self):
        return {"model": "NSQL-6B"}

    @property
    def _llm_type(self):
        """Define the LLM type for LangChain."""
        return "nsql-6b"