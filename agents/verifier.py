from utils.safe_groq import SafeChatGroq
from dotenv import load_dotenv
import os
from typing import List, Dict
import json

load_dotenv()

class Verifier:
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        self.llm = SafeChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )

    def check_hallucinations(self, context: List[Dict], response: str) -> Dict:
        """Verify response accuracy"""
        context_text = "\n".join([item["content"] for item in context if item["type"] == "text"])
        
        prompt = f"""Verify this response against the context:
        Context: {context_text}
        Response: {response}
        Return JSON with: unsupported_claims (list), is_consistent (bool), confidence (0-1)"""
        
        try:
            result = self.llm.invoke(prompt).content
            return json.loads(result)
        except Exception as e:
            print(f"Verification failed: {e}")
            return {
                "unsupported_claims": ["Verification error"],
                "is_consistent": False,
                "confidence": 0.0
            }