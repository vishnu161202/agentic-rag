from utils.safe_groq import SafeChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing import List, Dict

load_dotenv()

class Generator:
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        self.llm = SafeChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.7
        )
        self.output_parser = StrOutputParser()

    def generate_response(self, query: str, context: List[Dict], image_metadata: List[Dict] = None) -> str:
        """Generate context-aware response"""
        context_text = "\n".join([item["content"] for item in context if item["type"] == "text"])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer using only the provided context. Acknowledge missing information."""),
            ("human", "Question: {question}\nContext: {context}")
        ])
        
        try:
            chain = prompt | self.llm | self.output_parser
            return chain.invoke({"question": query, "context": context_text})
        except Exception as e:
            return f"Response generation failed: {str(e)}"