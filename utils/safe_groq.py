from langchain_groq import ChatGroq as BaseChatGroq
import httpx

class SafeChatGroq(BaseChatGroq):
    """Streamlit-compatible Groq client with proper HTTP handling"""
    
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.7, timeout: int = 60):
        self.http_client = httpx.Client(timeout=timeout)
        self.http_async_client = httpx.AsyncClient(timeout=timeout)
        
        super().__init__(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            http_client=self.http_client,
            http_async_client=self.http_async_client
        )