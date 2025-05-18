from utils.safe_groq import SafeChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from typing import Dict, Optional
from PIL import Image

load_dotenv()

class ImageRouter:
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        self.llm = SafeChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0
        )
        self.image_types = {
            "chart": "Data visualization",
            "diagram": "Technical schematic",
            "photograph": "Real-world image",
            "screenshot": "Screen capture",
            "formula": "Mathematical notation",
            "unknown": "Unclassified content"
        }

    def classify_image(self, image: Image, context: Optional[str] = None) -> Dict:
        """Classify image without text extraction"""
        import base64
        from io import BytesIO
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        prompt = f"Classify this image as one of: {list(self.image_types.keys())}. Return only the type name."
        
        try:
            classification = self.llm.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{img_str}"}
                ])
            ]).content.lower().strip()
            
            return {
                "type": classification if classification in self.image_types else "unknown",
                "description": self.image_types.get(classification, "Unclassified"),
                "action": self._determine_action(classification, context)
            }
        except Exception as e:
            print(f"Classification error: {e}")
            return {
                "type": "unknown",
                "description": "Classification failed",
                "action": "defer"
            }

    def _determine_action(self, image_type: str, context: Optional[str]) -> str:
        """Determine image handling strategy"""
        if image_type in ["chart", "diagram"]:
            return "flag_important"
        return "store_reference"