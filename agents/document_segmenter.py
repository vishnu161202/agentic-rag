import fitz  # PyMuPDF
from PIL import Image
import io
import os
from typing import Dict, List, Tuple, Optional
import magic

class DocumentSegmenter:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
    
    def process_document(self, file_path: str) -> Dict:
        """Process document and separate text and image sections"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        mime_type = self.mime.from_file(file_path)
        
        if mime_type == 'application/pdf':
            return self._process_pdf(file_path)
        elif mime_type.startswith('image/'):
            return self._process_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")
    
    def _process_pdf(self, file_path: str) -> Dict:
        """Process PDF document"""
        doc = fitz.open(file_path)
        content = {
            "text_sections": [],
            "image_sections": [],
            "mixed_sections": []
        }
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            images = page.get_images()
            
            if text and not images:
                content["text_sections"].append({
                    "page": page_num + 1,
                    "content": text,
                    "type": "text"
                })
            elif images and not text:
                for img in images:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    content["image_sections"].append({
                        "page": page_num + 1,
                        "content": image,
                        "type": "image",
                        "metadata": {
                            "format": base_image["ext"],
                            "size": image.size
                        }
                    })
            else:
                content["mixed_sections"].append({
                    "page": page_num + 1,
                    "text": text,
                    "images": [img[0] for img in images],
                    "type": "mixed"
                })
        
        doc.close()
        return content
    
    def _process_image(self, file_path: str) -> Dict:
        """Process standalone image file"""
        image = Image.open(file_path)
        return {
            "text_sections": [],
            "image_sections": [{
                "page": 1,
                "content": image,
                "type": "image",
                "metadata": {
                    "format": image.format,
                    "size": image.size
                }
            }],
            "mixed_sections": []
        }