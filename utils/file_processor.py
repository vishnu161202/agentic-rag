import os
import tempfile
from typing import Dict
from agents.document_segmenter import DocumentSegmenter
from agents.image_router import ImageRouter

class FileProcessor:
    def __init__(self):
        self.segmenter = DocumentSegmenter()
        self.image_router = ImageRouter()
    
    def process_uploaded_file(self, uploaded_file) -> Dict:
        """Process an uploaded file through the full pipeline"""
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Segment document
            segmented = self.segmenter.process_document(tmp_path)
            
            # Process images
            image_metadata = []
            for img_section in segmented["image_sections"]:
                classification = self.image_router.classify_image(img_section["content"])
                image_metadata.append({
                    "page": img_section["page"],
                    "type": classification["type"],
                    "description": classification["description"],
                    "action": classification["action"]
                })
            
            # Process mixed sections
            for mixed_section in segmented["mixed_sections"]:
                classification = self.image_router.classify_image(
                    None,  # Can't process image here without extraction
                    context=mixed_section["text"]
                )
                image_metadata.append({
                    "page": mixed_section["page"],
                    "type": classification["type"],
                    "description": f"Mixed content: {classification['description']}",
                    "action": classification["action"]
                })
            
            return {
                "text_sections": segmented["text_sections"],
                "image_metadata": image_metadata,
                "file_path": tmp_path
            }
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass