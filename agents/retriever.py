from typing import List, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

class Retriever:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
    
    def index_documents(self, documents: List[Dict]):
        """Index text documents for retrieval"""
        texts = []
        metadatas = []
        
        for doc in documents:
            if doc["type"] == "text":
                texts.append(doc["content"])
                metadatas.append({
                    "page": doc["page"],
                    "type": "text",
                    "source": "document"
                })
        
        if texts:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas
            )
    
    def retrieve_relevant_text(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant text passages"""
        if not self.vector_store:
            return []
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [{
            "content": doc.page_content,
            "page": doc.metadata["page"],
            "score": self._calculate_score(doc, query),
            "type": "text"
        } for doc in docs]
    
    def _calculate_score(self, doc, query) -> float:
        """Calculate relevance score between document and query"""
        query_embedding = self.embedding_model.embed_query(query)
        doc_embedding = self.embedding_model.embed_query(doc.page_content)
        return np.dot(query_embedding, doc_embedding)