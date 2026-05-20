# search_engine.py (Optimized version for BGE-M3)
import numpy as np
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
import time
import re
# 1. Logger Integration Setup
try:
    from logger import FAQLogger
    LOGGER_AVAILABLE = True
    print(" Custom FAQLogger successfully activated.")
except ImportError:
    LOGGER_AVAILABLE = False
    print(" Custom logger not found - falling back to standard Python logging.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# 2. Vector Search Engine Class
class FAQSearchEngine:    
    def __init__(self, 
                 model_name: str = "BAAI/bge-m3", 
                 index_path: str = "faiss_index/faq_index.bin", 
                 metadata_path: str = "faiss_index/faq_metadata.json"):
        
        logger.info("Initializing FAQ Search Engine with BGE-M3...")
        # Loading the embedding model on CPU
        self.model = SentenceTransformer(model_name, device="cpu")
        # Loading the FAISS vector database index
        logger.info(f"📦 Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        # Loading matching JSON knowledge base metadata
        logger.info(f" Loading metadata from {metadata_path}...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        logger.info(f" Search engine ready! {len(self.metadata)} FAQ pairs loaded successfully.")
    
    def _normalize_query(self, query: str) -> str:
        """Basic text normalization and query cleaning."""
        query = query.strip()
        query = re.sub(r'[؟\?]+', '؟', query)
        return query
    def search(self, 
               query: str, 
               top_k: int = 3, 
               threshold: float = 0.50, 
               user_ip: Optional[str] = None) -> List[Dict]:
        
        start_time = time.time()
        timings = {}
        
        try:
            # Step 1: Normalization
            norm_start = time.time()
            normalized_query = self._normalize_query(query)
            timings["normalization"] = (time.time() - norm_start) * 1000
            
            # Step 2: Generate Normalized Embeddings (Required for FlatIP Index consistency)
            embed_start = time.time()
            query_embedding = self.model.encode(
                [normalized_query], 
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            ).astype("float32")
            timings["embedding"] = (time.time() - embed_start) * 1000
            
            # Step 3: Vector Similarity Search via FAISS
            search_start = time.time()
            distances, indices = self.index.search(query_embedding, top_k)
            timings["faiss_search"] = (time.time() - search_start) * 1000
            
            # Step 4: Metric Score Filtering
            results = []
            for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
                if idx == -1 or score < threshold:
                    continue
                
                item = self.metadata[idx]
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "question": item["question"],
                    "short_answer": item.get("short_answer", ""),
                    "full_answer": item.get("full_answer", ""),
                    "topic": item.get("topic", ""),
                    "id": item.get("id", str(idx))
                })
            
            timings["total"] = (time.time() - start_time) * 1000
            
            # Step 5: Metric & Performance Logging Middleware
            if LOGGER_AVAILABLE:
                try:
                    FAQLogger.log_query(query=query, normalized_query=normalized_query, results=results, user_ip=user_ip, timings=timings)
                    FAQLogger.log_performance(operation="search", duration_ms=timings["total"], details={"result_count": len(results), "top_score": results[0]["score"] if results else 0, "threshold": threshold})
                except Exception as log_error:
                    logger.warning(f"Error executing custom logging pipeline: {log_error}")
            
            logger.info(f"🔍 Search: '{query[:30]}...' → {len(results)} results in {timings['total']:.0f}ms")
            return results
            
        except Exception as e:
            if LOGGER_AVAILABLE:
                try: FAQLogger.log_error(error_type=type(e).__name__, error_message=str(e), query=query, user_ip=user_ip)
                except: pass
            logger.error(f"❌ Exception raised during query execution pipeline ({type(e).__name__}): {e}")
            raise
    
    def get_stats(self) -> Dict:
        return {
            "total_faqs": len(self.metadata),
            "index_size": self.index.ntotal,
            "model": "BAAI/bge-m3"
        }

# 3. Local Script Execution Test Runtime
if __name__ == "__main__":
    search_engine = FAQSearchEngine()