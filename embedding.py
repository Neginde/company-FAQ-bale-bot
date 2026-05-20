import json
from sentence_transformers import SentenceTransformer
import torch
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تنطیم پوشه ذخیره‌سازی هماهنگ با اسکریپت FAISS
DATA_DIR = "faq_system"
os.makedirs(DATA_DIR, exist_ok=True)

# ===== Load FAQ JSON =====
with open("data/faq_normalized.json", "r", encoding="utf-8") as f:
    faq_items = json.load(f)

logger.info(f"✅ Loaded {len(faq_items)} FAQ items")

# ===== Load embedding model =====
device = "cuda" if torch.cuda.is_available() else "cpu"

# 💡 تغییر به مدل سبک و بهینه برای زبان فارسی و CPU
model_name = "BAAI/bge-m3"
logger.info(f"🚀 Loading embedding model: {model_name} on {device}")

model = SentenceTransformer(model_name, device=device)

# warmup
try:
    with torch.inference_mode():
        _ = model.encode(["سلام"], convert_to_numpy=True, normalize_embeddings=True)
    logger.info("✅ Embedding model warmup done.")
except Exception as e:
    logger.warning(f"Warmup failed: {e}")

# ===== Build embeddings (QUESTION ONLY) =====
question_embeddings = []
all_metadata = [] 
batch_size = 8  # روی CPU هم می‌توان مقدار بچ را کمی بیشتر کرد

logger.info(f"🔄 Generating QUESTION-ONLY embeddings...")

for i in range(0, len(faq_items), batch_size):
    batch_chunks = faq_items[i:i + batch_size]
    texts = []

    for c in batch_chunks:
        text = c['question'].strip()
        texts.append(text)
    
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False  
        )

    for idx, emb in enumerate(embeddings):
        question_embeddings.append(emb.astype("float32"))
        chunk_data = batch_chunks[idx]
        
        metadata_item = {
            "id": chunk_data.get("id", f"Q{idx + i:03d}"),
            "question": chunk_data["question"].strip(),
            "short_answer": chunk_data.get("short_answer", ""),
            "full_answer": chunk_data.get("full_answer", ""),
            "topic": chunk_data.get("topic", ""),
            "tags": chunk_data.get("tags", []),
            "embedding_text": chunk_data["question"].strip()
        }
        all_metadata.append(metadata_item)

logger.info(f"✅ Generated embeddings for {len(question_embeddings)} FAQs")

# ===== ذخیره در مسیر هماهنگ با اسکریپت FAISS =====
embeddings_array = np.array(question_embeddings)
np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings_array)
logger.info(f"💾 Saved embeddings: shape={embeddings_array.shape}")

with open(os.path.join(DATA_DIR, "faqs.json"), "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)
logger.info(f"📄 Saved metadata to {DATA_DIR}/faqs.json")