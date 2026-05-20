# build_faiss_index.py
import numpy as np
import faiss
import json
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_faiss_index():
    """ساخت FAISS index بهینه"""
    
    logger.info("=" * 60)
    logger.info("🔧 BUILDING FAISS INDEX")
    logger.info("=" * 60)
    
    # تنظیمات مسیرها
    DATA_DIR = "faq_system"
    INDEX_DIR = "faiss_index"
    
    # ایجاد پوشه خروجی
    os.makedirs(INDEX_DIR, exist_ok=True)
    
    # ==== 1. بارگذاری embeddings ====
    logger.info("📥 Loading embeddings...")
    embeddings_path = os.path.join(DATA_DIR, "embeddings.npy")
    if not os.path.exists(embeddings_path):
        logger.error(f"❌ Embeddings file not found: {embeddings_path}")
        return None
    
    embeddings = np.load(embeddings_path)
    logger.info(f"✅ Loaded embeddings: shape={embeddings.shape}")
    logger.info(f"   - Dimension: {embeddings.shape[1]}")
    logger.info(f"   - Num vectors: {embeddings.shape[0]}")
    
    # ==== 2. بارگذاری metadata ====
    logger.info("\n📋 Loading metadata...")
    metadata_path = os.path.join(DATA_DIR, "faqs.json")
    if not os.path.exists(metadata_path):
        logger.error(f"❌ Metadata file not found: {metadata_path}")
        return None
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    logger.info(f"✅ Loaded {len(metadata)} FAQ items")
    
    # ==== 3. ساخت index ====
    logger.info("\n🔨 Creating FAISS index...")
    dimension = embeddings.shape[1]
    
    # انتخاب نوع index بر اساس تعداد داده‌ها
    if len(embeddings) < 1000:
        # برای داده‌های کوچک، index ساده
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        logger.info("📊 Using IndexFlatIP (exact search)")
    else:
        # برای داده‌های بزرگتر، index سریع‌تر
        nlist = min(100, len(embeddings) // 10)  # تعداد clusterها
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 10  # تعداد clusterهای جستجو شده
        logger.info(f"📊 Using IndexIVFFlat (approximate search, nlist={nlist})")
    
    # آموزش index (برای IVF)
    if hasattr(index, 'is_trained') and not index.is_trained:
        logger.info("🎓 Training index...")
        index.train(embeddings)
    
    # اضافه کردن داده‌ها
    logger.info("➕ Adding vectors to index...")
    index.add(embeddings)
    
    # ==== 4. ذخیره index ====
    index_path = os.path.join(INDEX_DIR, "faq_index.bin")
    faiss.write_index(index, index_path)
    logger.info(f"💾 Index saved: {index_path}")
    logger.info(f"   - Size: {os.path.getsize(index_path) / 1024:.1f} KB")
    
    # ==== 5. ذخیره metadata با index ====
    logger.info("\n📝 Saving enhanced metadata...")
    for i, item in enumerate(metadata):
        item["faiss_index"] = i
        item["embedding_norm"] = float(np.linalg.norm(embeddings[i]))
    
    metadata_path = os.path.join(INDEX_DIR, "faq_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 Metadata saved: {metadata_path}")
    
    # ==== 6. اطلاعات سیستم ====
    system_info = {
        "created_at": datetime.now().isoformat(),
        "index_type": type(index).__name__,
        "total_vectors": index.ntotal,
        "dimension": dimension,
        "metric": "METRIC_INNER_PRODUCT (cosine similarity)",
        "embedding_source": DATA_DIR,
        "faq_count": len(metadata),
        "index_size_kb": os.path.getsize(index_path) / 1024
    }
    
    info_path = os.path.join(INDEX_DIR, "index_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(system_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 System info saved: {info_path}")
    
    # ==== 7. تست اعتبارسنجی ====
    logger.info("\n🧪 Running validation tests...")
    
    # تست 1: جستجوی هر بردار با خودش
    test_indices = [0, len(embeddings)//2, -1]  # اولی، وسطی، آخری
    for idx in test_indices:
        test_query = embeddings[idx].reshape(1, -1)
        distances, indices = index.search(test_query, k=1)
        
        if indices[0][0] == idx and distances[0][0] > 0.99:
            logger.info(f"   ✅ Self-match test {idx}: score={distances[0][0]:.4f}")
        else:
            logger.warning(f"   ⚠️ Self-match test {idx} failed: index={indices[0][0]}, score={distances[0][0]:.4f}")
    
    # تست 2: جستجوی چندین نتیجه
    test_query = embeddings[0].reshape(1, -1)
    k = min(5, len(embeddings))
    distances, indices = index.search(test_query, k=k)
    
    logger.info(f"\n📊 Top-{k} search test for vector 0:")
    for i in range(k):
        logger.info(f"   #{i+1}: index={indices[0][i]}, score={distances[0][i]:.4f}")
    
    # ==== 8. خلاصه ====
    logger.info("\n" + "=" * 60)
    logger.info("🎉 FAISS INDEX BUILT SUCCESSFULLY!")
    logger.info("=" * 60)
    
    logger.info(f"\n📁 Output directory: {INDEX_DIR}/")
    logger.info(f"📊 Summary:")
    logger.info(f"   - Index type: {system_info['index_type']}")
    logger.info(f"   - Total vectors: {system_info['total_vectors']}")
    logger.info(f"   - Dimension: {system_info['dimension']}")
    logger.info(f"   - FAQ count: {system_info['faq_count']}")
    logger.info(f"   - Index size: {system_info['index_size_kb']:.1f} KB")
    
    logger.info(f"\n📋 Files created:")
    for file in os.listdir(INDEX_DIR):
        file_path = os.path.join(INDEX_DIR, file)
        size_kb = os.path.getsize(file_path) / 1024
        logger.info(f"   - {file} ({size_kb:.1f} KB)")
    
    logger.info(f"\n💡 Usage example:")
    logger.info(f"   from faiss_search import FAQSearchSystem")
    logger.info(f"   system = FAQSearchSystem('{INDEX_DIR}')")
    logger.info(f"   results = system.search('سوال شما...')")
    
    return index

if __name__ == "__main__":
    build_faiss_index()