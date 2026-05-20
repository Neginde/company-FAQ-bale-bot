import json
import re

with open("faq.json", "r", encoding="utf-8") as f:
    data = json.load(f)

normalized_chunks = []

for chunk in data:
    # ===== Topic =====
    topic = chunk.get("topic", "")
    # پاک کردن برچسب Topic: اگر مانده
    topic = topic.replace("Topic:", "").strip()
    
    # ===== Question =====
    question = chunk.get("question", "")
    # پاک کردن شماره‌های ابتدای سوال (مثلاً "2-1- ")
    # پاک کردن تمام شماره‌های ابتدای سوال (مثلاً "1- ", "2-1- ", "10- ")
    question = re.sub(r"^\d+([-\.\d]*)[-\.\s]+", "", question).strip()

    
    # ===== Short Answer =====
    short = chunk.get("short_answer", None)
    if short:
        short = re.sub(r"^(پاسخ کوتاه\s*[:：]?)", "", short).strip()
    else:
        short = None
    
    # ===== Full Answer =====
    full = chunk.get("full_answer", None)
    if full:
        full = re.sub(r"^(پاسخ کامل\s*[:：]?)", "", full).strip()
    else:
        full = None

    # اگر خالی بود → null
    if short == "":
        short = None
    if full == "":
        full = None

    normalized_chunks.append({
        "id": chunk.get("id"),
        "topic": topic if topic else None,
        "question": question if question else None,
        "short_answer": short,
        "full_answer": full
    })

# ذخیره JSON نرمال‌شده
with open("faq_normalized.json", "w", encoding="utf-8") as f:
    json.dump(normalized_chunks, f, ensure_ascii=False, indent=2)

print(f"{len(normalized_chunks)} chunks normalized and saved to faq_normalized.json")
