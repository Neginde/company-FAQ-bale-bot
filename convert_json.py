from docx import Document
import json
import re  

doc = Document("data/Rasta FAQ.docx")

chunks = []
current_topic = None
current_chunk = None

for para in doc.paragraphs:
    text = para.text.strip()
    style = para.style.name

    if not text:
        continue

    if style == "Heading 1":
        current_topic = re.sub(r'^Topic\s*[:：]?\s*', '', text).strip()
        continue

    if style == "Heading 2":
        if current_chunk:
            chunks.append(current_chunk)

        current_chunk = {
            "id": f"Q{len(chunks) + 1:03d}",  
            "topic": current_topic,
            "question": text,
            "short_answer": "",
            "full_answer": ""
        }
        continue

    if text.startswith("پاسخ کوتاه"):
        if current_chunk:
            current_chunk["short_answer"] = text.replace("پاسخ کوتاه:", "").strip()
        continue

    if text.startswith("پاسخ کامل"):
        if current_chunk:
            current_chunk["full_answer"] = text.replace("پاسخ کامل:", "").strip()
        continue

    if current_chunk and current_chunk["full_answer"]:
        current_chunk["full_answer"] += "\n" + text  # بهبود: استفاده از \n بجای space

if current_chunk:
    chunks.append(current_chunk)

# ذخیره با UTF-8
with open("faq.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"{len(chunks)} questions extracted successfully.")
print("First item preview:", json.dumps(chunks[0], ensure_ascii=False, indent=2)[:200] + "...")