import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from transformers import pipeline
import re
import os
import chromadb

# ====== CONFIGURATION ======
BASE_URL = "https://devgan.in"
LAW_TYPE = "bns"
CHROMA_DB_PATH = "./chroma_db"
OUTPUT_DIR = "./Legal_AEye-Opener"
JSON_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "bns_laws_data.json")
FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "bns_faiss.index")
COLLECTION_NAME = "bns_laws"

# ====== MODEL SETUP ======
print("🔄 Loading models...")
model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# ====== CHROMADB SETUP ======
print("🔗 Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ====== FAISS SETUP (optional backup) ======
embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(embedding_dim)

# ====== OUTPUT DIR ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== HELPER FUNCTIONS ======
def extract_sections(content):
    sections = []
    content = re.sub(r"IPC\s*Section\s*\d+\s*", "", content, flags=re.IGNORECASE)
    section_pattern = re.split(r"(?=Section\s+\d+\s*–)", content)
    for sec in section_pattern:
        match = re.match(r"Section\s+(\d+)\s*–\s*(.*)", sec.strip())
        if match:
            section_number = match.group(1)
            section_title = match.group(2)
            cleaned_content = re.sub(
                rf"^Section\s+{section_number}\s*–\s*{re.escape(section_title)}\s*",
                "",
                sec.strip(),
                count=1,
            )
            sections.append({
                "section_number": section_number,
                "section_title": section_title,
                "section_content": cleaned_content.strip()
            })
    return sections

def summarize_content(content):
    return summarizer(content[:1024], max_length=150, min_length=50, do_sample=False)[0]['summary_text']

# ====== START SCRAPING ======
print("🌐 Fetching BNS chapters...")
response = requests.get(f"{BASE_URL}/{LAW_TYPE}/")
soup = BeautifulSoup(response.text, "html.parser")

chapters = []
for row in soup.select("table.menu tr"):
    columns = row.find_all("td")
    if len(columns) == 2:
        chapter_number = columns[0].text.strip()
        chapter_title = columns[1].text.strip()
        chapter_link = BASE_URL + columns[1].find("a")["href"]
        chapters.append((chapter_number, chapter_title, chapter_link))

all_laws_data = []

for chapter_number, chapter_title, chapter_link in chapters:
    print(f"\n📥 Fetching Chapter {chapter_number}: {chapter_title}")
    chapter_response = requests.get(chapter_link)
    chapter_soup = BeautifulSoup(chapter_response.text, "html.parser")
    content_div = chapter_soup.find("div", id="content") or chapter_soup.find("div", class_="main-content")

    chapter_content = content_div.get_text(separator="\n", strip=True) if content_div else "Content not found."
    sections = extract_sections(chapter_content)

    for section in sections:
        section_summary = summarize_content(section["section_content"])
        formatted_summary = f"Section {section['section_number']} - {section['section_title']}: {section_summary}"

        # Embedding
        combined_text = section["section_title"] + " - " + section["section_content"]
        embedding = model.encode(combined_text).astype(np.float32)
        index.add(np.array([embedding]))

        # Store in ChromaDB
        section_id = f"{LAW_TYPE}_{chapter_number}_{section['section_number']}"
        collection.add(
            ids=[section_id],
            documents=[formatted_summary],
            metadatas=[{
                "law_type": LAW_TYPE.upper(),
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "section_summary": formatted_summary,
                "source_url": chapter_link
            }]
        )

        all_laws_data.append({
            "law_type": LAW_TYPE.upper(),
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "section_number": section["section_number"],
            "section_title": section["section_title"],
            "section_content": section["section_content"],
            "section_summary": formatted_summary,
            "source_url": chapter_link
        })

        print(f"✅ Section {section['section_number']} added to ChromaDB")

# ====== SAVE BACKUPS ======
with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(all_laws_data, f, ensure_ascii=False, indent=4)

faiss.write_index(index, FAISS_INDEX_PATH)

print(f"\n🎉 BNS Laws Scraping Completed!")
print(f"📁 Data saved to: {JSON_OUTPUT_PATH}")
print(f"📦 FAISS index saved to: {FAISS_INDEX_PATH}")
