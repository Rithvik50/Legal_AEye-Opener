from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import chromadb
import re

app = Flask(__name__)

# Load model and connect to ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bns_laws")

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    user_input = ""

    if request.method == 'POST':
        user_input = request.form['user_input']

        # Extract all section numbers from user query
        section_numbers = re.findall(r"\bsection\s+(\d+)\b", user_input, re.IGNORECASE)

        # Get all metadata from ChromaDB
        all_docs = collection.get()
        all_metadata = all_docs["metadatas"]

        # Look for exact matches
        found_exact = []
        for sec_num in section_numbers:
            matches = [
                meta for meta in all_metadata
                if meta.get("section_number", "").strip() == sec_num
            ]
            found_exact.extend(matches)

        if found_exact:
            formatted_results = []
            for metadata in found_exact:
                formatted_results.append({
                    "law_type": metadata.get('law_type', 'Unknown'),
                    "section_summary": metadata.get('section_summary', 'No summary available'),
                    "source_url": metadata.get('source_url', 'No source available')
                })
            result = formatted_results
        else:
            # Fallback to semantic search
            query_embedding = model.encode(user_input).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=5)

            if results and results.get('documents') and results['documents'][0]:
                formatted_results = []
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    formatted_results.append({
                        "law_type": metadata.get('law_type', 'Unknown'),
                        "section_summary": metadata.get('section_summary', 'No summary available'),
                        "source_url": metadata.get('source_url', 'No source available')
                    })
                result = formatted_results
            else:
                result = "❌ No matches found."

    return render_template('index.html', result=result, user_input=user_input)

@app.route('/debug')
def debug():
    all_data = collection.get()
    records = []

    for i in range(len(all_data['ids'])):
        records.append({
            "id": all_data['ids'][i],
            "document": all_data['documents'][i],
            "metadata": all_data['metadatas'][i]
        })

    return render_template('debug.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)
