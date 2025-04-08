from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import chromadb
import re
from groq import Groq
import os

# Set your Groq API key here or through an environment variable
groq_api_key = os.getenv("GROQ_API_KEY", "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")
groq_client = Groq(api_key=groq_api_key)

app = Flask(__name__)

# Load model and connect to ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="bns_laws")

# --- Groq helper function ---
def generate_response_with_groq(query, context_docs):
    context_text = "\n\n".join(context_docs)

    system_prompt = (
        "You are a helpful legal assistant. Use the following legal context to answer the user’s question accurately. "
        "If the context is not sufficient, say so."
    )

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuery:\n{query}"}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ Groq API Error: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    user_input = ""

    if request.method == 'POST':
        user_input = request.form['user_input']
        
        if not user_input.strip():
            result = [{
                "law_type": "Input Error",
                "section_summary": "⚠️ Please enter a query in the text box.",
                "source_url": "#",
                "match_type": "Error"
            }]
            return render_template('index.html', result=result, user_input=user_input)

        # Normalize "till section" to "from section"
        user_input = re.sub(
            r'till\s+section\s*(\d+)\s*from\s+(?:section\s*)?(\d+)',
            r'from section \2 to section \1',
            user_input,
            flags=re.IGNORECASE
        )

        MIN_SECTION = 1
        MAX_SECTION = 358
        extracted_sections = set()
        invalid_sections = set()

        combined_range_matches = re.findall(
            r'(?:from\s+section\s*(\d+)\s*(?:to|till)\s*section?\s*(\d+))|(?:till\s+section\s*(\d+)\s*from\s+section\s*(\d+))',
            user_input,
            re.IGNORECASE
        )
        for m in combined_range_matches:
            nums = [n for n in m if n]
            if len(nums) == 2:
                start, end = int(nums[0]), int(nums[1])
                if start > end:
                    start, end = end, start
                for i in range(start, end + 1):
                    if MIN_SECTION <= i <= MAX_SECTION:
                        extracted_sections.add(str(i))
                    else:
                        invalid_sections.add(str(i))

        range_matches = re.findall(r'\b(\d+)\s*(?:to|-)\s*(\d+)\b', user_input)
        for start, end in range_matches:
            try:
                start_int = int(start)
                end_int = int(end)
                if start_int > end_int:
                    start_int, end_int = end_int, start_int
                for i in range(start_int, end_int + 1):
                    if MIN_SECTION <= i <= MAX_SECTION:
                        extracted_sections.add(str(i))
                    else:
                        invalid_sections.add(str(i))
            except:
                invalid_sections.add(f"{start}-{end}")

        grouped_matches = re.findall(r'\bsection(?:s)?[^\d]*(\d+(?:[\s,]*(?:and)?[\s,]*\d+)*)', user_input, re.IGNORECASE)
        for group in grouped_matches:
            for num in re.findall(r'\d+', group):
                try:
                    normalized = str(int(num))
                    if MIN_SECTION <= int(normalized) <= MAX_SECTION:
                        extracted_sections.add(normalized)
                    else:
                        invalid_sections.add(normalized)
                except:
                    continue

        all_docs = collection.get()
        all_metadata = all_docs["metadatas"]

        found_exact = []
        formatted_results = []

        for sec_num in extracted_sections:
            matches = [
                meta for meta in all_metadata
                if meta.get("section_number", "").strip() == sec_num
            ]
            found_exact.extend(matches)

        found_exact_sorted = sorted(
            found_exact,
            key=lambda x: int(x.get("section_number", "0").strip())
        )

        for metadata in found_exact_sorted:
            formatted_results.append({
                "law_type": metadata.get('law_type', 'Unknown'),
                "section_summary": metadata.get('section_summary', 'No summary available'),
                "source_url": metadata.get('source_url', 'No source available'),
                "match_type": "Exact"
            })

        if invalid_sections:
            formatted_results.append({
                "law_type": "Invalid",
                "section_summary": f"The following section(s) do not exist: {', '.join(sorted(invalid_sections))} (valid range is 1 to 358).",
                "source_url": "#",
                "match_type": "Error"
            })

        if not found_exact and not extracted_sections:
            query_embedding = model.encode(user_input).tolist()
            results = collection.query(query_embeddings=[query_embedding], n_results=5)

            if results and results.get('documents') and results['documents'][0]:
                top_docs = results['documents'][0]
                top_contexts = [doc for doc in top_docs]

                groq_answer = generate_response_with_groq(user_input, top_contexts)

                formatted_results.append({
                    "law_type": "Groq AI",
                    "section_summary": groq_answer,
                    "source_url": "#",
                    "match_type": "Generated"
                })

                for doc, metadata in zip(top_docs, results['metadatas'][0]):
                    formatted_results.append({
                        "law_type": metadata.get('law_type', 'Unknown'),
                        "section_summary": metadata.get('section_summary', 'No summary available'),
                        "source_url": metadata.get('source_url', 'No source available'),
                        "match_type": "Source"
                    })
            else:
                formatted_results.append({
                    "law_type": "None",
                    "section_summary": "❌ No matches found at all.",
                    "source_url": "#",
                    "match_type": "Error"
                })

        result = formatted_results

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
