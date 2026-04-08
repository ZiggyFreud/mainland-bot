import os
import json
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import voyageai
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CHROMA_PATH = "/data/chroma_db"
KB_FILE = "./mainland_knowledge_base.json"

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name="mainland_bot",
    embedding_function=VoyageEmbeddingFunction()
)

def main():
    with open(KB_FILE, "r") as f:
        data = json.load(f)

    chunks = []
    clinic = data["clinic"]

    clinic_text = f"""Mainland Pain Management Associates
Doctor: {clinic['doctor']}
Specialty: {clinic['specialty']}
Location: {clinic['location']}
Phone: {clinic['phone']}
Email: {clinic['email']}
Website: {clinic['website']}
Appointment Booking: {clinic['appointment_url']}"""

    chunks.append({
        "id": "clinic_info",
        "text": clinic_text,
        "metadata": {"source": "clinic_info"}
    })

    for section in data["sections"]:
        chunks.append({
            "id": section["id"],
            "text": f"{section['title']}\n\n{section['content']}",
            "metadata": {"source": section["id"]}
        })

    print(f"Built {len(chunks)} chunks.")

    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"Removed {len(existing['ids'])} existing chunks.")
    except Exception as e:
        print(f"No existing chunks to remove: {e}")

    collection.add(
        documents=[c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    print(f"Added {len(chunks)} chunks to ChromaDB.")
    print(f"Collection now has {collection.count()} total chunks.")

main()
