import os
import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import voyageai
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
CHROMA_PATH = "/data/chroma_db"

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

URLS = [
    "https://www.mainland-pain-management.com/",
    "https://www.mainland-pain-management.com/about-doctor/",
    "https://www.mainland-pain-management.com/about-clinic/",
    "https://www.mainland-pain-management.com/back-pain/",
    "https://www.mainland-pain-management.com/neck-and-head-pain/",
    "https://www.mainland-pain-management.com/abdominal-pain/",
    "https://www.mainland-pain-management.com/pelvic-pain/",
    "https://www.mainland-pain-management.com/epidural/",
    "https://www.mainland-pain-management.com/selective-nerve-root-block/",
    "https://www.mainland-pain-management.com/radiofrequency-ablation/",
    "https://www.mainland-pain-management.com/sacroiliac-joint-injections/",
    "https://www.mainland-pain-management.com/vertiflex/",
    "https://www.mainland-pain-management.com/spinal-cord-stimulator/",
    "https://www.mainland-pain-management.com/kyphoplasty/",
    "https://www.mainland-pain-management.com/tumour-ablations/",
    "https://www.mainland-pain-management.com/peripheral-nerve-stimulator/",
    "https://www.mainland-pain-management.com/contact-us/",
]

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def scrape_page(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        text = " ".join(text.split())
        return text
    except Exception as e:
        print(f"  Error scraping {url}: {e}")
        return ""

def chunk_text(text, source):
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append({
            "id": f"{source}_chunk_{idx}",
            "text": chunk,
            "metadata": {"source": source}
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks

def main():
    all_chunks = []

    for url in URLS:
        print(f"Scraping: {url}")
        text = scrape_page(url)
        if not text:
            print("  No content found, skipping.")
            continue
        source = url.replace("https://www.mainland-pain-management.com/", "").strip("/") or "home"
        chunks = chunk_text(text, source)
        print(f"  {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No content scraped.")
        return

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Adding to ChromaDB...")

    collection.add(
        documents=[c["text"] for c in all_chunks],
        ids=[c["id"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks]
    )

    print(f"Done! Collection now has {collection.count()} chunks.")

if __name__ == "__main__":
    main()
