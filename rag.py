import os
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import voyageai
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

FALLBACK = "I'm sorry, I don't have specific information about that. Please contact our office at (609) 788-3625 or email mainlandpain@mainland-pain.com for assistance."

class VoyageEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input):
        result = voyage_client.embed(input, model="voyage-2")
        return result.embeddings

chroma_client = chromadb.PersistentClient(path="/data/chroma_db")
collection = chroma_client.get_or_create_collection(
    name="mainland_bot",
    embedding_function=VoyageEmbeddingFunction()
)

def query_rag(user_message):
    results = collection.query(
        query_texts=[user_message],
        n_results=5
    )

    docs = results["documents"][0] if results["documents"] else []

    if not docs:
        return FALLBACK

    context = "\n\n".join(docs)

    response = anthropic_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=f"""You are a helpful and compassionate assistant for Mainland Pain Management Associates, a pain management clinic in New Jersey led by Dr. Dipty Mangla.
Answer questions based ONLY on the provided context from the clinic website.
Be empathetic, professional, and concise. This is a medical practice so maintain a calm and caring tone.
Never provide specific medical advice or diagnoses.
If the specific topic is NOT found in the context, respond with ONLY this exact message and nothing else:
\"{FALLBACK}\"
Do NOT combine this fallback message with any other information.""",
        messages=[
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"}
        ]
    )

    return response.content[0].text
