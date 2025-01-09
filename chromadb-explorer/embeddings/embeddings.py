import os
import json
import google.generativeai as genai
import chromadb
from chromadb.api.models import Collection
from dotenv import load_dotenv

def get_tables_info():
    tables = [
        {
            "table_name": "Customers",
            "columns": ["customer_id", "name", "email"],
            "description": "Stores customer information including their personal details."
        },
        {
            "table_name": "Orders",
            "columns": ["order_id", "customer_id", "order_date", "total_amount"],
            "description": "Contains order information for each customer."
        },
        {
            "table_name": "Products",
            "columns": ["product_id", "product_name", "price"],
            "description": "Stores product details including product name and price."
        }
    ]
    return tables

def generate_embedding(text):
    """Generates embedding for a text using the Gemini API."""
    embedding_result = genai.embed_content(
        model="models/text-embedding-004", content={"parts": [{"text": text}]}
    )
    return embedding_result["embedding"]

def store_embeddings(table_collection: Collection):
    tables = get_tables_info()
    for table in tables:
        table_metadata = f"Table: {table['table_name']}; Columns: {', '.join(table['columns'])}; Description: {table['description']}"
        embedding = generate_embedding(table_metadata)
        table_collection.add(
            documents=[table_metadata],
            metadatas=[
                {"table_name": table["table_name"], "columns": ', '.join(table['columns']), "description": table["description"]}],
            ids=[table["table_name"]],
            embeddings=[embedding]
        )
    print("Tables embeddings are stored in ChromaDB.")


def search_tables(query: str, table_collection, top_k=1):
    print("Searching for related tables...")
    query_embedding = generate_embedding(query)
    results = table_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    for metadatas in results['metadatas']:
        for metadata in metadatas:
            print(json.dumps(metadata))
            print()


if __name__ == "__main__":
    load_dotenv()
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_client.heartbeat()
    print(f"Connected to ChromaDB, version: {chroma_client.get_version()}")

    collection = chroma_client.create_collection(name="tables_collection", get_or_create=True);

    store_embeddings(collection)

    search_query = "Find tables that store customer data."
    search_tables(search_query, collection)
