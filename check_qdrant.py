import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load the environment variables
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "Rag_project")

print("Connecting to Qdrant...")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

print(f"Fetching 1 document from the '{qdrant_collection_name}' collection...")
# We use scroll to just grab the very first point in the database
results, _ = qdrant_client.scroll(
    collection_name=qdrant_collection_name,
    limit=1,
    with_vectors=True, # THIS TELLS QDRANT TO SEND THE VECTOR BACK
    with_payload=True
)

if results:
    point = results[0]
    print("\n--- WE FOUND THE POINT! ---")
    print(f"ID: {point.id}")
    
    print("\n--- HERE IS THE PAYLOAD (What you saw in the screenshot) ---")
    print(point.payload)
    
    print("\n--- HERE IS THE HIDDEN VECTOR! ---")
    print(f"Total numbers in the vector: {len(point.vector)}")
    print(f"First 10 numbers: {point.vector[:10]}")
    print("...")
else:
    print("No points found in the database.")
