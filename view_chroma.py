import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Similar to selecting a table in MySQL
collection = client.get_collection("pdf_chunks")

print("Total records:", collection.count())

# # Similar to SELECT * FROM pdf_chunks LIMIT 5
# results = collection.get(
#     limit=10,
#     include=["documents", "metadatas", "embeddings"]
# )

# for i in range(len(results["ids"])):
#     print("\n-------------------------")
#     print("ID:", results["ids"][i])
#     print("Metadata:", results["metadatas"][i])
#     print("Document:", results["documents"][i][:300])

#     embedding = results["embeddings"][i]
#     print("Vector dimensions:", len(embedding))
#     print("First 10 vector values:", embedding[:10])

# results = collection.get(
#     where={"page": 75},
#     include=["documents"]
# )

# for i, document in enumerate(results["documents"]):
#     print(f"\n--- Chunk {i + 1} ---")
#     print(document)

terms = [
    "chocolate cake",
    "FIFA",
    "photosynthesis",
    "Python 3.12"
]

results = collection.get(
    include=["documents"]
)

all_text = " ".join(results["documents"]).lower()

for term in terms:
    print(
        term,
        "FOUND" if term.lower() in all_text else "NOT FOUND"
    )