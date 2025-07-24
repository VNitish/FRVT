import os
import json
import numpy as np
import faiss
import faiss.contrib.torch_utils  # Important to ensure GPU context
from Utilities.config import*

INDEX_DIR = "faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)
gpu_res = faiss.StandardGpuResources()   # Optional: Can be reused globally

def build_faiss_index_for_collection(collection_name: str):
    docs = list(db.embeddings.find(
            {"collection_name": collection_name},
            {"_id": 0, "id": 1, "embedding": 1, "collection_name": 1}
        ))
    if not docs:
        dim = 512
        ids_metadata = []
        # Create FAISS index (cosine similarity)
        cpu_index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)  # 0 = GPU ID

        # Save index and metadata
        faiss.write_index(faiss.index_gpu_to_cpu(index), os.path.join(INDEX_DIR, f"{collection_name}.index"))
        with open(os.path.join(INDEX_DIR, f"{collection_name}.json"), "w") as f:
            json.dump(ids_metadata, f)
        print(f"[OK] Built FAISS index for '{collection_name}' with {len(docs)} embeddings (dim={dim}).")
        return

    # Extract embeddings and determine dimension
    embeddings = np.array([doc["embedding"] for doc in docs], dtype='float32')
    dim = embeddings.shape[1]  # Automatically detect dimension

    # Prepare metadata
    ids_metadata = [
        {
            "id": doc["id"],
            "collection_name": doc["collection_name"],
        }
        for doc in docs
    ]

    # Create FAISS index (cosine similarity)
    cpu_index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)  # 0 = GPU ID
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(faiss.index_gpu_to_cpu(index), os.path.join(INDEX_DIR, f"{collection_name}.index"))
    with open(os.path.join(INDEX_DIR, f"{collection_name}.json"), "w") as f:
        json.dump(ids_metadata, f)

    print(f"[OK] Built FAISS index for '{collection_name}' with {len(docs)} embeddings (dim={dim}).")

def initialize_faiss_indexes():
    print("Rebuilding FAISS indexes from MongoDB...")
    # -------------------------------------------------------------------
    collection_names = db.collections.distinct("collection_name")
    # -------------------------------------------------------------------
    for collection_name in collection_names:
        build_faiss_index_for_collection(collection_name)

    print(" All indexes are up to date.")
