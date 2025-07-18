# All essential functions for service.py
import time
import numpy as np
from frt.faiss import * 
from Utilities.config import *

def search_and_update_threshold(vec, index, metadata, collection_name,image_id, max_threshold = 0.30, min_threshold = 0.2):
    """
    Searches for the closest match in FAISS index and updates thresholds if necessary.

    Returns:
        updated_thresholds: dict of {id: new_threshold}
        threshold: float used for the new image
        search_time: float
    """
    thres = max_threshold  # Setting up threshold for comparison
    updated_thresholds = {}

    search_start_time = time.time()
    if len(metadata) > 0:
        D, I = index.search(vec.numpy().reshape(1, -1), 1)
        best_distance = (1 - D[0][0])*0.5 # D gives us cosine similarity but our logic looks for cosine distance
        best_idx = I[0][0]

        if best_idx != -1:
            matched_entry = metadata[best_idx]
            if best_distance < min_threshold:
                return None, None, None, {"status": True, 
                        "code": ALREADY_EXISTS_CODE, 
                        "message": f"Identity already exists under the ID: {matched_entry.get('id')}", 
                        "errorMessage": "", 
                        "data": {"input_id": image_id,
                                 "matched_id": matched_entry.get('id'),
                                 "distance": best_distance
                                }
                        }
            if best_distance < matched_entry["at_threshold"]:
                updated_thresholds[matched_entry["id"]] = max(float(0.2), best_distance)
                db.embeddings.update_one(
                    {"id": matched_entry["id"], "collection_name": collection_name},
                    {"$set": {"at_threshold": max(float(0.2), best_distance)}}
                )
                matched_entry["at_threshold"] = max(float(0.2), best_distance)

            if best_distance < max_threshold:
                thres = max(float(0.2), best_distance)
    search_time = time.time() - search_start_time
    return updated_thresholds, thres, search_time, None

# ===============================================================================================
def load_faiss_index_and_metadata(collection_name):
    index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{collection_name}.json")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        return None, None, {
            "status": True, "code": NOT_FOUND_CODE,
            "message": EMPTY_COLLECTION.format(collection_name=collection_name), "data": {}
        }

    cpu_index = faiss.read_index(index_path)
    index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return index, metadata, None
