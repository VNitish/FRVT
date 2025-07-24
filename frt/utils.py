# All essential functions for service.py
import time
import numpy as np
from frt.faiss import * 
from Utilities.config import *

# Removed search_and_update_threshold function - adaptive thresholding logic has been removed

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
