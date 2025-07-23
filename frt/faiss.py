import os
import json
import time
import numpy as np
import faiss
import faiss.contrib.torch_utils  # Important to ensure GPU context
from Utilities.config import*
from frt.commons.cache import faiss_index_cache
import threading

INDEX_DIR = "faiss_indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# Optimized GPU resources with better memory management
gpu_res = faiss.StandardGpuResources()
gpu_res.setDefaultNullStreamAllDevices()  # Better GPU memory management
gpu_res.setTempMemory(1024 * 1024 * 512)  # 512MB temp memory

# Thread-safe locks for index operations
index_lock = threading.RLock()

def build_faiss_index_for_collection(collection_name: str, use_cache: bool = True):
    """Optimized FAISS index building with caching and better memory management"""
    with index_lock:
        # Check cache first
        cache_key = f"index_{collection_name}"
        if use_cache:
            cached_data = faiss_index_cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached FAISS index for '{collection_name}'")
                return cached_data
        
        start_time = time.time()
        docs = list(db.embeddings.find(
                {"collection_name": collection_name},
                {"_id": 0, "id": 1, "embedding": 1, "collection_name": 1, "at_threshold": 1}
            ))
        
        if not docs:
            dim = 512
            ids_metadata = []
            # Create optimized empty index
            cpu_index = faiss.IndexFlatIP(dim)
            # Use IVF index for better performance with larger datasets
            if dim == 512:  # Standard embedding dimension
                quantizer = faiss.IndexFlatIP(dim)
                cpu_index = faiss.IndexIVFFlat(quantizer, dim, min(100, max(1, len(docs) // 10)))
                cpu_index.nprobe = 10  # Search parameter
            
            try:
                index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
            except Exception as e:
                logger.warning(f"GPU index creation failed, using CPU: {e}")
                index = cpu_index

            # Save index and metadata with better compression
            index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
            metadata_path = os.path.join(INDEX_DIR, f"{collection_name}.json")
            
            faiss.write_index(faiss.index_gpu_to_cpu(index) if hasattr(index, 'device') else index, index_path)
            with open(metadata_path, "w") as f:
                json.dump(ids_metadata, f, separators=(',', ':'))  # Compact JSON
            
            logger.info(f"Built empty FAISS index for '{collection_name}' (dim={dim}) in {time.time()-start_time:.3f}s")
            return index, ids_metadata

        # Extract embeddings with optimized memory usage
        embeddings = np.array([doc["embedding"] for doc in docs], dtype=np.float32)
        dim = embeddings.shape[1]

        # Prepare metadata
        ids_metadata = [
            {
                "id": doc["id"],
                "collection_name": doc["collection_name"],
                "at_threshold": doc.get("at_threshold", 0.4)
            }
            for doc in docs
        ]

        # Create optimized FAISS index based on dataset size
        if len(docs) < 1000:
            # Use flat index for small datasets
            cpu_index = faiss.IndexFlatIP(dim)
        else:
            # Use IVF index for larger datasets
            nlist = min(4096, max(100, len(docs) // 10))  # Adaptive number of clusters
            quantizer = faiss.IndexFlatIP(dim)
            cpu_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            cpu_index.nprobe = min(50, max(10, nlist // 10))  # Adaptive search parameter
            
            # Train the index if needed
            if len(docs) >= 100:  # Only train if we have enough data
                cpu_index.train(embeddings)

        try:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
            logger.debug(f"Created GPU FAISS index for '{collection_name}'")
        except Exception as e:
            logger.warning(f"GPU index creation failed, using CPU: {e}")
            index = cpu_index

        # Add embeddings in batches for better memory management
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            index.add(batch)

        # Save with optimized I/O
        index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
        metadata_path = os.path.join(INDEX_DIR, f"{collection_name}.json")
        
        # Convert back to CPU for saving
        cpu_index_for_save = faiss.index_gpu_to_cpu(index) if hasattr(index, 'device') else index
        faiss.write_index(cpu_index_for_save, index_path)
        
        with open(metadata_path, "w") as f:
            json.dump(ids_metadata, f, separators=(',', ':'))  # Compact JSON

        # Cache the result
        if use_cache:
            faiss_index_cache.put(cache_key, (index, ids_metadata))

        build_time = time.time() - start_time
        logger.info(f"Built FAISS index for '{collection_name}' with {len(docs)} embeddings (dim={dim}) in {build_time:.3f}s")
        
        return index, ids_metadata

def initialize_faiss_indexes():
    print("Rebuilding FAISS indexes from MongoDB...")
    # -------------------------------------------------------------------
    collection_names = db.collections.distinct("collection_name")
    # -------------------------------------------------------------------
    for collection_name in collection_names:
        build_faiss_index_for_collection(collection_name)

    print(" All indexes are up to date.")
