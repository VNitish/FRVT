from typing import List, Optional
import numpy as np
import time
import logging
import psutil
import os

logger = logging.getLogger(__name__)

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def find(
    stored_embeddings: dict,
    img_embeddings: list,
    model_name: str = "Facenet",
    distance_metric: str = "cosine",
    threshold: Optional[float] = None,
    silent: bool = False,
    debug: bool = False,
) -> list:
    tic = time.time()
    if debug:
        logger.info("Running find()...")

    # Preload data
    ids, at_thresholds, emb_list = [], [], []
    for item in stored_embeddings:
        if item and "embedding" in item:
            ids.append(item["id"])
            at_thresholds.append(item.get("at_threshold", threshold or 1.0))
            emb_list.append(item["embedding"])

    if len(emb_list) == 0:
        return [{"query_embedding_index": i, "results": []} for i in range(len(img_embeddings))]

    mem_before = get_memory_usage_mb()
    t1 = time.time()
    stored_matrix = np.stack(emb_list).astype(np.float32)
    query_matrix = np.stack(img_embeddings).astype(np.float32)
    t2 = time.time()

    if debug:
        logger.info(f"Memory before dot: {mem_before:.2f} MB")
        logger.info(f"Stored matrix shape: {stored_matrix.shape}, Query matrix shape: {query_matrix.shape}")
        logger.info(f"Embedding stack time: {t2 - t1:.4f} sec")

    if distance_metric != "cosine":
        raise NotImplementedError("Only 'cosine' is optimized for now.")

    # Fast cosine (dot since normalized)
    t3 = time.time()
    sim_matrix = np.dot(query_matrix, stored_matrix.T)
    dist_matrix = 1.0 - sim_matrix
    t4 = time.time()

    if model_name.lower() == "adaface":
        dist_matrix = dist_matrix / 2.0
        if debug:
            logger.info("Divided distances by 2 for AdaFace")

    if debug:
        logger.info(f"Dot product time: {t4 - t3:.4f} sec")

    t5 = time.time()
    results = []
    for i, dists in enumerate(dist_matrix):
        query_results = [
            {
                "id": ids[j],
                "distance": float(dist),
                "threshold": min(at_thresholds[j], threshold or 1.0)
            }
            for j, dist in enumerate(dists)
            if dist <= min(at_thresholds[j], threshold or 1.0)
        ]

        query_results.sort(key=lambda x: x["distance"])
        results.append({
            "query_embedding_index": i,
            "results": query_results
        })

        if debug and query_results:
            distances = [r["distance"] for r in query_results]
            logger.info(f"Query {i}: match count={len(query_results)} | min={min(distances):.4f}, max={max(distances):.4f}, avg={np.mean(distances):.4f}")

    t6 = time.time()

    if not silent:
        logger.info(f"find() took {time.time() - tic:.3f} sec for {len(img_embeddings)}×{len(emb_list)}")
        if debug:
            logger.info(f"Filtering+sorting time: {t6 - t5:.4f} sec")
            logger.info(f"Total memory after processing: {get_memory_usage_mb():.2f} MB")

    return results
