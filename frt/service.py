# built-in dependencies
import time
import random
from frt.utils import *
from frt.faiss import *
import tensorflow as tf
from Utilities.config import *
from frt.modules import recognition
from frt.adaface import detect_face_scrfd, compare_embeddings_scrfd, get_embedding_scrfd_async

#-----------------------------------------------------
#------------- Supporting Services --------------------
#-----------------------------------------------------

INDEX_DIR = "faiss_indexes"
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1:1 Supporting services for the main services =============================================================================================================

async def verify_response(input_args):
    logger.info("Verifing image......")
    loop = asyncio.get_running_loop()
    img1_base64 = input_args.get("img1")
    img2_base64 = input_args.get("img2")

    if not img1_base64 or not img2_base64:
        return {
            "status": True,
            "data": "",
            "code": BAD_REQUEST_CODE,
            "message": MISSING_IMAGES,
            "errorMessage": ""
        }
    
    threshold = float(input_args.get("threshold", 0.25))
    img1 = await loop.run_in_executor(executor, load_image_from_base64, img1_base64)
    img2 = await loop.run_in_executor(executor, load_image_from_base64, img2_base64)
        # Extract parameters from the payload

    
    verification= await compare_embeddings_scrfd(img1, img2, threshold)
    if verification == None:
        return_data = {
        "status": False,
        "data": "",
        "code": EMBEDDING_FAILED,
        "message": EMBEDDING_FAILED,
        "errorMessage": ""
    }

    logger.info(f"Verification result: {verification}")
    
    return_data = {
        "status": True,
        "data": verification,
        "code": SUCCESS_CODE,
        "message": VERIFY_SUCCESS,
        "errorMessage": ""
    }
    return return_data

# 1: N Supporting services for the main services =============================================================================================================

async def create_collection_response(data):
    collection_name = data.get('collection_name')
    if not collection_name:
        return {
            "status": True,
            "data": "",
            "code": BAD_REQUEST_CODE,
            "message": COLLECTION_NOT_FOUND.format(collection_name=collection_name),
            "errorMessage": ""
        }
    if db.collections.find_one({"collection_name": collection_name}):
        logger.warning(f"Collection '{collection_name}' already exists.")
        return {
            "status": True,
            "data": "",
            "code": BAD_REQUEST_CODE,
            "message": COLLECTION_EXISTS.format(collection_name=collection_name),
            "errorMessage": ""
        }
    

    db.collections.insert_one({
        "_id": unique_id(),
        "collection_name": collection_name,
        "created_at": current_timestamp()
    })

    try: 
        build_faiss_index_for_collection(collection_name, docs=[])
    except Exception as e:
        logger.warning(f"FAISS index creation failed for collection '{collection_name}': {e}")
    
    return {
        "status": True,
        "data": {"collection_name": collection_name},
        "code": CREATED_CODE,
        "message": COLLECTION_CREATED.format(collection_name=collection_name),
        "errorMessage": ""
    }

# ===========================================================================================================================================================
async def list_collections_response():
    collections_cursor = db.collections.find({}, {"_id": 0, "collection_name": 1})
    collection_names = [col["collection_name"] for col in collections_cursor]

    if not collection_names:
        logger.info("No collections found in the database.")
        return {
            "status": True,
            "data": {"collections": []},
            "code": NOT_FOUND_CODE,
            "message": COLLECTIONS_NOT_FOUND,
            "errorMessage": ""
        }
    return {
        "status": True,
        "data": {"collections": collection_names},
        "code": SUCCESS_CODE,
        "message": COLLECTION_LISTED if collections_cursor else COLLECTIONS_NOT_FOUND,
        "errorMessage": ""
    }

# ========================================================================================
async def delete_collection_response(data):
    collection_name = data.get("collection_name")
    if not collection_name:
        logger.warning("Missing collection_name in request.")
        return {
            "status": True,
            "code": BAD_REQUEST_CODE,
            "message": COLLECTION_NAME_REQUIRED,
            "errorMessage": "",
            "data": {}
        }

    res = db.collections.delete_one({"collection_name": collection_name})
    if res.deleted_count == 0:
        return {
            "status": True,
            "code": NOT_FOUND_CODE,
            "message": COLLECTION_NOT_FOUND.format(collection_name=collection_name),
            "errorMessage": "",
            "data": {}
        }
    del_obj = db.embeddings.delete_many({"collection_name": collection_name})

    if del_obj.deleted_count == 0:
        logger.warning(f"No embeddings found for collection '{collection_name}'.")
    # Delete FAISS index and metadata file
    index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{collection_name}.json")

    index_deleted = False
    meta_deleted = False

    if os.path.exists(index_path):
        os.remove(index_path)
        index_deleted = True

    if os.path.exists(metadata_path):
        os.remove(metadata_path)
        meta_deleted = True

    if index_deleted or meta_deleted:
        logger.info(f"Deleted index and metadata files for collection '{collection_name}'.")
    else:
        logger.warning(f"No index/metadata files found for collection '{collection_name}'.")

    return {
        "status": True,
        "code": SUCCESS_CODE,
        "message": COLLECTION_DELETED.format(collection_name=collection_name),
        "errorMessage": "",
        "data": {
            "deleted_collection": collection_name,
            "deleted_embeddings": del_obj.deleted_count,
            "index_deleted": index_deleted,
            "metadata_deleted": meta_deleted
        }
    }

# ===========================================================================================================================================================
async def list_images_response(collection_name):
    if not collection_name:
        return {
            "status": True,
            "data": "",
            "code": BAD_REQUEST_CODE,
            "message": COLLECTION_NAME_REQUIRED,
            "errorMessage": ""
        }
    if not db.collections.find_one({"collection_name": collection_name}):
        logger.warning(f"Collection '{collection_name}' not found.")
        return {
            "status": True,
            "data": "",
            "code": NOT_FOUND_CODE,
            "message": COLLECTION_NOT_FOUND.format(collection_name=collection_name),
            "errorMessage": ""
        }

    docs = list(db.embeddings.find(
        {"collection_name": collection_name},
        {"_id": 0, "id": 1}
        ))
    ids = [d["id"] for d in docs]
    msg = IDENTITIES_LISTED if ids else NO_IDENTITIES.format(collection_name=collection_name)
    return_data = {
        "collection": collection_name,
        "number_of_identities": len(ids),
        "identities": ids
    }

    return {
        "status": True,
        "data": return_data,
        "code": SUCCESS_CODE,
        "message": msg,
        "errorMessage": ""
    }
    
# ===========================================================================================================================================================
async def delete_image_response(input_args):
    image_id = input_args.get("id")
    collection_name = input_args.get("collection_name")

    if not image_id or not collection_name:
        logger.warning("Missing required fields in request.")
        return {"status": True, "code": BAD_REQUEST_CODE, "message": BAD_REQUEST, "errorMessage": "", "data": input_args}
    
    collection_obj = db.collections.find_one({"collection_name": collection_name})
    if not collection_obj:
        logger.warning(f"Collection '{collection_name}' not found.")
        return {"status": True, "code": NOT_FOUND_CODE, "message": COLLECTIONS_NOT_FOUND.format(collection_name=collection_name), "errorMessage": "", "data": {}}

    res = db.embeddings.delete_one({"id": image_id, "collection_name": collection_name})
    if res.deleted_count == 0:
        logger.warning(f"Failed to delete image '{image_id}' from collection '{collection_name}'.")
        return {"status": True, "code": NOT_FOUND_CODE, "message": IMAGE_NOT_FOUND.format(id=image_id, collection_name=collection_name), "errorMessage": "", "data": {}}
    logger.info(f"Deleted image '{image_id}' from collection '{collection_name}'.")

    build_faiss_index_for_collection(collection_name)

    return {
        "status": True,
        "code": SUCCESS_CODE,
        "message": IMAGE_DELETED.format(id=image_id, collection_name=collection_name),
        "errorMessage": "",
        "data": {"id": image_id, "collection_name": collection_name}
    }

# ===========================================================================================================================================================
async def add_image_response(input_args):
    overall_start_time = time.time()
    logger.info(f"Adding image ...")
    
    loop = asyncio.get_running_loop()

    force = input_args.get("force", False)
    collection_name = input_args.get("collection_name")
    base64_image = input_args.get("base64_code")
    image_id = input_args.get("id")
    
    if not collection_name or not base64_image or not image_id:
        logger.warning("Missing required fields in request.")
        return {"status": True, "code": BAD_REQUEST_CODE, "message": BAD_REQUEST, "errorMessage": "", "data" : {}}

    if not db.collections.find_one({"collection_name": collection_name}):
        return {"status": True, "code": NOT_FOUND_CODE, "message": COLLECTION_NOT_FOUND.format(collection_name=collection_name), "errorMessage": "", "data": {}}

    if db.embeddings.find_one({"id": image_id, "collection_name": collection_name}):
        logger.info(f"id {image_id} already exists.")
        return {"status": True, "code": ALREADY_EXISTS_CODE, "message": ID_ALREADY_EXISTS.format(id=image_id, collection_name=collection_name), "errorMessage": "", "data": {}}

    embedding_start_time = time.time()
    img = await loop.run_in_executor(executor, load_image_from_base64, base64_image)
    model_name=input_args.get("model_name", "adaface_scrfd")

    vec = await get_embedding_scrfd_async(img)
    
    if vec is None or (hasattr(vec, 'size') and vec.size == 0):
        logger.warning("Failed to generate face embedding.")
        return {"status": True, "code": BAD_REQUEST_CODE, "message": EMBEDDING_FAILED, "errorMessage": EMBEDDING_FAILED, "data": {}}
    embedding_time = time.time() - embedding_start_time

# ----------------------------------------------------------------------------
    db_retrieval_start_time = time.time()
    index, metadata, response = load_faiss_index_and_metadata(collection_name)
    db_retrieval_time = time.time() - db_retrieval_start_time
# -----------------------------------------------------------------------------
    if force == "True": # for handling edge case like twins
        updated_thresholds, thres, search_time, response = {}, float(0.3), 0, None
    else:
        updated_thresholds, thres, search_time, response = search_and_update_threshold(vec, index, metadata, collection_name, image_id)    
        if updated_thresholds is None and thres is None and search_time is None:
            return response
# ------------------------------------------------------------------------------    
    # Save updated metadata
    index_path = os.path.join(INDEX_DIR, f"{collection_name}.index")
    metadata_path = os.path.join(INDEX_DIR, f"{collection_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
            
    index_update_start_time = time.time()
    db.embeddings.insert_one({
        "_id": unique_id(),
        "id": image_id,
        "collection_name": collection_name,
        "embedding": vec.numpy().tolist(),  # Convert to list for MongoDB storage
        "at_threshold": float(thres),
        "created_at": current_timestamp(),
        })

    # ---------- FAISS Index & Metadata Update ----------
    index.add(np.expand_dims(vec, axis=0))  # Add single vector
    cpu_index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index, index_path)

    # Update existing thresholds if needed
    for meta in metadata:
        if meta["id"] in updated_thresholds:
            meta["at_threshold"] = updated_thresholds[meta["id"]]

    # Add new entry
    metadata.append({
        "id": image_id,
        "collection_name": collection_name,
        "at_threshold": thres
    })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    index_update_time = time.time() - index_update_start_time
    # ------------------------------------------------------
        
    logger.info(f"Added image '{image_id}' to collection '{collection_name}'.")
    return {"status": True, 
            "code": SUCCESS_CODE, 
            "message": IMAGE_ADDED.format(id=image_id, collection_name=collection_name), 
            "errorMessage": "", 
            "data": {"id": image_id,
                     "collection_name": collection_name,
                     "threshold": thres,
                     "timing": {
                                 "embedding_time": round(embedding_time , 4),
                                 "DB_retrieval_time": round(db_retrieval_time, 4),
                                 "search_time": round(search_time, 4),
                                 "index_update_time": round(index_update_time, 4),
                                 "total_time": round(time.time() - overall_start_time, 4),
                                 }
                    }
            }

# ===========================================================================================================================================================
async def find_response(input_args):
    logger.info(f"Finding image ....")
    loop = asyncio.get_running_loop()
    overall_start_time = time.time()
    img_base64 = input_args.get("img_base64")
    collection_name = input_args.get("collection_name")

    if not img_base64:
        return {
            "status": True,
            "data": {},
            "code": BAD_REQUEST_CODE,
            "message": MISSING_IMAGES,
            "errorMessage": ""
        }
    if not db.collections.find_one({"collection_name": collection_name}):
        return {
            "status": True,
            "data": {"collection_name": collection_name},
            "code": BAD_REQUEST_CODE,
            "message": COLLECTION_NOT_FOUND.format(collection_name=collection_name),
            "errorMessage": ""
        }
        
    db_retrieval_start_time = time.time()
    index, metadata, response = load_faiss_index_and_metadata(collection_name)
    db_retrieval_end_time = time.time()

    # ---- TIMING: Preprocessing ----
    img = await loop.run_in_executor(executor, load_image_from_base64, img_base64)
    # face_pp = get_full_face(img)

    # ---- TIMING: Embedding Generation ----
    embedding_start_time = time.time()
    
    img_embeddings = await get_embedding_scrfd_async(img)
    if img_embeddings is None:
        logger.warning("Failed to generate face embedding.")
        return {"status": True, "code": BAD_REQUEST_CODE, "message": EMBEDDING_FAILED, "errorMessage": "", "data": {}}

    embedding_end_time = time.time()

    search_start_time = time.time()
    
    img_embeddings_tf = tf.cast(tf.reshape(img_embeddings, [1, -1]), tf.float32)  # do everything in-graph
    img_embeddings_np = img_embeddings_tf.numpy()
    scores, indices = index.search(img_embeddings_np, 1)
    search_end_time = time.time()

    best_score = (1 - scores[0][0])/2
    best_idx = indices[0][0]
    print("scores: ",scores)
    print("indices: ",indices)
    print("best_score: ",best_score)
    print("best_idx: ",best_idx)

    match = None
    if best_idx != -1 and best_score >= 0:
        if best_idx < len(metadata):
            meta = metadata[best_idx]
            if best_score < meta["at_threshold"]:
                match = {
                    "id": meta["id"],
                    "score": float(best_score),
                    "threshold": meta["at_threshold"]
                }
        else:
            logger.warning(f"Metadata missing for index {best_idx}. Possible deletion or desync.")

    total_time = round(time.time() - overall_start_time, 4)
    logger.info(f"find executed in: {total_time}")

    if not match:
        return {
            "status": True,
            "data": {
                "timing": {
                    "embedding_time": round(embedding_end_time - embedding_start_time, 4),
                    "DB_retrieval_time": round(db_retrieval_end_time - db_retrieval_start_time, 4),
                    "search_time": round(search_end_time - search_start_time, 4),
                    "total_time": total_time
                }
            },
            "code": BAD_REQUEST_CODE,
            "message": NOT_FOUND_IN_COLLECTION,
            "errorMessage": ""
        }

    response = [{
        "query": 0,
        "results": [{
            "id": match["id"],
            "distance": match["score"],
            "threshold": match["threshold"]
        }]
    }]

    return {
        "status": True,
        "data": {
            "response": response,
            "timing": {
                    "embedding_time": round(embedding_end_time - embedding_start_time, 4),
                    "DB_retrieval_time": round(db_retrieval_end_time - db_retrieval_start_time, 4),
                    "search_time": round(search_end_time - search_start_time, 4),
                    "total_time": total_time
            }
        },
        "code": SUCCESS_CODE,
        "message": FOUND_IN_COLLECTION,
        "errorMessage": ""
    }

# ===========================================================================================================================================================
async def rebuid_index_response(input_args):
    collection_name = input_args.get("collection_name", "")
    build_faiss_index_for_collection(collection_name)
    return{
            "status": True,
            "data": "",
            "code": SUCCESS_CODE,
            "message": f"index and metadata rebuilt for: {collection_name}",
            "errorMessage": ""
        }

# ===========================================================================================================================================================
async def detect_face_response(input_args):
    img_base64 = input_args.get("img_base64")
    return_data = {
        "type": "REAL",
        "spoof_confidence": 0.0,
        "num_faces": 0,
        "face_detected": False
    }
    if not img_base64:
        return {
            "status": True,
            "data": "",
            "code": BAD_REQUEST_CODE,
            "message": "Something went wrong",
            "errorMessage": "Base64-encoded image is required"
        }
    img = load_image_from_base64(img_base64)
    if img is None:
        return {
            "status": True,
            "data": "",
            "code": BAD_REQUEST_CODE,
            "message": "Could not read image",
            "errorMessage": "Could not read image"
        }
    try:
        num_faces = detect_face_scrfd(img)
    except:
        return {
            "status": False,
            "data": return_data,
            "code": BAD_REQUEST_CODE,
            "message": "Error in face detection",
            "errorMessage": "Error in face detection"
        }
    return_data["num_faces"] = num_faces
    if num_faces == 0:
        return {
            "status": True,
            "data": return_data,
            "code": BAD_REQUEST_CODE,
            "message": "No face detected",
            "errorMessage": "No face detected"
        }
    
    return_data["face_detected"] = True
    if num_faces > 1:
        return {
            "status": True,
            "data": return_data,
            "code": BAD_REQUEST_CODE,
            "message": "More than one face detected",
            "errorMessage": "More than one face detected"
        }
    return {
        "status": True,
        "data": return_data,
        "code": SUCCESS_CODE,
        "message": "Face detected",
        "errorMessage": ""
    }
