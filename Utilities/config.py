from Utilities.utils import *
import numpy as np
import logging
import base64
import cv2
import PIL.Image as Image
import io

import os
import asyncio
from Utilities.database import db

#initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


from concurrent.futures import ThreadPoolExecutor

# Performance Configuration - Optimized
import multiprocessing

# Auto-detect optimal worker counts based on system resources
cpu_count = multiprocessing.cpu_count()
max_workers = int(os.getenv("MAX_WORKERS", min(32, cpu_count * 4)))  # Cap at 32 for memory
detection_workers = int(os.getenv("DETECTION_WORKERS", min(8, cpu_count)))
recognition_workers = int(os.getenv("RECOGNITION_WORKERS", min(8, cpu_count)))

# Optimized ThreadPoolExecutor with better resource management
executor = ThreadPoolExecutor(
    max_workers=max_workers,
    thread_name_prefix="FRT-Worker"
)

# Limit concurrent GPU inferences with optimized semaphores
detection_semaphore = asyncio.Semaphore(detection_workers)    # CPU/GPU face detection
recognition_semaphore = asyncio.Semaphore(recognition_workers)  # GPU embedding inference

# Additional performance semaphores
db_semaphore = asyncio.Semaphore(20)  # Database operations
image_processing_semaphore = asyncio.Semaphore(16)  # Image decode/encode operations

# Codes
SUCCESS_CODE         = 200
CREATED_CODE         = 201
EXCEPTION_CODE       = 500
NOT_FOUND_CODE       = 404
BAD_REQUEST_CODE     = 400
ALREADY_EXISTS_CODE  = 409

DATABASE_PATH = "./frt_db.db"

# Messages
# Generic messages
SUCCESS_MESSAGE      = "Success"
EXCEPTION_MESSAGE    = "Something went wrong!"
CREATED              = "{name} created successfully."
UPDATED              = "{name} updated successfully."
DELETED              = "{name} deleted successfully."
NOT_DELETED          = "Could not delete the {name}."
FILE_ERROR           = "{name} file does not contain valid data."
WRONG_ID             = "Please provide a correct {name}."

# Feature-specific messages
# — Verification
VERIFY_SUCCESS       = "Face verification successful."
VERIFY_FAILURE       = "Face verification failed."
MISSING_IMAGES       = "You must provide base64-encoded image."

# — Collections
COLLECTION_CREATED   = "Created collection '{collection_name}' successfully."
COLLECTION_EXISTS    = "Collection '{collection_name}' already exists."
COLLECTION_NAME_REQUIRED = "Collection name is required."
COLLECTION_LISTED    = "Collections retrieved successfully."
NO_COLLECTIONS       = "No collections found."
COLLECTION_DELETED   = "Collection '{collection_name}' deleted successfully."
COLLECTION_NOT_FOUND = "Collection '{collection_name}' not found."
COLLECTIONS_NOT_FOUND = "No collections found in the database."
EMPTY_COLLECTION     = "Collection '{collection_name}' is empty."

# — Identities / Images
IDENTITIES_LISTED    = "Identities retrieved successfully."
ID_ALREADY_EXISTS    = "Identity with ID '{id}' already exists in collection '{collection_name}'."
NO_IDENTITIES        = "No identities found in '{collection_name}'."
IMAGE_ADDED          = "Image '{id}' added to collection '{collection_name}'."
IMAGE_DELETED        = "Image '{id}' deleted from collection '{collection_name}'."
IMAGE_NOT_FOUND      = "Image '{id}' not found in collection '{collection_name}'."
EMBEDDING_FAILED     = "No Face Detected"
IMAGE_NOT_CENTERED   = "Face in image is not centered. Please ensure the face is centered in the image."
# — Recognition
FOUND_IN_COLLECTION   = "Face match found."
NOT_FOUND_IN_COLLECTION = "No match found."
# — Detection
DETECTION_SUCCESS    = "Face detection successful."
DETECTION_FAILURE    = "Face detection failed."


# — Requests / Validation
BAD_REQUEST          = "Missing or invalid request data."
UNAUTHORIZED         = "Unauthorized: invalid or missing token."


def load_image_from_base64(base64_code: str) -> np.ndarray:
    """Optimized image loading with validation and caching support"""
    try:
        # Fast decode without extra validation for performance
        decoded_bytes = base64.b64decode(base64_code, validate=True)
        
        # Quick format validation using first few bytes (magic numbers)
        if decoded_bytes[:2] == b'\xff\xd8':  # JPEG
            pass  # Valid JPEG
        elif decoded_bytes[:8] == b'\x89PNG\r\n\x1a\n':  # PNG
            pass  # Valid PNG
        else:
            # Fallback to PIL for thorough validation only if needed
            with Image.open(io.BytesIO(decoded_bytes)) as img:
                file_type = img.format.lower()
                if file_type not in {"jpeg", "png"}:
                    raise ValueError(f"Input image can be jpg or png, but it is {file_type}")

        # Optimized decoding - use frombuffer instead of fromstring (deprecated)
        nparr = np.frombuffer(decoded_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise ValueError("Failed to decode image data")
            
        return img_bgr
        
    except Exception as e:
        logger.error(f"Image loading failed: {str(e)}")
        raise ValueError(f"Invalid image data: {str(e)}")


from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from Utilities.authentication import verify_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_current_user(token: str = Depends(oauth2_scheme)):
    # hard coded token for pooja maam
    if token == "9beaa95b-c59f-4ec9-bb60-5f4ee1986311":
        user = db.users.find_one({"_id": "4512f457845e5r474gui47k4m6"})
        return user
    payload = verify_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.users.find_one({"email": payload["sub"]})
    if not user or user.get("is_deleted", False) or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="Inactive or deleted user")
    return user