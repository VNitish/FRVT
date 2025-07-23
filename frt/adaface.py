import cv2
import time
import requests
import numpy as np
import tensorflow as tf
import onnxruntime as ort
from Utilities.config import *
from frt.resnet100 import build_adaface
from insightface.model_zoo.scrfd import SCRFD
from insightface.utils.face_align import norm_crop
from frt.commons.cache import embedding_cache, image_cache
import base64

# ───── Optimized GPU Setup ─────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enhanced GPU memory configuration
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # Set virtual GPU memory limit to prevent OOM
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]  # 4GB limit
            )
        logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s).")
    except Exception as e:
        logger.warning(f"Could not set GPU memory configuration: {e}")
else:
    logger.warning("No GPU found. Running on CPU.")

# Enhanced TensorFlow optimizations
tf.config.optimizer.set_jit(True)  # Enable XLA (JIT)
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "arithmetic_optimization": True,
    "remapping": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "scoped_allocator_optimization": True,  # Additional optimization
    "pin_to_host_optimization": True,       # Additional optimization
})

os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"

ort.set_default_logger_severity(3)

# ───── Load Models ───────────────────────────────────────────────
model = build_adaface()
model.trainable = False

@tf.function(input_signature=[tf.TensorSpec([112, 112, 3], tf.float32)])
def sync_model_infer(face):
    face = face / 127.5 - 1.0
    batched = tf.expand_dims(face, axis=0)  # (1, 112, 112, 3)
    emb = model(batched, training=False)    # (1, 512)
    emb = tf.linalg.l2_normalize(emb, axis=1)
    return tf.squeeze(emb, axis=0)
    
# =========================================================================================
async def adaface_infer_async(face: tf.Tensor) -> list:
    async with recognition_semaphore:
        return await asyncio.get_running_loop().run_in_executor(
            executor, sync_model_infer, face
        )

logger.info(">>> AdaFace model loaded.")

detector = SCRFD("./model_weights_recognition/scrfd.onnx")
logger.info(">>> Scrfd model loaded.")
ctx_id = 0 if "CUDAExecutionProvider" in detector.session.get_providers() else -1
detector.prepare(ctx_id, input_size=(320,320))

# ───── Helper Functions ──────────────────────────────────────────

def tighten_crop(img, margin_frac=0.07):
    h, w = img.shape[:2]
    mh, mw = int(h*margin_frac), int(w*margin_frac)
    return img[mh:h-mh, mw:w-mw]

# =========================================================================================
def sync_detect_and_crop_face(img: np.ndarray) -> tf.Tensor | None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, kpss = detector.detect(img)
    if len(bboxes) == 0:
        return None
    areas = (bboxes[:,2]-bboxes[:,0])*(bboxes[:,3]-bboxes[:,1])
    idx_max = int(np.argmax(areas))
    bbox    = bboxes[idx_max].astype(int)
    kps     = kpss[idx_max]
    # ── align, tighten, resize to 112×112 ───────────────────────────
    aligned = norm_crop(img, kps.reshape(5,2))
    aligned = tighten_crop(aligned, margin_frac=0.07)
    aligned = cv2.resize(aligned, (112,112), interpolation=cv2.INTER_CUBIC)
    return tf.convert_to_tensor(aligned, dtype=tf.float32)

# =========================================================================================
async def detect_and_crop_face_async(image_bgr: np.ndarray) -> tf.Tensor | None:
    async with detection_semaphore:
        return await asyncio.get_running_loop().run_in_executor(
            executor, sync_detect_and_crop_face, image_bgr
        )
        
# =========================================================================================
async def get_embedding_scrfd_async(face_bgr: np.ndarray, use_cache: bool = True):
    """Optimized embedding generation with caching support"""
    try:
        # Generate cache key from image data
        if use_cache:
            image_bytes = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            cached_embedding = embedding_cache.get_embedding(image_bytes)
            if cached_embedding is not None:
                logger.debug("Using cached embedding")
                return cached_embedding
        
        face_rgb = await detect_and_crop_face_async(face_bgr)
        if face_rgb is None:
            return None
            
        embedding = await adaface_infer_async(face_rgb)
        
        # Cache the result if enabled
        if use_cache and embedding is not None:
            embedding_cache.put_embedding(image_bytes, np.array(embedding))
            
        return embedding
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return None

# =========================================================================================
async def compare_embeddings_scrfd(image1_bgr, image2_bgr, threshold=0.4):
    """
    Takes two BGR images, computes embeddings, compares with cosine distance.
    """
    emb1= await get_embedding_scrfd_async(image1_bgr)
    emb2 = await get_embedding_scrfd_async(image2_bgr)

    if emb1 is None or emb2 is None:
        return None

    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    cosine_similarity = np.dot(emb1, emb2)
    distance = (1 - cosine_similarity) * 0.5

    verified = distance < threshold

    return {
        "distance": float(distance),
        "threshold": threshold,
        "verified": bool(verified)
    }

# =========================================================================================
def detect_face_scrfd(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, kpss = detector.detect(img)
    return len(bboxes)
