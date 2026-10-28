"""Face-registration business logic for MaghMela KhoyaPaya.

This module intentionally contains no FastAPI routes. main.py owns the API layer.
"""
import base64
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from bson import ObjectId
from pymongo.errors import DuplicateKeyError, PyMongoError

from database import COLLECTION_NAME, MODULE_CODE, PROJECT_CODE, get_collection
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
# Import the utility directly. Do not import face_align.py if it runs a webcam demo.
from face_alignment.alignment import norm_crop

logger = logging.getLogger(__name__)

DETECTOR_PATH = os.getenv(
    "FACE_DETECTOR_PATH", "face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx"
)
RECOGNIZER_PATH = os.getenv(
    "FACE_RECOGNIZER_PATH", "face_recognition/arcface/weights/arcface_r100.pth"
)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads/missing_persons"))
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

detector = None
recognizer = None


def load_face_models() -> None:
    """Load SCRFD and ArcFace once during application startup."""
    global detector, recognizer
    if detector is not None and recognizer is not None:
        return
    if not Path(DETECTOR_PATH).is_file():
        raise FileNotFoundError(f"SCRFD model file not found: {DETECTOR_PATH}")
    if not Path(RECOGNIZER_PATH).is_file():
        raise FileNotFoundError(f"ArcFace model file not found: {RECOGNIZER_PATH}")
    detector = SCRFD(model_file=DETECTOR_PATH)
    recognizer = iresnet_inference(
        model_name="r100", path=RECOGNIZER_PATH, device=DEVICE
    )
    logger.info("Face models loaded on %s", DEVICE)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_object_id(field_name: str, value: Optional[str]) -> Optional[ObjectId]:
    if value is None or not value.strip():
        return None
    if not ObjectId.is_valid(value):
        raise ValueError(f"{field_name} must be a valid 24-character MongoDB ObjectId")
    return ObjectId(value)


def decode_base64_image(value: str) -> np.ndarray:
    encoded = value.strip()
    if encoded.startswith("data:"):
        if "," not in encoded:
            raise ValueError("Invalid image data URL")
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Image is not valid base64") from exc
    if not raw:
        raise ValueError("Image data is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("Uploaded data could not be decoded as an image")
    height, width = image.shape[:2]
    if width < 40 or height < 40:
        raise ValueError("Image dimensions are too small")
    if width * height > 25_000_000:
        raise ValueError("Image dimensions are too large")
    return image


@torch.no_grad()
def extract_embedding(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """SCRFD detection -> five-point alignment -> normalized ArcFace embedding."""
    if detector is None or recognizer is None:
        raise RuntimeError("Face models are not loaded")
    boxes, landmarks = detector.detect(image_bgr, max_num=1)
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    score = float(boxes[0][4])
    if landmarks is None or len(landmarks) == 0:
        return None, score

    aligned = norm_crop(image_bgr, landmark=landmarks[0])
    aligned_rgb = aligned[..., ::-1].copy()
    tensor = torch.from_numpy(aligned_rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    tensor = (tensor / 255.0 - 0.5) / 0.5
    feature = recognizer(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
    if feature.size == 0 or not np.all(np.isfinite(feature)):
        return None, score
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-12:
        return None, score
    return feature / norm, score


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    return value.strip()


def _normalized_name(value: Optional[str]) -> Optional[str]:
    cleaned = _clean_text(value)
    return " ".join(cleaned.casefold().split()) if cleaned else None


def register_missing_person(data) -> dict:
    """Validate submitted images, create an embedding, save image and insert case."""
    collection = get_collection()
    missing_person_id = _clean_text(data.missing_person_id)
    if not missing_person_id:
        missing_person_id = f"MP-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:10].upper()}"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", missing_person_id):
        raise ValueError("missing_person_id may contain only letters, numbers, hyphens and underscores")

    camera_id = parse_object_id("cameraId", data.cameraId)
    user_id = parse_object_id("userId", data.userId)
    dvr_id = parse_object_id("dvrId", data.dvrId)

    best_embedding = None
    best_image = None
    best_score = -1.0
    for index, image_b64 in enumerate(data.images, start=1):
        try:
            image = decode_base64_image(image_b64)
        except ValueError as exc:
            raise ValueError(f"Image {index}: {exc}") from exc
        try:
            embedding, score = extract_embedding(image)
        except Exception:
            logger.exception("Face processing failed for image %s", index)
            continue
        if embedding is not None and score > best_score:
            best_embedding, best_image, best_score = embedding, image, score

    if best_embedding is None or best_image is None:
        raise FaceNotDetectedError(
            "No usable face was detected. Upload a clear, front-facing photo and try again."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    image_filename = f"{missing_person_id}_{uuid.uuid4().hex[:8]}.jpg"
    image_path = UPLOAD_DIR / image_filename
    ok, encoded = cv2.imencode(".jpg", best_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError("Could not encode face image")
    try:
        image_path.write_bytes(encoded.tobytes())
    except OSError as exc:
        logger.exception("Could not save face image")
        raise RuntimeError("Could not save face image") from exc

    now = utc_now()
    document = {
        "missing_person_id": missing_person_id,
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "name": _clean_text(data.name),
        "name_normalized": _normalized_name(data.name),
        "age": data.age,
        "gender": _clean_text(data.gender),
        "description": _clean_text(data.description),
        "last_seen_location": _clean_text(data.last_seen_location),
        "missing_since": data.missing_since,
        "centre_id": _clean_text(data.centre_id),
        "cameraId": camera_id,
        "userId": user_id,
        "dvrId": dvr_id,
        "face_embedding": best_embedding.tolist(),
        "embedding_dimension": int(best_embedding.shape[0]),
        "embedding_model": "ArcFace iresnet_r100",
        "embedding_version": "arcface_r100_v1",
        "face_image": {
            "filename": image_filename,
            "storage_path": str(image_path),
            "content_type": "image/jpeg",
        },
        "face_quality": {"detector_score": float(best_score)},
        "registered_by": data.registered_by.strip(),
        "registered_by_role": data.registered_by_role.strip(),
        "status": "missing",
        "created_at": now,
        "updated_at": now,
        "is_active": True,
        "schema_version": 1,
    }

    try:
        collection.insert_one(document)
    except DuplicateKeyError as exc:
        image_path.unlink(missing_ok=True)
        raise DuplicateRegistrationError("missing_person_id already exists") from exc
    except PyMongoError as exc:
        image_path.unlink(missing_ok=True)
        logger.exception("MongoDB insert failed")
        raise RuntimeError("Could not save missing-person record") from exc
    except Exception:
        image_path.unlink(missing_ok=True)
        logger.exception("Unexpected registration failure")
        raise

    return {
        "message": "Missing-person face registered successfully",
        "missing_person_id": missing_person_id,
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "collection": COLLECTION_NAME,
        "embedding_dimension": int(best_embedding.shape[0]),
        "face_quality_score": float(best_score),
        "cameraId": str(camera_id) if camera_id else None,
        "userId": str(user_id) if user_id else None,
        "dvrId": str(dvr_id) if dvr_id else None,
    }


class FaceNotDetectedError(ValueError):
    """No usable face could be extracted from supplied images."""


class DuplicateRegistrationError(ValueError):
    """The case ID already exists."""
