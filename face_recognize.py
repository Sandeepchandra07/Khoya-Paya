"""MongoDB-backed face recognition for MaghMela KhoyaPaya.

Reuses face_register.extract_embedding() so registration and recognition use
identical SCRFD detection, five-point alignment and ArcFace preprocessing.
"""
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from pymongo.errors import PyMongoError

from database import COLLECTION_NAME, MODULE_CODE, PROJECT_CODE, get_collection
from face_register import FaceNotDetectedError, decode_base64_image, extract_embedding

logger = logging.getLogger(__name__)
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
TOP_MATCHES_LIMIT = max(1, int(os.getenv("FACE_TOP_MATCHES_LIMIT", "5")))


def _normalized_vector(value: Any) -> Optional[np.ndarray]:
    """Return a finite unit vector, or None for malformed stored data."""
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return vector / norm


def _safe_iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def recognize_missing_person(image_base64: str, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Compare an image against active missing-person embeddings in MongoDB."""
    if threshold is None:
        threshold = FACE_MATCH_THRESHOLD
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    image = decode_base64_image(image_base64)
    query_embedding, detector_score = extract_embedding(image)
    if query_embedding is None:
        raise FaceNotDetectedError(
            "No usable face was detected. Upload a clear, front-facing image and try again."
        )
    query_vector = _normalized_vector(query_embedding)
    if query_vector is None:
        raise FaceNotDetectedError("Could not create a valid face embedding from the image.")

    query = {
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "status": "missing",
        "is_active": True,
    }
    try:
        cursor = get_collection().find(query, {
            "missing_person_id": 1, "name": 1, "age": 1, "gender": 1,
            "last_seen_location": 1, "missing_since": 1, "centre_id": 1,
            "face_embedding": 1, "status": 1,
        })
        candidates: List[Dict[str, Any]] = []
        for record in cursor:
            stored = _normalized_vector(record.get("face_embedding"))
            if stored is None or stored.size != query_vector.size:
                continue
            # Dot product equals cosine similarity because both vectors are normalized.
            candidates.append({"record": record, "similarity": float(np.dot(query_vector, stored))})
    except PyMongoError as exc:
        logger.exception("MongoDB face-recognition query failed")
        raise RuntimeError("Could not query registered face embeddings") from exc

    candidates.sort(key=lambda item: item["similarity"], reverse=True)

    def public_match(item: Dict[str, Any]) -> Dict[str, Any]:
        record = item["record"]
        return {
            "missing_person_id": record.get("missing_person_id"),
            "name": record.get("name"),
            "age": record.get("age"),
            "gender": record.get("gender"),
            "last_seen_location": record.get("last_seen_location"),
            "missing_since": _safe_iso(record.get("missing_since")),
            "centre_id": record.get("centre_id"),
            "status": record.get("status"),
            "similarity_score": round(float(item["similarity"]), 4),
        }

    best = candidates[0] if candidates else None
    matched = best is not None and best["similarity"] >= threshold
    return {
        "message": "Face recognition completed",
        "recognition_status": "match" if matched else "no_match",
        "match_found": bool(matched),
        "matched_person": public_match(best) if matched else None,
        # These are ranked candidates, not confirmed identities.
        "top_candidates": [public_match(item) for item in candidates[:TOP_MATCHES_LIMIT]],
        "similarity_threshold": threshold,
        "face_quality_score": round(float(detector_score), 4),
        "embedding_dimension": int(query_vector.size),
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "collection": COLLECTION_NAME,
        "candidates_compared": len(candidates),
    }
