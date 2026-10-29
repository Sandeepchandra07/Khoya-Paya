"""CCTV / recorded-video face matching for MaghMela KhoyaPaya.

Pipeline (per sampled frame):
    SCRFD full-frame detection (ALL faces + 5 landmarks, configurable input size)
      -> adaptive frame upscale for small-face detection (tiny CCTV faces)
      -> quality gates (detector score, face size, pose/yaw, blur)
      -> five-point alignment via norm_crop (IDENTICAL to registration)
      -> batched ArcFace embedding (IDENTICAL preprocessing to registration)
      -> one matrix multiply against ALL registered embeddings at once
      -> per-person "sightings" confirmed by independent observations in time
      -> possible-match events with video timestamp

Nothing is registered and no CCTV embedding is stored. Every result is a
POSSIBLE match that needs authorised human review.

Public entry point:
    process_video_file(video_path, *, video_filename=None, camera_id=...,
                       threshold=None, frame_skip=3, confirm_hits=None,
                       save_evidence=True, reference_image_base64=None,
                       video_start_time=None) -> dict
"""
from __future__ import annotations

import logging
import os
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from bson import ObjectId
from pymongo.errors import PyMongoError

import face_register
from database import MODULE_CODE, PROJECT_CODE, get_collection
from face_alignment.alignment import norm_crop

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration (all overridable with environment variables)
# --------------------------------------------------------------------------- #
EVIDENCE_DIR = Path(os.getenv("CCTV_EVIDENCE_DIR", "uploads/cctv_matches"))
EVENT_COLLECTION_NAME = os.getenv("CCTV_EVENT_COLLECTION", "cctv_possible_match_events")

DEFAULT_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
DEFAULT_CONFIRM_HITS = int(os.getenv("CCTV_CONFIRM_HITS", "2"))
MAX_VIDEO_BYTES = int(os.getenv("CCTV_MAX_VIDEO_BYTES", str(500 * 1024 * 1024)))

# Detector. Bigger input size = more small/far faces found, but slower.
# 1280 is a good accuracy default for 1080p CCTV. Use 960 or 640 for speed.
DET_SIZE = int(os.getenv("CCTV_DET_SIZE", "1280"))
DET_THRESH = float(os.getenv("CCTV_DET_THRESH", "0.30"))        # detector internal NMS/score cut
MIN_DET_SCORE = float(os.getenv("CCTV_MIN_DET_SCORE", "0.30"))  # our own quality gate
MIN_FACE_SIZE = int(os.getenv("CCTV_MIN_FACE_SIZE", "32"))      # min(face width, height) in pixels
MAX_YAW_RATIO = float(os.getenv("CCTV_MAX_YAW_RATIO", "0.70"))  # skip extreme side profiles
MIN_BLUR_VARIANCE = float(os.getenv("CCTV_MIN_BLUR_VARIANCE", "10"))  # measured on aligned 112x112 crop
MAX_FACES_PER_FRAME = int(os.getenv("CCTV_MAX_FACES_PER_FRAME", "60"))
EMBED_BATCH_SIZE = int(os.getenv("CCTV_EMBED_BATCH_SIZE", "16"))

# Adaptive upscale: if the original frame is smaller than DET_SIZE we upscale
# it before detection so tiny CCTV faces survive the detector's internal resize.
# Cap avoids runaway memory on odd inputs. 3.0x is safe for 1080p -> 4K work.
MAX_UPSCALE = float(os.getenv("CCTV_MAX_UPSCALE", "3.0"))
MIN_UPSCALE = float(os.getenv("CCTV_MIN_UPSCALE", "1.0"))

# Temporal confirmation
CONFIRM_WINDOW_SEC = float(os.getenv("CCTV_CONFIRM_WINDOW_SEC", "10"))  # hits must fall inside this window
MIN_HIT_SPACING_SEC = float(os.getenv("CCTV_MIN_HIT_SPACING_SEC", "0.5"))  # hits closer than this count once
SIGHTING_GAP_SEC = float(os.getenv("CCTV_SIGHTING_GAP_SEC", "30"))      # silence longer than this = new sighting

MAX_FRAME_ERRORS = int(os.getenv("CCTV_MAX_FRAME_ERRORS", "50"))
PROGRESS_LOG_EVERY = int(os.getenv("CCTV_PROGRESS_LOG_EVERY", "200"))

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]")
_indexes_ready = False


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def make_json_safe(value: Any) -> Any:
    """Recursively convert MongoDB/BSON and NumPy values to JSON-safe values."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key) if isinstance(key, ObjectId) else key: make_json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def _unit_vector(value: Any) -> Optional[np.ndarray]:
    """Finite, L2-normalised float32 vector, or None if malformed."""
    try:
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / norm


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _fmt_ts(seconds: float) -> str:
    """Seconds -> HH:MM:SS.mmm"""
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _confidence_band(similarity: float, threshold: float) -> str:
    if similarity >= threshold + 0.15:
        return "high"
    if similarity >= threshold + 0.05:
        return "medium"
    return "low"


def _public_candidate(record: Dict[str, Any], score: float) -> Dict[str, Any]:
    return {
        "missing_person_id": record.get("missing_person_id"),
        "name": record.get("name"),
        "age": record.get("age"),
        "gender": record.get("gender"),
        "last_seen_location": record.get("last_seen_location"),
        "missing_since": _iso(record.get("missing_since")),
        "centre_id": record.get("centre_id"),
        "status": record.get("status"),
        "similarity_score": round(float(score), 4),
    }


# --------------------------------------------------------------------------- #
# Robust SCRFD result parsing
# --------------------------------------------------------------------------- #
def _coerce_detector_output(raw: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize whatever the SCRFD wrapper returns into (boxes[N,5], kps[N,5,2]).

    Supported shapes (in order of what your wrapper might produce):
      * (boxes, kps)                     -> boxes[N,>=5] with score at index 4
      * (boxes, scores, kps)             -> boxes[N,4], scores[N], kps[N,5,2]
      * (boxes, kps, scores)             -> boxes[N,4], kps[N,5,2], scores[N]
      * dict {"bbox":..., "kps":..., "scores":...}
      * (outputs, img_info, bboxes, landmarks)  <- BYTETracker-era tuple, tolerated
      * tuple with single tensor that has [x1,y1,x2,y2,score,(kps...)] columns
    Returns (None, None) if it can't make sense of the payload (logged once).
    """
    boxes: Optional[np.ndarray] = None
    kps: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None

    if isinstance(raw, dict):
        boxes = raw.get("bbox") if raw.get("bbox") is not None else raw.get("boxes")
        kps = raw.get("kps") if raw.get("kps") is not None else raw.get("landmarks")
        scores = raw.get("scores") if raw.get("scores") is not None else raw.get("score")
    elif isinstance(raw, (tuple, list)):
        # Unwrap BYTETracker-style 4-tuple: (outputs, img_info, bboxes, landmarks)
        if len(raw) == 4 and isinstance(raw[1], dict):
            raw = (raw[2], raw[3])
        if len(raw) == 0:
            return None, None
        if len(raw) == 1:
            payload = raw[0]
            if isinstance(payload, np.ndarray) and payload.ndim == 2 and payload.shape[1] >= 5:
                boxes = payload
            else:
                return None, None
        elif len(raw) == 2:
            a, b = raw
            a_arr = None if a is None else np.asarray(a, dtype=np.float32)
            b_arr = None if b is None else np.asarray(b, dtype=np.float32)

            def _is_boxes(x):
                return x is not None and x.ndim == 2 and 4 <= x.shape[1] <= 6

            def _is_kps(x):
                return x is not None and x.ndim == 3 and x.shape[1] == 5 and x.shape[2] == 2

            if _is_boxes(a_arr) and (_is_kps(b_arr) or b_arr is None):
                boxes, kps = a_arr, b_arr
            elif _is_boxes(b_arr) and (_is_kps(a_arr) or a_arr is None):
                boxes, kps = b_arr, a_arr
            else:
                boxes, kps = a_arr, b_arr
        elif len(raw) == 3:
            a, b, c = raw
            a_arr = None if a is None else np.asarray(a, dtype=np.float32)
            b_arr = None if b is None else np.asarray(b, dtype=np.float32)
            c_arr = None if c is None else np.asarray(c, dtype=np.float32)
            if a_arr is not None and a_arr.ndim == 2 and a_arr.shape[1] >= 4:
                boxes = a_arr
                if b_arr is not None and b_arr.ndim == 3:
                    kps, scores = b_arr, (None if c_arr is None else c_arr.reshape(-1))
                else:
                    scores, kps = (None if b_arr is None else b_arr.reshape(-1)), c_arr
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None

    if boxes is None:
        return None, None
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 4 or len(boxes) == 0:
        return None, None

    # Ensure there is a score column at index 4.
    if boxes.shape[1] >= 5:
        col4 = boxes[:, 4]
        if scores is not None and (not np.any(col4) or np.nanmax(np.abs(col4)) < 1e-6):
            boxes = boxes[:, :4]
        else:
            boxes = boxes[:, :5]
    if boxes.shape[1] == 4:
        if scores is None:
            # No score available anywhere: default to 1.0 so faces are NOT silently dropped.
            scores = np.ones(len(boxes), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(scores) != len(boxes):
            scores = np.ones(len(boxes), dtype=np.float32)
        boxes = np.concatenate([boxes, scores.reshape(-1, 1)], axis=1)

    if kps is not None:
        kps = np.asarray(kps, dtype=np.float32)
        if kps.ndim != 3 or kps.shape[1] != 5 or kps.shape[2] != 2 or len(kps) != len(boxes):
            kps = None

    return boxes, kps


# --------------------------------------------------------------------------- #
# Gallery (registered embeddings)
# --------------------------------------------------------------------------- #
class _Gallery:
    """Registered embeddings stacked into one (N, D) matrix for fast matching."""

    def __init__(self, records: List[Dict[str, Any]], matrix: np.ndarray):
        self.records = records
        self.matrix = matrix  # float32, rows are unit vectors

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[1])


def _load_registered_gallery() -> _Gallery:
    query = {
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "status": "missing",
        "is_active": True,
    }
    projection = {
        "missing_person_id": 1, "name": 1, "age": 1, "gender": 1,
        "last_seen_location": 1, "missing_since": 1, "centre_id": 1,
        "face_embedding": 1, "status": 1,
    }
    records: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    try:
        for record in get_collection().find(query, projection):
            vec = _unit_vector(record.get("face_embedding"))
            if vec is None:
                logger.warning(
                    "Skipping malformed embedding for person_id=%s",
                    record.get("missing_person_id"),
                )
                continue
            record.pop("face_embedding", None)
            records.append(record)
            vectors.append(vec)
    except PyMongoError as exc:
        logger.exception("Could not load missing-person embeddings")
        raise RuntimeError(
            "Could not load registered missing-person embeddings from MongoDB"
        ) from exc

    if not records:
        raise ValueError(
            "No active registered missing-person face embeddings were found to compare against."
        )

    # Keep only the dominant embedding dimension (guards against mixed model versions).
    dominant_dim, _ = Counter(v.size for v in vectors).most_common(1)[0]
    kept = [(r, v) for r, v in zip(records, vectors) if v.size == dominant_dim]
    if len(kept) != len(records):
        logger.warning(
            "Skipped %s embeddings with dimension != %s",
            len(records) - len(kept),
            dominant_dim,
        )
    records = [r for r, _ in kept]
    matrix = np.ascontiguousarray(np.stack([v for _, v in kept]).astype(np.float32))
    logger.info("CCTV gallery loaded: %s people, dim=%s", len(records), dominant_dim)
    return _Gallery(records, matrix)


def _build_reference_gallery(reference_image_base64: str) -> _Gallery:
    image = face_register.decode_base64_image(reference_image_base64)
    vector, score = face_register.extract_embedding(image)
    unit = _unit_vector(vector) if vector is not None else None
    if unit is None:
        raise face_register.FaceNotDetectedError(
            "No usable face was detected. Provide a clear, front-facing reference photo."
        )
    logger.info("CCTV reference image ready: detector_score=%.3f dim=%s", score, unit.size)
    record = {
        "missing_person_id": "provided_reference_image",
        "name": "Provided reference image",
        "status": "reference_only",
    }
    return _Gallery([record], np.ascontiguousarray(unit.reshape(1, -1)))


# --------------------------------------------------------------------------- #
# Detection (with adaptive frame upscale for tiny CCTV faces)
# --------------------------------------------------------------------------- #
_det_kwargs: Dict[str, Any] = {"max_num": 0}  # max_num=0 -> return ALL faces
if DET_SIZE > 0:
    _det_kwargs["input_size"] = (DET_SIZE, DET_SIZE)
if DET_THRESH > 0:
    _det_kwargs["thresh"] = DET_THRESH
_det_kwargs_ok = True

_upscale_warned = False


def _compute_upscale(frame_h: int, frame_w: int) -> float:
    """Return the upscale factor to reach DET_SIZE, clamped to [MIN_UPSCALE, MAX_UPSCALE]."""
    longest = float(max(frame_h, frame_w))
    if longest <= 0 or DET_SIZE <= 0:
        return 1.0
    target = float(DET_SIZE)
    if longest >= target:
        return 1.0  # already big enough
    factor = target / longest
    if factor < MIN_UPSCALE:
        return MIN_UPSCALE
    if factor > MAX_UPSCALE:
        return MAX_UPSCALE
    return factor


def _detect_all_faces(frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (boxes[N,5], landmarks[N,5,2]) for every face in the frame.

    Robust against the exact SCRFD wrapper signature by:
      1. Upscaling small frames before detection so tiny CCTV faces survive the
         detector's internal resize to DET_SIZE. Boxes/landmarks are rescaled
         back to original coordinates so alignment runs on the best-quality
         pixels available (the original frame).
      2. Trying the richest kwargs, then degrading once, permanently.
      3. Normalizing whatever payload comes back through _coerce_detector_output.
      4. If landmarks are missing, synthesizing a soft alignment guide from the
         bounding box so ArcFace still runs (instead of silently returning []).
    """
    global _det_kwargs, _det_kwargs_ok, _upscale_warned
    detector = face_register.detector

    # --- Adaptive upscale for small-face detection ---
    h, w = frame.shape[:2]
    upscale = _compute_upscale(h, w)
    det_frame = frame
    if upscale > 1.01:
        new_w = int(round(w * upscale))
        new_h = int(round(h * upscale))
        try:
            det_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        except cv2.error:
            # Memory failure on extreme inputs: fall back to original frame once.
            if not _upscale_warned:
                logger.warning(
                    "CCTV upscale to %sx failed for frame %sx%s; falling back to original",
                    upscale, w, h,
                )
                _upscale_warned = True
            det_frame = frame
            upscale = 1.0

    raw = None
    if _det_kwargs_ok:
        try:
            raw = detector.detect(det_frame, **_det_kwargs)
        except TypeError:
            logger.warning(
                "SCRFD.detect rejected kwargs %s; degrading permanently",
                list(_det_kwargs),
            )
            _det_kwargs = {"max_num": 0}
            _det_kwargs_ok = False
            try:
                raw = detector.detect(det_frame, **_det_kwargs)
            except TypeError:
                raw = None

    if raw is None:
        try:
            raw = detector.detect(det_frame)
        except Exception:
            logger.exception("SCRFD.detect failed entirely on a frame")
            return None, None

    boxes, kps = _coerce_detector_output(raw)
    if boxes is None:
        return None, None

    # --- Scale boxes and landmarks back to original frame coordinates ---
    if upscale != 1.0:
        boxes = boxes.copy()
        boxes[:, :4] /= upscale
        if kps is not None:
            kps = kps.copy() / upscale

    if kps is None:
        kps = _synthesize_landmarks(boxes)
    return boxes, kps


def _synthesize_landmarks(boxes: np.ndarray) -> np.ndarray:
    """Approximate the 5 landmarks from box geometry, per ArcFace's reference layout."""
    out = np.zeros((len(boxes), 5, 2), dtype=np.float32)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        w = x2 - x1
        h = y2 - y1
        # ArcFace's 112x112 template normalized to [0,1]:
        # left eye (0.316), right eye (0.707), nose (0.514), L mouth (0.378), R mouth (0.651)
        # vertical: eyes ~0.377, nose ~0.566, mouth ~0.769
        out[i, 0] = (x1 + 0.316 * w, y1 + 0.377 * h)  # left eye
        out[i, 1] = (x1 + 0.707 * w, y1 + 0.377 * h)  # right eye
        out[i, 2] = (x1 + 0.514 * w, y1 + 0.566 * h)  # nose
        out[i, 3] = (x1 + 0.378 * w, y1 + 0.769 * h)  # left mouth
        out[i, 4] = (x1 + 0.651 * w, y1 + 0.769 * h)  # right mouth
    return out


@torch.no_grad()
def _embed_aligned(frame: np.ndarray, landmarks: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
    """Align each face and run ArcFace in batches (identical preprocessing to registration)."""
    all_feats: List[np.ndarray] = []
    blur_values: List[float] = []
    for start in range(0, len(landmarks), EMBED_BATCH_SIZE):
        chunk = landmarks[start:start + EMBED_BATCH_SIZE]
        crops = []
        for lm in chunk:
            aligned = norm_crop(frame, landmark=lm)
            gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            blur_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            crops.append(aligned[..., ::-1])  # BGR -> RGB
        batch = np.stack(crops).astype(np.float32)
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(face_register.DEVICE)
        tensor = (tensor / 255.0 - 0.5) / 0.5
        feats = face_register.recognizer(tensor).detach().cpu().numpy().astype(np.float32)
        all_feats.append(feats.reshape(len(chunk), -1))
    return np.concatenate(all_feats, axis=0), blur_values


class _Face:
    __slots__ = ("bbox", "det_score", "blur", "embedding")

    def __init__(self, bbox, det_score, blur, embedding):
        self.bbox = bbox
        self.det_score = det_score
        self.blur = blur
        self.embedding = embedding


def _extract_faces(frame: np.ndarray, stats: Dict[str, Any]) -> List[_Face]:
    boxes, landmarks = _detect_all_faces(frame)
    if boxes is None:
        return []

    stats["frames_with_face_detections"] += 1
    stats["faces_detected"] += len(boxes)
    try:
        max_score = float(np.max(boxes[:, 4]))
        if np.isfinite(max_score):
            stats["max_det_score_seen"] = max(stats["max_det_score_seen"], max_score)
    except Exception:
        pass

    if landmarks is None:
        stats["rejected_no_landmarks"] += len(boxes)
        return []

    frame_h, frame_w = frame.shape[:2]
    order = np.argsort(-boxes[:, 4])[:MAX_FACES_PER_FRAME]

    keep_boxes: List[Tuple[int, int, int, int]] = []
    keep_scores: List[float] = []
    keep_lms: List[np.ndarray] = []
    for i in order:
        if i >= len(landmarks):
            stats["rejected_no_landmarks"] += 1
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes[i][:4]]
        # Detector score can legitimately be 0 for some SCRFD wrappers; treat
        # a zero as "neutral" so we don't reject valid faces on bad metadata.
        score = float(boxes[i][4]) if boxes.shape[1] >= 5 else 1.0
        if not np.isfinite(score):
            score = 1.0

        if score < MIN_DET_SCORE:
            stats["rejected_low_det_score"] += 1
            continue
        if min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
            stats["rejected_small_faces"] += 1
            continue

        lm = landmarks[i]
        eye_dist = float(np.linalg.norm(lm[1] - lm[0]))
        if eye_dist < 2.0:
            stats["rejected_small_faces"] += 1
            continue
        yaw_ratio = abs(float(lm[2][0]) - float((lm[0][0] + lm[1][0]) / 2.0)) / eye_dist
        if yaw_ratio > MAX_YAW_RATIO:
            stats["rejected_pose"] += 1
            continue

        keep_boxes.append((
            int(max(0, x1)), int(max(0, y1)), int(min(frame_w, x2)), int(min(frame_h, y2)),
        ))
        keep_scores.append(score)
        keep_lms.append(lm)

    if not keep_lms:
        return []

    embeddings, blurs = _embed_aligned(frame, keep_lms)
    faces: List[_Face] = []
    for bbox, score, blur, feat in zip(keep_boxes, keep_scores, blurs, embeddings):
        if blur < MIN_BLUR_VARIANCE:
            stats["rejected_blurry"] += 1
            continue
        unit = _unit_vector(feat)
        if unit is None:
            stats["rejected_invalid_embeddings"] += 1
            continue
        faces.append(_Face(bbox, score, blur, unit))
    stats["faces_embedded"] += len(faces)
    return faces


# --------------------------------------------------------------------------- #
# Sightings: group above-threshold hits per person, confirm over time
# --------------------------------------------------------------------------- #
class _Sighting:
    __slots__ = (
        "gallery_idx", "first_t", "last_t", "total_hits", "counted", "confirmed",
        "confirm_start_t", "best_sim", "best_t", "best_frame", "best_bbox",
        "best_det_score", "best_blur", "frame_jpg", "face_jpg",
    )

    def __init__(self, gallery_idx: int, t: float):
        self.gallery_idx = gallery_idx
        self.first_t = t
        self.last_t = t
        self.total_hits = 0
        self.counted: List[float] = []
        self.confirmed = False
        self.confirm_start_t = t
        self.best_sim = -1.0
        self.best_t = t
        self.best_frame = 0
        self.best_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self.best_det_score = 0.0
        self.best_blur = 0.0
        self.frame_jpg: Optional[bytes] = None
        self.face_jpg: Optional[bytes] = None


def _encode_evidence(frame: np.ndarray, bbox, label: str) -> Tuple[Optional[bytes], Optional[bytes]]:
    x1, y1, x2, y2 = bbox
    annotated = frame.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(annotated, label, (max(0, x1), max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
    ok_frame, frame_buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    pad_x, pad_y = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
    ok_face, face_buf = (False, None)
    if crop.size:
        ok_face, face_buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return (frame_buf.tobytes() if ok_frame else None), (face_buf.tobytes() if ok_face else None)


class _SightingTracker:
    def __init__(self, gallery: _Gallery, threshold: float, confirm_hits: int, save_evidence: bool):
        self.gallery = gallery
        self.threshold = threshold
        self.confirm_hits = confirm_hits
        self.save_evidence = save_evidence
        self.active: Dict[int, _Sighting] = {}
        self.finished: List[_Sighting] = []

    def add_hit(self, gidx: int, t: float, sim: float, frame_idx: int, frame: np.ndarray, face: _Face) -> None:
        s = self.active.get(gidx)
        if s is not None and (t - s.last_t) > SIGHTING_GAP_SEC:
            self.finished.append(self.active.pop(gidx))
            s = None
        if s is None:
            s = _Sighting(gidx, t)
            self.active[gidx] = s

        s.last_t = t
        s.total_hits += 1
        if not s.counted or (t - s.counted[-1]) >= MIN_HIT_SPACING_SEC:
            s.counted.append(t)

        if not s.confirmed:
            recent = [x for x in s.counted if (t - x) <= CONFIRM_WINDOW_SEC]
            if len(recent) >= self.confirm_hits:
                s.confirmed = True
                s.confirm_start_t = recent[0]

        if sim > s.best_sim:
            s.best_sim = sim
            s.best_t = t
            s.best_frame = frame_idx
            s.best_bbox = face.bbox
            s.best_det_score = face.det_score
            s.best_blur = face.blur
            if self.save_evidence:
                pid = str(self.gallery.records[gidx].get("missing_person_id", ""))
                label = f"POSSIBLE MATCH - REVIEW {pid} {sim:.2f}"
                s.frame_jpg, s.face_jpg = _encode_evidence(frame, face.bbox, label)

    def close_all(self) -> Tuple[List[_Sighting], int]:
        self.finished.extend(self.active.values())
        self.active = {}
        confirmed = [s for s in self.finished if s.confirmed]
        unconfirmed = len(self.finished) - len(confirmed)
        confirmed.sort(key=lambda s: s.confirm_start_t)
        return confirmed, unconfirmed


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _ensure_event_indexes(collection) -> None:
    global _indexes_ready
    if _indexes_ready:
        return
    collection.create_index([("review_status", 1), ("created_at", -1)])
    collection.create_index([("camera_id", 1), ("created_at", -1)])
    collection.create_index([("candidate.missing_person_id", 1), ("created_at", -1)])
    _indexes_ready = True


def _save_evidence_files(s: _Sighting, person_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not (s.frame_jpg or s.face_jpg):
        return None, None
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"match_{_SAFE_NAME.sub('_', person_id)}_{uuid.uuid4().hex[:8]}"
    frame_path = face_path = None
    try:
        if s.frame_jpg:
            p = EVIDENCE_DIR / f"{stem}_frame.jpg"
            p.write_bytes(s.frame_jpg)
            frame_path = str(p)
        if s.face_jpg:
            p = EVIDENCE_DIR / f"{stem}_face.jpg"
            p.write_bytes(s.face_jpg)
            face_path = str(p)
    except OSError:
        logger.exception("Could not write CCTV evidence images")
    return frame_path, face_path


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def process_video_file(
    video_path: str,
    *,
    video_filename: Optional[str] = None,
    camera_id: str = "manual-upload",
    threshold: Optional[float] = None,
    frame_skip: int = 3,
    confirm_hits: Optional[int] = None,
    save_evidence: bool = True,
    reference_image_base64: Optional[str] = None,
    video_start_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Scan a video and report where/when registered (or reference) faces appear.

    frame_skip=3   -> analyse every 3rd frame.
    confirm_hits=2 -> a person needs 2 independent observations (>= MIN_HIT_SPACING_SEC
                      apart, inside CONFIRM_WINDOW_SEC) before an event is raised.
    video_start_time -> optional wall-clock time of video frame 0, used to compute
                      "absolute_time" for every event.
    The similarity score is NOT proof of identity; every event requires human review.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    video_size = os.path.getsize(video_path)
    if video_size > MAX_VIDEO_BYTES:
        raise ValueError(f"Video exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)} MiB limit")

    threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    frame_skip = int(frame_skip)
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    confirm_hits = max(1, int(confirm_hits or DEFAULT_CONFIRM_HITS))

    if face_register.detector is None or face_register.recognizer is None:
        face_register.load_face_models()

    # ---- gallery -------------------------------------------------------- #
    if reference_image_base64:
        gallery = _build_reference_gallery(reference_image_base64)
        match_mode = "reference_image_vs_video"
    else:
        gallery = _load_registered_gallery()
        match_mode = "registered_missing_persons_vs_video"

    # ---- video ---------------------------------------------------------- #
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open video file. Check the file format and codec.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0 or fps > 240:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_name = video_filename or Path(video_path).name

    logger.info(
        "CCTV scan start: video=%s mode=%s threshold=%.3f frame_skip=%s confirm_hits=%s "
        "det_size=%s upscale_range=[%.2f,%.2f] fps=%.2f frames=%s gallery=%s",
        video_name, match_mode, threshold, frame_skip, confirm_hits, DET_SIZE,
        MIN_UPSCALE, MAX_UPSCALE, fps, total_frames, gallery.size,
    )

    stats: Dict[str, Any] = {
        "frames_with_face_detections": 0,
        "faces_detected": 0,
        "faces_embedded": 0,
        "rejected_no_landmarks": 0,
        "rejected_low_det_score": 0,
        "rejected_small_faces": 0,
        "rejected_pose": 0,
        "rejected_blurry": 0,
        "rejected_invalid_embeddings": 0,
        "rejected_below_threshold": 0,
        "frames_decode_failed": 0,
        "frame_errors": 0,
        "max_det_score_seen": 0.0,
    }
    tracker = _SightingTracker(gallery, threshold, confirm_hits, save_evidence)
    best_seen_sim = -1.0
    best_seen_person: Optional[str] = None

    frame_index = 0
    processed_frames = 0
    dim_checked = False

    try:
        while True:
            if not cap.grab():
                break
            current = frame_index
            frame_index += 1
            if current % frame_skip != 0:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                stats["frames_decode_failed"] += 1
                continue
            processed_frames += 1
            t = current / fps

            try:
                faces = _extract_faces(frame, stats)
                if faces:
                    emb = np.stack([f.embedding for f in faces]).astype(np.float32)
                    if not dim_checked:
                        if emb.shape[1] != gallery.dimension:
                            raise RuntimeError(
                                f"Embedding dimension mismatch: video={emb.shape[1]} database={gallery.dimension}. "
                                "Registered embeddings were created with a different model."
                            )
                        dim_checked = True

                    sims = emb @ gallery.matrix.T            # (faces, people), cosine similarity
                    best_idx = sims.argmax(axis=1)
                    best_sim = sims[np.arange(len(faces)), best_idx]

                    for face, gidx, sim in zip(faces, best_idx, best_sim):
                        sim = float(sim)
                        gidx = int(gidx)
                        if sim > best_seen_sim:
                            best_seen_sim = sim
                            best_seen_person = str(gallery.records[gidx].get("missing_person_id"))
                        if sim < threshold:
                            stats["rejected_below_threshold"] += 1
                            continue
                        tracker.add_hit(gidx, t, sim, current, frame, face)
            except RuntimeError:
                raise
            except Exception:
                stats["frame_errors"] += 1
                logger.exception("CCTV frame processing failed at frame=%s", current)
                if stats["frame_errors"] > MAX_FRAME_ERRORS:
                    raise RuntimeError("Too many frame-processing errors; scan aborted")
                continue

            if processed_frames % PROGRESS_LOG_EVERY == 0:
                logger.info(
                    "CCTV progress: frame=%s/%s t=%s faces_embedded=%s best_sim=%.3f",
                    current, total_frames or "?", _fmt_ts(t), stats["faces_embedded"], best_seen_sim,
                )
    finally:
        cap.release()

    confirmed, unconfirmed = tracker.close_all()

    # ---- build events ---------------------------------------------------- #
    now = datetime.now(timezone.utc)
    events: List[Dict[str, Any]] = []
    for s in confirmed:
        record = gallery.records[s.gallery_idx]
        person_id = str(record.get("missing_person_id", ""))
        frame_path, face_path = _save_evidence_files(s, person_id) if save_evidence else (None, None)
        event: Dict[str, Any] = {
            "event_type": "possible_match",
            "review_status": "pending_review",
            "project_code": PROJECT_CODE,
            "module_code": MODULE_CODE,
            "camera_id": camera_id,
            "video_filename": video_name,
            "match_mode": match_mode,
            "candidate": _public_candidate(record, s.best_sim),
            "confidence_band": _confidence_band(s.best_sim, threshold),
            "similarity_threshold": threshold,
            "video_timestamp_seconds": round(s.best_t, 3),
            "video_timestamp": _fmt_ts(s.best_t),
            "first_seen_seconds": round(s.confirm_start_t, 3),
            "first_seen": _fmt_ts(s.confirm_start_t),
            "last_seen_seconds": round(s.last_t, 3),
            "last_seen": _fmt_ts(s.last_t),
            "frame_index": s.best_frame,
            "supporting_hits": s.total_hits,
            "independent_observations": len(s.counted),
            "face_detector_score": round(s.best_det_score, 4),
            "blur_variance": round(s.best_blur, 2),
            "bbox_xyxy": list(s.best_bbox),
            "evidence_path": frame_path,
            "face_crop_path": face_path,
            "human_verification_required": True,
            "created_at": now,
        }
        if video_start_time is not None:
            event["absolute_time"] = (video_start_time + timedelta(seconds=s.best_t)).isoformat()
            event["absolute_first_seen"] = (video_start_time + timedelta(seconds=s.confirm_start_t)).isoformat()
            event["absolute_last_seen"] = (video_start_time + timedelta(seconds=s.last_t)).isoformat()
        events.append(event)

    # ---- persist (never lose results if MongoDB write fails) ------------- #
    warnings: List[str] = []
    if events:
        try:
            event_collection = get_collection().database[EVENT_COLLECTION_NAME]
            _ensure_event_indexes(event_collection)
            for event in events:
                doc = dict(event)  # insert_one mutates its input with _id
                event_collection.insert_one(doc)
                event["event_id"] = str(doc["_id"])
                event["persisted"] = True
        except PyMongoError:
            logger.exception("Could not persist CCTV possible-match events")
            warnings.append("Matches were found but could not be saved to MongoDB; returned in this response only.")
            for event in events:
                event.setdefault("persisted", False)
    for event in events:
        event["created_at"] = event["created_at"].isoformat()
        logger.warning(
            "CCTV_MATCH_EVENT person_id=%s sim=%.4f at=%s event_id=%s",
            event["candidate"]["missing_person_id"], event["candidate"]["similarity_score"],
            event["video_timestamp"], event.get("event_id"),
        )

    # ---- per-person summary --------------------------------------------- #
    people: Dict[str, Dict[str, Any]] = {}
    for event in events:
        cand = event["candidate"]
        pid = cand["missing_person_id"]
        entry = people.setdefault(pid, {
            "missing_person_id": pid, "name": cand.get("name"),
            "best_similarity": 0.0, "sightings": 0, "first_seen": event["first_seen"],
            "first_seen_seconds": event["first_seen_seconds"],
        })
        entry["sightings"] += 1
        entry["best_similarity"] = max(entry["best_similarity"], cand["similarity_score"])
        if event["first_seen_seconds"] < entry["first_seen_seconds"]:
            entry["first_seen"] = event["first_seen"]
            entry["first_seen_seconds"] = event["first_seen_seconds"]
    people_found = sorted(people.values(), key=lambda p: p["best_similarity"], reverse=True)

    logger.info(
        "CCTV scan done: video=%s frames_read=%s processed=%s %s sightings_confirmed=%s unconfirmed=%s",
        video_name, frame_index, processed_frames, stats, len(events), unconfirmed,
    )

    return make_json_safe({
        "message": "CCTV video scan completed",
        "match_mode": match_mode,
        "reference_image_used": bool(reference_image_base64),
        "recognition_status": "possible_matches_found" if events else "no_possible_match",
        "video_filename": video_name,
        "camera_id": camera_id,
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "similarity_threshold": threshold,
        "confirm_hits": confirm_hits,
        "frame_skip": frame_skip,
        "detector_input_size": DET_SIZE,
        "upscale_range": [MIN_UPSCALE, MAX_UPSCALE],
        "video_fps": round(fps, 3),
        "video_duration_seconds": round(frame_index / fps, 3),
        "frames_read": frame_index,
        "frames_processed": processed_frames,
        "frames_with_face_detections": stats["frames_with_face_detections"],
        "faces_detected": stats["faces_detected"],
        "faces_embedded": stats["faces_embedded"],
        "max_detector_score_seen": round(stats["max_det_score_seen"], 4),
        "rejected_no_landmarks": stats["rejected_no_landmarks"],
        "rejected_low_det_score": stats["rejected_low_det_score"],
        "rejected_small_faces": stats["rejected_small_faces"],
        "rejected_pose": stats["rejected_pose"],
        "rejected_blurry": stats["rejected_blurry"],
        "rejected_invalid_embeddings": stats["rejected_invalid_embeddings"],
        "rejected_below_threshold": stats["rejected_below_threshold"],
        "unconfirmed_sightings": unconfirmed,
        "frames_decode_failed": stats["frames_decode_failed"],
        "frame_errors": stats["frame_errors"],
        "highest_similarity_seen": round(best_seen_sim, 4) if best_seen_sim >= 0 else None,
        "highest_similarity_person_id": best_seen_person,
        "registered_missing_people_compared": 0 if reference_image_base64 else gallery.size,
        "reference_embeddings_compared": 1 if reference_image_base64 else 0,
        "people_found": people_found,
        "possible_match_count": len(events),
        "possible_matches": events,
        "warnings": warnings,
        "identity_confirmation": "Not confirmed; authorized human review is required.",
    })