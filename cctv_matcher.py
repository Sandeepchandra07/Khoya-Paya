"""CCTV/video face matching against existing KhoyaPaya missing-person records.

This module does NOT register people and does NOT save CCTV face embeddings.
It reads the existing active missing-person embeddings from MongoDB once per
video, tracks faces with BYTETracker, and emits possible-match review events.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from pymongo.errors import PyMongoError

import face_register
from database import MODULE_CODE, PROJECT_CODE, get_collection
from face_tracking.tracker.byte_tracker import BYTETracker

logger = logging.getLogger(__name__)

TRACKING_CONFIG = Path(os.getenv("FACE_TRACKING_CONFIG", "face_tracking/config/config_tracking.yaml"))
EVIDENCE_DIR = Path(os.getenv("CCTV_EVIDENCE_DIR", "uploads/cctv_matches"))
DEFAULT_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.45"))
MIN_FACE_SIZE = int(os.getenv("CCTV_MIN_FACE_SIZE", "40"))
MIN_BLUR_VARIANCE = float(os.getenv("CCTV_MIN_BLUR_VARIANCE", "20"))
DEFAULT_CONFIRM_HITS = int(os.getenv("CCTV_CONFIRM_HITS", "2"))
DEFAULT_EMBED_INTERVAL = int(os.getenv("CCTV_EMBED_INTERVAL_FRAMES", "10"))
MAX_VIDEO_BYTES = int(os.getenv("CCTV_MAX_VIDEO_BYTES", str(500 * 1024 * 1024)))


def _unit_vector(value: Any) -> Optional[np.ndarray]:
    try:
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if not vec.size or not np.all(np.isfinite(vec)):
        return None
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return None
    return vec / norm


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _load_tracking_config() -> Dict[str, Any]:
    if not TRACKING_CONFIG.is_file():
        raise FileNotFoundError(f"Tracking config not found: {TRACKING_CONFIG}")
    with TRACKING_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = {"track_thresh", "track_buffer", "match_thresh", "min_box_area", "aspect_ratio_thresh"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Tracking config is missing keys: {', '.join(sorted(missing))}")
    return config


def _load_missing_embeddings() -> List[Dict[str, Any]]:
    """Load active cases once; only metadata and embeddings are kept in memory."""
    query = {
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "status": "missing",
        "is_active": True,
    }
    projection = {
        "missing_person_id": 1, "name": 1, "age": 1, "gender": 1,
        "last_seen_location": 1, "missing_since": 1, "centre_id": 1,
        "face_embedding": 1, "embedding_dimension": 1, "status": 1,
    }
    records: List[Dict[str, Any]] = []
    try:
        for record in get_collection().find(query, projection):
            vector = _unit_vector(record.get("face_embedding"))
            if vector is None:
                continue
            record["_unit_embedding"] = vector
            records.append(record)
    except PyMongoError as exc:
        logger.exception("Could not load missing-person embeddings")
        raise RuntimeError("Could not load registered missing-person embeddings from MongoDB") from exc
    return records


def _best_candidate(embedding: np.ndarray, records: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], float]]:
    best_record = None
    best_score = -1.0
    for record in records:
        stored = record["_unit_embedding"]
        if stored.size != embedding.size:
            continue
        score = float(np.dot(embedding, stored))
        if score > best_score:
            best_record, best_score = record, score
    if best_record is None:
        return None
    return best_record, best_score


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


def process_video_file(
    video_path: str,
    *,
    video_filename: Optional[str] = None,
    camera_id: str = "manual-upload",
    threshold: Optional[float] = None,
    frame_skip: int = 3,
    confirm_hits: Optional[int] = None,
    embedding_interval_frames: Optional[int] = None,
    save_evidence: bool = True,
) -> Dict[str, Any]:
    """Scan an uploaded video and return possible matches for human review.

    `frame_skip=3` means detect on every third frame. A track must produce
    `confirm_hits` above-threshold observations before an event is created.
    The similarity threshold is an initial operating value, not a calibrated
    identity guarantee; validate it on representative footage before use.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    confirm_hits = max(1, int(confirm_hits or DEFAULT_CONFIRM_HITS))
    embedding_interval_frames = max(1, int(embedding_interval_frames or DEFAULT_EMBED_INTERVAL))

    if face_register.detector is None or face_register.recognizer is None:
        face_register.load_face_models()

    config = _load_tracking_config()
    records = _load_missing_embeddings()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video file. Check the file format and codec.")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    tracker = BYTETracker(args=config, frame_rate=max(1, int(fps / frame_skip)))

    frame_index = 0
    processed_frames = 0
    detected_tracks_seen = set()
    track_observations: Dict[int, int] = {}
    last_embedding_frame: Dict[int, int] = {}
    votes: Dict[int, Dict[str, int]] = {}
    emitted_tracks = set()
    events: List[Dict[str, Any]] = []
    video_name = video_filename or Path(video_path).name

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_frame = frame_index
            frame_index += 1
            if current_frame % frame_skip != 0:
                continue
            processed_frames += 1

            outputs, img_info, _bboxes, _landmarks = face_register.detector.detect_tracking(image=frame)
            if outputs is None or len(outputs) == 0:
                # No detections on this sampled frame; continue to next frame.
                continue

            online_targets = tracker.update(
                outputs,
                [img_info["height"], img_info["width"]],
                (128, 128),
            )

            for target in online_targets:
                track_id = int(target.track_id)
                tlwh = target.tlwh
                x, y, w, h = [float(v) for v in tlwh]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                frame_h, frame_w = frame.shape[:2]
                pad_x, pad_y = int(max(w, 1) * 0.25), int(max(h, 1) * 0.25)
                x1p, y1p = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2p, y2p = min(frame_w, x2 + pad_x), min(frame_h, y2 + pad_y)
                if x2p <= x1p or y2p <= y1p:
                    continue
                face_crop = frame[y1p:y2p, x1p:x2p]
                if face_crop.size == 0 or face_crop.shape[1] < MIN_FACE_SIZE or face_crop.shape[0] < MIN_FACE_SIZE:
                    continue

                track_observations[track_id] = track_observations.get(track_id, 0) + 1
                detected_tracks_seen.add(track_id)
                if track_id in emitted_tracks:
                    continue
                if current_frame - last_embedding_frame.get(track_id, -embedding_interval_frames) < embedding_interval_frames:
                    continue
                last_embedding_frame[track_id] = current_frame

                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                if blur_variance < MIN_BLUR_VARIANCE:
                    continue

                try:
                    embedding, detector_score = face_register.extract_embedding(face_crop)
                except Exception:
                    logger.exception("Embedding extraction failed for track_id=%s", track_id)
                    continue
                if embedding is None or float(detector_score) < float(config.get("track_thresh", 0.5)):
                    continue
                unit = _unit_vector(embedding)
                if unit is None:
                    continue

                candidate = _best_candidate(unit, records)
                if candidate is None:
                    continue
                record, similarity = candidate
                person_id = str(record.get("missing_person_id", ""))
                if not person_id or similarity < threshold:
                    continue

                votes.setdefault(track_id, {})[person_id] = votes.setdefault(track_id, {}).get(person_id, 0) + 1
                if votes[track_id][person_id] < confirm_hits:
                    continue

                # Use the same candidate whose repeated observations passed the vote threshold.
                match_record = record
                match_similarity = similarity
                match_detector_score = float(detector_score)
                match_blur_variance = blur_variance
                match_bbox = [x1, y1, x2, y2]
                event_time_seconds = round(current_frame / fps, 3)
                evidence_path = None
                if save_evidence:
                    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
                    evidence_name = f"possible_match_{person_id}_track{track_id}_{current_frame}.jpg"
                    evidence_file = EVIDENCE_DIR / evidence_name
                    annotated = frame.copy()
                    bx1, by1, bx2, by2 = match_bbox
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 165, 255), 2)
                    cv2.putText(annotated, "POSSIBLE MATCH - REVIEW", (max(0, bx1), max(25, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                    if cv2.imwrite(str(evidence_file), annotated):
                        evidence_path = str(evidence_file)

                event = {
                    "event_type": "possible_match",
                    "review_status": "pending_review",
                    "project_code": PROJECT_CODE,
                    "module_code": MODULE_CODE,
                    "camera_id": camera_id,
                    "video_filename": video_name,
                    "video_timestamp_seconds": event_time_seconds,
                    "frame_index": current_frame,
                    "track_id": track_id,
                    "candidate": _public_candidate(match_record, float(match_similarity)),
                    "similarity_threshold": threshold,
                    "face_detector_score": round(match_detector_score, 4),
                    "blur_variance": round(match_blur_variance, 2),
                    "bbox_xyxy": match_bbox,
                    "evidence_path": evidence_path,
                    "human_verification_required": True,
                    "created_at": datetime.now(timezone.utc),
                }
                # Persist an event separately from person registration; never save the CCTV embedding.
                try:
                    event_collection = get_collection().database["cctv_possible_match_events"]
                    event_collection.create_index([("review_status", 1), ("created_at", -1)])
                    event_collection.create_index([("camera_id", 1), ("created_at", -1)])
                    insert_result = event_collection.insert_one(event)
                    event["event_id"] = str(insert_result.inserted_id)
                except PyMongoError as exc:
                    logger.exception("Could not persist CCTV possible-match event")
                    raise RuntimeError("Match was detected but the possible-match event could not be saved") from exc

                event["created_at"] = event["created_at"].isoformat()
                events.append(event)
                emitted_tracks.add(track_id)
    finally:
        cap.release()

    return {
        "message": "CCTV video scan completed",
        "recognition_status": "possible_matches_found" if events else "no_possible_match", 
        "video_filename": video_name,
        "camera_id": camera_id,
        "project_code": PROJECT_CODE,
        "module_code": MODULE_CODE,
        "similarity_threshold": threshold,
        "frame_skip": frame_skip,
        "video_fps": round(fps, 3),
        "frames_read": frame_index,
        "frames_processed": processed_frames,
        "tracks_seen": len(detected_tracks_seen),
        "registered_missing_people_compared": len(records),
        "possible_match_count": len(events),
        "possible_matches": events,
        "identity_confirmation": "Not confirmed; authorized human review is required.",
    }
