"""Camera registry + RTSP snapshot helpers for MaghMela KhoyaPaya.

Reads cameras from `servilience_db.cameras` (same MongoDB server, different DB).
Provides:
  - list_cameras()             : all cameras (with connection filtering)
  - get_camera(camera_id)      : single camera by _id / code / streamPath
  - grab_frame(camera)         : one BGR frame from an RTSP camera (fast timeout)
  - snapshot_and_recognize()   : grab + match against registered missing persons
"""
from __future__ import annotations

import logging
import os
import socket
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from bson import ObjectId
from pymongo.errors import PyMongoError

import face_register
from face_recognize import recognize_missing_person
from database import get_collection

logger = logging.getLogger(__name__)

# servilience_db (camera registry) — same MongoDB server as KhoyaPaya, different DB
SERVILIENCE_DB_NAME = os.getenv("SERVILIENCE_DB_NAME", "servilience_db")
CAMERA_COLLECTION = os.getenv("CAMERA_COLLECTION", "cameras")
RTSP_OPEN_TIMEOUT_MS = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "5000"))
RTSP_READ_TIMEOUT_MS = int(os.getenv("RTSP_READ_TIMEOUT_MS", "5000"))
RTSP_GRAB_ATTEMPTS = int(os.getenv("RTSP_GRAB_ATTEMPTS", "3"))
SNAPSHOT_CACHE_SEC = int(os.getenv("CAMERA_SNAPSHOT_CACHE_SEC", "2"))


# --------------------------------------------------------------------------- #
# Camera registry access
# --------------------------------------------------------------------------- #
def _camera_collection():
    """Return the cameras collection from servilience_db (same MongoDB client)."""
    base = get_collection()  # KhoyaPaya collection -> has .database.client
    client = base.database.client
    return client[SERVILIENCE_DB_NAME][CAMERA_COLLECTION]


def _public_camera(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a camera document into a JSON-friendly dict (hide credentials)."""
    rtsp = doc.get("rtspUrl") or ""
    # Redact basic-auth credentials from any URL shown to the frontend
    safe_rtsp = _redact_rtsp(rtsp)
    return {
        "camera_id": str(doc["_id"]) if doc.get("_id") is not None else None,
        "name": doc.get("name"),
        "code": doc.get("code"),
        "location": doc.get("location"),
        "camera_type": doc.get("cameraType"),
        "stream_path": doc.get("streamPath"),
        "rtsp_url": safe_rtsp,
        "is_active": bool(doc.get("isActive", False)),
        "is_online": bool(doc.get("isOnline", False)),
        "offline_reason": doc.get("offlineReason") or "",
        "last_checked": _iso(doc.get("lastChecked")),
        "last_online": _iso(doc.get("lastOnline")),
        "people_count": int(doc.get("peopleCount", 0) or 0),
        "dvr_id": str(doc["dvrId"]) if doc.get("dvrId") else None,
        "user_id": str(doc["userId"]) if doc.get("userId") else None,
        "asset_id": doc.get("assetId") or "",
        "created_at": _iso(doc.get("createdAt")),
        "updated_at": _iso(doc.get("updatedAt")),
    }


def _redact_rtsp(url: str) -> str:
    """Return RTSP URL with user:pass@ stripped to user:***@."""
    if not url or "://" not in url:
        return url
    try:
        scheme, rest = url.split("://", 1)
        if "@" not in rest:
            return url
        credentials, host_part = rest.rsplit("@", 1)
        if ":" in credentials:
            user, _ = credentials.split(":", 1)
            return f"{scheme}://{user}:***@{host_part}"
        return f"{scheme}://***@{host_part}"
    except Exception:
        return url


def _iso(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def list_cameras(
    *,
    only_active: bool = True,
    only_online: bool = False,
    camera_type: Optional[str] = None,
    location_contains: Optional[str] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Return camera registry entries from servilience_db."""
    query: Dict[str, Any] = {}
    if only_active:
        query["isActive"] = True
    if only_online:
        query["isOnline"] = True
    if camera_type:
        query["cameraType"] = camera_type
    if location_contains:
        query["location"] = {"$regex": location_contains, "$options": "i"}

    try:
        cursor = _camera_collection().find(query).sort("name", 1).limit(max(1, limit))
        return [_public_camera(doc) for doc in cursor]
    except PyMongoError as exc:
        logger.exception("Could not list cameras")
        raise RuntimeError("Could not read camera registry") from exc


def _parse_camera_lookup(value: str) -> Dict[str, Any]:
    """Accept ObjectId string, camera code, or streamPath."""
    if ObjectId.is_valid(value):
        return {"_id": ObjectId(value)}
    return {"$or": [{"code": value}, {"streamPath": value}]}


def get_camera(camera_ref: str) -> Optional[Dict[str, Any]]:
    """Lookup a single camera by _id, code, or streamPath."""
    if not camera_ref:
        return None
    try:
        doc = _camera_collection().find_one(_parse_camera_lookup(camera_ref))
    except PyMongoError as exc:
        logger.exception("Could not read camera %s", camera_ref)
        raise RuntimeError("Could not read camera registry") from exc
    if doc is None:
        return None
    return {
        "_raw": doc,                # keep original for internal use (has rtspUrl)
        "public": _public_camera(doc),
    }


# --------------------------------------------------------------------------- #
# RTSP frame grab
# --------------------------------------------------------------------------- #
_snapshot_cache: Dict[str, Tuple[float, bytes]] = {}  # camera_id -> (ts, jpg_bytes)


def _cap_open(rtsp_url: str) -> cv2.VideoCapture:
    """Open an RTSP stream with fast timeouts and TCP transport."""
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS,
    ]
    # Try TCP transport first (more reliable for DVRs), then UDP
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG, params)
    if cap.isOpened():
        return cap

    # Some builds do not accept extra params cleanly; retry with plain call
    try:
        cap.release()
    except Exception:
        pass
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(rtsp_url)


def grab_frame(rtsp_url: str) -> np.ndarray:
    """Grab a single BGR frame from an RTSP stream, or raise RuntimeError."""
    if not rtsp_url:
        raise RuntimeError("Camera has no RTSP URL")

    last_err: Optional[Exception] = None
    for attempt in range(1, RTSP_GRAB_ATTEMPTS + 1):
        cap = _cap_open(rtsp_url)
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            last_err = RuntimeError(
                "Could not open RTSP stream (is the camera online?)"
            )
            time.sleep(0.4 * attempt)
            continue

        try:
            # Warm up: read a few frames so codec + buffering stabilise.
            # Some DVRs need 3-5 reads before the first usable frame.
            frame = None
            for _ in range(5):
                ok, buf = cap.read()
                if ok and buf is not None and buf.size > 0:
                    frame = buf
            if frame is None:
                last_err = RuntimeError("RTSP stream opened but no frames arrived")
                continue
            return frame
        finally:
            try:
                cap.release()
            except Exception:
                pass

    raise last_err or RuntimeError("Could not grab a frame from the camera")


def grab_frame_cached(rtsp_url: str, cache_key: str) -> Tuple[np.ndarray, bool]:
    """Return a recent frame if cached within SNAPSHOT_CACHE_SEC, else fresh grab."""
    now = time.time()
    cached = _snapshot_cache.get(cache_key)
    if cached and (now - cached[0]) <= SNAPSHOT_CACHE_SEC:
        arr = np.frombuffer(cached[1], dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img, True

    img = grab_frame(rtsp_url)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if ok:
        _snapshot_cache[cache_key] = (now, buf.tobytes())
    return img, False


# --------------------------------------------------------------------------- #
# Snapshot + recognize
# --------------------------------------------------------------------------- #
def _frame_to_base64_jpeg(frame: np.ndarray) -> str:
    import base64
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Could not encode camera frame to JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def snapshot_and_recognize(
    camera_ref: str,
    *,
    threshold: Optional[float] = None,
    persist_events: bool = True,
) -> Dict[str, Any]:
    """Grab one frame from the camera, run recognition, and optionally persist."""
    cam = get_camera(camera_ref)
    if cam is None:
        raise ValueError(f"Camera not found: {camera_ref}")

    raw = cam["_raw"]
    public = cam["public"]
    rtsp_url = raw.get("rtspUrl")
    if not rtsp_url:
        raise RuntimeError(f"Camera {camera_ref} has no RTSP URL configured")

    started = time.time()
    frame, was_cached = grab_frame_cached(rtsp_url, public["camera_id"] or camera_ref)
    grab_ms = int((time.time() - started) * 1000)

    img_b64 = _frame_to_base64_jpeg(frame)
    try:
        result = recognize_missing_person(img_b64, threshold=threshold)
    except Exception as exc:
        logger.exception("Recognition failed for camera %s", camera_ref)
        raise RuntimeError(f"Recognition failed: {exc}") from exc

    now = datetime.now(timezone.utc)
    response: Dict[str, Any] = {
        "camera": public,
        "captured_at": now.isoformat(),
        "grab_duration_ms": grab_ms,
        "used_cached_frame": was_cached,
        "frame_size": {"width": int(frame.shape[1]), "height": int(frame.shape[0])},
        "recognition": result,
        "event_persisted": False,
    }

    if persist_events and result.get("match_found"):
        try:
            event = {
                "event_type": "camera_face_match",
                "review_status": "pending_review",
                "project_code": result.get("project_code"),
                "module_code": result.get("module_code"),
                "camera_id": public["camera_id"],
                "camera_code": public["code"],
                "camera_name": public["name"],
                "camera_location": public["location"],
                "captured_at": now,
                "matched_person": result.get("matched_person"),
                "similarity_score": result["matched_person"]["similarity_score"] if result.get("matched_person") else None,
                "similarity_threshold": result.get("similarity_threshold"),
                "top_candidates": result.get("top_candidates") or [],
                "human_verification_required": True,
            }
            event_coll = get_collection().database["camera_face_match_events"]
            event_coll.create_index([("review_status", 1), ("captured_at", -1)])
            event_coll.create_index([("camera_id", 1), ("captured_at", -1)])
            event_coll.create_index([("matched_person.missing_person_id", 1), ("captured_at", -1)])
            insert_result = event_coll.insert_one(dict(event))
            response["event_id"] = str(insert_result.inserted_id)
            response["event_persisted"] = True
        except PyMongoError:
            logger.exception("Could not persist camera face-match event")
            response["event_persist_warning"] = (
                "Match was detected but could not be saved to MongoDB."
            )

    return response


# --------------------------------------------------------------------------- #
# Continuous monitor (optional background thread)
# --------------------------------------------------------------------------- #
def monitor_camera_once(
    camera_ref: str,
    *,
    threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Monitor helper: returns match event if found, else None. Never raises."""
    try:
        result = snapshot_and_recognize(
            camera_ref, threshold=threshold, persist_events=True,
        )
        if result.get("recognition", {}).get("match_found"):
            return result
        return None
    except Exception:
        logger.exception("Camera monitor iteration failed for %s", camera_ref)
        return None