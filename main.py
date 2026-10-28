"""FastAPI routes for MaghMela KhoyaPaya registration, recognition and CCTV scan."""
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from database import (
    COLLECTION_NAME, DB_NAME, MODULE_CODE, PROJECT_CODE, close_database,
    get_collection, initialize_database, ping_database,
)
from face_register import (
    DEVICE, DuplicateRegistrationError, FaceNotDetectedError, load_face_models,
    register_missing_person as register_missing_person_logic,
)
from face_recognize import recognize_missing_person
from cctv_matcher import MAX_VIDEO_BYTES, process_video_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("khoyapaya-api")
app = FastAPI(
    title="MaghMela KhoyaPaya API",
    description="Missing-person registration, image recognition, CCTV video matching and case management API.",
    version="1.2.0",
)


class MissingPersonRegistration(BaseModel):
    missing_person_id: Optional[str] = Field(default=None, min_length=1, max_length=80)
    name: Optional[str] = Field(default=None, max_length=150)
    age: Optional[int] = Field(default=None, ge=0, le=125)
    gender: Optional[str] = Field(default=None, max_length=30)
    description: Optional[str] = Field(default=None, max_length=2000)
    last_seen_location: Optional[str] = Field(default=None, max_length=500)
    missing_since: Optional[datetime] = None
    centre_id: Optional[str] = Field(default=None, max_length=100)
    cameraId: Optional[str] = Field(default=None, max_length=24)
    userId: Optional[str] = Field(default=None, max_length=24)
    dvrId: Optional[str] = Field(default=None, max_length=24)
    registered_by: str = Field(..., min_length=1, max_length=100)
    registered_by_role: str = Field(..., min_length=1, max_length=80)
    images: List[str] = Field(..., min_length=1, max_length=10)


class FaceRecognitionRequest(BaseModel):
    image: str = Field(..., min_length=1, description="Base64 image or data:image/...;base64,... URL")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class StatusUpdate(BaseModel):
    status: str = Field(..., description="missing, found, reunited, or inactive")


@app.on_event("startup")
def startup_event() -> None:
    initialize_database()
    try:
        load_face_models()
    except Exception:
        close_database()
        logger.exception("Face model initialization failed")
        raise


@app.on_event("shutdown")
def shutdown_event() -> None:
    close_database()


@app.get("/")
def root():
    return {"service": "MaghMela KhoyaPaya API", "status": "running", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    try:
        import face_register
        models_ok = face_register.detector is not None and face_register.recognizer is not None
    except Exception:
        logger.exception("Could not inspect face model state")
        models_ok = False
    mongo_ok = ping_database()
    return {
        "status": "healthy" if mongo_ok and models_ok else "degraded",
        "database": DB_NAME, "collection": COLLECTION_NAME,
        "mongo_connected": mongo_ok, "models_loaded": models_ok, "device": str(DEVICE),
    }


@app.post("/missing-persons/register", status_code=status.HTTP_201_CREATED)
def register_missing_person(request: MissingPersonRegistration):
    try:
        return register_missing_person_logic(request)
    except FaceNotDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DuplicateRegistrationError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Registration runtime failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected registration error")
        raise HTTPException(status_code=500, detail="Registration failed") from exc


@app.post("/missing-persons/recognize")
def recognize_person(request: FaceRecognitionRequest):
    try:
        return recognize_missing_person(request.image, request.threshold)
    except FaceNotDetectedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Recognition runtime failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected recognition error")
        raise HTTPException(status_code=500, detail="Face recognition failed") from exc


@app.post("/cctv/match-video")
def cctv_match_video(
    video: UploadFile = File(..., description="Recorded CCTV video file"),
    camera_id: str = Form(default="manual-upload", min_length=1, max_length=100),
    threshold: float = Form(default=0.45, ge=0.0, le=1.0),
    frame_skip: int = Form(default=3, ge=1, le=60),
    confirm_hits: int = Form(default=2, ge=1, le=10),
):
    """Scan a recorded video; does not register people or store CCTV embeddings."""
    suffix = Path(video.filename or "upload.mp4").suffix.lower()
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm"}
    if suffix not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension: {suffix or '(none)'}")

    temp_path = None
    try:
        total = 0
        with tempfile.NamedTemporaryFile(prefix="khoyapaya_cctv_", suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = video.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail=f"Video exceeds {MAX_VIDEO_BYTES // (1024 * 1024)} MB limit")
                temp_file.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty")
        return process_video_file(
            temp_path,
            video_filename=Path(video.filename or "uploaded-video").name,
            camera_id=camera_id,
            threshold=threshold,
            frame_skip=frame_skip,
            confirm_hits=confirm_hits,
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("CCTV matching runtime failure")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected CCTV matching error")
        raise HTTPException(status_code=500, detail="CCTV matching failed") from exc
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Could not remove temporary uploaded video: %s", temp_path)
        try:
            video.file.close()
        except Exception:
            pass


@app.get("/missing-persons/{missing_person_id}")
def get_missing_person(missing_person_id: str):
    try:
        record = get_collection().find_one(
            {"missing_person_id": missing_person_id, "project_code": PROJECT_CODE, "module_code": MODULE_CODE},
            {"face_embedding": 0},
        )
    except PyMongoError as exc:
        logger.exception("MongoDB read failed")
        raise HTTPException(status_code=500, detail="Could not read missing-person record") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Missing person not found")
    record["_id"] = str(record["_id"])
    for field in ("cameraId", "userId", "dvrId"):
        if record.get(field) is not None:
            record[field] = str(record[field])
    for field in ("created_at", "updated_at", "missing_since", "resolved_at"):
        if isinstance(record.get(field), datetime):
            record[field] = record[field].isoformat()
    return record


@app.patch("/missing-persons/{missing_person_id}/status")
def update_missing_person_status(missing_person_id: str, body: StatusUpdate):
    allowed = {"missing", "found", "reunited", "inactive"}
    if body.status not in allowed:
        raise HTTPException(status_code=400, detail=f"status must be one of: {', '.join(sorted(allowed))}")
    now = datetime.now().astimezone()
    update_fields = {"status": body.status, "updated_at": now}
    update_fields["resolved_at"] = now if body.status in {"found", "reunited", "inactive"} else None
    try:
        result = get_collection().update_one(
            {"missing_person_id": missing_person_id, "project_code": PROJECT_CODE, "module_code": MODULE_CODE},
            {"$set": update_fields},
        )
    except PyMongoError as exc:
        logger.exception("MongoDB status update failed")
        raise HTTPException(status_code=500, detail="Could not update case status") from exc
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Missing person not found")
    return {"message": "Case status updated", "missing_person_id": missing_person_id, "status": body.status}
