"""MongoDB configuration and lifecycle for MaghMela KhoyaPaya."""
import logging
import os
from typing import Optional
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

# Load .env from the current project directory (existing shell variables take precedence).
load_dotenv()

# Prefer a complete URI if configured. Otherwise build it from MONGO_HOST/MONGO_PORT.
MONGO_HOST = os.getenv("MONGO_HOST", "MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_PORT", "MONGO_PORT")
MONGO_USERNAME = os.getenv("MONGO_USERNAME")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_AUTH_SOURCE = os.getenv("MONGO_AUTH_SOURCE", "admin")
MONGO_URL = os.getenv("MONGO_URL")

if not MONGO_URL:
    if MONGO_USERNAME and MONGO_PASSWORD:
        MONGO_URL = (
            f"mongodb://{quote_plus(MONGO_USERNAME)}:{quote_plus(MONGO_PASSWORD)}"
            f"@{MONGO_HOST}:{MONGO_PORT}/?authSource={quote_plus(MONGO_AUTH_SOURCE)}"
        )
    else:
        MONGO_URL = f"mongodb://{MONGO_HOST}:{MONGO_PORT}"

DB_NAME = os.getenv("MONGO_DB", os.getenv("DB_NAME", "servilience_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "missing_person_embeddings")
PROJECT_CODE = os.getenv("PROJECT_CODE", "MAGHMELA")
MODULE_CODE = os.getenv("MODULE_CODE", "KHOYAPAYA")

_client: Optional[MongoClient] = None
_db = None


def initialize_database() -> None:
    """Connect to MongoDB, verify connectivity, and create metadata indexes."""
    global _client, _db
    if _client is not None:
        return

    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # Standard metadata indexes; these are not vector-search indexes.
        collection.create_index([("missing_person_id", ASCENDING)], unique=True)
        collection.create_index([("project_code", ASCENDING), ("module_code", ASCENDING)])
        collection.create_index([("status", ASCENDING), ("created_at", DESCENDING)])
        collection.create_index([("centre_id", ASCENDING), ("status", ASCENDING)])
        collection.create_index([("name_normalized", ASCENDING)])
        collection.create_index([("cameraId", ASCENDING)])
        collection.create_index([("userId", ASCENDING)])
        collection.create_index([("dvrId", ASCENDING)])
    except Exception:
        client.close()
        logger.exception("MongoDB initialization failed")
        raise

    _client = client
    _db = db
    logger.info("MongoDB connected: database=%s collection=%s", DB_NAME, COLLECTION_NAME)


def get_collection() -> Collection:
    """Return the configured collection, requiring startup initialization first."""
    if _db is None:
        raise RuntimeError("MongoDB is not initialized. Call initialize_database() first.")
    return _db[COLLECTION_NAME]


def ping_database() -> bool:
    try:
        if _client is None:
            return False
        _client.admin.command("ping")
        return True
    except PyMongoError:
        return False


def close_database() -> None:
    global _client, _db
    if _client is not None:
        _client.close()
    _client = None
    _db = None
