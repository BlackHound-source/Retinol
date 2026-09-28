"""
Cloud synchronization and database connectors (MongoDB and Cloudinary).
"""

import os
import io
import datetime as dt
from PIL import Image
import numpy as np

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    import cloudinary
    import cloudinary.uploader
    CLOUDINARY_AVAILABLE = True
except ImportError:
    CLOUDINARY_AVAILABLE = False

# Default configurations (fallback to environment or Kaggle Secrets)
MONGO_URI = "your_mongodb_connection_string"
MONGO_DB_NAME = "retinal_screening"
CLOUDINARY_CLOUD_NAME = "your_cloud_name"
CLOUDINARY_API_KEY = "your_api_key"
CLOUDINARY_API_SECRET = "your_api_secret"
CLOUDINARY_FOLDER = "retinol_screenings"

try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    def _sec(name, default):
        try:
            val = _secrets.get_secret(name)
            return val if val else default
        except Exception:
            return default

    MONGO_URI = _sec("MONGO_URI", MONGO_URI)
    CLOUDINARY_CLOUD_NAME = _sec("CLOUDINARY_CLOUD_NAME", CLOUDINARY_CLOUD_NAME)
    CLOUDINARY_API_KEY = _sec("CLOUDINARY_API_KEY", CLOUDINARY_API_KEY)
    CLOUDINARY_API_SECRET = _sec("CLOUDINARY_API_SECRET", CLOUDINARY_API_SECRET)
except Exception:
    MONGO_URI = os.environ.get("MONGO_URI", MONGO_URI)
    CLOUDINARY_CLOUD_NAME = os.environ.get("CLOUDINARY_CLOUD_NAME", CLOUDINARY_CLOUD_NAME)
    CLOUDINARY_API_KEY = os.environ.get("CLOUDINARY_API_KEY", CLOUDINARY_API_KEY)
    CLOUDINARY_API_SECRET = os.environ.get("CLOUDINARY_API_SECRET", CLOUDINARY_API_SECRET)

mongo_client, mongo_db, patients_collection, screenings_collection = None, None, None, None

if PYMONGO_AVAILABLE and MONGO_URI and "your_mongodb_connection_string" not in MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        mongo_client.admin.command("ping")
        mongo_db = mongo_client[MONGO_DB_NAME]
        patients_collection = mongo_db["patients"]
        screenings_collection = mongo_db["screenings"]
    except Exception as e:
        print(f"[Storage] MongoDB connection skipped: {e}")

if CLOUDINARY_AVAILABLE and CLOUDINARY_CLOUD_NAME and "your_cloud_name" not in CLOUDINARY_CLOUD_NAME:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )

def execute_database_save(patient_id, name, age, gender, contact, state_image, state_record):
    """Persists patient record and screening report with uploaded image to Cloudinary & MongoDB."""
    if not patient_id or not str(patient_id).strip():
        return "✕ Action Denied: Valid Patient ID is mandatory."
    if state_image is None or state_record is None:
        return "✕ Action Denied: Perform screening before persisting records."
    if not CLOUDINARY_AVAILABLE or not screenings_collection:
        return "✕ Service Offline: Cloudinary/MongoDB configuration missing."

    try:
        # Prepare and upload fundus image
        pil_img = Image.fromarray(np.ascontiguousarray(state_image).astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        safe_pid = "".join(c for c in str(patient_id) if c.isalnum() or c in "-_") or "retinol_case"
        upload_resp = cloudinary.uploader.upload(
            buf,
            folder=CLOUDINARY_FOLDER,
            public_id=f"{safe_pid}_{int(dt.datetime.utcnow().timestamp())}",
            resource_type="image"
        )
        cloud_url = upload_resp.get("secure_url")

        # Find or create patient record
        patient = patients_collection.find_one({"patient_id": patient_id})
        if not patient:
            now = dt.datetime.utcnow()
            patient = {
                "patient_id": patient_id.strip(),
                "name": name.strip() if name else "Unspecified",
                "age": int(age) if age else None,
                "gender": gender if gender else "Unspecified",
                "contact": contact.strip() if contact else "N/A",
                "createdAt": now,
                "updatedAt": now
            }
            res_p = patients_collection.insert_one(patient)
            patient["_id"] = res_p.inserted_id

        # Insert diagnostic screening log
        screenings_collection.insert_one({
            "patient": patient["_id"],
            "image_url": cloud_url,
            "prediction": "DR" if int(state_record["grade"]) > 0 else "No_DR",
            "confidence": round(float(state_record["confidence"]) / 100.0, 4),
            "disease": "Diabetic Retinopathy",
            "grade": int(state_record["grade"]),
            "triage_status": state_record["triage_status"],
            "quality_status": "FLAGGED" if state_record["safety_flags"].get("ood_flag") else "VERIFIED",
            "createdAt": dt.datetime.utcnow(),
            "updatedAt": dt.datetime.utcnow()
        })
        return f"✓ Record Persisted: Patient {patient_id} synced to registry."
    except Exception as ex:
        return f"✕ Sync Failed: {str(ex)[:100]}"
