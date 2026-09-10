import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import io
import os
import logging
from pathlib import Path
from typing import Optional, Union
from unified_rag.config import settings

logger = logging.getLogger(__name__)

# Legacy local-storage convention this project used before Cloudinary was
# added (2026-04-12) — app/assistant_api.py's _absolute_image_urls() still
# maps a "data/..." path onto "{API_URL}/static/...", it just had nothing
# writing into data/ or mounting /static anymore. Reviving it here as the
# fallback for local/offline runs rather than inventing a new convention.
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

class CloudinaryService:
    def __init__(self):
        self.cloud_name = settings.cloudinary_cloud_name
        self.api_key = settings.cloudinary_api_key
        self.api_secret = settings.cloudinary_api_secret

        if all([self.cloud_name, self.api_key, self.api_secret]):
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True
            )
            self.enabled = True
        else:
            self.enabled = False
            LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
            logger.warning(
                "Cloudinary is not configured — falling back to local storage under %s "
                "(served at /static). Fine for local testing; set CLOUDINARY_* for real deployments.",
                LOCAL_DATA_DIR,
            )

    def upload_image(self, file_source: Union[str, bytes], public_id: str, folder: str = "phoenix/manuals") -> Optional[str]:
        return self.upload_file(file_source, public_id, folder, resource_type="image")

    def upload_file(self, file_source: Union[str, bytes], public_id: str, folder: str = "phoenix/data", resource_type: str = "raw") -> Optional[str]:
        """
        Generic upload method for Cloudinary.
        resource_type: 'image', 'video', or 'raw' (for PDFs, JSON, models)
        """
        if not self.enabled:
            return self._upload_local(file_source, public_id, folder, resource_type)

        try:
            upload_result = cloudinary.uploader.upload(
                file_source,
                public_id=public_id,
                folder=folder,
                overwrite=True,
                resource_type=resource_type
            )
            return upload_result.get("secure_url")
        except Exception as e:
            logger.error(f"Cloudinary upload failed for {public_id} ({resource_type}): {e}")
            return None

    def _upload_local(
        self, file_source: Union[str, bytes], public_id: str, folder: str, resource_type: str
    ) -> Optional[str]:
        """Offline stand-in for cloudinary.uploader.upload: writes under LOCAL_DATA_DIR
        and returns a "data/..." path, which _absolute_image_urls() turns into a real
        URL under the /static mount. Same signature/return shape as the Cloudinary path
        so callers (parser.py, endpoints.py) don't need to know which one ran."""
        try:
            if isinstance(file_source, (bytes, bytearray)):
                data = bytes(file_source)
            elif hasattr(file_source, "read"):
                pos = file_source.tell() if hasattr(file_source, "tell") else None
                data = file_source.read()
                if pos is not None:
                    file_source.seek(pos)
            elif isinstance(file_source, str) and os.path.exists(file_source):
                data = Path(file_source).read_bytes()
            else:
                logger.error(f"Local upload for {public_id}: unsupported file_source type {type(file_source)}")
                return None

            ext = ".png" if resource_type == "image" else (".pdf" if resource_type == "raw" else "")
            target = LOCAL_DATA_DIR / folder / f"{public_id}{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

            rel_path = target.relative_to(LOCAL_DATA_DIR.parent).as_posix()  # "data/<folder>/<name>.ext"
            return rel_path
        except Exception as e:
            logger.error(f"Local upload failed for {public_id} ({resource_type}): {e}")
            return None

    def delete_local_file(self, local_path: str):
        """Clean up local file after successful upload if requested."""
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"Cleaned up local file: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to delete local file {local_path}: {e}")
