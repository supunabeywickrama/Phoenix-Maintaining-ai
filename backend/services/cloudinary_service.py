import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import os
import logging
from typing import Optional, Union
from unified_rag.config import settings

logger = logging.getLogger(__name__)

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
            logger.warning("Cloudinary is not configured. Media storage will be local-only or fail.")

    def upload_image(self, file_source: Union[str, bytes], public_id: str, folder: str = "phoenix/manuals") -> Optional[str]:
        return self.upload_file(file_source, public_id, folder, resource_type="image")

    def upload_file(self, file_source: Union[str, bytes], public_id: str, folder: str = "phoenix/data", resource_type: str = "raw") -> Optional[str]:
        """
        Generic upload method for Cloudinary.
        resource_type: 'image', 'video', or 'raw' (for PDFs, JSON, models)
        """
        if not self.enabled:
            return None
        
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

    def delete_local_file(self, local_path: str):
        """Clean up local file after successful upload if requested."""
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"Cleaned up local file: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to delete local file {local_path}: {e}")
