"""
Modal GPU Agent - Image Generation + Drive Processing + General GPU Compute
"""

import base64
import hmac
import io
import json
import logging
import math
import mimetypes
import os
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import modal
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

app = modal.App("gpu-agent")
logger = logging.getLogger(__name__)

MAX_PROMPT_LENGTH = 2_000
MAX_DRIVE_FILES = 500
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 36_000_000
MAX_IMAGE_SIDE = 8_192
MAX_GENERATED_IMAGE_BYTES = 20 * 1024 * 1024
MAX_INFERENCE_STEPS = 20
MAX_GUIDANCE_SCALE = 10.0
MAX_SANDBOX_CODE_LENGTH = 200_000
MAX_SANDBOX_TIMEOUT = 600
MAX_SANDBOX_OUTPUT_BYTES = 10_000_000
MAX_SANDBOX_FILE_BYTES = 20_000_000
MAX_SANDBOX_TOTAL_FILE_BYTES = 50_000_000
MAX_SANDBOX_FILES = 50
endpoint_bearer = HTTPBearer(auto_error=False)
endpoint_auth_dependency = Depends(endpoint_bearer)
_pose_model = None

# NOTE: Full file is too large for this interface. Please apply the changes manually or I can give you a patch.
