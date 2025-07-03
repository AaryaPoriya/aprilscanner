import cv2
import numpy as np
import os
import math
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pupil_apriltags import Detector

app = FastAPI(title="Lightning Fast AprilTag Detection API")

# --- Detector Setup ---
CPU_THREADS = os.cpu_count() or 2  # Use all available CPU cores
at_detector = Detector(families='tag36h11', nthreads=CPU_THREADS)

# --- Calibration Constants ---
REAL_TAG_SIZE_METERS = 0.10     # Example: 100mm tag size = 0.10 meters
FOCAL_LENGTH = 700              # Approximate focal length (adjust for accuracy)
MAX_IMAGE_WIDTH = 800           # Resize large images for speed


@app.post("/detect")
async def detect_apriltags(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"found": False, "error": "Invalid image"})

        # Resize for faster detection if needed
        height, width = img_bgr.shape[:2]
        if width > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / width
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        tags = at_detector.detect(img_gray)

        if not tags:
            return {"found": False, "tags": []}

        results = []

        for tag in tags:
            tag_id = tag.tag_id
            center_x, center_y = map(int, tag.center)

            tag_width_px = np.linalg.norm(tag.corners[0] - tag.corners[1])
            distance_m = None
            if tag_width_px > 0:
                distance_m = (REAL_TAG_SIZE_METERS * FOCAL_LENGTH) / tag_width_px

            rotation_matrix = tag.pose_R
            yaw_rad = math.atan2(-rotation_matrix[2, 0], rotation_matrix[0, 0])
            yaw_deg = math.degrees(yaw_rad)

            results.append({
                "tag_id": tag_id,
                "position": {"x": center_x, "y": center_y},
                "distance_m": round(distance_m, 3) if distance_m else None,
                "rotation_deg": round(yaw_deg, 2)
            })

        return {"found": True, "tags": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"found": False, "error": str(e)})

