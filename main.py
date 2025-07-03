import cv2
import numpy as np
import os
import math
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pupil_apriltags import Detector

app = FastAPI(title="Lightning Fast AprilTag Detection API")

CPU_THREADS = os.cpu_count() or 2
at_detector = Detector(families='tag36h11', nthreads=CPU_THREADS)

REAL_TAG_SIZE_METERS = 0.10     # 100mm
FOCAL_LENGTH_X = 700            # Calibrate for your camera
FOCAL_LENGTH_Y = 700
PRINCIPAL_POINT_X = 320         # Usually width/2
PRINCIPAL_POINT_Y = 240         # Usually height/2
MAX_IMAGE_WIDTH = 800


@app.post("/detect")
async def detect_apriltags(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return JSONResponse(status_code=400, content={"found": False, "error": "Invalid image"})

        height, width = img_bgr.shape[:2]
        if width > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / width
            img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            height, width = img_bgr.shape[:2]  # update dimensions after resizing

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Set proper camera parameters for pose estimation:
        camera_params = (FOCAL_LENGTH_X, FOCAL_LENGTH_Y, width / 2, height / 2)
        tags = at_detector.detect(img_gray, estimate_tag_pose=True,
                                  camera_params=camera_params,
                                  tag_size=REAL_TAG_SIZE_METERS)

        if not tags:
            return {"found": False, "tags": []}

        results = []

        for tag in tags:
            tag_id = tag.tag_id
            center_x, center_y = map(int, tag.center)

            tag_width_px = np.linalg.norm(tag.corners[0] - tag.corners[1])
            distance_m = None
            if tag_width_px > 0:
                distance_m = (REAL_TAG_SIZE_METERS * FOCAL_LENGTH_X) / tag_width_px

            if tag.pose_R is not None:
                rotation_matrix = tag.pose_R
                yaw_rad = math.atan2(-rotation_matrix[2, 0], rotation_matrix[0, 0])
                yaw_deg = math.degrees(yaw_rad)
            else:
                yaw_deg = None  # Safe fallback

            results.append({
                "tag_id": tag_id,
                "position": {"x": center_x, "y": center_y},
                "distance_m": round(distance_m, 3) if distance_m else None,
                "rotation_deg": round(yaw_deg, 2) if yaw_deg is not None else None
            })

        return {"found": True, "tags": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"found": False, "error": str(e)})
