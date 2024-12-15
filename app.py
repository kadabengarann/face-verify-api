import os
import json
import base64
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import gradio as gr
from deepface import DeepFace
from gradio.routes import App as GradioApp
from uuid import uuid4

# Constants
UPLOAD_DIR = "uploads"
DEFAULT_MODEL = "Facenet"
DEFAULT_DIST = "cosine"
DEFAULT_DETECTOR = "ssd"

# Ensure uploads directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Environment Configurations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI()

# Request model
class ImageRequest(BaseModel):
    img1_base64: str
    img2_base64: str
    dist: str = DEFAULT_DIST
    model: str = DEFAULT_MODEL
    detector: str = DEFAULT_DETECTOR

# Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    try:
        body_data = json.loads(body.decode("utf-8"))
        # Log errors with truncated base64 fields
        for key in ["img1_base64", "img2_base64"]:
            if key in body_data:
                body_data[key] = f"{body_data[key][:20]}... [truncated]"
        logger.error(f"Validation error: {exc.errors()}")
        logger.error(f"Request body: {json.dumps(body_data)}")
    except Exception as e:
        logger.warning(f"Error decoding request body: {e}")
    return JSONResponse(status_code=400, content={"detail": exc.errors()})

# Utility Functions
def save_base64_image(base64_string: str, file_name: str) -> str:
    """
    Decodes a base64 string and saves it to a file.
    Returns the file path.
    """
    file_path = os.path.join(UPLOAD_DIR, file_name)
    with open(file_path, "wb") as f:
        f.write(base64.b64decode(base64_string))
    return file_path

def cleanup_files(*file_paths):
    """Deletes files from the file system."""
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

# Face Verification Endpoint
@app.post("/face_verification")
async def face_verification(request: ImageRequest):
    try:
        img1_path = save_base64_image(request.img1_base64, f"{uuid4()}_img1.png")
        img2_path = save_base64_image(request.img2_base64, f"{uuid4()}_img2.png")

        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            distance_metric=request.dist,
            model_name=request.model,
            detector_backend=request.detector,
            enforce_detection=False,
        )
        cleanup_files(img1_path, img2_path)

        return {
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": result["model"],
            "detector_backend": result["detector_backend"],
            "similarity_metric": result["similarity_metric"],
        }

    except Exception as e:
        logger.error(f"An error occurred during face verification: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during face verification.")

# Gradio Interface
def face_verification_ui(img1, img2, dist=DEFAULT_DIST, model=DEFAULT_MODEL, detector=DEFAULT_DETECTOR):
    try:
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            distance_metric=dist,
            model_name=model,
            detector_backend=detector,
            enforce_detection=False,
        )
        return {
            "Verified": result["verified"],
            "Distance": result["distance"],
            "Threshold": result["threshold"],
            "Model": result["model"],
            "Detector Backend": result["detector_backend"],
            "Similarity Metric": result["similarity_metric"],
        }
    except Exception as e:
        return {"Error": str(e)}

# Gradio Blocks
with gr.Blocks() as demo:
    with gr.Row():
        img1 = gr.Image(label="Image 1", sources=["upload", "webcam", "clipboard"])
        img2 = gr.Image(label="Image 2", sources=["upload", "webcam", "clipboard"])
    with gr.Row():
        dist = gr.Dropdown(choices=["cosine", "euclidean", "euclidean_l2"], label="Distance Metric", value=DEFAULT_DIST)
        model = gr.Dropdown(choices=["VGG-Face", "Facenet", "Facenet512", "ArcFace"], label="Model", value=DEFAULT_MODEL)
        detector = gr.Dropdown(choices=["opencv", "ssd", "mtcnn", "retinaface", "mediapipe"], label="Detector", value=DEFAULT_DETECTOR)
    with gr.Row():
        btn = gr.Button("Verify")
        output = gr.Textbox(label="Output")

    btn.click(face_verification_ui, inputs=[img1, img2, dist, model, detector], outputs=output)

# Running Server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)
