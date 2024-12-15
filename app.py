from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
import gradio as gr
from deepface import DeepFace
import os
import base64
from gradio.routes import App as GradioApp
import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# FastAPI instance
app = FastAPI()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(f"Validation error: {exc.errors()}")
    try:
        logger.error(f"Request body: {body.decode('utf-8')}")
    except UnicodeDecodeError:
        logger.warning("Request body contains non-text data and could not be decoded.")
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    ) 

@app.middleware("http")
async def log_request_body(request: Request, call_next):
    # Check the Content-Type of the request
    content_type = request.headers.get('Content-Type', '')
    
    if 'multipart/form-data' in content_type:
        # For binary data, log a placeholder message
        logger.info(f"Incoming request: {request.method} {request.url}")
        logger.info("Request body contains binary data and is not logged.")
    else:
        # For non-binary data, log the actual body
        body = await request.body()
        logger.info(f"Incoming request: {request.method} {request.url}")
        logger.info(f"Request body: {body.decode('utf-8')}")
    
    response = await call_next(request)
    return response

# Gradio Interface Function
def face_verification_uii(img1, img2, dist="cosine", model="Facenet", detector="ssd"):
    """
    Gradio function for face verification
    """
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
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": result["model"],
            "detector_backend": result["detector_backend"],
            "similarity_metric": result["similarity_metric"],
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/face_verification")
async def face_verification(
    img1: UploadFile = File(None),
    img2: UploadFile = File(None),
    img1_base64: str = Form(None),
    img2_base64: str = Form(None),
    dist: str = Form("cosine"),
    model: str = Form("Facenet"),
    detector: str = Form("ssd")
):
    if not img1 and not img2 and not img1_base64 and not img2_base64:
        raise HTTPException(status_code=400, detail="Invalid input: At least one image input is required.")

    try:
        # Ensure uploads directory exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        img1_path = None
        img2_path = None

        # Process img1
        if img1:
            img1_path = os.path.join("uploads", img1.filename)
            with open(img1_path, "wb") as f:
                f.write(await img1.read())
        elif img1_base64:
            img1_path = os.path.join("uploads", "img1_base64.png")
            with open(img1_path, "wb") as f:
                f.write(base64.b64decode(img1_base64))

        # Process img2
        if img2:
            img2_path = os.path.join("uploads", img2.filename)
            with open(img2_path, "wb") as f:
                f.write(await img2.read())
        elif img2_base64:
            img2_path = os.path.join("uploads", "img2_base64.png")
            with open(img2_path, "wb") as f:
                f.write(base64.b64decode(img2_base64))
        # Run DeepFace verification
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            distance_metric=dist,
            model_name=model,
            detector_backend=detector,
            enforce_detection=False,
        )

        # Delete uploaded images after processing
        os.remove(img1_path)
        os.remove(img2_path)

        # Return verification results
        return {
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "model": result["model"],
            "detector_backend": result["detector_backend"],
            "similarity_metric": result["similarity_metric"]
        }

    except Exception as e:
        logger.error(f"An error occurred during face verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Define Gradio Blocks
with gr.Blocks() as demo:
    img1 = gr.Image(label="Image 1",sources=["upload", "webcam", "clipboard"])
    img2 = gr.Image(label="Image 2",sources=["upload", "webcam", "clipboard"])
    dist = gr.Dropdown(choices=["cosine", "euclidean", "euclidean_l2"], label="Distance Metric", value="cosine")
    model = gr.Dropdown(choices=["VGG-Face", "Facenet", "Facenet512", "ArcFace"], label="Model", value="Facenet")
    detector = gr.Dropdown(choices=["opencv", "ssd", "mtcnn", "retinaface", "mediapipe"], label="Detector", value="ssd")
    btn = gr.Button("Verify")
    output = gr.Textbox()

    btn.click(face_verification_uii, inputs=[img1, img2, dist, model, detector], outputs=output)

# Running Servers
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)