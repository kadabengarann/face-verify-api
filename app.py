from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import gradio as gr
from deepface import DeepFace
import os
from threading import Thread
import asyncio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# FastAPI instance
app = FastAPI()

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

# FastAPI Endpoint
@app.get("/", response_class=HTMLResponse)
async def gradio_ui():
    html_content = """
    <html>
        <head>
            <title>Gradio UI</title>
        </head>
        <body>
            <iframe src="http://0.0.0.0:7861" width="100%" height="100%" frameborder="0"></iframe>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/face_verification")
async def face_verification(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...),
    dist: str = Form("cosine"),
    model: str = Form("Facenet"),
    detector: str = Form("ssd")
):
    """
    Endpoint to verify if two images belong to the same person.
    """
    try:
        # Ensure uploads directory exists
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Save uploaded images to disk
        img1_path = os.path.join("uploads", img1.filename)
        img2_path = os.path.join("uploads", img2.filename)
        
        with open(img1_path, "wb") as f:
            f.write(await img1.read())
        with open(img2_path, "wb") as f:
            f.write(await img2.read())

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
        raise HTTPException(status_code=500, detail=str(e))


def run_gradio_ui():
    """
    Function to run Gradio in a separate thread
    """
    # Create and set an event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def face_verification_ui(img1, img2, dist, model, detector):
        result = face_verification_uii(img1, img2, dist, model, detector)
        return result

    with gr.Blocks() as demo:
        img1 = gr.Image(type="filepath", label="Image 1")
        img2 = gr.Image(type="filepath", label="Image 2")
        dist = gr.Dropdown(choices=["cosine", "euclidean", "euclidean_l2"], label="Distance Metric", value="cosine")
        model = gr.Dropdown(choices=["VGG-Face", "Facenet", "Facenet512", "ArcFace"], label="Model", value="Facenet")
        detector = gr.Dropdown(choices=["opencv", "ssd", "mtcnn", "retinaface", "mediapipe"], label="Detector", value="ssd")
        btn = gr.Button("Verify")
        output = gr.Textbox()

        btn.click(face_verification_ui, inputs=[img1, img2, dist, model, detector], outputs=output)
    demo.launch(server_name="0.0.0.0", server_port=7861, show_api=False)
    
# FastAPI Startup Event
# FastAPI Startup Event
@app.on_event("startup")
def startup_event():
    """
    Start Gradio UI in a separate thread
    """
    thread = Thread(target=run_gradio_ui)
    thread.start()  
# Running Both Servers
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=True)