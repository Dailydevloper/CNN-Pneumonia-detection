from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load model
print("LOADING MODEL FROM:", "model/pneumonia_model.keras")
model = tf.keras.models.load_model("model/pneumonia_model.keras")
print("MODEL LOADED")

# Templates
templates = Jinja2Templates(directory="templates")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")



def preprocess_image(image: Image.Image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize
    image = image.resize((224, 224))

    # Convert to numpy (NO scaling!)
    arr = np.array(image, dtype=np.float32)

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)

    # Apply SAME preprocessing as training
    arr = preprocess_input(arr)

    return arr



@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    img = preprocess_image(image)

    pred = model.predict(img, verbose=0)[0][0]


    if pred > 0.85:
        result = "PNEUMONIA DETECTED"
    else:
        result = "NORMAL"

    confidence = round(pred*100, 2)    

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": result, "confidence": confidence}
    )
