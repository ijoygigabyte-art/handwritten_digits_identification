import base64
import io
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image, ImageOps
from neural_network import NeuralNetwork
from PIL import ImageFilter

app = FastAPI(title="NeuralNumber - Digit Recognizer")

# Load the trained model once at startup
model = NeuralNetwork.load("model.pkl")

# Serve the static frontend files (HTML/CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ImagePayload(BaseModel):
    image: str  # base64-encoded PNG from the canvas


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.post("/predict")
def predict(payload: ImagePayload):
    # 1. Decode the base64 image from the browser canvas
    header, data = payload.image.split(",", 1)
    image_bytes = base64.b64decode(data)

    # 2. Open image as RGBA to preserve alpha channel
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # 3. Composite onto a BLACK background.
    #    The canvas has a transparent bg (CSS makes it look black, but pixels are alpha=0).
    #    The user draws WHITE strokes. So we need: black bg + white strokes = MNIST format.
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    img = Image.alpha_composite(background, img).convert("L")

    

    # After compositing and before resizing:
    # Crop tight around the digit, then pad to center it like MNIST
    bbox = img.getbbox()   # Find the bounding box of non-black pixels
    if bbox:
        img = img.crop(bbox)
        # Pad to square with margin
        size = max(img.size)
        padded = Image.new("L", (size, size), 0)
        offset = ((size - img.size[0]) // 2, (size - img.size[1]) // 2)
        padded.paste(img, offset)
        img = padded

    img = img.resize((28, 28), Image.LANCZOS)

    # Optional: slight blur to smooth pixelation from scaling
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    

    # NO inversion needed: after compositing, we already have white digit on black bg (MNIST format)

    # 5. Convert to numpy, normalize, and flatten
    arr = np.array(img, dtype=np.float64) / 255.0           # Values in [0, 1]
    arr = arr.reshape(1, 784)                               # Shape: (1, 784)

    # 5. Run through the model
    probabilities = model.forward(arr)[0]                   # Shape: (10,)
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return {
        "digit": predicted_digit,
        "confidence": round(confidence * 100, 2),
        "probabilities": [round(float(p) * 100, 2) for p in probabilities]
    }
